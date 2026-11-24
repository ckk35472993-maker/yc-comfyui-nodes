"""VLObjects_Layerize：逐对象 mask 过滤 + 层清单 manifest 产出。

解决 img_vl_objects 实测的两个问题（L2 汽车图 17 名词 → 29 实例）：

1) 层数爆炸：卡钳/反光镜等小碎片各成一层，画布不可用。
   - min_area_ratio：按 mask 面积占比过滤小碎片
   - max_layers：上限截断（上游 seg 已按 score 降序，截断保留高分检测）

2) 层↔名词对齐：消费 easy-sam3 fork 新增的 labels 输出
   （逐检测所属 prompt 词，与 obj_masks/boxes/scores 同序），
   按 keep 索引同序下采样 → manifest 每层带确定性 label。

manifest（layers_*.json，经 VLObjects_SaveText ext=json 回传 text_content）：
{
  "version": 1,
  "image": {"w": 1024, "h": 576},
  "detected": 29, "kept": 14,
  "vocab": ["car", "wheel", ...],
  "layers": [{"index":0, "src_index":0, "label":"car", "score":0.91,
              "area":0.312, "bbox":{"x":0.05,"y":0.2,"w":0.7,"h":0.6}}]
}
bbox 为归一化坐标（0-1，相对缩放后图像），前端乘画布尺寸还原。
"""

import json

import torch


def _flat_scalars(v):
    """scores/labels：easy-sam3 输出为 per-image 列表 [[..]]，也兼容平铺/JSON 串。"""
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
        except Exception:
            # 逗号串兜底（labels 被改接原始名词串的场景）
            return [p.strip() for p in s.split(",") if p.strip()]
    if not isinstance(v, (list, tuple)) or not v:
        return []
    if isinstance(v[0], (list, tuple)):
        return list(v[0])
    return list(v)


def _flat_rows(v):
    """boxes：[[x1,y1,x2,y2],...] 或 per-image 包裹 [[[..]]] / JSON 串。"""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return []
    if not isinstance(v, (list, tuple)) or not v:
        return []
    if isinstance(v[0], (list, tuple)) and v[0] and isinstance(v[0][0], (list, tuple)):
        return list(v[0])
    return list(v)


def _mask_bbox_norm(mask_hw):
    """单张 (H,W) mask → 非零区归一化 bbox {x,y,w,h}；空 mask 返回 None。"""
    ys, xs = torch.nonzero(mask_hw > 0.5, as_tuple=True)
    if ys.numel() == 0:
        return None
    h, w = mask_hw.shape
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {
        "x": round(x0 / w, 5),
        "y": round(y0 / h, 5),
        "w": round((x1 - x0 + 1) / w, 5),
        "h": round((y1 - y0 + 1) / h, 5),
    }


class VLObjects_Layerize:
    """逐对象 mask 集 → 过滤后 mask batch + JSON manifest。"""

    RETURN_TYPES = ("MASK", "STRING", "INT")
    RETURN_NAMES = ("kept_masks", "manifest", "kept_count")
    FUNCTION = "layerize"
    CATEGORY = "AILab/vl_objects"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "image": ("IMAGE",),
                "min_area_ratio": ("FLOAT", {
                    "default": 0.004, "min": 0.0, "max": 0.5, "step": 0.001,
                    "tooltip": "mask 面积占比下限，小于该值的碎片层被过滤",
                }),
                "max_layers": ("INT", {
                    "default": 16, "min": 1, "max": 100, "step": 1,
                    "tooltip": "层数上限（seg 输出已按 score 降序，截断保高分）",
                }),
            },
            "optional": {
                "boxes": ("BBOX", {"forceInput": True}),
                "scores": ("FLOAT", {"forceInput": True}),
                "labels": ("STRING", {"forceInput": True}),
            },
        }

    def layerize(self, masks, image, min_area_ratio=0.004, max_layers=16,
                 boxes=None, scores=None, labels=None):
        m = masks.detach().float().cpu()
        # 实测 easy-sam3 obj_masks 以 4D (1,N,H,W) 送达；统一压成 (N,H,W)
        while m.dim() > 3:
            m = m.reshape(-1, m.shape[-2], m.shape[-1])
        if m.dim() == 2:
            m = m.unsqueeze(0)
        n, mh, mw = m.shape

        # image 为 (B,H,W,C)；bbox 归一化基于 mask 自身尺寸，与 image 同尺度（同接 src_s）
        img_h = int(image.shape[1]) if image is not None and image.dim() >= 3 else mh
        img_w = int(image.shape[2]) if image is not None and image.dim() >= 3 else mw

        labels_l = _flat_scalars(labels)
        scores_l = _flat_scalars(scores)
        boxes_l = _flat_rows(boxes)

        areas = (m.reshape(n, -1).sum(dim=1) / float(mh * mw))

        keep = [i for i in range(n) if float(areas[i]) >= float(min_area_ratio)]
        if not keep and n > 0:
            # 全被过滤时保底留最大一块，避免下游 SaveImage 拿到空 batch
            keep = [int(torch.argmax(areas))]
        keep = keep[: max(1, int(max_layers))]

        kept = m[keep]

        vocab = []
        layers = []
        for out_i, src_i in enumerate(keep):
            label = str(labels_l[src_i]) if src_i < len(labels_l) else ""
            if label and label not in vocab:
                vocab.append(label)
            score = float(scores_l[src_i]) if src_i < len(scores_l) else None
            bbox = _mask_bbox_norm(kept[out_i])
            if bbox is None and src_i < len(boxes_l):
                b = boxes_l[src_i]
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    x0, y0, x1, y1 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
                    bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
                    # sam3 box 为像素坐标 → 归一化
                    if bw > 2 or bh > 2:
                        x0, y0, bw, bh = x0 / img_w, y0 / img_h, bw / img_w, bh / img_h
                    bbox = {"x": round(x0, 5), "y": round(y0, 5),
                            "w": round(bw, 5), "h": round(bh, 5)}
            layers.append({
                "index": out_i,
                "src_index": src_i,
                "label": label,
                "score": round(score, 4) if score is not None else None,
                "area": round(float(areas[src_i]), 5),
                "bbox": bbox,
            })

        manifest = json.dumps({
            "version": 1,
            "image": {"w": img_w, "h": img_h},
            "detected": n,
            "kept": len(keep),
            "vocab": vocab,
            "layers": layers,
        }, ensure_ascii=False)

        return (kept, manifest, len(keep))


NODE_CLASS_MAPPINGS = {"VLObjects_Layerize": VLObjects_Layerize}
NODE_DISPLAY_NAME_MAPPINGS = {"VLObjects_Layerize": "VL Objects Layerize"}
