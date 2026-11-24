# -*- coding: utf-8 -*-
"""VSR 裁区/贴回 ComfyUI 节点——薄封装生产几何 `local_remove_box_utils`。

`VSR_ContextCrop`：frames(+masks)+bbox -> 条带 frames/masks + 几何 JSON
  strategy=watermark_context → 调生产 `compute_watermark_context_crop`
    （0.5半屏上下文、贴边扩展、8对齐、384×256地板——与 Bridge 裁区逐字节同源）
  strategy=box_expand8      → 框±expand_px后8对齐（=生产 region 条带）
`VSR_PasteBack`：原帧 + 条带 + 几何 JSON -> 全帧（与 Bridge overlay 同语义，
  整贴条带；修复引擎本身保证条带非 mask 区原样）

mask 与图像共用同一矩形——F 策略 mask-video 裁对齐的节点等价物。
"""
import os
import sys
import json
import numpy as np
import torch

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from local_remove_box_utils import compute_watermark_context_crop  # noqa: E402
from local_remove_subtitle_ocr_smart import align_to_8x  # noqa: E402


def _parse_bbox(s: str):
    x1, y1, x2, y2 = [int(float(v)) for v in s.split(",")]
    return x1, y1, x2, y2


def _crop_t(t, x, y, w, h):
    return t[:, y:y + h, x:x + w, ...]


class VSRContextCrop:
    """上下文裁区（生产 Bridge 裁区同款几何；mask 同步裁）。"""

    CATEGORY = "video/remove"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("crop_images", "crop_masks", "geom")
    DESCRIPTION = ("frames(+masks)+bbox -> 条带。watermark_context=生产半屏"
                   "上下文裁区；box_expand8=框扩展8对齐条带。geom JSON 喂 PasteBack。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "bbox": ("STRING", {"default": "", "tooltip": "x1,y1,x2,y2"}),
            "strategy": (["watermark_context", "box_expand8"], {"default": "watermark_context"}),
            "expand_px": ("INT", {"default": 10, "min": 0, "max": 120,
                                  "tooltip": "box_expand8 模式的框外扩（=生产 CROP_DEVIATION_PX）"}),
        }, "optional": {
            "masks": ("MASK",),
        }}

    def run(self, images, bbox, strategy, expand_px, masks=None):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        n, h, w = images.shape[0], images.shape[1], images.shape[2]
        x1, y1, x2, y2 = _parse_bbox(bbox)

        if strategy == "watermark_context":
            cx, cy, cw, ch = compute_watermark_context_crop(x1, y1, x2, y2, w, h)
        else:
            e = align_to_8x(max(0, x1 - expand_px), max(0, y1 - expand_px),
                            min(w, x2 + expand_px), min(h, y2 + expand_px), w, h)
            cx, cy, cw, ch = e["x"], e["y"], e["w"], e["h"]

        crop_imgs = _crop_t(images, cx, cy, cw, ch)
        if masks is None:
            crop_masks = torch.zeros((n, ch, cw), dtype=torch.float32)
        else:
            if masks.ndim == 2:
                masks = masks.unsqueeze(0)
            if masks.shape[0] == 1 and n > 1:
                masks = masks.repeat(n, 1, 1)
            if masks.shape[0] != n:
                raise ValueError(f"mask 帧数 {masks.shape[0]} 与图像帧数 {n} 不一致")
            crop_masks = _crop_t(masks, cx, cy, cw, ch)

        geom = json.dumps({"x": cx, "y": cy, "w": cw, "h": ch,
                           "frame_w": w, "frame_h": h, "frames": n,
                           "strategy": strategy})
        return (crop_imgs, crop_masks, geom)


class VSRPasteBack:
    """条带贴回原帧（与 Bridge overlay 同语义：整贴条带回原坐标）。"""

    CATEGORY = "video/remove"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = ("orig frames + crop frames + geom -> 全帧。"
                   "条带写回 (x,y,w,h)；其余像素原样保留。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "crop_images": ("IMAGE",),
            "geom": ("STRING", {"default": ""}),
        }}

    def run(self, images, crop_images, geom):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if crop_images.ndim == 3:
            crop_images = crop_images.unsqueeze(0)
        g = json.loads(geom)
        x, y, w, h = g["x"], g["y"], g["w"], g["h"]
        n = images.shape[0]
        if crop_images.shape[0] == 1 and n > 1:
            crop_images = crop_images.repeat(n, 1, 1, 1)
        if crop_images.shape[0] != n:
            raise ValueError(f"条带帧数 {crop_images.shape[0]} 与原帧 {n} 不一致")
        if crop_images.shape[1] != h or crop_images.shape[2] != w:
            raise ValueError(f"条带尺寸 {crop_images.shape[2]}x{crop_images.shape[1]}"
                             f" 与 geom {w}x{h} 不一致")

        out = images.clone()
        out[:, y:y + h, x:x + w, :] = crop_images
        return (out,)


NODE_CLASS_MAPPINGS = {
    "VSR_ContextCrop": VSRContextCrop,
    "VSR_PasteBack": VSRPasteBack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VSR_ContextCrop": "VSR 上下文裁区",
    "VSR_PasteBack": "VSR 贴回原帧",
}
