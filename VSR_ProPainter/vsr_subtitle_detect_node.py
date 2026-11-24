# -*- coding: utf-8 -*-
"""VSR 字幕检测 ComfyUI 节点——薄封装自家生产引擎 `SubtitleDetect`。

复刻 VSR main.py 的 sttn-det 检测链，逐段同源：
  IMAGE batch[N,H,W,C] -> 临时 mp4 -> SubtitleDetect(采样OCR+框并集填充+区间mask加宽)
  -> find_continuous_ranges_with_same_mask
  -> expand_frame_ranges(backward, forward)   # 默认3/3 = 生产配置
  -> filter_and_merge_intervals(ref_length)   # 默认10 = 生产配置
  -> start_end_map(首帧>=1钳位)               # 含片头负起点修复补丁
  -> 区间并集框 + subtitleYXAxisDifferencePixel 过滤 -> create_mask -> MASK batch

输出：
  masks  = MASK batch（区间内每帧=区间并集mask，区间外=全黑）——可直接喂
           VSR_STTNInpaint / VSR_ProPainter / MMR_Remover。
  info   = JSON 字符串：检测区间/白像素统计/provider，供审计。

检测 provider：默认 auto = OpenVINO PP-OCRv4 det（VSR_OV_SRC/VSR_OCR_DET_MODEL
指向包内模型；不可用且 provider=paddle 时回退 PaddleOCR）。
"""
import os
import sys
import json
import tempfile
import numpy as np
import torch
import cv2

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

# OV 资源默认指向包内模型（可用 env 覆盖；与生产 vsr_runner 同名变量）
# VSR_OV_SRC = 含 ppocrv4_ov 包的目录（subtitle_detect 内部 sys.path.insert 后
# `from ppocrv4_ov.predict_det import TextDetector`）——不是包目录本身。
os.environ.setdefault("VSR_OV_SRC", _NODE_DIR)
os.environ.setdefault(
    "VSR_OCR_DET_MODEL",
    os.path.join(_NODE_DIR, "models", "ocr", "inference.pdmodel"))


class _RemoverStub:
    """find_subtitle_frame_no 需要 sub_remover 接口的最小桩。"""

    def __init__(self):
        self.ab_sections = []
        self.progress_total = 0.0

    def append_output(self, *_a, **_k):
        pass


def _write_frames_to_mp4(frames_rgb: np.ndarray, fps: float, path: str):
    """帧 -> 临时 mp4。libx264 不可用则 mp4v（仅检测用，不产出给用户）。"""
    h, w = frames_rgb.shape[1:3]
    for fourcc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if vw.isOpened():
            break
        vw.release()
    if not vw.isOpened():
        raise RuntimeError("临时检测视频写入失败：无可用编码器")
    for i in range(frames_rgb.shape[0]):
        vw.write(cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR))
    vw.release()


class VSRSubtitleDetect:
    """字幕 OCR 检测 -> 逐帧 MASK（生产 sttn-det 检测侧同款代码路径）。"""

    CATEGORY = "video/remove"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("masks", "info")
    DESCRIPTION = ("frames -> OCR 字幕检测 -> 逐帧并集 MASK。"
                   "区间逻辑与生产 sttn-det 相同（采样/填充/加宽/扩展/合并/首帧钳位）。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 240.0,
                             "tooltip": "原视频帧率（决定检测采样步长，须与源一致）"}),
            "sub_area": ("STRING", {"default": "",
                                    "tooltip": "可选字幕区 x1,y1,x2,y2；留空=全帧检测"}),
            "lang": (["ch", "en", "japan", "korea"], {"default": "ch"}),
            "det_provider": (["auto", "ov", "paddle"], {"default": "auto"}),
            "expand_backward": ("INT", {"default": 3, "min": 0, "max": 60,
                                        "tooltip": "区间向前扩帧（=生产 SubtitleTimelineBackwardFrameCount）"}),
            "expand_forward": ("INT", {"default": 3, "min": 0, "max": 60,
                                       "tooltip": "区间向后扩帧（=生产 subtitleTimelineForwardFrameCount）"}),
            "ref_length": ("INT", {"default": 10, "min": 1, "max": 100,
                                   "tooltip": "区间合并参考长（=生产 sttnReferenceLength）"}),
        }}

    def run(self, images, fps, sub_area, lang, det_provider,
            expand_backward, expand_forward, ref_length):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        n, h, w = images.shape[0], images.shape[1], images.shape[2]

        sub_areas = None
        if sub_area.strip():
            x1, y1, x2, y2 = [int(float(v)) for v in sub_area.split(",")]
            # 生产约定 sub_areas=(ymin,ymax,xmin,xmax)
            sub_areas = [(y1, y2, x1, x2)]

        if det_provider != "auto":
            os.environ["VSR_DET_PROVIDER"] = det_provider

        img_np = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        from backend.tools.subtitle_detect import SubtitleDetect
        from backend.tools.inpaint_tools import create_mask, expand_frame_ranges
        from backend.config import config

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            tmp_path = tf.name
        try:
            _write_frames_to_mp4(img_np, fps, tmp_path)
            det = SubtitleDetect(tmp_path, sub_areas, lang=lang)
            sub_list = det.find_subtitle_frame_no(_RemoverStub())
            if not sub_list:
                empty = torch.zeros((n, h, w), dtype=torch.float32)
                return (empty, json.dumps({"intervals": [], "provider": "none",
                                           "note": "no subtitle detected"}))
            continuous = det.find_continuous_ranges_with_same_mask(sub_list)
            continuous = expand_frame_ranges(continuous, expand_backward, expand_forward)
            continuous = det.filter_and_merge_intervals(continuous, ref_length)
            # main.py start_end_map：首帧>=1钳位 + 坍缩同键取更大 end
            start_end_map = {}
            for s, e in continuous:
                key = max(1, s)
                new_end = min(e, n)
                start_end_map[key] = max(new_end, start_end_map.get(key, new_end))
            del det

            masks_np = np.zeros((n, h, w), dtype=np.uint8)
            intervals_info = []
            for s, e in sorted(start_end_map.items()):
                coords = []
                for fidx in range(s, e):  # 与 main.py 相同：e 为开区间尾
                    if fidx in sub_list:
                        for area in sub_list[fidx]:
                            xmin, xmax, ymin, ymax = area
                            if (ymax - ymin) - (xmax - xmin) > \
                                    config.subtitleYXAxisDifferencePixel.value:
                                continue
                            if area not in coords:
                                coords.append(area)
                mask = create_mask((h, w), coords)
                for fidx in range(s - 1, min(e, n)):  # 1-based -> 0-based
                    masks_np[fidx] = mask
                intervals_info.append({"start": s, "end": e,
                                       "white_px": int((mask > 0).sum())})
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        info = {"intervals": intervals_info,
                "provider": os.environ.get("VSR_DET_PROVIDER", "auto"),
                "frames": n}
        return (torch.from_numpy(masks_np).float() / 255.0, json.dumps(info))


NODE_CLASS_MAPPINGS = {"VSR_SubtitleDetect": VSRSubtitleDetect}
NODE_DISPLAY_NAME_MAPPINGS = {"VSR_SubtitleDetect": "VSR 字幕检测"}
