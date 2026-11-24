# -*- coding: utf-8 -*-
"""VSR STTN-DET 修复 ComfyUI 节点——薄封装自家生产引擎 `STTNDetInpaint`。

接口 = 生产 `STTNDetInpaint(frames, mask)`：
  IMAGE batch[N,H,W,C] + MASK batch[N,H,W] -> IMAGE batch
区间语义与生产 sttn-det 相同：
  连续且 mask 逐像素相同的帧 = 一个修复区间（= main.py same-mask 分段），
  每区间用该 mask 经 STTNDetInpaint 时序修复；全黑 mask 帧原样透传。
  区间内部按 batch_generator(max_load) 分块（=生产 MaxLoadNum 50）。
配对节点：VSR_SubtitleDetect 输出（区间并集 mask）为本节点契约输入。
"""
import os
import sys
import numpy as np
import torch
import cv2

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

_INSTANCE_CACHE = {}


def _resolve_model_path(model_path: str) -> str:
    if os.path.isfile(model_path):
        return model_path
    cands = [
        os.path.join(_NODE_DIR, "models", "sttn-det", "sttn.pth"),
        os.path.join(_NODE_DIR, "models", "sttn", "sttn.pth"),
    ]
    try:
        import folder_paths
        for key in ("sttn", "inpaint"):
            for base in folder_paths.get_folder_paths(key):
                cands.append(os.path.join(base, "sttn.pth"))
        cands.append(os.path.join(folder_paths.models_dir, "sttn-det", "sttn.pth"))
    except Exception:
        pass
    cands.append("/root/ComfyUI/models/sttn-det/sttn.pth")
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def _get_engine(device_str: str, model_path: str):
    key = (device_str, model_path)
    eng = _INSTANCE_CACHE.get(key)
    if eng is None:
        from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
        dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
        eng = STTNDetInpaint(dev, model_path)
        _INSTANCE_CACHE[key] = eng
    return eng


def _same_mask_runs(mask_np: np.ndarray):
    """连续 mask 相同帧分组 -> [(start_idx, end_idx_inclusive, mask)]，0-based。"""
    runs = []
    n = mask_np.shape[0]
    i = 0
    while i < n:
        if not mask_np[i].any():
            i += 1
            continue
        j = i
        while j + 1 < n and np.array_equal(mask_np[j + 1], mask_np[i]):
            j += 1
        runs.append((i, j, mask_np[i]))
        i = j + 1
    return runs


class VSRSTTNInpaint:
    """STTN-DET 时序字幕修复（生产 VSR STTN 同款代码路径）。"""

    CATEGORY = "video/remove"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = ("frames + per-frame masks -> STTN 时序修复。"
                   "连续相同 mask=一个区间；全黑 mask 帧透传。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
            "model_path": ("STRING", {"default": "sttn.pth"}),
            "max_load": ("INT", {"default": 50, "min": 2, "max": 300,
                                 "tooltip": "单批帧数上限（=生产 Sttn/MaxLoadNum 50）"}),
        }}

    def run(self, images, masks, model_path, max_load):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        n = images.shape[0]
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] == 1 and n > 1:
            masks = masks.repeat(n, 1, 1)
        if masks.shape[0] != n:
            raise ValueError(f"mask 帧数 {masks.shape[0]} 与图像帧数 {n} 不一致")

        img_np = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_np = (masks.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        runs = _same_mask_runs(mask_np)
        if not runs:
            return (images.clamp(0, 1),)

        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        mpath = _resolve_model_path(model_path)
        eng = _get_engine(dev_str, mpath)

        from backend.tools.inpaint_tools import batch_generator

        out_np = img_np.copy()
        for s, e, m in runs:
            frames_bgr = [cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR)
                          for i in range(s, e + 1)]
            repaired = []
            for batch in batch_generator(frames_bgr, max_load):
                repaired.extend(eng(batch, m))
            for k, fr in enumerate(repaired):
                out_np[s + k] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        return (torch.from_numpy(out_np).float() / 255.0,)


NODE_CLASS_MAPPINGS = {"VSR_STTNInpaint": VSRSTTNInpaint}
NODE_DISPLAY_NAME_MAPPINGS = {"VSR_STTNInpaint": "VSR STTN 填充"}
