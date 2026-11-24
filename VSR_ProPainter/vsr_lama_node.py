# -*- coding: utf-8 -*-
"""VSR LaMa ComfyUI 节点——薄封装自家生产引擎 `LamaInpaint`。

接口 = 生产 `LamaInpaint.inpaint(image, mask)`：
  IMAGE batch[N,H,W,C](RGB 0~1) + MASK batch[N,H,W](0~1, 1=待修复) -> IMAGE batch
语义与 VSR 生产链逐字节同源：
  - 空 mask 帧原样透传（LamaInpaint 内部 mask sum==0 直接返回原图）
  - >2048px 大图自动窗口化修复+贴回（防 OOM/索引越界，LAMA_MAX_INPAINT_SIDE）
与社区 AILab_LamaRemover 的区别：无额外阈值/高斯模糊重处理 mask——
mask 按上游（检测/用户框）原样消费，与本地 `inpaint_image_lama` 一致。
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
    """big-lama.pt 解析序：绝对路径 > models/lama/(包内) > ComfyUI models/inpaint > models/RMBG。"""
    if os.path.isfile(model_path):
        return model_path
    cands = [
        os.path.join(_NODE_DIR, "models", "lama", "big-lama.pt"),
    ]
    try:
        import folder_paths
        for key in ("inpaint", "RMBG", "lama"):
            for base in folder_paths.get_folder_paths(key):
                cands.append(os.path.join(base, "big-lama.pt"))
        cands.append(os.path.join(folder_paths.models_dir, "inpaint", "big-lama.pt"))
        cands.append(os.path.join(folder_paths.models_dir, "RMBG", "Lama", "big-lama.pt"))
    except Exception:
        pass
    cands += [
        "/root/ComfyUI/models/inpaint/big-lama.pt",
        "/root/ComfyUI/models/RMBG/Lama/big-lama.pt",
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return cands[0]


def _get_engine(device_str: str, model_path: str):
    key = (device_str, model_path)
    eng = _INSTANCE_CACHE.get(key)
    if eng is None:
        from backend.inpaint.lama_inpaint import LamaInpaint
        dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
        eng = LamaInpaint(dev, model_path)
        _INSTANCE_CACHE[key] = eng
    return eng


class VSRLaMa:
    """LaMa 单帧/图片修复（生产 VSR LaMa 同款代码路径，含大图窗口保护）。"""

    CATEGORY = "video/remove"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = ("images + masks -> big-lama 修复。mask=1 区域被重建；"
                   "空 mask 帧原样透传；>2048px 自动窗口化防 OOM。"
                   "模型 = 生产同款 big-lama.pt（JIT）。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
            "model_path": ("STRING", {"default": "big-lama.pt",
                                      "tooltip": "留空/文件名走包内与 ComfyUI 标准目录解析"}),
        }}

    def run(self, images, masks, model_path):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        n = images.shape[0]
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] == 1 and n > 1:
            masks = masks.repeat(n, 1, 1)
        if masks.shape[0] != n:
            raise ValueError(f"mask 帧数 {masks.shape[0]} 与图像帧数 {n} 不一致")

        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        mpath = _resolve_model_path(model_path)
        eng = _get_engine(dev_str, mpath)

        img_np = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_np = (masks.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        out_frames = []
        for i in range(n):
            frame_bgr = cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR)
            out = eng.inpaint(frame_bgr, mask_np[i])
            out_frames.append(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        out_np = np.stack(out_frames)
        return (torch.from_numpy(out_np).float() / 255.0,)


NODE_CLASS_MAPPINGS = {"VSR_LaMa": VSRLaMa}
NODE_DISPLAY_NAME_MAPPINGS = {"VSR_LaMa": "VSR LaMa 填充"}
