# -*- coding: utf-8 -*-
"""VSR ProPainter ComfyUI 节点——薄封装自家生产引擎。

接口 = 生产 `PropainterInpaint.__call__(frames_list, per_frame_masks)`：
  IMAGE batch[N,H,W,C](RGB 0~1) + MASK batch[N,H,W](0~1) -> IMAGE batch
检测/裁区/overlay 回贴由 Bridge 层负责（循环在外），本节点只做时序填充核。
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


def _resolve_model_dir(model_dir: str) -> str:
    if os.path.isdir(model_dir):
        return model_dir
    try:
        import folder_paths
        for base in folder_paths.get_folder_paths("propainter"):
            if os.path.isdir(base):
                return base
        cand = os.path.join(folder_paths.models_dir, "propainter")
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    for cand in ("/root/ComfyUI/models/propainter",
                 os.path.join(_NODE_DIR, "models", "propainter")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(_NODE_DIR, "models", "propainter")


def _get_engine(device_str, model_dir, sub_video_length, mask_dilation, fp16):
    key = (device_str, model_dir, sub_video_length, mask_dilation, fp16)
    eng = _INSTANCE_CACHE.get(key)
    if eng is None:
        os.environ["VSR_MASK_DILATION"] = str(mask_dilation)
        from backend.inpaint.propainter_inpaint import PropainterInpaint
        dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
        eng = PropainterInpaint(dev, model_dir,
                                sub_video_length=sub_video_length,
                                use_fp16=fp16)
        _INSTANCE_CACHE[key] = eng
    return eng


class VSRProPainter:
    """逐帧 mask 动态水印/字幕填充（生产 VSR ProPainter 同款代码路径）。"""

    CATEGORY = "video/remove"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    DESCRIPTION = ("frames + per-frame masks -> ProPainter 时序填充。"
                   "未遮罩帧原样透传；mask=1 的像素区域被重建。")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "masks": ("MASK",),
            "model_dir": ("STRING", {"default": "propainter"}),
            "sub_video_length": ("INT", {"default": 50, "min": 2, "max": 90,
                                       "tooltip": "单批帧数上限（=生产 VSR_PROP_MAX_LOAD，≤16G 卡建议 ≤50）"}),
            "mask_dilation": ("INT", {"default": 4, "min": 0, "max": 32}),
            "fp16": ("BOOLEAN", {"default": True}),
        }}

    def run(self, images, masks, model_dir, sub_video_length, mask_dilation, fp16):
        if images.ndim == 3:
            images = images.unsqueeze(0)
        n = images.shape[0]
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] == 1 and n > 1:
            masks = masks.repeat(n, 1, 1)
        if masks.shape[0] != n:
            raise ValueError(f"mask 帧数 {masks.shape[0]} 与图像帧数 {n} 不一致")

        # RAFT/ProPainter 要求 H/W 均被 8 整除；非整除时右/下 pad，输出再裁回原尺寸
        # （960x540 等常见片源否则会 grid_sampler batch 错配，见 README）
        h0, w0 = int(images.shape[1]), int(images.shape[2])
        h8 = ((h0 + 7) // 8) * 8
        w8 = ((w0 + 7) // 8) * 8
        if h8 != h0 or w8 != w0:
            # BHWC: pad=(C_l,C_r, W_l,W_r, H_l,H_r)；BHW mask: (W_l,W_r, H_l,H_r)
            images = torch.nn.functional.pad(images, (0, 0, 0, w8 - w0, 0, h8 - h0))
            masks = torch.nn.functional.pad(masks, (0, w8 - w0, 0, h8 - h0))

        dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        mdir = _resolve_model_dir(model_dir)
        eng = _get_engine(dev_str, mdir, sub_video_length, mask_dilation, fp16)

        img_np = (images.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frames_bgr = [cv2.cvtColor(img_np[i], cv2.COLOR_RGB2BGR) for i in range(n)]
        mask_np = (masks.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        mask_list = [mask_np[i] for i in range(n)]

        out_frames = eng.inpaint(frames_bgr, mask_list)
        out_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in out_frames])
        if h8 != h0 or w8 != w0:
            out_np = out_np[:, :h0, :w0, :]
        return (torch.from_numpy(out_np).float() / 255.0,)


NODE_CLASS_MAPPINGS = {"VSR_ProPainter": VSRProPainter}
NODE_DISPLAY_NAME_MAPPINGS = {"VSR_ProPainter": "VSR ProPainter 填充"}
