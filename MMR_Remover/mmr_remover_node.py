# -*- coding: utf-8 -*-
"""MMR_Remover — MiniMax-Remover 薄封装节点（IMAGE+MASK批 -> IMAGE批）。

设计同 VSR_ProPainter：检测/裁区/回贴/81帧切片归 Bridge，节点只做
"帧+逐帧mask -> 修复帧"。内部按官方规范处理：
  - num_frames 以 81 为一窗（Wan VAE 时序压缩 (N-1)/4+1，N=81 帧数守恒）；
    批长非 81 倍数时尾窗复制末帧补齐，输出裁回原长。
  - 官方参数 num_inference_steps=12（6步蒸馏亦可）、iterations=6（mask膨胀轮数）。
  - images [-1,1] RGB、masks [0,1] 白=移除——与 ComfyUI IMAGE/MASK 语义天然一致。
模型目录需含 transformer/ vae/ scheduler 三个子目录。
"""
import os
import sys

import numpy as np
import torch

_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
_MMR_SRC = os.path.join(_NODE_DIR, "minimax_remover")
if _MMR_SRC not in sys.path:
    sys.path.insert(0, _MMR_SRC)

_WINDOW = 81  # Wan VAE 帧数守恒对齐值


def _resolve_model_dir(model_dir: str) -> str:
    if os.path.isdir(model_dir):
        return model_dir
    try:
        import folder_paths
        cand = os.path.join(folder_paths.models_dir, "minimax_remover")
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    for cand in ("/root/ComfyUI/models/minimax_remover",
                 os.path.join(_NODE_DIR, "models", "minimax_remover"),
                 os.path.join(os.path.dirname(_NODE_DIR), "_models_store", "minimax_remover")):
        if os.path.isdir(cand):
            return cand
    return os.path.join(_NODE_DIR, "models", "minimax_remover")


_PIPE_CACHE = {}


def _get_pipe(model_dir: str):
    pipe = _PIPE_CACHE.get(model_dir)
    if pipe is not None:
        return pipe
    for sub in ("transformer", "vae", "scheduler"):
        if not os.path.isdir(os.path.join(model_dir, sub)):
            raise FileNotFoundError(
                f"MMR_Remover 模型目录缺 {sub}/：{model_dir}（需 transformer/ vae/ scheduler）")
    from diffusers.models import AutoencoderKLWan
    from diffusers.schedulers import UniPCMultistepScheduler
    from transformer_minimax_remover import Transformer3DModel
    from pipeline_minimax_remover import Minimax_Remover_Pipeline
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_dir, "vae"), torch_dtype=torch.float16)
    tr = Transformer3DModel.from_pretrained(
        os.path.join(model_dir, "transformer"), torch_dtype=torch.float16)
    sch = UniPCMultistepScheduler.from_pretrained(os.path.join(model_dir, "scheduler"))
    pipe = Minimax_Remover_Pipeline(transformer=tr, vae=vae, scheduler=sch).to("cuda:0")
    _PIPE_CACHE[model_dir] = pipe
    return pipe


def _round16(x: int) -> int:
    return max(16, int(round(x / 16.0)) * 16)


class MMRRemover:
    """MiniMax-Remover：IMAGE[B,H,W,C]+MASK[B,H,W] -> IMAGE[B,H,W,C]"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            },
            "optional": {
                "model_dir": ("STRING", {"default": "minimax_remover"}),
                "num_inference_steps": ("INT", {"default": 12, "min": 4, "max": 50}),
                "iterations": ("INT", {"default": 6, "min": 0, "max": 30}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "video/remove"

    @torch.no_grad()
    def run(self, images, masks, model_dir="minimax_remover",
            num_inference_steps=12, iterations=6, seed=42):
        mdir = _resolve_model_dir(model_dir)
        pipe = _get_pipe(mdir)

        # IMAGE[B,H,W,C] float[0,1] RGB -> tensor f h w c in [-1,1]
        imgs = images.detach().to("cpu", torch.float32) * 2.0 - 1.0
        n, h0, w0, _ = imgs.shape
        # MASK[B,H,W] -> f h w 1 in [0,1]
        mks = masks.detach().to("cpu", torch.float32)
        if mks.dim() == 3:
            mks = mks[..., None]
        if mks.shape[0] != n:  # mask 批与图像批不等长：广播/截断对齐
            if mks.shape[0] == 1:
                mks = mks.expand(n, -1, -1, -1)
            else:
                k = min(n, mks.shape[0])
                mks = mks[:k]
                imgs = imgs[:k]
                n = k
        H, W = _round16(h0), _round16(w0)

        outs = []
        for s in range(0, n, _WINDOW):
            e = min(s + _WINDOW, n)
            img_w, mk_w = imgs[s:e], mks[s:e]
            pad = _WINDOW - (e - s)  # 尾窗补末帧保 81 对齐
            if pad > 0:
                img_w = torch.cat([img_w, img_w[-1:].expand(pad, -1, -1, -1)], 0)
                mk_w = torch.cat([mk_w, mk_w[-1:].expand(pad, -1, -1, -1)], 0)
            res = pipe(images=img_w, masks=mk_w, num_frames=_WINDOW,
                       height=H, width=W,
                       num_inference_steps=num_inference_steps,
                       iterations=iterations,
                       generator=torch.Generator(device="cuda:0").manual_seed(seed)
                       ).frames[0]
            arr = np.asarray(res, dtype=np.float32)  # f h w c in [0,1]
            outs.append(torch.from_numpy(arr[: e - s]))
        out = torch.cat(outs, 0)
        if (H, W) != (h0, w0):  # 内部分辨率被16对齐时，重采样回原尺寸
            out = torch.nn.functional.interpolate(
                out.permute(0, 3, 1, 2), size=(h0, w0),
                mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        return (out.clamp(0.0, 1.0),)


NODE_CLASS_MAPPINGS = {"MMR_Remover": MMRRemover}
NODE_DISPLAY_NAME_MAPPINGS = {"MMR_Remover": "MMR 视频移除（MiniMax-Remover）"}
