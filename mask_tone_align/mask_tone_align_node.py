import cv2
import numpy as np
import torch


def _to_uint8(t):
    a = t[0].detach().cpu().numpy()
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def _to_tensor(u8):
    return torch.from_numpy(u8.astype(np.float32) / 255.0).unsqueeze(0)


def _mask_u8(t):
    m = t[0].detach().cpu().numpy()
    return (np.clip(m, 0, 1) * 255).astype(np.uint8)


def tone_align(img, ref, m_u8, erode_px=6, strength=1.0):
    """mask 内 img 的 LAB 三通道均值对齐到 ref 同 mask 区(=原图该区域肤色)。
    只做均值平移、不重设对比度——保留重绘自身的明暗结构(乳沟/下缘阴影合法存在)。"""
    mb = (m_u8 > 127).astype(np.uint8)
    if mb.max() == 0:
        return img.copy()
    inner = cv2.erode(mb, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_px * 2 + 1,) * 2)) > 0   # mask 内部(避开羽化边)
    if inner.sum() < 100:
        return img.copy()
    lab_i = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    for c in range(3):
        lab_i[..., c] += (lab_r[..., c][inner].mean() - lab_i[..., c][inner].mean()) * strength
    out = cv2.cvtColor(np.clip(lab_i, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    alpha = cv2.GaussianBlur(mb.astype(np.float32), (0, 0), 3)[..., None]
    return (out * alpha + img * (1 - alpha)).astype(np.uint8)


class MaskToneAlign:
    """局部色调对齐：把 mask 内重绘区的肤色/亮度对齐到 mask 外一圈原图，消除 i2i 接缝偏色。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "reference": ("IMAGE",),
            "mask": ("MASK",),
            "strength": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "inpaint"

    def run(self, image, reference, mask, strength=0.85):
        img = _to_uint8(image)
        ref = _to_uint8(reference)
        ref = cv2.resize(ref, (img.shape[1], img.shape[0])) if ref.shape[:2] != img.shape[:2] else ref
        m = _mask_u8(mask)
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) \
            if m.shape[:2] != img.shape[:2] else m
        return (_to_tensor(tone_align(img, ref, m, 6, strength)),)


NODE_CLASS_MAPPINGS = {"MaskToneAlign": MaskToneAlign}
NODE_DISPLAY_NAME_MAPPINGS = {"MaskToneAlign": "Mask Tone Align (ring match)"}
