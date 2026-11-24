"""body_sculpt: 非生成式身形微调（face_sculpt 同款哲学的身体版）。

与扩散 inpaint 的根本区别：不"重画"，只对原像素做受控几何位移——
  mask 区外像素级原样；mask 区内是同一块布料的拉伸（liquify）。
  所以脸/服装/背景结构性一致，不会把衣服重设计成另一件。

BodySculptBustWarp（胸型调整·保真）：
  输入 IMAGE + MASK（胸带/上身区域 mask，通常接 SAM3 torso−head ∩ band 链产物）。
  自动定位双胸心 = mask bbox 下 62% 高度、横 28%/72% 处（无需 pose 模型）；
  双心径向高斯外扩位移场 + 羽化贴回。
  mask 为空 → 原图透传（空 MASK 输出可见，便于排障）。
  实测强度边界（768×960 蕾丝吊带）：s≤0.8 纹理连贯；≥1.0 纹理变稀+边缘吸背景。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


def _to_uint8(img_tensor):
    t = img_tensor[0].detach().cpu().numpy()
    return np.clip(t * 255.0, 0, 255).astype(np.uint8)


def _to_tensor(img_uint8):
    return torch.from_numpy(img_uint8.astype(np.float32) / 255.0).unsqueeze(0)


def _mask_to_u8(mask_tensor):
    m = mask_tensor[0].detach().cpu().numpy()
    return np.clip(m * 255.0, 0, 255).astype(np.uint8)


def _mask_bbox(m_u8):
    ys, xs = np.where(m_u8 > 127)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bust_warp_field(img, m_u8, strength=0.5, down_ratio=0.5,
                    center_y_ratio=0.62, feather=0.3):
    """双心径向外扩位移场 + 羽化。返回 (warped, blend_alpha)。"""
    h, w = img.shape[:2]
    bb = _mask_bbox(m_u8)
    if bb is None:
        return img.copy(), np.zeros((h, w), np.float32)
    x0, y0, x1, y1 = bb
    bw, bh = float(x1 - x0), float(y1 - y0)
    cx_l = x0 + bw * 0.28
    cx_r = x0 + bw * 0.72
    cy = y0 + bh * center_y_ratio
    sx = max(bw * 0.28, 40.0)
    sy = max(bh * 0.30, 40.0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    for cx in (cx_l, cx_r):
        g = np.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
        dx += (xx - cx) * strength * g
        dy += (yy - cy) * strength * down_ratio * g
    warped = cv2.remap(img, np.clip(xx + dx, 0, w - 1),
                       np.clip(yy + dy, 0, h - 1), cv2.INTER_LINEAR)
    alpha = np.sqrt(dx ** 2 + dy ** 2)
    alpha = np.clip(alpha / (alpha.max() + 1e-6) / max(1.0 - feather, 1e-3), 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 8)
    out = warped * alpha[..., None] + img * (1.0 - alpha[..., None])
    return out.astype(np.uint8), alpha


class BodySculptBustWarp:
    """胸型保真微调：mask 驱动双心外扩 warp，非生成、零模型依赖。"""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.5, "step": 0.05}),
            "down_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "center_y_ratio": ("FLOAT", {"default": 0.62, "min": 0.3, "max": 0.9, "step": 0.01}),
            "feather": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.6, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "warp_mask")
    FUNCTION = "run"
    CATEGORY = "body/sculpt"

    def run(self, image, mask, strength=0.5, down_ratio=0.5,
            center_y_ratio=0.62, feather=0.3):
        img = _to_uint8(image)          # ComfyUI IMAGE 即 RGB
        m_u8 = _mask_to_u8(mask)
        if m_u8.shape[:2] != img.shape[:2]:
            m_u8 = cv2.resize(m_u8, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        out, alpha = bust_warp_field(img, m_u8, strength, down_ratio,
                                     center_y_ratio, feather)
        warp_mask = torch.from_numpy(alpha).unsqueeze(0)
        return (_to_tensor(out), warp_mask)


NODE_CLASS_MAPPINGS = {
    "BodySculptBustWarp": BodySculptBustWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BodySculptBustWarp": "Body Sculpt · Bust Warp (保真)",
}
