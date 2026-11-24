# -*- coding: utf-8 -*-
"""ComfyUI 节点包内 shim：仅含 local_remove_box_utils 需要的两个几何函数，
逐字复制自 video_ai_core/local/local_remove_subtitle_ocr_smart.py（原文件
含 OCR/cv2 重依赖，节点包不需要）。改动须与源文件保持同步。"""
from __future__ import annotations

import logging
import math
import os
from typing import Dict, Optional

logger = logging.getLogger("video_ai_core.local_remove_subtitle_ocr_smart")


def align_to_8x(x1: int, y1: int, x2: int, y2: int, video_w: int, video_h: int) -> Dict[str, int]:
    """将矩形区域强制对齐到 8 的倍数，并确保不超出视频边界"""
    w = x2 - x1
    h = y2 - y1
    # 限制对齐后的宽高不超过视频边界（防止越界）
    aligned_w = min(math.ceil(w / 8.0) * 8, video_w)
    aligned_h = min(math.ceil(h / 8.0) * 8, video_h)
    delta_w = aligned_w - w
    delta_h = aligned_h - h
    new_x1 = int(x1 - delta_w / 2)
    new_y1 = int(y1 - delta_h / 2)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    if new_x1 + aligned_w > video_w:
        new_x1 = video_w - aligned_w
    if new_y1 + aligned_h > video_h:
        new_y1 = video_h - aligned_h
    # 最终兜底：确保坐标非负且框在边界内
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    aligned_w = min(aligned_w, video_w - new_x1)
    aligned_h = min(aligned_h, video_h - new_y1)
    return {"x": new_x1, "y": new_y1, "w": aligned_w, "h": aligned_h}


def expand_and_align_box(
    box: Dict[str, int],
    video_w: int,
    video_h: int,
    pad_y_ratio: float = 0.20,
    pad_x_ratio: float = 0.05,
    expand_px: Optional[int] = None,
) -> Dict[str, int]:
    """扩大选框并强制对齐到 8 的倍数（语义与源文件一致）"""
    if expand_px is None:
        expand_px = int(os.environ.get("SUBTITLE_BOX_EXPAND_PX", "20"))

    if expand_px == 0:
        pad_x = 0
        pad_y = 0
    elif expand_px > 0:
        pad_y = expand_px
        pad_x = expand_px
        logger.info(f"[expand_box] 使用固定像素扩展: {expand_px}px")
    else:
        pad_y = int(box["h"] * pad_y_ratio)
        pad_x = int(box["w"] * pad_x_ratio)
        logger.info(f"[expand_box] 使用比例扩展: y={pad_y_ratio*100:.0f}%, x={pad_x_ratio*100:.0f}%")

    new_x1 = max(0, box["x"] - pad_x)
    new_y1 = max(0, box["y"] - pad_y)
    new_x2 = min(video_w, box["x"] + box["w"] + pad_x)
    new_y2 = min(video_h, box["y"] + box["h"] + pad_y)

    aligned_box = align_to_8x(new_x1, new_y1, new_x2, new_y2, video_w, video_h)

    logger.info(
        f"[expand_box] 原始: x={box['x']}, y={box['y']}, w={box['w']}, h={box['h']} "
        f"→ 扩展后: x={aligned_box['x']}, y={aligned_box['y']}, "
        f"w={aligned_box['w']}, h={aligned_box['h']}"
    )

    return aligned_box
