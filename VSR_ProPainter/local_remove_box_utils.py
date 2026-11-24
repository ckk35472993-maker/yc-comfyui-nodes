"""
去字幕 / 去水印 region 框选规范化（与 VSR backend/tools/inpaint_tools.py 对齐）

规则（与 ASR 智能检测、主预览手动框选一致）：
1. 手动 region 框：REMOVE_REGION_EXPAND_PX（默认 0，仅 8 对齐）
2. OCR 自动检测：SUBTITLE_BOX_EXPAND_PX（默认 20px）
3. VLOG 去水印：默认 REMOVE_WATERMARK_CONTEXT_RATIO（0.5=半屏）滑动窗口；贴边时扩展上下文
   - 竖屏贴水平边 → 全宽 + ratio 高（如右下角水印 → 720×640 下半屏，对齐 Test-23）
   - 横屏贴垂直边 → 全高；水平贴边 → ratio 宽（四角 → ratio×ratio，非全高条带）
   - 图片与视频均走同一半屏逻辑（已废弃横屏 +20px 紧凑裁剪）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Tuple

from local_remove_subtitle_ocr_smart import align_to_8x, expand_and_align_box

logger = logging.getLogger("video_ai_core.local_remove_box_utils")

ALIGN_MULTIPLE = 8
MIN_BOX_W = int(os.environ.get("REMOVE_BOX_MIN_W", "16"))
MIN_BOX_H = int(os.environ.get("REMOVE_BOX_MIN_H", "16"))
CROP_DEVIATION_PX = int(os.environ.get("REMOVE_REGION_CROP_DEVIATION", "10"))
REMOVE_REGION_EXPAND_PX = int(os.environ.get("REMOVE_REGION_EXPAND_PX", "0"))
MIN_REGION_CROP_W = int(os.environ.get("REMOVE_REGION_MIN_CROP_W", "384"))
MIN_REGION_CROP_H = int(os.environ.get("REMOVE_REGION_MIN_CROP_H", "256"))
WATERMARK_MIN_CROP_FLOOR = int(os.environ.get("REMOVE_WATERMARK_MIN_CROP_FLOOR", "384"))
WATERMARK_EDGE_PX = int(os.environ.get("REMOVE_WATERMARK_EDGE_PX", "12"))


_RATIO_CLAMP_LOGGED = False


def watermark_context_ratio() -> float:
    """去水印上下文占比：默认且下限 0.5（半屏）。0.3 会把竖屏底边裁成 384 地板。"""
    global _RATIO_CLAMP_LOGGED
    try:
        r = float(os.environ.get("REMOVE_WATERMARK_CONTEXT_RATIO", "0.5"))
    except (TypeError, ValueError):
        r = 0.5
    r = max(0.2, min(1.0, r))
    if r < 0.5:
        if not _RATIO_CLAMP_LOGGED:
            logger.warning(
                "[watermark_crop] CONTEXT_RATIO=%.2f 过小，强制 0.5（否则竖屏底边会裁成 384 地板）",
                r,
            )
            _RATIO_CLAMP_LOGGED = True
        return 0.5
    return r


def dino_sam2_box_expand_ratio() -> float:
    """
    DINO 检测框传给 SAM2 前的外扩比例（相对框宽高各边）。
    默认 0.12（12%）；设为 0 关闭。推荐 0.10–0.15。
    环境变量：DINO_SAM2_BOX_EXPAND_RATIO
    """
    try:
        r = float(os.environ.get("DINO_SAM2_BOX_EXPAND_RATIO", "0.12"))
    except (TypeError, ValueError):
        r = 0.12
    return max(0.0, min(0.5, r))


def expand_norm_box_xyxy_for_sam2(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    expand_ratio: float | None = None,
) -> Tuple[float, float, float, float]:
    """
    将归一化框 (0–1) 按宽高比例外扩，供 SAM2 提示框使用。
    DINO 框往往贴边过紧，略扩可让 SAM2 捕获完整轮廓与边缘上下文。
    """
    ratio = dino_sam2_box_expand_ratio() if expand_ratio is None else max(0.0, min(0.5, float(expand_ratio)))
    if ratio <= 0:
        return float(x1), float(y1), float(x2), float(y2)

    fx1, fy1, fx2, fy2 = float(x1), float(y1), float(x2), float(y2)
    w = max(1e-6, fx2 - fx1)
    h = max(1e-6, fy2 - fy1)
    pad_w = w * ratio
    pad_h = h * ratio

    nx1 = max(0.0, fx1 - pad_w)
    ny1 = max(0.0, fy1 - pad_h)
    nx2 = min(1.0, fx2 + pad_w)
    ny2 = min(1.0, fy2 + pad_h)
    if nx2 <= nx1 + 0.005:
        nx2 = min(1.0, nx1 + 0.01)
    if ny2 <= ny1 + 0.005:
        ny2 = min(1.0, ny1 + 0.01)
    return nx1, ny1, nx2, ny2


def expand_norm_box_dict_for_sam2(
    box: Dict[str, float],
    *,
    expand_ratio: float | None = None,
) -> Dict[str, float]:
    """对 {x1,y1,x2,y2} 字典做 SAM2 外扩，保留其它键。"""
    if not box:
        return box
    x1, y1, x2, y2 = expand_norm_box_xyxy_for_sam2(
        float(box.get("x1") or 0),
        float(box.get("y1") or 0),
        float(box.get("x2") or 0),
        float(box.get("y2") or 0),
        expand_ratio=expand_ratio,
    )
    return {**box, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def probe_video_display_dimensions(video_path: str) -> Tuple[int, int, int, int]:
    """
    返回 (display_w, display_h, stored_w, stored_h)。
    display_* 为应用 rotation 元数据后的显示尺寸（与前端框选一致）；
    stored_* 为解码器原始帧尺寸（OpenCV 读取）。
    """
    import json
    import subprocess

    stored_w, stored_h = 0, 0
    try:
        from local_service import get_bin_path
        ffprobe = get_bin_path("ffprobe")
    except Exception:
        ffprobe = "ffprobe"
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height:stream_side_data=rotation",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        stored_w = int(stream.get("width") or 0)
        stored_h = int(stream.get("height") or 0)
        side_data_list = stream.get("side_data_list") or []
        side_data = side_data_list[0] if side_data_list else {}
        rotation = int(side_data.get("rotation") or 0)
    except Exception:
        rotation = 0

    if stored_w <= 0 or stored_h <= 0:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            stored_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            stored_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    display_w, display_h = stored_w, stored_h
    if rotation in (90, 270, -90, -270):
        display_w, display_h = stored_h, stored_w
    return display_w, display_h, stored_w, stored_h


def probe_video_rotation_deg(video_path: str) -> int:
    """ffprobe rotation 元数据（度），无则 0。"""
    import json
    import subprocess

    try:
        from local_service import get_bin_path
        ffprobe = get_bin_path("ffprobe")
    except Exception:
        ffprobe = "ffprobe"
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream_side_data=rotation",
                "-of", "json",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(result.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        side_data_list = stream.get("side_data_list") or []
        side_data = side_data_list[0] if side_data_list else {}
        return int(side_data.get("rotation") or 0)
    except Exception:
        return 0


def map_display_xyxy_to_stored(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    display_w: int,
    display_h: int,
    stored_w: int,
    stored_h: int,
    rotation: int,
) -> Tuple[int, int, int, int]:
    """前端/显示坐标 bbox → FFmpeg 解码帧（stored）坐标。

    输入 x2/y2 为排他边界（exclusive bound），映射时先用 x2-1/y2-1 还原真实像素，
    映射后再 +1 转回排他边界，避免旋转翻转时差一像素。
    """
    if (display_w, display_h) == (stored_w, stored_h) or int(rotation) % 360 == 0:
        return int(x1), int(y1), int(x2), int(y2)
    r = int(rotation) % 360
    # 用真实的内含边界像素 (x2-1, y2-1) 进行空间映射
    true_x2 = max(x1, x2 - 1)
    true_y2 = max(y1, y2 - 1)
    corners = [
        (float(x1), float(y1)),
        (float(true_x2), float(y1)),
        (float(true_x2), float(true_y2)),
        (float(x1), float(true_y2)),
    ]

    def _disp_to_store(dx: float, dy: float) -> Tuple[int, int]:
        if r == 90:
            return int(stored_w - 1 - dy), int(dx)
        if r == 270:
            return int(dy), int(stored_h - 1 - dx)
        if r == 180:
            return int(stored_w - 1 - dx), int(stored_h - 1 - dy)
        return int(dx), int(dy)

    mapped = [_disp_to_store(dx, dy) for dx, dy in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    ox1 = max(0, min(xs))
    oy1 = max(0, min(ys))
    # 映射回来后 +1 恢复为排他性边界
    ox2 = min(stored_w, max(xs) + 1)
    oy2 = min(stored_h, max(ys) + 1)
    return ox1, oy1, max(ox1 + 2, ox2), max(oy1 + 2, oy2)


def bbox_iou_xyxy(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(max(1, union))


WATERMARK_EDGE_ZONE_RATIO = float(os.environ.get("REMOVE_WATERMARK_EDGE_ZONE_RATIO", "0.15"))


def _watermark_edge_flags(
    x1: int, y1: int, x2: int, y2: int, media_w: int, media_h: int,
) -> Tuple[bool, bool, bool, bool]:
    """
    判断水印是否贴近画面边缘（绝对像素 + 外圈比例带，默认 15%）。
    8 对齐后的 bbox 可能差几像素未贴边，但仍处于角落死角 → 需扩展上下文。
    """
    edge = WATERMARK_EDGE_PX
    zone_x = max(edge, int(media_w * WATERMARK_EDGE_ZONE_RATIO))
    zone_y = max(edge, int(media_h * WATERMARK_EDGE_ZONE_RATIO))
    near_left = x1 <= zone_x
    near_right = x2 >= media_w - zone_x
    near_top = y1 <= zone_y
    near_bottom = y2 >= media_h - zone_y
    return near_left, near_right, near_top, near_bottom


def _xyxy_from_box(box: Dict[str, int]) -> Tuple[int, int, int, int]:
    x = int(box.get("x", 0))
    y = int(box.get("y", 0))
    w = max(0, int(box.get("w", 0)))
    h = max(0, int(box.get("h", 0)))
    return x, y, x + w, y + h


def bbox_str_from_box(box: Dict[str, int]) -> str:
    x1, y1, x2, y2 = _xyxy_from_box(box)
    return f"{x1},{y1},{x2},{y2}"


def bbox_in_crop_coords(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    crop_x: int,
    crop_y: int,
) -> str:
    """全图 bbox 转为裁剪后片段内的相对坐标（与 removeBoxNormalize.js upload_bbox 一致）。"""
    ux1 = max(0, int(x1) - int(crop_x))
    uy1 = max(0, int(y1) - int(crop_y))
    ux2 = max(ux1 + 2, int(x2) - int(crop_x))
    uy2 = max(uy1 + 2, int(y2) - int(crop_y))
    return f"{ux1},{uy1},{ux2},{uy2}"


def clamp_bbox_to_crop(bbox_str: str, crop_w: int, crop_h: int) -> str:
    """将裁剪后坐标系内的 bbox 限制在 [0,crop_w]×[0,crop_h]，避免 VSR -c 超出画面导致黑框。"""
    raw = (bbox_str or "").strip()
    if not raw or crop_w <= 0 or crop_h <= 0:
        return raw
    parts = [int(float(p)) for p in raw.split(",")]
    if len(parts) != 4:
        return raw
    x1, y1, x2, y2 = parts
    x1 = max(0, min(x1, crop_w - 2))
    y1 = max(0, min(y1, crop_h - 2))
    x2 = max(x1 + 2, min(x2, crop_w))
    y2 = max(y1 + 2, min(y2, crop_h))
    return f"{x1},{y1},{x2},{y2}"


def bbox_xyxy_area(bbox_str: str) -> int:
    raw = (bbox_str or "").strip()
    if not raw:
        return 0
    parts = [int(float(p)) for p in raw.split(",")]
    if len(parts) != 4:
        return 0
    x1, y1, x2, y2 = parts
    return max(0, x2 - x1) * max(0, y2 - y1)


def mask_bounding_box_from_png(mask_path: str, *, threshold: int = 32) -> str:
    """从 mask PNG 读取实际涂抹区域外接框（裁剪后视频坐标 x1,y1,x2,y2）。"""
    import cv2
    import numpy as np

    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return ""
    ys, xs = np.where(m > int(threshold))
    if xs.size == 0 or ys.size == 0:
        return ""
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return f"{x1},{y1},{x2},{y2}"


def parse_bbox_str(bbox: str) -> Dict[str, int]:
    parts = [p.strip() for p in str(bbox or "").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("bbox 须为 x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(float(parts[0])), int(float(parts[1])), int(float(parts[2])), int(float(parts[3])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {"x": x1, "y": y1, "w": max(2, x2 - x1), "h": max(2, y2 - y1)}


def normalize_region_user_box(
    box: Dict[str, int],
    media_w: int,
    media_h: int,
    *,
    expand_px: int | None = None,
) -> Dict[str, int]:
    """扩展 + 8 对齐 + 最小框约束（手动框选默认不扩展）。"""
    media_w = max(1, int(media_w))
    media_h = max(1, int(media_h))
    if expand_px is None:
        expand_px = REMOVE_REGION_EXPAND_PX
    raw = {
        "x": max(0, int(box.get("x", 0))),
        "y": max(0, int(box.get("y", 0))),
        "w": max(1, int(box.get("w", 0))),
        "h": max(1, int(box.get("h", 0))),
    }
    expanded = expand_and_align_box(raw, media_w, media_h, expand_px=expand_px)
    if expanded["w"] < MIN_BOX_W or expanded["h"] < MIN_BOX_H:
        cx = expanded["x"] + expanded["w"] // 2
        cy = expanded["y"] + expanded["h"] // 2
        tw = min(media_w, max(expanded["w"], MIN_BOX_W))
        th = min(media_h, max(expanded["h"], MIN_BOX_H))
        x1 = max(0, cx - tw // 2)
        y1 = max(0, cy - th // 2)
        expanded = align_to_8x(x1, y1, x1 + tw, y1 + th, media_w, media_h)
    return expanded


def _align_crop_dimensions(
    crop_w: int,
    crop_h: int,
    media_w: int,
    media_h: int,
) -> Tuple[int, int]:
    """8 像素对齐 + 偶数宽高，且不超过原片尺寸。"""
    crop_w = min(media_w, int(crop_w))
    crop_h = min(media_h, int(crop_h))
    crop_w = ((crop_w + 7) // 8) * 8
    crop_h = ((crop_h + 7) // 8) * 8
    crop_w = min(crop_w, media_w)
    crop_h = min(crop_h, media_h)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    crop_w = max(ALIGN_MULTIPLE, crop_w)
    crop_h = max(ALIGN_MULTIPLE, crop_h)
    return crop_w, crop_h


def _clamp_crop_origin(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    media_w: int,
    media_h: int,
) -> Tuple[int, int]:
    """滑动窗口边界约束 + YUV420 偶数起点对齐。"""
    crop_x = max(0, min(int(crop_x), media_w - crop_w))
    crop_y = max(0, min(int(crop_y), media_h - crop_h))
    crop_x -= crop_x % 2
    crop_y -= crop_y % 2
    crop_x = max(0, min(crop_x, media_w - crop_w))
    crop_y = max(0, min(crop_y, media_h - crop_h))
    return crop_x, crop_y


def _ensure_crop_contains_box(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    media_w: int,
    media_h: int,
) -> Tuple[int, int]:
    """在已 clamp 的窗口上微调起点，保证水印框 100% 落在裁剪区内。"""
    cx, cy = int(crop_x), int(crop_y)
    if x1 < cx:
        cx = x1
    if y1 < cy:
        cy = y1
    if x2 > cx + crop_w:
        cx = x2 - crop_w
    if y2 > cy + crop_h:
        cy = y2 - crop_h
    return _clamp_crop_origin(cx, cy, crop_w, crop_h, media_w, media_h)


def compute_region_context_crop(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    media_w: int,
    media_h: int,
    *,
    min_crop_w: int | None = None,
    min_crop_h: int | None = None,
) -> Tuple[int, int, int, int]:
    """
    ASR / 去字幕 region：以框为中心扩展，默认最小上下文 384×256。
    """
    media_w = max(1, int(media_w))
    media_h = max(1, int(media_h))
    floor_w = int(min_crop_w if min_crop_w is not None else MIN_REGION_CROP_W)
    floor_h = int(min_crop_h if min_crop_h is not None else MIN_REGION_CROP_H)
    deviation = CROP_DEVIATION_PX
    crop_x = max(0, x1 - deviation)
    crop_y = max(0, y1 - deviation)
    crop_w = min(media_w, x2 + deviation) - crop_x
    crop_h = min(media_h, y2 + deviation) - crop_y

    crop_w = ((crop_w + 7) // 8) * 8
    crop_h = ((crop_h + 7) // 8) * 8
    if crop_x + crop_w > media_w:
        crop_x = max(0, media_w - crop_w)
    if crop_y + crop_h > media_h:
        crop_y = max(0, media_h - crop_h)
    crop_w = min(crop_w, media_w - crop_x)
    crop_h = min(crop_h, media_h - crop_y)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    box_cx = (int(x1) + int(x2)) // 2
    box_cy = (int(y1) + int(y2)) // 2
    target_w = max(crop_w, min(floor_w, media_w))
    target_h = max(crop_h, min(floor_h, media_h))
    crop_x = max(0, min(box_cx - target_w // 2, media_w - target_w))
    crop_y = max(0, min(box_cy - target_h // 2, media_h - target_h))

    edge = 12
    if x2 >= media_w - edge:
        crop_x = media_w - target_w
    elif x1 <= edge:
        crop_x = 0
    if y2 >= media_h - edge:
        crop_y = media_h - target_h
    elif y1 <= edge:
        crop_y = 0

    crop_w = ((target_w + 7) // 8) * 8
    crop_h = ((target_h + 7) // 8) * 8
    if crop_x + crop_w > media_w:
        crop_x = max(0, media_w - crop_w)
    if crop_y + crop_h > media_h:
        crop_y = max(0, media_h - crop_h)
    crop_w = min(crop_w, media_w - crop_x)
    crop_h = min(crop_h, media_h - crop_y)
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    return crop_x, crop_y, crop_w, crop_h


def compute_watermark_tight_crop(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    media_w: int,
    media_h: int,
    *,
    pad_px: int = 20,
    min_h: int = 128,
    min_w: int = 192,
    _dilation_hint_px: int = 10,
    _max_mask_ratio: float = 0.30,
) -> Tuple[int, int, int, int]:
    """
    横屏小水印紧凑裁剪：围绕水印 box 外扩固定像素，8x 对齐。
    用于本地 ProPainter/云端 VSR 需要更精准上下文而非全宽/半宽条带的场景。

    自动上下文保护：当水印面积超过 crop 面积 _max_mask_ratio 时，自动扩大 crop，
    保证 ProPainter 有足够的背景像素可供参考，避免整块涂黑。
    上限为原片半屏（media_w//2 × media_h//2），不超过全屏。
    """
    media_w = max(1, int(media_w))
    media_h = max(1, int(media_h))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 外扩并限制在画面内
    cx1 = max(0, x1 - pad_px)
    cy1 = max(0, y1 - pad_px)
    cx2 = min(media_w, x2 + pad_px)
    cy2 = min(media_h, y2 + pad_px)

    crop_w = max(min_w, cx2 - cx1)
    crop_h = max(min_h, cy2 - cy1)

    # 自动上下文保护：估算 mask 白区占比，若超限则等比扩大 crop
    # mask 尺寸 ≈ 水印 box + 两侧膨胀各 _dilation_hint_px
    mask_est_w = (x2 - x1) + 2 * _dilation_hint_px
    mask_est_h = (y2 - y1) + 2 * _dilation_hint_px
    mask_est_area = float(max(1, mask_est_w) * max(1, mask_est_h))
    crop_area = float(crop_w * crop_h)
    if crop_area > 0 and (mask_est_area / crop_area) > _max_mask_ratio:
        # 以当前 mask 尺寸为基础，推算满足 _max_mask_ratio 的 crop 边长
        needed_area = mask_est_area / _max_mask_ratio
        scale = (needed_area / max(1.0, mask_est_area)) ** 0.5
        auto_crop_w = int(mask_est_w * scale)
        auto_crop_h = int(mask_est_h * scale)
        # 最多扩到半屏
        half_w = max(crop_w, media_w // 2)
        half_h = max(crop_h, media_h // 2)
        crop_w = min(half_w, max(crop_w, auto_crop_w))
        crop_h = min(half_h, max(crop_h, auto_crop_h))
        logger.info(
            "[watermark_crop] tight_crop 上下文扩张: mask≈%dx%d ratio=%.1f%% > %.0f%%, "
            "crop扩至%dx%d",
            mask_est_w, mask_est_h, 100 * mask_est_area / crop_area,
            100 * _max_mask_ratio, crop_w, crop_h,
        )

    # 以水印中心为锚点，向两侧扩展至目标尺寸
    box_cx = (x1 + x2) // 2
    box_cy = (y1 + y2) // 2
    crop_x = max(0, box_cx - crop_w // 2)
    crop_y = max(0, box_cy - crop_h // 2)
    if crop_x + crop_w > media_w:
        crop_x = max(0, media_w - crop_w)
    if crop_y + crop_h > media_h:
        crop_y = max(0, media_h - crop_h)

    crop_w = min(crop_w, media_w - crop_x)
    crop_h = min(crop_h, media_h - crop_y)

    # 8x 对齐（向上取整）
    crop_w = ((crop_w + 7) // 8) * 8
    crop_h = ((crop_h + 7) // 8) * 8
    if crop_x + crop_w > media_w:
        crop_x = max(0, media_w - crop_w)
    if crop_y + crop_h > media_h:
        crop_y = max(0, media_h - crop_h)

    crop_x, crop_y = _clamp_crop_origin(crop_x, crop_y, crop_w, crop_h, media_w, media_h)
    crop_x, crop_y = _ensure_crop_contains_box(
        crop_x, crop_y, crop_w, crop_h, x1, y1, x2, y2, media_w, media_h,
    )
    logger.info(
        "[watermark_crop] tight_crop box=(%d,%d,%d,%d) media=%dx%d → crop=(%d,%d,%d,%d)",
        x1, y1, x2, y2, media_w, media_h,
        crop_x, crop_y, crop_w, crop_h,
    )
    return crop_x, crop_y, crop_w, crop_h


def compute_watermark_context_crop(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    media_w: int,
    media_h: int,
) -> Tuple[int, int, int, int]:
    """
    VLOG 去水印上下文截取（边缘感知滑动窗口）。

    默认：宽高各为原片 REMOVE_WATERMARK_CONTEXT_RATIO（默认 0.5 = 半屏），水印中心锚定 + 边界 clamp。
    贴边扩展（ProPainter 需借上下文像素，避免死角黑块）：
      - 竖屏 (h≥w)：水平贴边 → 全宽；垂直贴边 → ratio 高（锚定贴边侧）
      - 横屏 (w>h)：垂直贴边 → 全高；水平贴边 → ratio 宽（锚定贴边侧）
    例：720×1280 右下角水印 → 720×640 下半屏（与 Test-23 一致）。

    图片与视频均走本函数，不再使用横屏紧凑裁剪（+20px）。
    """
    media_w = max(1, int(media_w))
    media_h = max(1, int(media_h))

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    deviation = CROP_DEVIATION_PX
    is_portrait = media_h >= media_w
    ctx_ratio = watermark_context_ratio()
    floor_w = min(media_w, MIN_REGION_CROP_W)
    floor_h = min(media_h, MIN_REGION_CROP_H)

    target_w = min(media_w, max(floor_w, max(64, int(media_w * ctx_ratio))))
    target_h = min(media_h, max(floor_h, max(64, int(media_h * ctx_ratio))))

    near_left, near_right, near_top, near_bottom = _watermark_edge_flags(
        x1, y1, x2, y2, media_w, media_h,
    )

    strategy = "half_half"
    if is_portrait:
        if near_left or near_right:
            target_w = media_w
            strategy = "portrait_full_w"
        if near_top or near_bottom:
            target_h = min(media_h, max(floor_h, max(64, int(media_h * ctx_ratio))))
            strategy = (
                "portrait_full_w_half_h"
                if target_w >= media_w
                else ("portrait_half_h" if strategy == "half_half" else strategy)
            )
    else:
        corner_wm = (near_left or near_right) and (near_top or near_bottom)
        if near_top or near_bottom:
            # 四角水印：ratio 高即可；仅贴上下边非角时才全高
            target_h = (
                min(media_h, max(floor_h, max(64, int(media_h * ctx_ratio))))
                if corner_wm else media_h
            )
            strategy = "landscape_corner_half_h" if corner_wm else "landscape_full_h"
        if near_left or near_right:
            target_w = min(media_w, max(floor_w, max(64, int(media_w * ctx_ratio))))
            if corner_wm:
                strategy = "landscape_corner_half"
            elif target_h >= media_h:
                strategy = "landscape_half_w_full_h"
            else:
                strategy = "landscape_half_w" if strategy == "half_half" else strategy

    box_w = (x2 - x1) + deviation * 2
    box_h = (y2 - y1) + deviation * 2
    crop_w, crop_h = _align_crop_dimensions(
        max(target_w, box_w),
        max(target_h, box_h),
        media_w,
        media_h,
    )

    box_cx = (x1 + x2) // 2
    box_cy = (y1 + y2) // 2

    if is_portrait:
        if crop_w >= media_w:
            crop_x = 0
        elif near_right and not near_left:
            crop_x = media_w - crop_w
        elif near_left and not near_right:
            crop_x = 0
        else:
            crop_x = box_cx - crop_w // 2
        if near_bottom and not near_top:
            crop_y = media_h - crop_h
        elif near_top and not near_bottom:
            crop_y = 0
        else:
            crop_y = box_cy - crop_h // 2
    else:
        if crop_h >= media_h:
            crop_y = 0
        elif near_bottom and not near_top:
            crop_y = media_h - crop_h
        elif near_top and not near_bottom:
            crop_y = 0
        else:
            crop_y = box_cy - crop_h // 2
        if near_right and not near_left:
            crop_x = media_w - crop_w
        elif near_left and not near_right:
            crop_x = 0
        else:
            crop_x = box_cx - crop_w // 2

    crop_x, crop_y = _clamp_crop_origin(crop_x, crop_y, crop_w, crop_h, media_w, media_h)
    crop_x, crop_y = _ensure_crop_contains_box(
        crop_x, crop_y, crop_w, crop_h, x1, y1, x2, y2, media_w, media_h,
    )

    logger.info(
        "[watermark_crop] strategy=%s ctx_ratio=%.2f box=(%d,%d,%d,%d) media=%dx%d "
        "edge_zone(LR/TB)=(%s,%s,%s,%s) → crop=(%d,%d,%d,%d) %dx%d",
        strategy, ctx_ratio, x1, y1, x2, y2, media_w, media_h,
        near_left, near_right, near_top, near_bottom,
        crop_x, crop_y, crop_w, crop_h, crop_w, crop_h,
    )
    if crop_w <= MIN_REGION_CROP_W + ALIGN_MULTIPLE and crop_h <= MIN_REGION_CROP_H + ALIGN_MULTIPLE:
        logger.warning(
            "[watermark_crop] crop %dx%d 贴近地板 %dx%d，ProPainter 上下文不足；"
            "请确认 REMOVE_WATERMARK_CONTEXT_RATIO（当前 %.2f），横屏半屏应约 %dx%d",
            crop_w, crop_h, MIN_REGION_CROP_W, MIN_REGION_CROP_H, ctx_ratio,
            max(MIN_REGION_CROP_W, media_w // 2), max(MIN_REGION_CROP_H, media_h // 2),
        )
    # 竖屏全宽时 crop_w=720 不会触发上面的 AND 条件，但仍可能是 384 高地板
    if (
        is_portrait
        and crop_h <= MIN_REGION_CROP_H + ALIGN_MULTIPLE
        and media_h >= MIN_REGION_CROP_H * 2
    ):
        half_h = max(MIN_REGION_CROP_H, int(media_h * max(0.5, ctx_ratio)))
        logger.warning(
            "[watermark_crop] 竖屏 crop 高度 %d 贴近 384 地板 ctx_ratio=%.2f media=%dx%d；"
            "半屏应为约 %dx%d (y≈%d)。若仍是 384，Bridge 进程未加载 .env=0.5，请重启",
            crop_h, ctx_ratio, media_w, media_h, media_w, half_h, max(0, media_h - half_h),
        )
    return crop_x, crop_y, crop_w, crop_h


def describe_watermark_crop_strategy(
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
    media_w: int,
    media_h: int,
) -> str:
    """根据最终 crop 尺寸推断策略标签（与前端 computeWatermarkContextCrop.strategy 对齐）。"""
    media_w = max(1, int(media_w))
    media_h = max(1, int(media_h))
    is_portrait = media_h >= media_w
    half_y = media_h // 2
    if is_portrait:
        if crop_w >= media_w and crop_h <= media_h // 2 + ALIGN_MULTIPLE:
            if crop_y >= half_y - ALIGN_MULTIPLE:
                return "portrait_full_w_half_h"
            if crop_y <= ALIGN_MULTIPLE:
                return "portrait_half_h"
            return "portrait_full_w_half_h"
        if crop_w >= media_w:
            return "portrait_full_w"
        if crop_h <= media_h // 2 + ALIGN_MULTIPLE:
            return "portrait_half_h"
    else:
        if crop_h >= media_h and crop_w <= media_w // 2 + ALIGN_MULTIPLE:
            return "landscape_half_w_full_h"
        if crop_h >= media_h:
            return "landscape_full_h"
        if crop_w <= media_w // 2 + ALIGN_MULTIPLE:
            return "landscape_half_w"
    if crop_w <= media_w // 2 + ALIGN_MULTIPLE and crop_h <= media_h // 2 + ALIGN_MULTIPLE:
        return "half_half"
    return "adaptive"


def normalize_region_request(
    box: Dict[str, int],
    media_w: int,
    media_h: int,
    *,
    expand_px: int | None = None,
    include_crop: bool = True,
    crop_strategy: str = "region",
) -> Dict[str, Any]:
    """统一规范化：用户框、全图 bbox、视频裁剪区、上传云端 bbox。"""
    media_w = max(1, int(media_w))
    media_h = max(1, int(media_h))
    raw = {
        "x": int(box.get("x", 0)),
        "y": int(box.get("y", 0)),
        "w": max(1, int(box.get("w", 0))),
        "h": max(1, int(box.get("h", 0))),
    }
    user = normalize_region_user_box(raw, media_w, media_h, expand_px=expand_px)
    x1, y1, x2, y2 = _xyxy_from_box(user)
    strategy = (crop_strategy or "region").strip().lower()
    result: Dict[str, Any] = {
        "box": user,
        "bbox": bbox_str_from_box(user),
        "adjusted": (
            user["x"] != raw["x"] or user["y"] != raw["y"]
            or user["w"] != raw["w"] or user["h"] != raw["h"]
        ),
        "crop_strategy": strategy,
        "rules": {
            "align_multiple": ALIGN_MULTIPLE,
            "min_box_w": MIN_BOX_W,
            "min_box_h": MIN_BOX_H,
            "min_crop_w": MIN_REGION_CROP_W,
            "min_crop_h": MIN_REGION_CROP_H,
            "crop_deviation_px": CROP_DEVIATION_PX,
            "expand_px": int(expand_px if expand_px is not None else REMOVE_REGION_EXPAND_PX),
        },
    }
    if include_crop:
        if strategy in ("watermark", "watermark_half", "half_screen", "watermark_context"):
            crop_x, crop_y, crop_w, crop_h = compute_watermark_context_crop(
                x1, y1, x2, y2, media_w, media_h,
            )
        else:
            crop_x, crop_y, crop_w, crop_h = compute_region_context_crop(
                x1, y1, x2, y2, media_w, media_h,
            )
        result["crop"] = {
            "x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h,
            "crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_w, "crop_h": crop_h,
        }
        if strategy in ("watermark", "watermark_half", "half_screen", "watermark_context"):
            result["crop"]["crop_strategy"] = describe_watermark_crop_strategy(
                crop_x, crop_y, crop_w, crop_h, media_w, media_h,
            )
        result["upload_bbox"] = bbox_in_crop_coords(x1, y1, x2, y2, crop_x, crop_y)
    return result


def probe_video_wh(video_path: str) -> Tuple[int, int]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w < 1 or h < 1:
            raise ValueError(f"无效视频尺寸: {video_path}")
        return w, h
    finally:
        cap.release()


def probe_image_wh(image_path: str) -> Tuple[int, int]:
    try:
        from PIL import Image, ImageOps
        with ImageOps.exif_transpose(Image.open(image_path)) as im:
            w, h = im.size
            if w > 0 and h > 0:
                return int(w), int(h)
    except Exception:
        pass
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img.shape[:2]
    return int(w), int(h)
