# -*- coding: utf-8 -*-
"""
ref_subject_guard：参考图单主体守护（检测多人 → 锁主脸 → 排他裁切）。

解决 H3 类 ref2va 工作流的"多实体参考图"坑：参考编码器不是"取最大脸"的几何逻辑，
整帧/含画中画（相框照片、镜面反射、海报人像）的图会让编码器选错身份。
本模块在图进工作流前统一处理：检出所有人脸 → 选出主脸 → 头肩裁切并把其他人脸挤出框外。

三处使用同一实现（勿分叉改）：
  1. ComfyUI 节点包  comfyui126/custom_nodes/ref_subject_crop/（自包含副本）
  2. Bridge 端点     ref_subject_api.py（上传时裁切，覆盖 RH/网页端）
  3. 复用方直接 import 本文件

依赖：仅 cv2 + numpy。模型 = OpenCV YuNet ONNX（~232KB，MIT，opencv_zoo），
路径解析顺序：REF_GUARD_YUNET 环境变量 → 本文件同级 assets/ → ComfyUI models/insightface 旁路不可用（不依赖）。
"""

import os
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("ref_subject_guard")

_YUNET_NAME = "face_detection_yunet_2023mar.onnx"
_detector = None
_detector_key = None  # (model_path, w, h) 输入尺寸变化时重建


class RefGuardUnavailable(Exception):
    """cv2 或模型缺失时抛出；调用方应 fail-open 透传原图。"""


def _resolve_model_path(extra_paths=None):
    cands = []
    env = os.environ.get("REF_GUARD_YUNET", "").strip()
    if env:
        cands.append(env)
    if extra_paths:
        cands.extend(extra_paths)
    here = Path(__file__).resolve().parent
    cands += [
        here / "assets" / _YUNET_NAME,
        here / "models" / _YUNET_NAME,
        here / _YUNET_NAME,
    ]
    for c in cands:
        p = Path(c)
        if p.is_file() and p.stat().st_size > 100_000:
            return str(p)
    raise RefGuardUnavailable(f"YuNet 模型缺失，已查: {[str(c) for c in cands]}")


def _get_detector(w, h, score_threshold, extra_paths=None):
    """按输入尺寸缓存检测器（FaceDetectorYN 绑定 input size）。"""
    global _detector, _detector_key
    try:
        import cv2
    except Exception as e:
        raise RefGuardUnavailable(f"cv2 不可用: {e}")
    model = _resolve_model_path(extra_paths)
    key = (model, int(w), int(h), float(score_threshold))
    if _detector is None or _detector_key != key:
        _detector = cv2.FaceDetectorYN_create(
            model, "", (int(w), int(h)),
            score_threshold=float(score_threshold),
            nms_threshold=0.3, top_k=50,
        )
        _detector_key = key
    return _detector


def detect_faces(img_bgr, min_score=0.5, extra_paths=None):
    """返回 [{bbox:[x,y,w,h], score}]，按 score 降序。"""
    h, w = img_bgr.shape[:2]
    det = _get_detector(w, h, min_score, extra_paths)
    _, faces = det.detect(img_bgr)
    out = []
    if faces is not None:
        for f in faces:
            x, y, fw, fh = [float(v) for v in f[:4]]
            out.append({"bbox": [x, y, fw, fh], "score": float(f[-1])})
    out.sort(key=lambda d: -d["score"])
    return out


def pick_primary_index(faces, mode="largest", img_w=0, img_h=0, primary_min_score=0.8):
    """候选 = score>=primary_min_score 的脸（误检大手/低分小图先滤掉），
    无候选时降级为全部脸。mode: largest|best|center|left|right。"""
    if not faces:
        return -1
    cand = [i for i, f in enumerate(faces) if f["score"] >= primary_min_score] or list(range(len(faces)))

    def area(i):
        b = faces[i]["bbox"]
        return b[2] * b[3]

    def cx(i):
        b = faces[i]["bbox"]
        return b[0] + b[2] / 2

    def cy(i):
        b = faces[i]["bbox"]
        return b[1] + b[3] / 2

    if mode == "best":
        return max(cand, key=lambda i: faces[i]["score"])
    if mode == "center":
        return min(cand, key=lambda i: (cx(i) - img_w / 2) ** 2 + (cy(i) - img_h / 2) ** 2)
    if mode == "left":
        return min(cand, key=cx)
    if mode == "right":
        return max(cand, key=cx)
    return max(cand, key=area)  # largest


def compute_crop_box(faces, primary_idx, img_w, img_h,
                     expand_top=1.0, expand_bottom=2.2, expand_side=1.3,
                     exclude_others=True, exclude_pad=0.15,
                     other_up=0.8, other_down=3.0, other_side=1.0):
    """以主脸为核扩出头肩框；其他人脸整体位于主脸某侧时，把该侧裁切边收到其边界外。

    方位判定用"脸框是否完全在主脸框某侧"（photo 在主脸上方→收上边；
    手部误检在主脸下方→收下边），无法归侧的（交叠人群）保留并记入 info。

    其他人脸先按 other_up/down/side 外扩再判定——脸框只是"人/照片"的头部，
    不扩的话相框人的身体/边框会残留在裁切图里（实测：照片人脸 16×21px，
    但相框实际覆盖 ~3×脸高的区域）。
    """
    fx, fy, fw, fh = faces[primary_idx]["bbox"]
    x0 = fx - fw * expand_side
    x1 = fx + fw * (1.0 + expand_side)
    y0 = fy - fh * expand_top
    y1 = fy + fh * (1.0 + expand_bottom)

    px0, py0, px1, py1 = fx, fy, fx + fw, fy + fh
    excluded, overlapped = [], []
    for i, f in enumerate(faces):
        if i == primary_idx:
            continue
        ox0, oy0, ow, oh = f["bbox"]
        # 其他人的"脸框→人形区域"外扩：下 3×(肩胸/照片框) 上 0.8× 侧 1×
        ox0 -= ow * other_side
        oy0 -= oh * other_up
        ox1 = ox0 + ow * (1.0 + 2 * other_side)
        oy1 = oy0 + oh * (1.0 + other_up + other_down)
        # 该脸与当前裁切框不相交则无需处理
        if ox1 <= x0 or ox0 >= x1 or oy1 <= y0 or oy0 >= y1:
            continue
        pad = max(ow, oh) * exclude_pad
        if exclude_others:
            if oy1 <= py0:   # 完全在主脸上方
                y0 = max(y0, oy1 + pad)
            elif oy0 >= py1:  # 完全在下方
                y1 = min(y1, oy0 - pad)
            elif ox1 <= px0:  # 完全在左
                x0 = max(x0, ox1 + pad)
            elif ox0 >= px1:  # 完全在右
                x1 = min(x1, ox0 - pad)
            else:
                overlapped.append(i)  # 与主脸方位交叠，挤不掉
                continue
            excluded.append(i)

    x0 = int(max(0, min(x0, img_w - 2)))
    y0 = int(max(0, min(y0, img_h - 2)))
    x1 = int(max(x0 + 2, min(x1, img_w)))
    y1 = int(max(y0 + 2, min(y1, img_h)))
    return [x0, y0, x1, y1], excluded, overlapped


def guard_image(img_bgr, pick_mode="largest", min_score=0.5, primary_min_score=0.8,
                expand_top=1.0, expand_bottom=2.2, expand_side=1.3,
                exclude_others=True, fallback="passthrough",
                extra_paths=None, debug_draw=False):
    """主入口：img_bgr → (out_bgr, info dict)。info 可直接 JSON 序列化/写日志。

    fallback: passthrough=0 脸或异常时原图返回 | center=中心 3:4 人像兜底裁切。
    """
    h, w = img_bgr.shape[:2]
    info = {"faces": 0, "picked": -1, "cropped": False, "reason": "",
            "crop_box": None, "img_wh": [w, h], "boxes": []}
    try:
        faces = detect_faces(img_bgr, min_score=min_score, extra_paths=extra_paths)
    except RefGuardUnavailable as e:
        info["reason"] = f"guard_unavailable: {e}"
        return img_bgr, info

    info["faces"] = len(faces)
    info["boxes"] = [{"bbox": [round(v, 1) for v in f["bbox"]],
                      "score": round(f["score"], 3)} for f in faces]
    if not faces:
        info["reason"] = "no_face"
        if fallback == "center":
            cw = int(w * 0.6)
            ch = int(cw * 4 / 3)
            if ch > h:
                ch = int(h * 0.9)
                cw = int(ch * 3 / 4)
            x0 = (w - cw) // 2
            y0 = max(0, int(h * 0.05))
            out = img_bgr[y0:y0 + ch, x0:x0 + cw].copy()
            info.update(cropped=True, crop_box=[x0, y0, x0 + cw, y0 + ch],
                        reason="no_face_center_fallback")
            return out, info
        return img_bgr, info

    pi = pick_primary_index(faces, pick_mode, w, h, primary_min_score)
    info["picked"] = pi
    (x0, y0, x1, y1), excluded, overlapped = compute_crop_box(
        faces, pi, w, h, expand_top, expand_bottom, expand_side,
        exclude_others=exclude_others)
    info["excluded_faces"] = excluded
    info["overlapped_faces"] = overlapped

    # 裁切框已接近全图（说明原图即单主体人像构图）就不动刀
    cover = (x1 - x0) * (y1 - y0) / float(w * h)
    if cover >= 0.985 and len(faces) == 1:
        info["reason"] = "single_face_full_frame"
        return img_bgr, info

    out = img_bgr[y0:y1, x0:x1].copy()
    info.update(cropped=True, crop_box=[x0, y0, x1, y1],
                reason="ok" if len(faces) == 1 else "multi_face_excluded")

    if debug_draw:
        try:
            import cv2
            dbg = img_bgr.copy()
            for i, f in enumerate(faces):
                bx, by, bw, bh = [int(v) for v in f["bbox"]]
                c = (0, 255, 0) if i == pi else (0, 0, 255)
                cv2.rectangle(dbg, (bx, by), (bx + bw, by + bh), c, 2)
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (255, 200, 0), 3)
            info["debug_image"] = dbg
        except Exception:
            pass
    return out, info
