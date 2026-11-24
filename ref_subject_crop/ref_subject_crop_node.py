# -*- coding: utf-8 -*-
"""RefSubjectCrop：参考图单主体守护节点（多人检出 → 锁主脸 → 排他裁切）。

用法：LoadImage → 本节点 → 下游参考图输入（ref_image / swap 参考等）。
零外部服务依赖：cv2 + 随包 YuNet ONNX（232KB，CPU ~30ms），无模型目录约定。
fail-open：cv2 缺失/模型缺失/0 人脸 → 原图透传 + info 告警，工作流不挂。
"""

import json
import logging

import numpy as np
import torch

from . import ref_subject_guard as guard

logger = logging.getLogger("RefSubjectCrop")


def _tensor_to_bgr(image):
    # ComfyUI IMAGE: [B,H,W,C] float 0-1
    t = image[0].detach().cpu().numpy()
    t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return t[:, :, ::-1].copy()  # RGB→BGR


def _bgr_to_tensor(img_bgr):
    t = img_bgr[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB
    return torch.from_numpy(t)[None, :, :, :]


class RefSubjectCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pick_mode": (["largest", "best", "center", "left", "right"],
                              {"default": "largest",
                               "tooltip": "主脸选择：largest=候选里面积最大（主体验证推荐）；"
                                          "best=检测分最高；center=离画面中心最近；left/right=最左/右"}),
                "primary_min_score": ("FLOAT", {"default": 0.8, "min": 0.3, "max": 1.0, "step": 0.05,
                                                "tooltip": "主脸候选的检测分门槛——低于此分的脸一律视为'待排除其他人'；"
                                                           "防低分误检（如手部假脸）抢主脸"}),
                "min_score": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05,
                                        "tooltip": "检出下限：调低可抓到更小的'照片人脸'以便排除"}),
                "expand_top": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1,
                                         "tooltip": "脸框向上扩展倍数（含发际/头顶）"}),
                "expand_bottom": ("FLOAT", {"default": 2.2, "min": 0.0, "max": 5.0, "step": 0.1,
                                            "tooltip": "脸框向下扩展倍数（含肩胸；全身参考图调大）"}),
                "expand_side": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 3.0, "step": 0.1,
                                          "tooltip": "脸框左右各扩倍数"}),
                "exclude_others": ("BOOLEAN", {"default": True,
                                             "tooltip": "把其他检出人脸（照片/镜面/路人）挤出裁切框"}),
                "fallback": (["passthrough", "center"],
                             {"default": "passthrough",
                              "tooltip": "0 人脸时：passthrough=原图透传；center=中心 3:4 人像兜底裁切"}),
                "debug_overlay": ("BOOLEAN", {"default": False,
                                              "tooltip": "输出画框调试图（绿=主脸 红=其他人 黄=裁切框）"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_image", "info")
    FUNCTION = "run"
    CATEGORY = "image/ref_guard"
    DESCRIPTION = "参考图单主体守护：多人检出→锁主脸→头肩裁切并排除其他人脸，防 ref 编码器选错身份"

    def run(self, image, pick_mode, primary_min_score, min_score,
            expand_top, expand_bottom, expand_side, exclude_others,
            fallback, debug_overlay):
        bgr = _tensor_to_bgr(image)
        out, info = guard.guard_image(
            bgr, pick_mode=pick_mode, min_score=min_score,
            primary_min_score=primary_min_score,
            expand_top=expand_top, expand_bottom=expand_bottom,
            expand_side=expand_side, exclude_others=exclude_others,
            fallback=fallback,
            extra_paths=[], debug_draw=debug_overlay)

        dbg = info.pop("debug_image", None)
        if dbg is None:
            dbg = out  # 没开调试时调试口也给一张（接线不断）
        info_str = json.dumps(info, ensure_ascii=False)
        if info.get("faces", 0) > 1:
            logger.warning("[RefSubjectCrop] 检出 %d 张人脸已裁切 info=%s", info["faces"], info_str)
        elif info.get("reason"):
            logger.info("[RefSubjectCrop] %s", info_str)
        return (_bgr_to_tensor(out), _bgr_to_tensor(dbg), info_str)


NODE_CLASS_MAPPINGS = {"RefSubjectCrop": RefSubjectCrop}
NODE_DISPLAY_NAME_MAPPINGS = {"RefSubjectCrop": "参考图主脸裁切(防多人串脸) RefSubjectCrop"}
