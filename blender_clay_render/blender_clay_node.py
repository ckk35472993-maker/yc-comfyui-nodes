# -*- coding: utf-8 -*-
# BlenderClayRender —— 容器内无头 Blender → clay/depth 控制序列（H3 control_video 原料）
# 调用: subprocess blender -b --python render_model.py -- --spec '{...}'
# 输入: model_path（绝对路径直接用；相对路径按 ComfyUI input/ 解析）+ 运镜预设枚举
# 输出: clay_images / depth_images (IMAGE batch) + frames_dir
# Blender 本体在 /data/ComfyUI/blender-5.3.0/（宿主挂载面，容器零镜像改动）
import json
import os
import re
import subprocess
import uuid

import numpy as np
import torch
from PIL import Image

import folder_paths

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "render_model.py")

# blender 二进制：env 覆盖优先；默认 /data 挂载面（宿主与容器同路径）
def _blender_bin():
    cand = [
        os.environ.get("BLENDER_BIN", ""),
        "/data/ComfyUI/blender-5.3.0/blender",
        "/opt/blender-5.3.0/blender",
    ]
    for c in cand:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError(
        "BlenderClayRender: blender 二进制未找到——"
        "期望 /data/ComfyUI/blender-5.3.0/blender（宿主挂载）或设 BLENDER_BIN")

# GPU EEVEE：容器 NVIDIA_DRIVER_CAPABILITIES 通常无 graphics → 只有 mesa llvmpipe 软渲。
# nvidia GL 栈已铺到 /data 挂载面（驱动同版 580.178.04），env 定向加载即可，
# 容器 compose/env 零改动；目录缺失时静默回落 mesa（慢但功能一致）
_NVGL = "/data/ComfyUI/nvidia_gl_580"
# blender 缺的 X 侧 .so 放挂载面：容器 fs 非持久（重建丢 zypper 装包），
# /data 为只读挂载但 .so 只需读——宿主侧装一次，容器重建不丢
_XLIBS = "/data/ComfyUI/blender_xlibs"
def _blender_env():
    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    icd = os.path.join(_NVGL, "egl_vendor.d", "10_nvidia.json")
    lib = os.path.join(_NVGL, "lib")
    if os.path.isfile(icd) and os.path.isdir(lib):
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = icd
        ld = lib + ":" + ld
    if os.path.isdir(_XLIBS):
        ld = _XLIBS + ":" + ld
    if ld != env.get("LD_LIBRARY_PATH", ""):
        env["LD_LIBRARY_PATH"] = ld
    return env

def _resolve_media_path(p):
    p = str(p or "").strip()
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join(folder_paths.get_input_directory(), p)

def _load_seq_pngs(dir_path, prefix):
    files = sorted(f for f in os.listdir(dir_path)
                   if f.startswith(prefix) and f.endswith(".png"))
    imgs = []
    for fn in files:
        with Image.open(os.path.join(dir_path, fn)) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        imgs.append(torch.from_numpy(arr))
    if not imgs:
        raise RuntimeError("BlenderClayRender: no frames in %s" % dir_path)
    return torch.stack(imgs, dim=0)


class BlenderClayRender:
    """外部 3D 模型 → clay/depth 白膜控制序列（EEVEE 无头渲染）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_path": ("STRING", {"default": "",
                "tooltip": "glb/gltf/fbx/obj/stl/ply；绝对路径或 ComfyUI input/ 相对名；"
                           "也支持 gen:bottle[:marker] 程序化生成（marker=非对称方位标记，orbit 必用）"}),
            "motion": (["orbit360", "orbit_half", "orbit_quarter",
                        "dolly_in", "dolly_out", "crane_up", "crane_down", "still",
                        "truck_l", "truck_r", "pan_l", "pan_r",
                        "tilt_up", "tilt_down", "pedestal_up", "pedestal_down",
                        "zoom_in", "zoom_out",
                        "helix_up", "helix_down", "reveal", "vertigo"],),
            "frames": ("INT", {"default": 72, "min": 8, "max": 480}),
            "fps": ("INT", {"default": 24, "min": 8, "max": 60}),
            "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
            "seed": ("INT", {"default": 7, "min": 0, "max": 2147483647}),
        }, "optional": {
            "pedestal": ("BOOLEAN", {"default": True,
                "tooltip": "False=无圆台只留地面——白膜布局应与参考图一致（纯商品图配无底座）"}),
            "anim_path": ("STRING", {"default": "",
                "tooltip": "可选动画 fbx：骨骼名与模型匹配时动作自动转移"}),
            "subject_spin_deg": ("FLOAT", {"default": 0.0, "min": -720.0, "max": 720.0}),
            "dist_scale": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 4.0}),
            "elev_deg": ("FLOAT", {"default": 22.0, "min": -10.0, "max": 85.0}),
            "target_size": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 50.0}),
            "scale_mode": (["height", "maxdim", "none"],),
            "depth_near": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0,
                "tooltip": "0=自动按相机距离估计"}),
            "depth_far": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0,
                "tooltip": "0=自动按相机距离估计"}),
            "keep_frames": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("clay_images", "depth_images", "frames_dir")
    FUNCTION = "render"
    CATEGORY = "video/blender"

    @staticmethod
    def _gen_model(spec_str, out_dir):
        """gen:bottle[:marker] → 跑 gen_bottle.py 产 glb，返回路径。"""
        parts = [p for p in spec_str.split(":") if p]
        name = parts[0] if parts else ""
        if name != "bottle":
            raise RuntimeError("BlenderClayRender: 未知 gen 类型 '%s'（现支持 gen:bottle[:marker]）" % name)
        gen_script = os.path.join(_HERE, "gen_bottle.py")
        glb = os.path.join(out_dir, "gen_%s.glb" % name)
        cmd = [_blender_bin(), "-b", "--python", gen_script, "--", glb] + parts[1:]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_blender_env(),
                              timeout=120, encoding="utf-8", errors="replace")
        if proc.returncode != 0 or not os.path.isfile(glb):
            tail = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()[-15:]
            raise RuntimeError("BlenderClayRender: gen 建模失败 rc=%s\n%s" % (proc.returncode, "\n".join(tail)))
        return glb

    def render(self, model_path, motion, frames, fps, width, height, seed,
               pedestal=True, anim_path="", subject_spin_deg=0.0, dist_scale=1.0,
               elev_deg=22.0, target_size=2.0, scale_mode="height",
               depth_near=0.0, depth_far=0.0, keep_frames=False):
        apath = _resolve_media_path(anim_path)

        out_dir = os.path.join(folder_paths.get_output_directory(),
                               "blender_clay", uuid.uuid4().hex[:12])
        os.makedirs(out_dir, exist_ok=True)

        # gen: 程序化建模伪路径——"gen:bottle" / "gen:bottle:marker"（marker=非对称方位标记）
        mpath = _resolve_media_path(model_path)
        if str(model_path).strip().startswith("gen:"):
            mpath = self._gen_model(str(model_path).strip()[4:], out_dir)
        if not mpath or not os.path.isfile(mpath):
            raise RuntimeError("BlenderClayRender: model_path 不存在: %s" % mpath)

        spec = {
            "model_path": mpath, "anim_path": apath, "out_dir": out_dir,
            "frames": int(frames), "fps": int(fps),
            "width": int(width), "height": int(height), "seed": int(seed),
            "motion": motion, "target_size": float(target_size),
            "scale_mode": scale_mode, "subject_spin_deg": float(subject_spin_deg),
            "dist_scale": float(dist_scale), "elev_deg": float(elev_deg),
            "depth_near": float(depth_near), "depth_far": float(depth_far),
            "pedestal": bool(pedestal),
        }
        cmd = [_blender_bin(), "-b", "--python", _SCRIPT, "--",
               "--spec", json.dumps(spec, ensure_ascii=False)]
        timeout = max(120, int(frames) * 3)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=_blender_env(),
                              timeout=timeout, encoding="utf-8", errors="replace")
        log = (proc.stdout or "") + (proc.stderr or "")

        m = re.search(r"SELFCHECK (\{.*?\})", log)
        ok = bool(m and json.loads(m.group(1)).get("ok"))
        if proc.returncode != 0 or not ok:
            tail = "\n".join(log.strip().splitlines()[-30:])
            raise RuntimeError("BlenderClayRender: 渲染失败 rc=%s selfcheck=%s\n%s"
                               % (proc.returncode, bool(m and ok), tail))

        clay = _load_seq_pngs(os.path.join(out_dir, "clay"), "clay_")
        depth = _load_seq_pngs(os.path.join(out_dir, "depth"), "depth_")

        if not keep_frames:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
            out_dir = ""
        return (clay, depth, out_dir)


NODE_CLASS_MAPPINGS = {"BlenderClayRender": BlenderClayRender}
NODE_DISPLAY_NAME_MAPPINGS = {"BlenderClayRender": "Blender Clay Render (白膜控制片)"}
