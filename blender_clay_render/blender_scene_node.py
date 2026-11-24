# -*- coding: utf-8 -*-
# BlenderSceneRun —— storyboard 分镜 → 容器内无头 Blender → 多演员 clay/depth 帧批
# 与本地 Bridge /api/local/blender/* 同一条引擎链（engine/ 目录=video_studio/blender
# 的 vendored 镜像，sync_engine.py 同步，勿手改）：
#   director_spec.translate(storyboard)→spec → blender -b --python scene_run.py
#   → render_scene(bpy) → out_dir/shots/<sid>/<chan>/<chan>_NNNN.png
# 与 BlenderClayRender 差异：吃整段 storyboard（多 actor/多 shot/运镜词汇表），
# 输出帧批交给下游 VHS_VideoCombine 编码（--no-concat，不依赖容器 ffmpeg）。
import importlib.util
import json
import os
import re
import subprocess
import uuid

import folder_paths
import torch

from .blender_clay_node import _blender_bin, _blender_env, _load_seq_pngs

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENGINE = os.path.join(_HERE, "engine")
# 云端素材库默认落 /root/ComfyUI/user/ 下（=宿主 /data/ComfyUI/node1_user/，
# /data/ComfyUI 整目录在容器里是 RO 挂载写不进；user 子目录 RW 且持久）；env 可覆盖
_ASSETS_DIR = os.environ.get(
    "BLENDER_SCENE_ASSETS_DIR", "/root/ComfyUI/user/blender_assets")


def _load_engine_mod(name):
    """按显式路径 import engine/<name>.py——避免与其他 custom_nodes 同名模块撞车"""
    path = os.path.join(_ENGINE, name + ".py")
    spec = importlib.util.spec_from_file_location("blender_scene_engine_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BlenderSceneRun:
    """storyboard JSON → 白膜分镜 clay/depth 帧批（H3 control_video 原料）"""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "storyboard": ("STRING", {"default": "", "multiline": True,
                "tooltip": "分镜词汇表 JSON（scene/shots/actors；director_spec 词汇表"
                           " 见 video_studio/blender/README.md）。云端就地 translate，"
                           "与本地 Bridge /blender/translate 同码同错。"}),
            "preview": (["", "still", "motion"],),
        }, "optional": {
            "keep_frames": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("clay_images", "depth_images", "info")
    FUNCTION = "render"
    CATEGORY = "video/blender"

    def render(self, storyboard, preview="", keep_frames=False):
        text = str(storyboard or "").strip()
        if not text:
            raise RuntimeError("BlenderSceneRun: storyboard 为空")
        try:
            sb = json.loads(text)
        except Exception as e:
            raise RuntimeError("BlenderSceneRun: storyboard JSON 解析失败 %s" % e)

        # 1. storyboard → spec（云端就地翻译——与本地 translate 端点同一份 director_spec）
        try:
            spec, errors = _load_engine_mod("director_spec").translate(sb)
        except Exception as e:
            raise RuntimeError("BlenderSceneRun: translate 崩溃 %s" % e)
        if errors:
            raise RuntimeError(
                "BlenderSceneRun: storyboard 校验失败（director_spec）：\n  - "
                + "\n  - ".join(errors[:12]))
        total_frames = sum(int(s.get("frames", 24)) for s in spec.get("shots") or [])

        out_dir = os.path.join(folder_paths.get_output_directory(),
                               "blender_scene", uuid.uuid4().hex[:12])
        os.makedirs(out_dir, exist_ok=True)
        spec_path = os.path.join(out_dir, "scene_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)

        # 2. blender -b --python engine/scene_run.py（与 Bridge _blender_spawn_scene 同构）
        blender = _blender_bin()
        os.makedirs(_ASSETS_DIR, exist_ok=True)
        env = _blender_env()
        env["PYTHONUTF8"] = "1"
        env.setdefault("VIDEOAI_ASSETS_DIR", _ASSETS_DIR)
        cmd = [blender, "-b", "--python", os.path.join(_ENGINE, "scene_run.py"), "--",
               spec_path, "--json", "--blender", blender,
               "--work-dir", out_dir, "--assets-dir", _ASSETS_DIR, "--no-concat"]
        if preview:
            cmd += ["--preview", preview]
        timeout = max(180, total_frames * 10)
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              timeout=timeout, encoding="utf-8", errors="replace")
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # 3. SCENE 契约行（与 Bridge jobs 轮询同口径——只认约定行）
        m = re.search(r"SCENE (\{.*\})", proc.stdout or "")
        scene = {}
        if m:
            try:
                scene = json.loads(m.group(1))
            except Exception:
                scene = {}
        if proc.returncode != 0 or not scene.get("ok"):
            tail = "\n".join(log.strip().splitlines()[-25:])
            raise RuntimeError("BlenderSceneRun: 渲染失败 rc=%s reason=%s\n%s"
                               % (proc.returncode, scene.get("reason") or "?", tail))

        # 4. 帧批：resolved spec 的 shots 顺序逐 shot 拼（shots/<sid>/<chan>/<chan>_*.png）
        # 帧根=resolved spec 的 out_dir（--preview 时 engine 会重定位到
        # out_dir/preview_<mode>——读契约落盘文件，不在节点里重复引擎规则）
        frames_root = out_dir
        rsp = scene.get("resolved_spec")
        if rsp and os.path.isfile(rsp):
            try:
                with open(rsp, "r", encoding="utf-8") as f:
                    frames_root = json.load(f).get("out_dir") or out_dir
            except Exception:
                pass
        clay_parts, depth_parts = [], []
        missing = []
        for i, sh in enumerate(spec.get("shots") or []):
            sid = str(sh.get("id") or "shot%d" % i)
            for chan, acc in (("clay", clay_parts), ("depth", depth_parts)):
                d = os.path.join(frames_root, "shots", sid, chan)
                if not os.path.isdir(d):
                    missing.append("%s/%s" % (sid, chan))
                    continue
                acc.append(_load_seq_pngs(d, chan + "_"))
        if missing:
            raise RuntimeError("BlenderSceneRun: 缺 shot 帧目录: %s" % ", ".join(missing))
        clay = torch.cat(clay_parts, dim=0) if clay_parts else None
        depth = torch.cat(depth_parts, dim=0) if depth_parts else None
        if clay is None or depth is None:
            raise RuntimeError("BlenderSceneRun: 无帧产出")

        info = {"nb_frames": int(clay.shape[0]), "fps": int(spec.get("fps", 24)),
                "preview": preview or None, "resolved_spec": scene.get("resolved_spec"),
                "h3_input": "depth", "clay_is_preview_only": True}
        if not keep_frames:
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            info["frames_dir"] = out_dir
        return (clay, depth, json.dumps(info, ensure_ascii=False))


NODE_CLASS_MAPPINGS = {"BlenderSceneRun": BlenderSceneRun}
NODE_DISPLAY_NAME_MAPPINGS = {"BlenderSceneRun": "Blender Scene Run (白膜分镜多演员)"}
