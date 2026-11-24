# scene_run.py —— 场景编排器（host 侧，不进 Blender）
# 用法:
#   python scene_run.py <scene_spec.json> [--blender <exe>] [--work-dir <dir>]
#                       [--json] [--no-concat] [--timeout <s>]
#                       [--preview still|motion]
# 流程:
#   1. 读 spec → actors[].query 走 ensure_model.ensure（本地库命中/联网补）→ model_path
#   2. 写 <work_dir>/scene_spec.resolved.json（确定性落盘，所见即所跑）
#   3. blender_resolve.resolve_blender → blender -b --python render_scene.py
#   4. 逐 shot SELFCHECK → ffmpeg filter_complex concat（§7 规范）→ nb_frames 验收
# 预览语义（产品一刀切，禁止用「全渲 frames=1」冒充预览）:
#   still  = 每镜 motion=still + frames=1；concat 用 -framerate 1（联系表节奏）
#   motion = 每镜 frames 封顶 6~8、分辨率压到 ≤384×256；运镜保留
# 契约: 末行 SCENE {"ok","resolved_spec","shots":{...},"videos":{...},"errors":[...]}
# 依赖: stdlib + ensure_model/blender_resolve/render_scene + ffmpeg（env/PATH/仓库 bin）
import contextlib
import copy
import json
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_resolve
import ensure_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RENDER_SCENE = os.path.join(SCRIPT_DIR, "render_scene.py")

# 预览档位常量（一刀切，勿再堆 fallback）
_PREVIEW_STILL_FPS = 1
_PREVIEW_MOTION_FRAMES = 8
_PREVIEW_MOTION_MAX_W = 384
_PREVIEW_MOTION_MAX_H = 256


def resolve_ffmpeg():
    """VIDEOAI_FFMPEG_PATH > PATH > 仓库 local/bin/{win,mac}/ffmpeg*（get_bin_path 同链简版）"""
    env = (os.environ.get("VIDEOAI_FFMPEG_PATH") or "").strip()
    cands = [env] if env else []
    w = shutil.which("ffmpeg")
    if w:
        cands.append(w)
    repo = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    for sub in ("win", "mac"):
        cands.append(os.path.join(
            repo, "installdocker_pro", "fastapi", "video_ai_core", "local",
            "bin", sub, "ffmpeg.exe" if sub == "win" else "ffmpeg"))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return ""


def apply_preview_mode(spec, mode):
    """
    就地改写 spec 为预览档。mode ∈ still|motion|""。
    返回写入 result 的 preview 元数据；非法 mode 抛 ValueError。
    """
    mode = (mode or "").strip().lower()
    if not mode:
        return None
    if mode not in ("still", "motion"):
        raise ValueError("preview_mode_invalid:%s" % mode)
    shots = spec.get("shots") or []
    if mode == "still":
        for sh in shots:
            sh["motion"] = "still"
            sh["frames"] = 1
            # dur_s 与 frames 冲突时以 frames 为准（render_scene 读 frames）
            sh.pop("dur_s", None)
        spec["fps"] = _PREVIEW_STILL_FPS
        # 联系表节奏：低分辨率即可
        w = int(spec.get("width") or 768)
        h = int(spec.get("height") or 512)
        if w > _PREVIEW_MOTION_MAX_W or h > _PREVIEW_MOTION_MAX_H:
            scale = min(_PREVIEW_MOTION_MAX_W / max(w, 1),
                        _PREVIEW_MOTION_MAX_H / max(h, 1))
            spec["width"] = max(64, int(w * scale) // 2 * 2)
            spec["height"] = max(64, int(h * scale) // 2 * 2)
        return {"mode": "still", "fps": _PREVIEW_STILL_FPS,
                "frames_per_shot": 1, "motion": "still"}
    # motion
    for sh in shots:
        fr = int(sh.get("frames") or _PREVIEW_MOTION_FRAMES)
        sh["frames"] = max(6, min(fr, _PREVIEW_MOTION_FRAMES))
        sh.pop("dur_s", None)
    w = int(spec.get("width") or 768)
    h = int(spec.get("height") or 512)
    if w > _PREVIEW_MOTION_MAX_W or h > _PREVIEW_MOTION_MAX_H:
        scale = min(_PREVIEW_MOTION_MAX_W / max(w, 1),
                    _PREVIEW_MOTION_MAX_H / max(h, 1))
        spec["width"] = max(64, int(w * scale) // 2 * 2)
        spec["height"] = max(64, int(h * scale) // 2 * 2)
    return {"mode": "motion",
            "frames_cap": _PREVIEW_MOTION_FRAMES,
            "width": spec.get("width"), "height": spec.get("height")}


def resolve_actors(spec, work_dir, assets_dir=None):
    """query → ensure_model.ensure → model_path；返回 (resolved_actors, errors)。
    assets_dir 缺省走 ensure 默认库（真实素材库），可传参隔离 per-scene"""
    out, errors = [], []
    for i, a in enumerate(spec.get("actors") or []):
        a = dict(a)
        aid = a.get("id") or "actor%d" % i
        mp = (a.get("model_path") or "").strip()
        if mp and os.path.isfile(mp):
            a["model_path"] = os.path.abspath(mp)
            out.append(a)
            continue
        q = (a.get("query") or "").strip()
        if not q:
            errors.append("actor %s: no model_path and no query" % aid)
            continue
        try:
            model, source = ensure_model.ensure(q, assets_dir or ensure_model.ASSETS_DIR)
        except Exception as e:
            model, source = None, "error:%s" % type(e).__name__
        if not model:
            errors.append("actor %s: query '%s' unresolved (%s)" % (aid, q, source))
            continue
        a["model_path"] = model["path"]
        a["resolved_from"] = {"query": q, "source": source,
                              "title": model.get("title"), "license": model.get("license")}
        print("  [resolve] actor %s: '%s' -> %s (%s)"
              % (aid, q, os.path.basename(model["path"]), source), file=sys.stderr)
        out.append(a)
    return out, errors


def concat_shots(spec, ffmpeg, work_dir):
    """逐 shot 帧序列 filter_complex concat → scene_clay/depth.mp4 + §7 nb_frames 验收"""
    fps = int(spec.get("fps", 24))
    shots = spec.get("shots") or []
    videos, errors = {}, []
    chans = ("clay", "depth", "beauty") if spec.get("beauty") else ("clay", "depth")
    for chan in chans:
        inputs, filters, expect = [], [], 0
        for i, sh in enumerate(shots):
            sid = str(sh.get("id") or "shot%d" % i)
            pat = os.path.join(spec["out_dir"], "shots", sid, chan, chan + "_%04d.png")
            if not os.path.isdir(os.path.dirname(pat)):
                continue
            # filter 下标必须跟实际 -i 顺序一致；勿用 shots 枚举 i（中间缺 shot 会指错输入）
            inp_idx = len(filters)
            inputs += ["-framerate", str(fps), "-i", pat]
            filters.append("[%d:v]" % inp_idx)
            expect += int(sh.get("frames", 24))
        if not inputs:
            continue
        n = len(filters)
        fc = "".join(filters) + "concat=n=%d:v=1[v]" % n
        out_mp4 = os.path.join(work_dir, "scene_%s.mp4" % chan)
        cmd = [ffmpeg, "-y"] + inputs + [
            "-filter_complex", fc, "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-r", str(fps), out_mp4]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            errors.append("concat %s ffmpeg rc=%d: %s" % (chan, r.returncode, (r.stderr or "")[-400:]))
            continue
        # §7 验收：看 nb_read_packets 不看 duration
        fp = shutil.which("ffprobe") or os.path.join(
            os.path.dirname(ffmpeg), "ffprobe" + (".exe" if os.name == "nt" else ""))
        nb = -1
        if os.path.isfile(fp):
            pr = subprocess.run(
                [fp, "-v", "error", "-count_packets", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", out_mp4],
                capture_output=True, text=True, timeout=60)
            try:
                nb = int((pr.stdout or "").strip().split(",")[0])
            except (ValueError, IndexError):
                nb = -1
        ok = (nb == expect)
        videos[chan] = {"path": out_mp4, "nb_frames": nb, "expected": expect, "ok": ok}
        if not ok:
            errors.append("concat %s frames %d != expected %d" % (chan, nb, expect))
    return videos, errors


def main():
    for _s in (sys.stdout, sys.stderr):
        if not _s.isatty() and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    out_json = "--json" in args
    no_concat = "--no-concat" in args
    blender = ""
    work_dir = ""
    assets_dir = None
    timeout_s = 0
    preview_mode = ""
    spec_path = None
    i = 0
    try:
        while i < len(args):
            a = args[i]
            if a == "--blender":
                blender = args[i + 1]; i += 2
            elif a == "--work-dir":
                work_dir = args[i + 1]; i += 2
            elif a == "--assets-dir":
                assets_dir = args[i + 1]; i += 2
            elif a == "--timeout":
                timeout_s = int(args[i + 1]); i += 2
            elif a == "--preview":
                preview_mode = args[i + 1]; i += 2
            elif a.startswith("-"):
                i += 1
            else:
                spec_path = a; i += 1
    except (IndexError, ValueError) as e:
        print("SCENE " + json.dumps({"ok": False, "reason": "bad_args:%s" % type(e).__name__},
                                    ensure_ascii=False))
        return 2
    if not spec_path or not os.path.isfile(spec_path):
        print("SCENE " + json.dumps({"ok": False, "reason": "spec_missing",
                                     "spec": spec_path}, ensure_ascii=False))
        return 2

    try:
        spec = json.load(open(spec_path, encoding="utf-8"))
    except Exception as e:
        print("SCENE " + json.dumps({"ok": False, "reason": "spec_bad_json:%s" % type(e).__name__},
                                    ensure_ascii=False))
        return 2

    # 预览档：深拷贝后改写，不污染用户原 spec 文件语义（resolved 落盘含 preview 标记）
    preview_meta = None
    if preview_mode:
        spec = copy.deepcopy(spec)
        try:
            preview_meta = apply_preview_mode(spec, preview_mode)
        except ValueError as e:
            print("SCENE " + json.dumps({"ok": False, "reason": str(e)},
                                        ensure_ascii=False))
            return 2

    work_dir = work_dir or spec.get("out_dir") or os.path.join(
        os.path.dirname(os.path.abspath(spec_path)), "scene_out")
    # 预览与全渲同 out_dir 会互相覆盖 shots/——预览默认落到 out_dir/preview_<mode>
    if preview_meta and not work_dir.endswith(
            ("preview_still", "preview_motion")) and "preview_" not in os.path.basename(
            os.path.abspath(work_dir)):
        work_dir = os.path.join(os.path.abspath(work_dir), "preview_%s" % preview_meta["mode"])
    spec["out_dir"] = os.path.abspath(work_dir)
    os.makedirs(spec["out_dir"], exist_ok=True)
    if preview_meta:
        spec["preview"] = preview_meta

    # 1. actors 解析（ensure 联网时 [ok] 进度会打 stdout——--json 下先挪 stderr）
    try:
        if out_json:
            with contextlib.redirect_stdout(sys.stderr):
                actors, errors = resolve_actors(spec, work_dir, assets_dir)
        else:
            actors, errors = resolve_actors(spec, work_dir, assets_dir)
    except Exception as e:
        print("SCENE " + json.dumps(
            {"ok": False, "reason": "resolve_crash:%s" % type(e).__name__},
            ensure_ascii=False))
        return 1
    if errors:
        print("SCENE " + json.dumps({"ok": False, "reason": "actors_unresolved",
                                     "errors": errors}, ensure_ascii=False))
        return 1
    spec["actors"] = actors

    # 2. resolved spec 落盘（文件契约）
    resolved_path = os.path.join(spec["out_dir"], "scene_spec.resolved.json")
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)

    # 3. Blender 渲染
    exe = blender_resolve.resolve_blender(blender)
    if not exe:
        print("SCENE " + json.dumps({"ok": False, "reason": "blender_exe_missing",
                                     "resolved_spec": resolved_path,
                                     "hint": "VIDEOAI_BLENDER_PATH/BLENDER_EXE 或设置页配置"},
                                    ensure_ascii=False))
        return 1
    total_frames = sum(int(s.get("frames", 24)) for s in spec.get("shots") or [])
    if timeout_s:
        pass
    elif preview_meta:
        # 预览秒~十秒级；按帧给宽限，下限 120s 防冷启动
        timeout_s = max(120, total_frames * 15)
    else:
        timeout_s = max(900, total_frames * 8)
    cmd = [exe, "-b", "--python", RENDER_SCENE, "--", "--spec-file", resolved_path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print("SCENE " + json.dumps({"ok": False, "reason": "render_timeout",
                                     "timeout_s": timeout_s,
                                     "preview": preview_meta,
                                     "resolved_spec": resolved_path}, ensure_ascii=False))
        return 1
    m = re.search(r"SELFCHECK (\{.*\})", r.stdout or "")
    selfcheck = json.loads(m.group(1)) if m else None
    result = {"ok": bool(selfcheck and selfcheck.get("ok")),
              "resolved_spec": resolved_path,
              "shots": (selfcheck or {}).get("shots"),
              "overlap_pairs": (selfcheck or {}).get("overlap_pairs"),
              "returncode": r.returncode}
    if preview_meta:
        result["preview"] = preview_meta
    if r.returncode != 0:
        # FATAL 打 stdout（blender 脚本惯例），翻出来当结构化 reason 给上层
        fm = re.search(r"FATAL (.+)", r.stdout or "")
        result["stderr_tail"] = (r.stderr or "")[-1500:]
        result["reason"] = "fatal:" + fm.group(1).strip() if fm else "render_failed"
        print("SCENE " + json.dumps(result, ensure_ascii=False))
        return 1

    # 4. concat（§7：filter_complex + nb_frames 验收）
    if not no_concat:
        ffmpeg = resolve_ffmpeg()
        if not ffmpeg:
            result["videos"] = {}
            result["errors"] = ["ffmpeg_missing"]
        else:
            videos, verrs = concat_shots(spec, ffmpeg, spec["out_dir"])
            result["videos"] = videos
            if verrs:
                result["errors"] = verrs
        # H3 控制输入默认=depth（2026-11-20 同视频 A/B 实锤：union 控制网认
        # depth/线稿不认灰模，depth 推近重构 vs clay 近静止；clay 只作 UI 预览）
        # preview 产物是降质片（≤384×256/封顶8帧/still=1帧）——不许冒名 h3_input，
        # UI 预览要拿 depth 自己读 videos.depth.path。
        dv = (result.get("videos") or {}).get("depth") or {}
        if dv.get("ok") and not preview_meta:
            result["h3_input"] = {"control_video": dv.get("path"),
                                  "variant": "depth",
                                  "clay_is_preview_only": True}
        result["ok"] = result["ok"] and not result.get("errors")

    print("SCENE " + json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
