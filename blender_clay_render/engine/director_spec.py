#!/usr/bin/env python3
# director_spec.py —— 导演分镜词汇表 → scene_spec.json（确定性枚举映射，LLM/手写共用入口）
# 定位：上游（导演/LLM/手写）只产"词汇表级"分镜——shot type/focus/dur_s；
#       本翻译器把词汇表确定性展开成 render_scene/scene_run 吃的 scene_spec.json，
#       并前置校验全部枚举与引用——**下游永远见不到非法枚举/悬空引用**。
# 用法: python director_spec.py storyboard.json [--out scene_spec.json]
# 输出: 末行 SPEC {...} JSON 契约；exit 0=ok 1=内容错误 2=参数错误
import json
import os
import sys

for _s in (sys.stdout, sys.stderr):   # 管道端中文报错不炸（ensure_model 同款防护）
    try:
        if not _s.isatty():
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── 枚举真源（与 render_scene.py/blender_lib.py 同值——改运镜须三处同步）──
VALID_MOTIONS = ("orbit360", "orbit_half", "orbit_quarter",
                 "dolly_in", "dolly_out", "crane_up", "crane_down", "still",
                 "truck_l", "truck_r", "pan_l", "pan_r",
                 "tilt_up", "tilt_down", "pedestal_up", "pedestal_down",
                 "zoom_in", "zoom_out",
                 "helix_up", "helix_down", "reveal", "vertigo")
SIZES = ("els", "ls", "ms", "mcu", "cu", "ecu")
ANGLES = ("worm", "low", "eye", "high", "top", "bird")
SCALE_MODES = ("height", "maxdim", "none")

# ── 导演词汇表：type → {size, angle, motion, dur_s 默认时长} ──
# 只列"语义明确"的拍法；要自由组合用逐字段 override（仍过枚举校验）
SHOT_PRESETS = {
    "establishing":   {"size": "ls",  "angle": "eye",  "motion": "orbit_quarter", "dur_s": 3.0},
    "wide":           {"size": "els", "angle": "low",  "motion": "orbit_half",    "dur_s": 4.0},
    "orbit":          {"size": "ms",  "angle": "eye",  "motion": "orbit360",      "dur_s": 4.0},
    "half_orbit":     {"size": "ms",  "angle": "eye",  "motion": "orbit_half",    "dur_s": 3.0},
    "approach":       {"size": "ls",  "angle": "eye",  "motion": "dolly_in",      "dur_s": 2.5},
    "pull_back":      {"size": "ms",  "angle": "eye",  "motion": "dolly_out",     "dur_s": 2.5},
    "detail":         {"size": "mcu", "angle": "eye",  "motion": "dolly_in",      "dur_s": 2.0},
    "closeup":        {"size": "cu",  "angle": "eye",  "motion": "dolly_in",      "dur_s": 2.0},
    "macro":          {"size": "ecu", "angle": "eye",  "motion": "still",         "dur_s": 1.5},
    "hero":           {"size": "cu",  "angle": "low",  "motion": "crane_up",      "dur_s": 2.5},
    "descend":        {"size": "ms",  "angle": "high", "motion": "crane_down",    "dur_s": 2.5},
    "reveal":         {"size": "ms",  "angle": "low",  "motion": "reveal",        "dur_s": 2.5},
    "scan_l":         {"size": "ms",  "angle": "eye",  "motion": "pan_l",         "dur_s": 2.0},
    "scan_r":         {"size": "ms",  "angle": "eye",  "motion": "pan_r",         "dur_s": 2.0},
    "track_l":        {"size": "ms",  "angle": "low",  "motion": "truck_l",       "dur_s": 2.0},
    "track_r":        {"size": "ms",  "angle": "low",  "motion": "truck_r",       "dur_s": 2.0},
    "tilt_up":        {"size": "ms",  "angle": "eye",  "motion": "tilt_up",       "dur_s": 2.0},
    "tilt_down":      {"size": "ms",  "angle": "eye",  "motion": "tilt_down",     "dur_s": 2.0},
    "rise":           {"size": "ms",  "angle": "low",  "motion": "pedestal_up",   "dur_s": 2.0},
    "drop":           {"size": "ms",  "angle": "high", "motion": "pedestal_down", "dur_s": 2.0},
    "spiral_up":      {"size": "ms",  "angle": "low",  "motion": "helix_up",      "dur_s": 3.0},
    "spiral_down":    {"size": "ms",  "angle": "high", "motion": "helix_down",    "dur_s": 3.0},
    "topdown":        {"size": "ms",  "angle": "top",  "motion": "still",         "dur_s": 2.0},
    "bird_orbit":     {"size": "ls",  "angle": "bird", "motion": "orbit_quarter", "dur_s": 3.0},
    "vertigo":        {"size": "mcu", "angle": "eye",  "motion": "vertigo",       "dur_s": 3.0},
    "zoom_in":        {"size": "cu",  "angle": "eye",  "motion": "zoom_in",       "dur_s": 2.0},
    "zoom_out":       {"size": "ms",  "angle": "eye",  "motion": "zoom_out",      "dur_s": 2.0},
    "still":          {"size": "ms",  "angle": "eye",  "motion": "still",         "dur_s": 1.5},
}

ACTOR_KEYS = {"id", "query", "model_path", "pos", "rot_deg", "on",
              "target_size", "scale_mode", "anim_path"}
SHOT_KEYS = {"id", "type", "focus", "dur_s", "frames", "t0",
             "size", "angle", "motion", "elev_deg", "dist_scale", "fill"}
TOP_KEYS = {"width", "height", "fps", "ground", "strict_layout",
            "depth_near", "depth_far", "actors", "shots"}


def _num(v, lo=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if (lo is None or x >= lo) else None


def validate_actor(a, i, seen_ids):
    errs = []
    if not isinstance(a, dict):
        return None, ["actor[%d]: not an object" % i]
    bad = set(a) - ACTOR_KEYS
    if bad:
        errs.append("actor[%d]: unknown keys %s (valid: %s)" % (i, sorted(bad), sorted(ACTOR_KEYS)))
    aid = str(a.get("id") or "actor%d" % i).strip()
    if not aid:
        errs.append("actor[%d]: empty id" % i)
    elif aid in seen_ids:
        errs.append("actor[%d]: duplicate id '%s'" % (i, aid))
    if not (a.get("model_path") or a.get("query")):
        errs.append("actor '%s': need model_path or query" % aid)
    on = a.get("on")
    if on is not None and str(on).strip() not in seen_ids:
        errs.append("actor '%s': on='%s' 必须引用排在前面的 actor" % (aid, on))
    pos = a.get("pos")
    if pos is not None and not (isinstance(pos, list) and 2 <= len(pos) <= 3
                                and all(_num(v) is not None for v in pos)):
        errs.append("actor '%s': pos 必须是 [x,y(,z_extra)] 数值数组" % aid)
    if a.get("scale_mode") is not None and str(a["scale_mode"]).lower() not in SCALE_MODES:
        errs.append("actor '%s': scale_mode 必须是 %s" % (aid, SCALE_MODES))
    if a.get("target_size") is not None and _num(a["target_size"], 1e-4) is None:
        errs.append("actor '%s': target_size 必须 >0" % aid)
    if a.get("rot_deg") is not None and _num(a["rot_deg"]) is None:
        errs.append("actor '%s': rot_deg 必须是数值" % aid)
    return aid, errs


def expand_shot(s, i, fps, actor_ids):
    errs = []
    if not isinstance(s, dict):
        return None, ["shot[%d]: not an object" % i]
    bad = set(s) - SHOT_KEYS
    if bad:
        errs.append("shot[%d]: unknown keys %s (valid: %s)" % (i, sorted(bad), sorted(SHOT_KEYS)))
    stype = str(s.get("type") or "").strip().lower()
    preset = SHOT_PRESETS.get(stype)
    if not preset:
        errs.append("shot[%d]: unknown type '%s' (valid: %s)"
                    % (i, stype or "(empty)", sorted(SHOT_PRESETS)))
        return None, errs

    size = str(s.get("size") or preset["size"]).lower()
    angle = str(s.get("angle") or preset["angle"]).lower()
    motion = str(s.get("motion") or preset["motion"]).lower()
    if size not in SIZES:
        errs.append("shot[%d]: unknown size '%s'" % (i, size))
    if angle not in ANGLES:
        errs.append("shot[%d]: unknown angle '%s'" % (i, angle))
    if motion not in VALID_MOTIONS:
        errs.append("shot[%d]: unknown motion '%s'" % (i, motion))

    focus = str(s.get("focus") or "scene").strip()
    if focus != "scene" and focus not in actor_ids:
        errs.append("shot[%d]: focus '%s' 不在 actors 里（%s）" % (i, focus, sorted(actor_ids)))

    dur = s.get("dur_s", preset["dur_s"])
    if s.get("frames") is not None:
        frames = int(_num(s["frames"], 1) or -1)
        if frames < 1:
            errs.append("shot[%d]: frames 必须 >=1" % i)
            frames = 2
    else:
        d = _num(dur, 0.001)
        if d is None:
            errs.append("shot[%d]: dur_s 必须 >0" % i)
            d = preset["dur_s"]
        frames = max(2, round(fps * d))

    out = {"id": str(s.get("id") or "shot%d" % (i + 1)).strip(),
           "focus": focus, "size": size, "angle": angle, "motion": motion,
           "frames": frames, "t0": int(_num(s.get("t0", 1), 0) or 1)}
    for k in ("elev_deg", "dist_scale", "fill"):
        if s.get(k) is not None:
            v = _num(s[k])
            if v is None:
                errs.append("shot[%d]: %s 必须是数值" % (i, k))
            else:
                out[k] = v
    return (out if not errs else None), errs


def translate(sb):
    """storyboard dict → (scene_spec dict, errors[])。纯函数，可单测"""
    errs = []
    bad = set(sb) - TOP_KEYS
    if bad:
        errs.append("top: unknown keys %s (valid: %s)" % (sorted(bad), sorted(TOP_KEYS)))
    fps = _num(sb.get("fps", 24), 1) or 24.0

    actors, actor_ids, seen = [], [], set()
    for i, a in enumerate(sb.get("actors") or []):
        aid, e = validate_actor(a, i, seen)
        errs += e
        if aid and aid not in seen:
            seen.add(aid)
            actor_ids.append(aid)
            actors.append(dict(a, id=aid))
    if not actors:
        errs.append("actors 为空")

    shots, shot_ids = [], set()
    for i, s in enumerate(sb.get("shots") or []):
        out, e = expand_shot(s, i, fps, set(actor_ids))
        errs += e
        if out:
            if out["id"] in shot_ids:
                errs.append("shot[%d]: duplicate id '%s'" % (i, out["id"]))
                out = None
            else:
                shot_ids.add(out["id"])
                shots.append(out)
    if not shots and not errs:
        errs.append("shots 为空")

    spec = {"out_dir": "", "width": int(_num(sb.get("width", 768), 8) or 768),
            "height": int(_num(sb.get("height", 512), 8) or 512),
            "fps": fps, "actors": actors, "shots": shots}
    for k in ("ground", "strict_layout", "depth_near", "depth_far"):
        if sb.get(k) is not None:
            spec[k] = sb[k]
    return spec, errs


def emit(obj, code):
    print("SPEC " + json.dumps(obj, ensure_ascii=False))
    sys.exit(code)


def main():
    args = sys.argv[1:]
    sb_path, out_path = None, None
    i = 0
    try:
        while i < len(args):
            if args[i] == "--out":
                out_path = args[i + 1]; i += 2
            elif args[i] == "--list-types":
                emit({"ok": True, "types": {k: v for k, v in SHOT_PRESETS.items()}}, 0)
            elif sb_path is None:
                sb_path = args[i]; i += 1
            else:
                i += 1
        if not sb_path or not os.path.isfile(sb_path):
            emit({"ok": False, "reason": "storyboard_file_missing"}, 2)
        sb = json.load(open(sb_path, "r", encoding="utf-8"))
    except Exception as e:
        emit({"ok": False, "reason": "bad_args:%s" % type(e).__name__}, 2)
    if not isinstance(sb, dict):
        emit({"ok": False, "reason": "storyboard_not_object"}, 1)

    spec, errs = translate(sb)
    if errs:
        emit({"ok": False, "reason": "validation", "errors": errs}, 1)

    out_path = out_path or (os.path.splitext(sb_path)[0] + ".scene.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(spec, f, ensure_ascii=False, indent=2)
    except Exception as e:
        emit({"ok": False, "reason": "write_failed:%s" % type(e).__name__}, 1)
    emit({"ok": True, "spec": os.path.abspath(out_path),
          "actors": len(spec["actors"]), "shots": len(spec["shots"]),
          "total_frames": sum(s["frames"] for s in spec["shots"])}, 0)


if __name__ == "__main__":
    main()
