# render_scene.py —— 多 actor 场景 + 多 shot 分镜 → 逐 shot clay/depth 控制序列
# 用法: blender --background --python render_scene.py -- --spec-file scene_spec.json
# spec(JSON):
#   out_dir, width, height, fps, seed
#   ground(bool 默认 true)  pedestal 场景模式不做（地面即可）
#   actors: [{id, model_path, pos:[x,y(,z)], rot_deg, target_size, scale_mode,
#             anim_path, on}]
#     model_path 必填（query 由 host 侧 scene_run.py 先解析成路径）
#     pos/rot_deg 缺省 [0,0]/0；target_size 缺省 1.0；scale_mode 同 render_model
#     on="<前置actor_id>"：叠放（min.z→目标顶面；pos 缺省=目标 XY 中心）；须排在目标之后
#   shots: [{id, focus, size, angle, motion, frames, t0}]
#     focus  actor id | "scene"(全体并集中心)，缺省 "scene"
#     size   els|ls|ms|mcu|cu|ecu → dist 倍率（缺省 ms）
#     angle  worm|low|eye|high|top|bird → elev_deg（缺省 eye）；
#            可用 elev_deg/dist_scale/fill 数值直接覆盖枚举
#     motion 与 render_model.py 相同 22 运镜枚举；frames 缺省 24
#     t0     时间轴起点（actor 动画从第几帧采，缺省 1）
# 产物: <out_dir>/shots/<shot_id>/clay|depth/*.png
# 契约: 末行 SELFCHECK {"shots":{id:{clay,depth,expected,ok}},
#                     "actors":{id:{objects,verts}},"overlap_pairs":[],"ok":bool}

import bpy
import json
import math
import os
import sys
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_lib as BL  # 共享：import_model/union_bbox/retarget_anim/运镜/depth 材质等


def parse_spec():
    spec = {
        "out_dir": "D:/_globixai/video_studio/out/blender_scene",
        "width": 768, "height": 512, "fps": 24, "seed": 7,
        "ground": True, "strict_layout": False,
        "actors": [], "shots": [],
    }
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    for i, a in enumerate(argv):
        if a == "--spec-file" and i + 1 < len(argv):
            with open(argv[i + 1], "r", encoding="utf-8") as f:
                spec.update(json.load(f))
        elif a == "--spec" and i + 1 < len(argv):
            spec.update(json.loads(argv[i + 1]))
    return spec


S = parse_spec()
OUT = S["out_dir"]
VALID_MOTIONS = BL.VALID_MOTIONS
SIZE_ENUM = {"els": 2.2, "ls": 1.6, "ms": 1.0, "mcu": 0.72, "cu": 0.5, "ecu": 0.32}
ANGLE_ENUM = {"worm": -25.0, "low": -8.0, "eye": 15.0,
              "high": 40.0, "top": 65.0, "bird": 78.0}
if not S["actors"]:
    print("FATAL actors[] empty")
    sys.exit(2)
if not S["shots"]:
    print("FATAL shots[] empty")
    sys.exit(2)

# ─── 场景 ───
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def link(obj):
    scene.collection.objects.link(obj)
    return obj


clay_mat = BL.new_mat("clay", (0.82, 0.82, 0.82))

# ─── actors 装配：每 actor 走 render_model 的"根空物体归一化"同款链 ───
ACTORS = {}   # id -> {root, subjects, bbox_mins, bbox_maxs}
ORIG_MATS = {}  # mesh datablock -> 原材质槽（spec.beauty 时第三通道恢复 PBR 用）
for idx, a in enumerate(S["actors"]):
    aid = str(a.get("id") or "actor%d" % idx)
    mp = str(a.get("model_path") or "").strip()
    if not mp or not os.path.isfile(mp):
        print("FATAL actor %s model_path missing/not found: %s" % (aid, mp))
        sys.exit(2)
    subjects = BL.import_model(mp)
    print("IMPORTED actor %s: %d objects from %s" % (aid, len(subjects), os.path.basename(mp)))

    # 动画 fbx 动作转移（共享库统一骨骼名匹配规则）
    ap = str(a.get("anim_path") or "").strip()
    if ap and os.path.isfile(ap):
        subjects = BL.retarget_anim(subjects, BL.import_model(ap), tag="actor %s" % aid)

    mins, maxs = BL.union_bbox(subjects)
    if mins is None:
        print("FATAL actor %s no mesh" % aid)
        sys.exit(2)
    size = maxs - mins
    mode = str(a.get("scale_mode", "height")).lower()
    ts = float(a.get("target_size", 1.0))
    if mode == "height":
        scale = ts / max(size.z, 1e-6)
    elif mode == "maxdim":
        scale = ts / max(size.x, size.y, size.z, 1e-6)
    else:
        scale = 1.0

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = "ACTOR_%s" % aid
    for o in subjects:
        if o.parent is None or o.parent not in subjects:
            o.parent = root
    root.scale = (scale, scale, scale)
    root.rotation_euler = (0, 0, math.radians(float(a.get("rot_deg", 0.0))))
    bpy.context.view_layer.update()
    wmins, wmaxs = BL.union_bbox(subjects)   # 旋转后实测世界 bbox，比"局部中心×scale"更准

    # pos=[x,y(,z_extra)]：actor 世界 bbox 的 XY 中心对准 pos，min.z 落 base_z。
    # on="<前置actor_id>"：base_z=目标顶面、pos 缺省=目标 XY 中心——"台灯放桌上"语义
    on_id = str(a.get("on") or "").strip()
    base_z = 0.0
    pos = a.get("pos")
    if on_id:
        tgt = ACTORS.get(on_id)
        if not tgt or tgt.get("wmaxs") is None:
            print("FATAL actor %s on='%s': 必须引用已装配的前置 actor" % (aid, on_id))
            sys.exit(2)
        base_z = float(tgt["wmaxs"].z)
        if pos is None:
            pos = [(tgt["wmins"].x + tgt["wmaxs"].x) * 0.5,
                   (tgt["wmins"].y + tgt["wmaxs"].y) * 0.5]
    pos = pos or [0.0, 0.0]
    cx_w = (wmins.x + wmaxs.x) * 0.5
    cy_w = (wmins.y + wmaxs.y) * 0.5
    root.location.x += float(pos[0]) - cx_w
    root.location.y += float(pos[1]) - cy_w
    root.location.z += base_z - wmins.z
    if len(pos) > 2:
        root.location.z += float(pos[2])          # pos[2]=额外抬高（悬空物/挂墙）
    bpy.context.view_layer.update()

    for o in subjects:
        if o.type == "MESH":
            if o.data not in ORIG_MATS:
                ORIG_MATS[o.data] = list(o.data.materials)
            o.data.materials.clear()
            o.data.materials.append(clay_mat)
    wmins2, wmaxs2 = BL.union_bbox(subjects)
    ACTORS[aid] = {"root": root, "subjects": subjects,
                   "wmins": wmins2, "wmaxs": wmaxs2,
                   "loc0": root.location.copy(),
                   "explode": a.get("explode")}

# 世界 bbox 已在装配时逐 actor 登记；这里只补元数据 + XY 重叠对报告
actor_meta = {}
for aid, rec in ACTORS.items():
    verts = sum(len(o.data.vertices) for o in rec["subjects"] if o.type == "MESH")
    actor_meta[aid] = {"objects": len(rec["subjects"]), "verts": verts}

overlap_pairs = []
ids = list(ACTORS)
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = ACTORS[ids[i]], ACTORS[ids[j]]
        ox = min(a["wmaxs"].x, b["wmaxs"].x) - max(a["wmins"].x, b["wmins"].x)
        oy = min(a["wmaxs"].y, b["wmaxs"].y) - max(a["wmins"].y, b["wmins"].y)
        if ox > 0.02 and oy > 0.02:
            overlap_pairs.append([ids[i], ids[j]])
if overlap_pairs:
    print("WARN overlap_pairs: %s" % overlap_pairs)
    if S.get("strict_layout"):
        print("FATAL strict_layout: overlapping actor AABBs")
        sys.exit(2)

# 地面 + 影棚灯（render_model 同款）
if S.get("ground", True):
    bpy.ops.mesh.primitive_plane_add(size=120, location=(0, 0, -0.01))
    gnd = bpy.context.active_object
    gnd.data.materials.append(BL.new_mat("gnd", (0.55, 0.55, 0.56)))

for name, loc, energy, sz in (("Key", (6, -4, 9), 1100, 6.0), ("Fill", (-6, 3, 7), 500, 7.0)):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.shape = "DISK"; ld.size = sz
    lo = link(bpy.data.objects.new(name, ld)); lo.location = loc
    BL.look_at(lo, (0, 0, 0.6))

w = bpy.data.worlds.new("w"); scene.world = w
w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.55, 0.58, 1)
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = int(S["width"])
scene.render.resolution_y = int(S["height"])
scene.render.fps = int(S["fps"])
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
layer = bpy.context.view_layer


def _smooth01(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def apply_explode(frame_no):
    """按帧号推 actor 炸开/归位位置。
    actor.explode = {"offset":[x,y,z], "range":[f0,f1], "mode":"out"|"in"}
    out: f0 合体 → f1 全炸开(+offset)；in 反向。范围外钳制。"""
    for rec in ACTORS.values():
        ex = rec.get("explode")
        if not ex:
            continue
        off = ex.get("offset") or [0, 0, 0]
        f0, f1 = (ex.get("range") or [1, 48])[:2]
        k = _smooth01((frame_no - f0) / max(1e-6, f1 - f0))
        if str(ex.get("mode", "out")).lower() == "in":
            k = 1.0 - k
        r = rec["root"]
        r.location = rec["loc0"] + Vector((off[0], off[1], off[2])) * k


def focus_bounds(focus):
    """返回 (cx, cy, subj_h, foot, tgt_z)——focus=actor id 用其世界 bbox，否则全体并集"""
    mins = Vector((1e9, 1e9, 1e9)); maxs = Vector((-1e9, -1e9, -1e9))
    if focus in ACTORS:
        mins, maxs = ACTORS[focus]["wmins"], ACTORS[focus]["wmaxs"]
    else:
        for rec in ACTORS.values():
            mins = Vector(map(min, mins, rec["wmins"]))
            maxs = Vector(map(max, maxs, rec["wmaxs"]))
    cx, cy = (mins.x + maxs.x) * 0.5, (mins.y + maxs.y) * 0.5
    h = max(maxs.z - mins.z, 0.2)
    foot = max(maxs.x - mins.x, maxs.y - mins.y)
    return cx, cy, h, foot, mins.z + h * 0.55


# ─── 逐 shot：独立相机架 → 运镜 → clay/depth 双通道 → 回收 ───
shot_results = {}
_used_sids = set()
for shot in S["shots"]:
    sid = str(shot.get("id") or "shot%d" % len(shot_results))
    if sid in _used_sids:  # 重复 id 会同目录帧交错——拒绝优于静默产出脏数据
        print("FATAL duplicate shot id: %s" % sid)
        sys.exit(2)
    _used_sids.add(sid)
    focus = str(shot.get("focus") or "scene")
    if focus != "scene" and focus not in ACTORS:
        print("FATAL shot %s focus '%s' not in actors" % (sid, focus))
        sys.exit(2)
    motion = str(shot.get("motion") or "still").strip().lower()
    if motion not in VALID_MOTIONS:
        print("FATAL shot %s unknown motion: %s" % (sid, motion))
        sys.exit(2)
    N = int(shot.get("frames", 24))
    t0 = int(shot.get("t0", 1))

    cx, cy, subj_h, foot, tgt_z = focus_bounds(focus)
    size_key = str(shot.get("size") or "ms").lower()
    dist_mult = SIZE_ENUM.get(size_key)
    if dist_mult is None:
        print("FATAL shot %s unknown size: %s (valid: %s)" % (sid, size_key, tuple(SIZE_ENUM)))
        sys.exit(2)
    angle_key = str(shot.get("angle") or "eye").lower()
    elev_deg = ANGLE_ENUM.get(angle_key)
    if elev_deg is None:
        print("FATAL shot %s unknown angle: %s (valid: %s)" % (sid, angle_key, tuple(ANGLE_ENUM)))
        sys.exit(2)
    elev_deg = float(shot.get("elev_deg", elev_deg))     # 数值覆盖枚举
    dist_scale = float(shot.get("dist_scale", dist_mult))

    # 相机距离反解（render_model 同公式，作用于 focus 尺寸）
    aspect = float(S["width"]) / max(1.0, float(S["height"]))
    half_fov_v = math.atan(18.0 / 50.0) if aspect < 1.0 else math.atan(18.0 / 50.0) / aspect
    fill = float(shot.get("fill", 0.65))
    dist_fit = (subj_h * 0.5) / max(0.05, math.tan(half_fov_v * fill))
    dist = max(3.0, foot * 1.35 + subj_h * 1.15, dist_fit) * dist_scale
    elev = math.radians(elev_deg)
    cam_r = dist * math.cos(elev)
    cam_z = max(BL.MIN_CAM_Z, dist * math.sin(elev) + tgt_z)   # 负俯角防钻地

    cam_d = bpy.data.cameras.new("cam_%s" % sid); cam_d.lens = 50
    cam = link(bpy.data.objects.new("cam_%s" % sid, cam_d))
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(cx, cy, 0))
    pivot = bpy.context.active_object
    cam.parent = pivot
    cam.location = (0, -cam_r, cam_z)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(cx, cy, tgt_z))
    ctgt = bpy.context.active_object
    con = cam.constraints.new("TRACK_TO")
    con.target = ctgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
    scene.camera = cam

    BL.apply_motion(motion, cam, cam_d, pivot, ctgt,
                    {"cam_r": cam_r, "cam_z": cam_z, "tgt_z": tgt_z,
                     "dist": dist, "subj_h": subj_h,
                     "t0": t0, "f1": t0 + N - 1, "cx": cx, "cy": cy})

    sdir = os.path.join(OUT, "shots", sid)
    for sub in ("clay", "depth"):
        os.makedirs(os.path.join(sdir, sub), exist_ok=True)
    layer.material_override = None
    w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.55, 0.58, 1)
    for i in range(N):
        scene.frame_set(t0 + i)
        apply_explode(t0 + i)
        scene.render.filepath = os.path.join(sdir, "clay", "clay_%04d.png" % (i + 1))
        bpy.ops.render.render(write_still=True)

    # depth 材质按本 shot 相机距离自动估计（可被 spec.depth_near/far 覆盖）
    dn = float(S.get("depth_near", 0.0)) or max(0.5, dist - subj_h * 1.2)
    df = float(S.get("depth_far", 0.0)) or (dist + subj_h * 2.5)
    dm = BL.make_depth_mat("depth_%s" % sid, dn, df)
    layer.material_override = dm
    w.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
    for i in range(N):
        scene.frame_set(t0 + i)
        apply_explode(t0 + i)
        scene.render.filepath = os.path.join(sdir, "depth", "depth_%04d.png" % (i + 1))
        bpy.ops.render.render(write_still=True)
    layer.material_override = None

    # beauty 通道（spec.beauty 才启用）：恢复 actor 原 PBR 材质渲第三遍，
    # 相机/灯光/世界与 clay 同架——广告"结构镜头"用，clay/depth 语义不变
    if S.get("beauty"):
        bdir = os.path.join(sdir, "beauty")
        os.makedirs(bdir, exist_ok=True)
        for me, mats in ORIG_MATS.items():
            me.materials.clear()
            for m in mats:
                me.materials.append(m)
        w.node_tree.nodes["Background"].inputs[0].default_value = (0.55, 0.55, 0.58, 1)
        # AgX 高光滚降 + 负曝光——Standard 变换下白产品 PBR 在影棚灯下全爆白
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.exposure = float(S.get("beauty_exposure", -0.6))
        for i in range(N):
            scene.frame_set(t0 + i)
            apply_explode(t0 + i)
            scene.render.filepath = os.path.join(bdir, "beauty_%04d.png" % (i + 1))
            bpy.ops.render.render(write_still=True)
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0.0
        for me in ORIG_MATS:
            me.materials.clear()
            me.materials.append(clay_mat)

    shot_results[sid] = {"clay": BL.count_seq(sdir, "clay", "clay_"),
                         "depth": BL.count_seq(sdir, "depth", "depth_"), "expected": N}
    if S.get("beauty"):
        shot_results[sid]["beauty"] = BL.count_seq(sdir, "beauty", "beauty_")
    shot_results[sid]["ok"] = (shot_results[sid]["clay"] == N
                             and shot_results[sid]["depth"] == N
                             and (not S.get("beauty")
                                  or shot_results[sid]["beauty"] == N))
    print("SHOT %s done: %s" % (sid, shot_results[sid]))

    # 回收本 shot 的相机架：cam/pivot/ctgt/cam_d/dm 全删——上一版漏删 cam 对象每 shot 泄漏一个
    for o in (cam, ctgt, pivot):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.data.cameras.remove(cam_d)
    bpy.data.materials.remove(dm)

res = {"shots": shot_results, "actors": actor_meta, "overlap_pairs": overlap_pairs}
res["ok"] = all(v["ok"] for v in shot_results.values())
print("SELFCHECK " + json.dumps(res))
sys.exit(0 if res["ok"] else 2)
