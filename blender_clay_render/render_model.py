# render_model.py —— 外部 3D 模型(glb/gltf/fbx/obj/stl/ply) → clay/depth 控制序列
# 用法: blender --background --python render_model.py -- --spec-file spec.json
# spec(JSON):
#   model_path   必填：模型文件路径（glb/gltf/fbx/obj/stl/ply）
#   anim_path    可选：动画 fbx（骨骼名与模型一致时动作自动转移；不匹配仅告警）
#   out_dir, frames, fps, width, height, seed
#   motion       运镜预设枚举：orbit360 / orbit_half / orbit_quarter /
#                dolly_in / dolly_out / crane_up / still
#   target_size  归一化目标高度（米，默认 2.0）；scale_mode="height"|"maxdim"|"none"
#   subject_spin_deg  主体自转角度（默认 0；turntable 也可相机静+主体转）
#   dist_scale   相机距离倍率（默认 1.0，越小越近）；elev_deg 相机仰角（默认 22）
#   depth_near/depth_far  缺省按相机距离自动估计
# 产物: <out_dir>/clay/clay_*.png + <out_dir>/depth/depth_*.png（H3 control_video 原料）

import bpy
import json
import math
import os
import sys
from mathutils import Vector

# ─── 参数 ───
def parse_spec():
    spec = {
        "model_path": "", "anim_path": "",
        "out_dir": "D:/_globixai/video_studio/out/blender_model",
        "frames": 72, "fps": 24, "width": 768, "height": 512, "seed": 7,
        "motion": "orbit360",
        "target_size": 2.0, "scale_mode": "height",
        "subject_spin_deg": 0.0,
        "dist_scale": 1.0, "elev_deg": 22.0,
        "depth_near": 0.0, "depth_far": 0.0,      # 0 = 自动按相机距离估计
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
F0, F1 = 1, int(S["frames"])
MOTION = str(S["motion"]).strip().lower()
VALID_MOTIONS = ("orbit360", "orbit_half", "orbit_quarter",
                 "dolly_in", "dolly_out", "crane_up", "crane_down", "still",
                 "truck_l", "truck_r", "pan_l", "pan_r",
                 "tilt_up", "tilt_down", "pedestal_up", "pedestal_down",
                 "zoom_in", "zoom_out",
                 "helix_up", "helix_down", "reveal", "vertigo")
if MOTION not in VALID_MOTIONS:
    print("FATAL unknown motion: %s (valid: %s)" % (MOTION, VALID_MOTIONS))
    sys.exit(2)
MODEL_PATH = str(S["model_path"]).strip()
if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    print("FATAL model_path missing/not found: %s" % MODEL_PATH)
    sys.exit(2)

# ─── 工具 ───
def new_mat(name, rgb, rough=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    return m

def look_at(obj, pt):
    d = Vector(pt) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

def kf(obj, frame, **paths):
    for p in paths:
        obj.keyframe_insert(data_path=p, frame=frame)

def import_model(path):
    before = set(bpy.data.objects)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        print("FATAL unsupported model ext: %s" % ext)
        sys.exit(2)
    return [o for o in bpy.data.objects if o not in before]

def union_bbox(objs):
    """世界系并集包围盒（含骨架变形前的网格体；空物体/相机/灯跳过）"""
    mins = Vector((1e9, 1e9, 1e9)); maxs = Vector((-1e9, -1e9, -1e9))
    hit = False
    for o in objs:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mins = Vector(map(min, mins, w)); maxs = Vector(map(max, maxs, w))
            hit = True
    if not hit:
        return None, None
    return mins, maxs

# ─── 场景 ───
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

def link(obj):
    scene.collection.objects.link(obj)
    return obj

clay_mat = new_mat("clay", (0.82, 0.82, 0.82))
dark_mat = new_mat("pedestal", (0.32, 0.32, 0.33))

# 模型导入（anim_path 先记录，最后处理动作转移）
subjects = import_model(MODEL_PATH)
print("IMPORTED %d objects from %s" % (len(subjects), os.path.basename(MODEL_PATH)))

anim_path = str(S["anim_path"]).strip()
anim_objs = []
if anim_path and os.path.isfile(anim_path):
    anim_objs = import_model(anim_path)
    print("IMPORTED %d anim objects from %s" % (len(anim_objs), os.path.basename(anim_path)))

# 动作转移：动画 fbx 的 action 按骨骼名匹配装到主体骨架上（同 rig 库直配）
model_arm = next((o for o in subjects if o.type == "ARMATURE"), None)
anim_arm = next((o for o in anim_objs if o.type == "ARMATURE"), None)
if anim_arm and anim_arm.animation_data and anim_arm.animation_data.action:
    act = anim_arm.animation_data.action
    if model_arm:
        # 骨骼名交集判定（同 rig 库应高重合）
        mb = {b.name for b in model_arm.data.bones}
        ab = {b.name for b in anim_arm.data.bones}
        hit = len(mb & ab)
        if hit >= max(3, int(0.5 * len(ab))):
            model_arm.animation_data_create()
            model_arm.animation_data.action = act
            print("ANIM retarget ok: %d/%d bones matched" % (hit, len(ab)))
            # 动画骨架本体不渲染（主体用自己的网格皮）
            anim_arm.hide_render = True
            for o in anim_objs:
                if o is not anim_arm:
                    o.hide_render = True
        else:
            print("ANIM retarget skip: bone match %d/%d too low, playing anim armature instead" % (hit, len(ab)))
            subjects = subjects + [o for o in anim_objs if not o.hide_render]
    else:
        print("ANIM no model armature; playing anim armature itself as subject")
        subjects = subjects + anim_objs

# ─── 主体归一化：包围盒 → 缩放 → 底面贴地/居中 ───
mins, maxs = union_bbox(subjects)
if mins is None:
    print("FATAL no mesh in model")
    sys.exit(2)
size = maxs - mins
center = (mins + maxs) * 0.5
mode = str(S["scale_mode"]).lower()
ts = float(S["target_size"])
if mode == "height":
    scale = ts / max(size.z, 1e-6)
elif mode == "maxdim":
    scale = ts / max(size.x, size.y, size.z, 1e-6)
else:
    scale = 1.0
# 挂到根空物体统一变换（保留导入层级/骨架绑定关系）
bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
root = bpy.context.active_object
root.name = "SUBJ_ROOT"
for o in subjects:
    if o.parent is None or o.parent not in subjects:
        o.parent = root
root.scale = (scale, scale, scale)
root.location = (-center.x * scale, -center.y * scale, -mins.z * scale)

# 世界尺寸（归一化后）→ 展台/相机基准
wsize = size * scale
subj_h = max(wsize.z, 0.2)
foot = max(wsize.x, wsize.y)
tgt_z = subj_h * 0.55                     # 注视点：主体腰部偏上

# 主体自转（可选）：绕 Z 全程匀速
spin = float(S["subject_spin_deg"])
if abs(spin) > 1e-6:
    root.rotation_euler = (0, 0, 0)
    kf(root, 1, rotation_euler="rotation_euler")
    root.rotation_euler = (0, 0, math.radians(spin))
    kf(root, F1, rotation_euler="rotation_euler")

# 应用 clay 材质到所有网格（动画骨架隐藏的不影响）
for o in subjects:
    if o.type == "MESH":
        o.data.materials.clear()
        o.data.materials.append(clay_mat)

# 展台 + 地面（与 product_collapse 同款视觉锚）
# pedestal=false 时省略圆台只留地面——白膜无底座，与"纯底商品参考图"语义一致（H3 首帧引导不冲突）
if S.get("pedestal", True):
    ped_r = max(0.9, foot * 0.85)
    bpy.ops.mesh.primitive_cylinder_add(radius=ped_r, depth=0.24, vertices=48, location=(0, 0, -0.12))
    ped = bpy.context.active_object
    ped.data.materials.append(dark_mat)
bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, -0.24 if S.get("pedestal", True) else -0.01))
gnd = bpy.context.active_object
gnd.data.materials.append(new_mat("gnd", (0.55, 0.55, 0.56)))

# ─── 相机：pivot 环绕 + TRACK_TO 注视（与 product_collapse 同构） ───
cam_d = bpy.data.cameras.new("cam"); cam_d.lens = 50
cam = link(bpy.data.objects.new("cam", cam_d))
# 取景：宽物体按 foot 估算，瘦高物体按"主体占竖幅 fill 比例"从镜头 FOV 反解
_aspect = float(S["width"]) / max(1.0, float(S["height"]))
_half_fov_v = math.atan(18.0 / 50.0) if _aspect < 1.0 else math.atan(18.0 / 50.0) / _aspect
_fill = float(S.get("fill", 0.65))
_dist_fit = (subj_h * 0.5) / max(0.05, math.tan(_half_fov_v * _fill))
dist = max(3.0, foot * 1.35 + subj_h * 1.15, _dist_fit) * float(S["dist_scale"])
elev = math.radians(float(S["elev_deg"]))
cam_r = dist * math.cos(elev)
cam_z = dist * math.sin(elev) + tgt_z

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
pivot = bpy.context.active_object
cam.parent = pivot
cam.location = (0, -cam_r, cam_z)

bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, tgt_z))
ctgt = bpy.context.active_object
con = cam.constraints.new("TRACK_TO")
con.target = ctgt; con.track_axis = "TRACK_NEGATIVE_Z"; con.up_axis = "UP_Y"
scene.camera = cam

# 运镜预设 → pivot 旋转 / 相机位/焦距 关键帧
def set_pivot(deg, frame):
    pivot.rotation_euler = (0, 0, math.radians(deg))
    kf(pivot, frame, rotation_euler="rotation_euler")

def set_cam(r, z, frame):
    cam.location = (0, -r, z)
    kf(cam, frame, location="location")

if MOTION == "orbit360":
    set_pivot(0, 1); set_pivot(360, F1)
elif MOTION == "orbit_half":
    set_pivot(0, 1); set_pivot(180, F1)
elif MOTION == "orbit_quarter":
    set_pivot(0, 1); set_pivot(90, F1)
elif MOTION == "dolly_in":
    set_cam(cam_r * 1.35, cam_z, 1); set_cam(cam_r * 0.72, cam_z, F1)
    set_pivot(0, 1); set_pivot(12, F1)
elif MOTION == "dolly_out":
    set_cam(cam_r * 0.72, cam_z, 1); set_cam(cam_r * 1.35, cam_z, F1)
    set_pivot(0, 1); set_pivot(-12, F1)
elif MOTION == "crane_up":
    set_cam(cam_r, tgt_z * 0.25 + 0.4, 1)
    set_cam(cam_r * 0.9, cam_z * 1.5, F1)
    set_pivot(0, 1); set_pivot(20, F1)
elif MOTION == "crane_down":
    set_cam(cam_r * 0.9, cam_z * 1.5, 1)
    set_cam(cam_r, tgt_z * 0.25 + 0.4, F1)
    set_pivot(0, 1); set_pivot(-20, F1)
elif MOTION in ("truck_l", "truck_r"):
    lat = cam_r * (0.55 if MOTION == "truck_l" else -0.55)
    cam.location = (-lat, -cam_r, cam_z); kf(cam, 1, location="location")
    cam.location = (lat, -cam_r, cam_z); kf(cam, F1, location="location")
elif MOTION in ("pan_l", "pan_r"):
    off = dist * (0.35 if MOTION == "pan_l" else -0.35)
    set_pivot(0, 1)                                    # 相机原地，注视点横移=摇镜
    ctgt.location = (0, 0, tgt_z); ctgt.keyframe_insert("location", frame=1)
    ctgt.location = (off, 0, tgt_z); ctgt.keyframe_insert("location", frame=F1)
elif MOTION in ("tilt_up", "tilt_down"):
    up = MOTION == "tilt_up"
    set_pivot(0, 1)
    ctgt.location = (0, 0, tgt_z * (0.15 if up else 1.7))
    ctgt.keyframe_insert("location", frame=1)
    ctgt.location = (0, 0, tgt_z * (1.7 if up else 0.15))
    ctgt.keyframe_insert("location", frame=F1)
elif MOTION in ("pedestal_up", "pedestal_down"):
    dz = subj_h * (0.9 if MOTION == "pedestal_up" else -0.9)
    for o, z0 in ((cam, cam_z), (ctgt, tgt_z)):
        o.location.z = z0 - dz * 0.5; o.keyframe_insert("location", frame=1)
        o.location.z = z0 + dz * 0.5; o.keyframe_insert("location", frame=F1)
    set_pivot(0, 1)
elif MOTION in ("zoom_in", "zoom_out"):
    l0, l1 = (50.0, 90.0) if MOTION == "zoom_in" else (50.0, 30.0)
    cam_d.lens = l0; cam_d.keyframe_insert("lens", frame=1)
    cam_d.lens = l1; cam_d.keyframe_insert("lens", frame=F1)
elif MOTION in ("helix_up", "helix_down"):
    z0, z1 = (cam_z * 0.45, cam_z * 1.6) if MOTION == "helix_up" else (cam_z * 1.6, cam_z * 0.45)
    set_cam(cam_r, z0, 1); set_cam(cam_r * 0.9, z1, F1)
    set_pivot(0, 1); set_pivot(360, F1)
elif MOTION == "reveal":
    set_cam(cam_r * 0.5, cam_z * 0.7, 1)
    set_cam(cam_r * 1.9, cam_z * 1.15, F1)
    set_pivot(0, 1); set_pivot(18, F1)
elif MOTION == "vertigo":
    # 希区柯克：dolly_out + lens 反解 lens_t=50·d_t/d0 → 主体等大、背景拉伸
    d0, d1 = cam_r * 0.75, cam_r * 1.8
    set_cam(d0, cam_z, 1); set_cam(d1, cam_z, F1)
    cam_d.lens = 50.0; cam_d.keyframe_insert("lens", frame=1)
    cam_d.lens = 50.0 * d1 / d0; cam_d.keyframe_insert("lens", frame=F1)
elif MOTION == "still":
    set_pivot(35, 1)                  # 3/4 侧后经典机位，无运动

# 灯光：极简影棚（与 product_collapse 同款）
for name, loc, energy, size in (("Key", (4, -3, 7), 900, 5.0), ("Fill", (-4, 2, 5), 400, 6.0)):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.shape = "DISK"; ld.size = size
    lo = link(bpy.data.objects.new(name, ld)); lo.location = loc; look_at(lo, (0, 0, tgt_z * 0.8))

# 世界/渲染设置
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
scene.frame_start, scene.frame_end = F0, F1
scene.view_settings.view_transform = "Standard"     # 亮膜防暗（白膜过暗会拖H3色调）
scene.view_settings.look = "None"

# depth 材质：近亮远暗（depth_near/far=0 → 按相机距离自动估计）
dn = float(S["depth_near"]) or max(0.5, dist - subj_h * 1.2)
df = float(S["depth_far"]) or (dist + subj_h * 2.5)
dm = bpy.data.materials.new("depth")
dm.use_nodes = True
nt = dm.node_tree; nt.nodes.clear()
no = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
mr = nt.nodes.new("ShaderNodeMapRange")
cd = nt.nodes.new("ShaderNodeCameraData")
mr.inputs["From Min"].default_value = dn
mr.inputs["From Max"].default_value = df
mr.inputs["To Min"].default_value = 1.0
mr.inputs["To Max"].default_value = 0.0
nt.links.new(cd.outputs["View Distance"], mr.inputs["Value"])
nt.links.new(mr.outputs["Result"], em.inputs["Color"])
nt.links.new(em.outputs["Emission"], no.inputs["Surface"])
layer = bpy.context.view_layer

# ─── 渲染两通道 ───
for sub_dir in ("clay", "depth"):
    os.makedirs(os.path.join(OUT, sub_dir), exist_ok=True)

for f in range(F0, F1 + 1):
    scene.frame_set(f)
    scene.render.filepath = os.path.join(OUT, "clay", "clay_%04d.png" % f)
    bpy.ops.render.render(write_still=True)

layer.material_override = dm
w.node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)
for f in range(F0, F1 + 1):
    scene.frame_set(f)
    scene.render.filepath = os.path.join(OUT, "depth", "depth_%04d.png" % f)
    bpy.ops.render.render(write_still=True)
layer.material_override = None

# ─── 自检 ───
def count_seq(d, prefix):
    p = os.path.join(OUT, d)
    fs = [x for x in os.listdir(p) if x.startswith(prefix) and x.endswith(".png")] if os.path.isdir(p) else []
    return len([x for x in fs if os.path.getsize(os.path.join(p, x)) > 0])

exp = F1 - F0 + 1
res = {"clay": count_seq("clay", "clay_"), "depth": count_seq("depth", "depth_"), "expected": exp}
res["ok"] = res["clay"] == exp and res["depth"] == exp
print("SELFCHECK " + json.dumps(res))
sys.exit(0 if res["ok"] else 2)
