# blender_lib.py —— blender -b 脚本的共享库（新脚本一律 import，别再复制）
# ⚠️ render_model.py 是 5090 ComfyUI 节点部署源、保持单文件自包含不 import 本库；
#   render_scene.py 及后续脚本统一从这里复用（两文件同目录部署即可）
# 用法（blender -b 脚本内）:
#   sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
#   import blender_lib as BL
import bpy
import math
import os
from mathutils import Vector

VALID_MOTIONS = ("orbit360", "orbit_half", "orbit_quarter",
                 "dolly_in", "dolly_out", "crane_up", "crane_down", "still",
                 "truck_l", "truck_r", "pan_l", "pan_r",
                 "tilt_up", "tilt_down", "pedestal_up", "pedestal_down",
                 "zoom_in", "zoom_out",
                 "helix_up", "helix_down", "reveal", "vertigo")


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
    """按扩展名分派导入（glb/gltf/fbx/obj/stl/ply），返回新增对象列表"""
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
        return []
    return [o for o in bpy.data.objects if o not in before]


def union_bbox(objs):
    """世界系并集包围盒；空物体/相机/灯跳过。返回 (mins,maxs) 或 (None,None)"""
    mins = Vector((1e9, 1e9, 1e9)); maxs = Vector((-1e9, -1e9, -1e9))
    hit = False
    for o in objs:
        if o.type != "MESH":
            continue
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            mins = Vector(map(min, mins, w)); maxs = Vector(map(max, maxs, w))
            hit = True
    return (mins, maxs) if hit else (None, None)


def retarget_anim(subjects, anim_objs, tag=""):
    """动画 fbx 动作转移：骨骼名交集>=50%且>=3 则 action 装主体骨架并隐藏动画源；
    否则把动画对象并入 subjects 让它自己演。返回新 subjects 列表（含/不含 anim 对象）"""
    model_arm = next((o for o in subjects if o.type == "ARMATURE"), None)
    anim_arm = next((o for o in anim_objs if o.type == "ARMATURE"), None)
    if anim_arm and anim_arm.animation_data and anim_arm.animation_data.action:
        act = anim_arm.animation_data.action
        if model_arm:
            mb = {b.name for b in model_arm.data.bones}
            ab = {b.name for b in anim_arm.data.bones}
            hit = len(mb & ab)
            if hit >= max(3, int(0.5 * len(ab))):
                model_arm.animation_data_create()
                model_arm.animation_data.action = act
                print("ANIM %s retarget ok: %d/%d" % (tag, hit, len(ab)))
                for o in anim_objs:
                    o.hide_render = True
                return subjects
            print("ANIM %s retarget skip: bone match %d/%d too low" % (tag, hit, len(ab)))
            return subjects + [o for o in anim_objs if not o.hide_render]
        print("ANIM %s no model armature; playing anim armature itself" % tag)
        return subjects + anim_objs
    return subjects


def make_depth_mat(name, dn, df):
    """近亮远暗的 depth 材质（CameraData.ViewDistance→MapRange→Emission）"""
    dm = bpy.data.materials.new(name)
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
    return dm


MIN_CAM_Z = 0.05   # 地面 z=0 上方硬下限：worm/low/负 elev 防镜头钻地（贴地仰拍语义保留）


def apply_motion(motion, cam, cam_d, pivot, ctgt, ctx):
    """22 运镜枚举 → pivot 旋转/相机位/焦距 关键帧（render_scene 移植版）。
    ctx = {cam_r, cam_z, tgt_z, dist, subj_h, t0, f1, cx, cy}
    pivot 位于 (cx,cy,0)、ctgt 位于 (cx,cy,tgt_z)；cam.location 为 pivot 局部系"""
    cam_r, cam_z, tgt_z = ctx["cam_r"], ctx["cam_z"], ctx["tgt_z"]
    dist, subj_h = ctx["dist"], ctx["subj_h"]
    t0, f1 = ctx["t0"], ctx["f1"]
    cx, cy = ctx["cx"], ctx["cy"]

    def set_pivot(deg, frame):
        pivot.rotation_euler = (0, 0, math.radians(deg))
        kf(pivot, frame, rotation_euler="rotation_euler")

    def set_cam(r, z, frame):
        cam.location = (0, -r, max(MIN_CAM_Z, z))
        kf(cam, frame, location="location")

    if motion == "orbit360":
        set_pivot(0, t0); set_pivot(360, f1)
    elif motion == "orbit_half":
        set_pivot(0, t0); set_pivot(180, f1)
    elif motion == "orbit_quarter":
        set_pivot(0, t0); set_pivot(90, f1)
    elif motion == "dolly_in":
        set_cam(cam_r * 1.35, cam_z, t0); set_cam(cam_r * 0.72, cam_z, f1)
        set_pivot(0, t0); set_pivot(12, f1)
    elif motion == "dolly_out":
        set_cam(cam_r * 0.72, cam_z, t0); set_cam(cam_r * 1.35, cam_z, f1)
        set_pivot(0, t0); set_pivot(-12, f1)
    elif motion == "crane_up":
        set_cam(cam_r, tgt_z * 0.25 + 0.4, t0)
        set_cam(cam_r * 0.9, cam_z * 1.5, f1)
        set_pivot(0, t0); set_pivot(20, f1)
    elif motion == "crane_down":
        set_cam(cam_r * 0.9, cam_z * 1.5, t0)
        set_cam(cam_r, tgt_z * 0.25 + 0.4, f1)
        set_pivot(0, t0); set_pivot(-20, f1)
    elif motion in ("truck_l", "truck_r"):
        lat = cam_r * (0.55 if motion == "truck_l" else -0.55)
        cam.location = (-lat, -cam_r, max(MIN_CAM_Z, cam_z)); kf(cam, t0, location="location")
        cam.location = (lat, -cam_r, max(MIN_CAM_Z, cam_z)); kf(cam, f1, location="location")
    elif motion in ("pan_l", "pan_r"):
        off = dist * (0.35 if motion == "pan_l" else -0.35)
        set_pivot(0, t0)
        ctgt.location = (cx, cy, tgt_z); ctgt.keyframe_insert("location", frame=t0)
        ctgt.location = (cx + off, cy, tgt_z); ctgt.keyframe_insert("location", frame=f1)
    elif motion in ("tilt_up", "tilt_down"):
        up = motion == "tilt_up"
        set_pivot(0, t0)
        ctgt.location = (cx, cy, tgt_z * (0.15 if up else 1.7))
        ctgt.keyframe_insert("location", frame=t0)
        ctgt.location = (cx, cy, tgt_z * (1.7 if up else 0.15))
        ctgt.keyframe_insert("location", frame=f1)
    elif motion in ("pedestal_up", "pedestal_down"):
        dz = subj_h * (0.9 if motion == "pedestal_up" else -0.9)
        for o, z0 in ((cam, cam_z), (ctgt, tgt_z)):
            lo, hi = z0 - dz * 0.5, z0 + dz * 0.5
            if o is cam:
                lo, hi = max(MIN_CAM_Z, lo), max(MIN_CAM_Z, hi)
            o.location.z = lo; o.keyframe_insert("location", frame=t0)
            o.location.z = hi; o.keyframe_insert("location", frame=f1)
        set_pivot(0, t0)
    elif motion in ("zoom_in", "zoom_out"):
        l0, l1 = (50.0, 90.0) if motion == "zoom_in" else (50.0, 30.0)
        cam_d.lens = l0; cam_d.keyframe_insert("lens", frame=t0)
        cam_d.lens = l1; cam_d.keyframe_insert("lens", frame=f1)
    elif motion in ("helix_up", "helix_down"):
        z0, z1 = (cam_z * 0.45, cam_z * 1.6) if motion == "helix_up" else (cam_z * 1.6, cam_z * 0.45)
        set_cam(cam_r, z0, t0); set_cam(cam_r * 0.9, z1, f1)
        set_pivot(0, t0); set_pivot(360, f1)
    elif motion == "reveal":
        set_cam(cam_r * 0.5, cam_z * 0.7, t0)
        set_cam(cam_r * 1.9, cam_z * 1.15, f1)
        set_pivot(0, t0); set_pivot(18, f1)
    elif motion == "vertigo":
        d0, d1 = cam_r * 0.75, cam_r * 1.8
        set_cam(d0, cam_z, t0); set_cam(d1, cam_z, f1)
        cam_d.lens = 50.0; cam_d.keyframe_insert("lens", frame=t0)
        cam_d.lens = 50.0 * d1 / d0; cam_d.keyframe_insert("lens", frame=f1)
    elif motion == "still":
        set_pivot(35, t0)


def count_seq(out_dir, d, prefix):
    p = os.path.join(out_dir, d)
    fs = [x for x in os.listdir(p) if x.startswith(prefix) and x.endswith(".png")] if os.path.isdir(p) else []
    return len([x for x in fs if os.path.getsize(os.path.join(p, x)) > 0])
