# -*- coding: utf-8 -*-
# 程序化 PET 瓶生成：车削(screw)剖面 + 瓶盖 + 标签环槽
# 用法: blender -b --python gen_bottle.py -- <out.glb>
import bpy, math, sys

bpy.ops.wm.read_factory_settings(use_empty=True)

# 剖面 (r, z)：瓶底→瓶身→标签环槽→肩→颈，高约 2.0（render_model 会归一化）
PROFILE = [
    (0.000, 0.000), (0.300, 0.000), (0.330, 0.030), (0.340, 0.080),
    (0.320, 0.110), (0.340, 0.150), (0.345, 0.300),
    (0.345, 0.450), (0.335, 0.470),                    # 标签槽上沿
    (0.335, 0.930), (0.345, 0.950),                    # 标签槽下沿
    (0.345, 1.000), (0.340, 1.120),
    (0.300, 1.300), (0.200, 1.500), (0.145, 1.600),    # 肩线
    (0.135, 1.780), (0.140, 1.820),                    # 瓶颈
]
verts = [(r, 0.0, z) for r, z in PROFILE]
edges = [(i, i + 1) for i in range(len(verts) - 1)]
me = bpy.data.meshes.new("bottle_profile")
me.from_pydata(verts, edges, [])
me.update()
body = bpy.data.objects.new("bottle_body", me)
bpy.context.collection.objects.link(body)
lathe = body.modifiers.new("lathe", "SCREW")
lathe.angle = math.radians(360.0)
lathe.steps = 64
lathe.render_steps = 64
lathe.use_smooth_shade = True

# 瓶盖：略宽于颈，顶面微收
bpy.ops.mesh.primitive_cylinder_add(radius=0.155, depth=0.20, vertices=64, location=(0, 0, 1.92))
cap = bpy.context.active_object
cap.name = "bottle_cap"
bev = cap.modifiers.new("cap_bevel", "BEVEL")
bev.width = 0.015
bev.segments = 2
for p in cap.data.polygons:
    p.use_smooth = True

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = argv[0] if argv else "/tmp/bottle.glb"
with_marker = "marker" in argv

# 非对称标记：标签带单侧凸起竖条——回转体各角度轮廓相同会导致 orbit 控制信号≈0，
# 一个单侧标记让白膜各帧结构不同，orbit 旋转才可被 controlnet 观测（生成图时被外观覆盖不影响）
if with_marker:
    # 竖棱凸起：贴标签带面(r=0.335)中心x=0.31，x半径0.06→棱面到0.37，出0.035
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, location=(0.31, 0.0, 0.70))
    mk = bpy.context.active_object
    mk.name = "bottle_marker"
    mk.scale = (0.06, 0.05, 0.36)
    for p in mk.data.polygons:
        p.use_smooth = True

# 修改器必须先应用再导出（glTF 默认 export_apply=False，边线剖面会丢面）
for o in list(bpy.context.scene.objects):
    bpy.context.view_layer.objects.active = o
    for m in list(o.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_yup=True)
print("BOTTLE_GLB", out)
