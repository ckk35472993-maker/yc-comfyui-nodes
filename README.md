# yc-comfyui-nodes

YingCe ComfyUI custom nodes collection (mirror).

每个目录为一个独立 custom node 包，安装时把对应目录放进 `ComfyUI/custom_nodes/` 后重启。

## 目录

| 目录 | 说明 |
|---|---|
| ComfyUI-YC-VideoKit | MiniMax H3 视频链路节点合集（PlanV2 计划链 + SAM3 选目标遮罩 + generation-mask 辅助）。RH 专用独立 repo：`github.com/ckk35472993-maker/yc-videokit` |
| comfyui-easy-sam3 | SAM3 视频/图像分割节点（detect/segment/objects） |
| ComfyUI-YingceDynamic | 动态变量/参数注入节点 |
| ComfyUI-VLObjects | VL 对象识别节点 |
| ComfyUI-CapRL | CapRL 图像描述节点 |
| face_sculpt | 面部微雕（Beauty/Warp）节点 |
| body_sculpt | 体型调整节点 |
| mask_tone_align | 遮罩色调对齐节点 |
| ref_subject_crop | 参考主体裁剪节点 |
| blender_clay_render | Blender 白膜渲染节点 |
| MMR_Remover | 视频修复/移除节点 |
| VSR_ProPainter | 视频去水印/修复节点 |

依赖说明：多数包零三方依赖（torch/numpy 由 ComfyUI 环境提供）；
comfyui-easy-sam3 依赖见其 pyproject.toml。
