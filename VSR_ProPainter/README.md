# VSR_ProPainter（ComfyUI 节点）

动态水印/字幕的**时序填充核**——薄封装自家生产引擎
`backend.inpaint.propainter_inpaint.PropainterInpaint`，与 `run_vsr_local`
/`cloud-remove-worker --inpaint-mode propainter --mask-video` 逐字节同行为。

## 架构分工（循环在外）

- **Bridge/编排层**：模板提取 → `local_vsr_masker.build_presence_mask_video`
  逐帧存在检测+迟滞窗 → `compute_watermark_context_crop` 裁区 → 分段 → 回贴。
- **本节点**：只吃 `IMAGE batch + MASK batch`，输出填充后的 IMAGE batch。
  mask=1 的像素区域被重建；mask=0 的像素**逐字节透传**（已验证 diff=0.0）。

## 节点

`VSR_ProPainter`（分类 `video/remove`）

| 输入 | 类型 | 说明 |
|---|---|---|
| images | IMAGE | [N,H,W,C] RGB 0~1 帧批 |
| masks | MASK | [N,H,W] 逐帧 mask；单帧 mask 自动广播 |
| model_dir | STRING | `propainter` 或绝对路径 |
| sub_video_length | INT | 单批帧数上限=生产 `VSR_PROP_MAX_LOAD`；≤16G 卡 ≤50 |
| mask_dilation | INT | =`VSR_MASK_DILATION`，默认 4 |
| fp16 | BOOLEAN | 默认开 |

## 权重（需手动部署，不进 git）

`/root/ComfyUI/models/propainter/`：
`ProPainter.pth`、`raft-things.pth`、`recurrent_flow_completion.pth`
（源：`~/VideoAI_Data/modules/models/remove/propainter/`，NTU S-Lab 非商用注意）

## 接线

```
VHS_LoadVideoPath(源段) ──IMAGE──┐
                                ├─ VSR_ProPainter ──IMAGE── VHS_VideoCombine
VHS_LoadVideoPath(mask视频) ──→MASK──┘   (mask 帧数须与源段一致)
```

## vendored backend/ 同步约定

`backend/` 是从 `video_ai_core/local/VSR-local/backend/` **整目录拷贝**的部署副本
（含 8 处桥接补丁 + `_widen_interval_masks` 等）。上游 backend 有改动时**整目录
重新拷贝**覆盖，禁止在副本里单独改——避免双源漂移。

## 已验证（4090 standalone）

24f@720×640 合成框 mask：mask 区 diff 0.0797（真填充）、区外 diff 0.0（透传）、
模型懒加载常驻（`_INSTANCE_CACHE` 按参数键复用）。

## 部署备忘（2026-11-14，4090/5090 双机实测）

- **pip 依赖**：`requirements.txt`（openvino + pyclipper + shapely，OCR det 走 OV 链；
  缺 openvino 会回退 `import paddle` 报 ModuleNotFoundError）。容器内 pip 装，
  重建容器后需重装（或打进镜像层）。
- **8 对齐硬约束**：`VSR_ProPainter` 内 vendored RAFT 要求帧宽/高均能被 8 整除，
  否则 grid_sampler batch 错配崩溃（960×540、967×544 均复现）。
  正确接法=先经 `VSR_ContextCrop`（`align_to_8x`）裁区再填充；整帧喂入须
  保证 LoadVideo 输出尺寸 W%8==0 且 H%8==0（如 960×544）。
- **VRAM**：960×544×40f 在占用 26G/32G 的 5090 上 OOM；16f 可跑。
  `sub_video_length` 与帧数是主要显存杠杆。
- 5090 `comfyui130_node_1` 已部署本目录 + `_models_store/minimax_remover`
  + `models/{propainter,sttn-det,ocr}`（big-lama 走 ComfyUI `models/inpaint`）。
