# MMR_Remover — MiniMax-Remover ComfyUI 薄封装节点

`IMAGE[B,H,W,C] + MASK[B,H,W] → IMAGE[B,H,W,C]`。封装 zibojia/MiniMax-Remover
（DiT 视频移除，6 步蒸馏、无 CFG）。检测/裁区/分段/回贴归 Bridge，
节点只做"帧+逐帧 mask → 修复帧"（与 VSR_ProPainter 同构同约定）。

## 输入

| 参数 | 默认 | 说明 |
|---|---|---|
| images | — | IMAGE 批，任意帧数 |
| masks | — | MASK 批；单帧 mask 自动广播全批 |
| model_dir | `minimax_remover` | 模型目录（需 `transformer/ vae/ scheduler/` 三子目录） |
| num_inference_steps | 12 | 官方口径 6-12 |
| iterations | 6 | **mask 膨胀轮数**（`expand_masks`），不是去噪迭代 |
| seed | 42 | 生成器种子 |

## 关键行为（与上游 pipeline 对齐，2026-11 实测）

- **81 帧窗口**：Wan VAE 时序压缩 `(N-1)/4+1`，N=81 帧数守恒。节点内部按 81
  分块，尾窗复制末帧补齐、输出裁回原长——任意批长输入帧数守恒。
- **单帧 mask 广播**：MASK 批为 1 帧时扩展到与 IMAGE 批同长；不等长取短并截齐。
- **分辨率**：任意尺寸，内部双线性缩到 16 倍数（输出插值回原尺寸）。
- **模型路径解析**：`model_dir` 绝对路径 > `ComfyUI/models/minimax_remover` >
  `custom_nodes/_models_store/minimax_remover` > 包内 `models/`。
- 管线 `pipeline_minimax_remover.py`/`transformer_minimax_remover.py` vendor 自
  上游仓库（部署机 `/root/mmr` 同文件 md5 一致）；上游更新→整目录重拷。

## 实测警示（避免踩坑）

1. **填充模式方差**：同输入同 seed 在不同进程环境可能落到"干净填充"或
   "幻觉填充"两种模式（fp16 选核差异放大）。生产建议 seed 重试+QC 挑优，
   或引擎侧加确定性开关。
2. **整帧过 VAE**：非遮罩区也重编码，非逐字节保真——合成时只贴回 mask 区
   （与 ProPainter 的透传特性不同）。
3. **尾部上下文**：mask 区压在窗口尾端时未来帧不足易留残影，Bridge 切片
   需补尾帧。
4. 权重位于部署机 `/data`（不落根盘）；许可同上游仓库，商用前需核。

## 部署

```
custom_nodes/MMR_Remover/
├── __init__.py
├── mmr_remover_node.py      # 节点本体
├── minimax_remover/         # vendored pipeline（勿改，重拷同步）
│   ├── pipeline_minimax_remover.py
│   └── transformer_minimax_remover.py
└── models/minimax_remover/  # 部署侧权重（git 忽略）：transformer/ vae/ scheduler/
```

验证：API `VHS_LoadVideoPath→ImageToMask→MMR_Remover→VHS_VideoCombine`
（2026-11-12 实测 success，81f@480×864≈28s/窗@4090）。
