# ComfyUI-YC-VideoKit

MiniMax H3 视频链路节点合集，单包安装。

## 节点

### PlanV2 计划链（`MiniMaxH3PlanV2*`）

把「分镜 + 参考素材 + 角色 + 台词时序」编译为合法 H3 prose，
并由 `ApplyReferencePlan` 路由进原生 I2V/R2V 采样路径：

`ProjectSetup` / `ImageReference` / `VideoReference` / `AudioReference` /
`SubjectBinding` / `CharacterReplacement` / `Shot` / `ShotKeyframe` /
`ShotMotionReference` / `DialogueEvent` / `FoleyTarget` /
`PromptMerge` / `PromptReview` / `PromptOverride` / `ApplyReferencePlan`

### 遮罩与媒体辅助

- `MiniMaxH3SelectObjectMask` — 从 SAM3 视频分割的多目标结果中按空间语义
  （largest / center / leftmost / rightmost / index）选定单一目标，
  输出逐帧对齐的 MASK，替代 obj_id 盲选。
- `MiniMaxH3TemporalUnionMask` / `MiniMaxH3MaskGridPreview` /
  `MiniMaxH3PerRowMaskPatch` / `MiniMaxH3SetGenerationMask` —
  generation-mask 补丁与预览工具。
- `MiniMaxH3TrimSourceAV` — 源片音视频定长裁剪。

## 依赖

零三方依赖；numpy / torch 由 ComfyUI 环境提供。
