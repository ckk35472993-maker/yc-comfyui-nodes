# ref_subject_crop — 参考图单主体守护节点

> 防"多人参考图选错身份"：H3/MiniMax ref2va 类换角色工作流的参考图编码器
> **不是"取画面最大人脸"**——参考图里若含画中画人物（相框照片、镜面反射、
> 海报人像、路人），编码器可能选中错误实体作为替换身份。本节点在
> `LoadImage → 参考图输入` 之间统一裁切：检出全部人脸 → 锁定主脸 →
> 头肩框裁切并把其他检出人脸挤出框外。

## 接线（任何需要人物参考图的工作流通用）

```
LoadImage → RefSubjectCrop.image → RefSubjectCrop.image → <下游 ref_image / 参考槽>
                                                       ↘ RefSubjectCrop.debug_image → PreviewImage（可选）
                                                       ↘ RefSubjectCrop.info       → 日志/调试
```

- `h3_char_swap`：`img1`/`img2` 各串一个实例再进 `sw_refimg0`
- `h3_product_replace_sam3`、换脸/换装类：同理插在被当作"身份来源"的 LoadImage 后

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| pick_mode | largest | 主脸选择：largest=候选里面积最大 / best=检测分最高 / center / left / right |
| primary_min_score | 0.8 | 主脸候选检测分门槛。低于此分的脸全部视为"待排除其他人"——防低分误检（如手部假脸 135k px² > 真脸 48k）抢主脸 |
| min_score | 0.5 | 检出下限；调低可抓更小的照片人脸以便排除 |
| expand_top/bottom/side | 1.0/2.2/1.3 | 脸框向上/下/侧扩展倍数（头肩构图；全身参考调大 bottom） |
| exclude_others | true | 其他人脸整体位于主脸某侧时，把该侧裁切边收到其边界外 |
| fallback | passthrough | 0 人脸时：透传 / 中心 3:4 兜底裁切 |
| debug_overlay | false | 输出画框调试图（绿=主脸 红=其他人 黄=裁切框） |

## 输出

- `image`：裁后图（0 脸/单脸全幅时=原图透传）
- `debug_image`：调试图（未开 overlay 时=输出图）
- `info`：JSON 串 `{faces, picked, cropped, reason, crop_box, img_wh, boxes, excluded_faces, overlapped_faces}`

## 部署

- **自包含**：`models/face_detection_yunet_2023mar.onnx`（232KB，opencv_zoo，MIT）随包发布，无外部模型路径约定；推理走 CPU（~30ms），不占显存。
- **依赖**：`cv2 + numpy + torch`（ComfyUI 生态常见依赖）。`install.py` 只提示不强装；缺失时节点照常注册、运行时 fail-open 透传（不会挂工作流，但失去防护）。
- **RunningHub**：节点未发布到 RH 前，**不要把本节点接进要跑 RH 的工作流**（unknown node = 整个任务失败，不是降级）。RH 侧防护请走"上传时裁切"（见下）。

## 上传时裁切（覆盖 RH/网页端）

同一算法已封装为共享模块 `video_ai_core/local/ref_subject_guard.py` +
Bridge 端点 `POST /api/media/ref_subject_crop`：前端在上传
`subject_guard.slots` 标记的图片槽前调端点拿 `crop_box`，本地裁切后上传
干净图——到达任何后端（含 RH）的参考图已免疫。节点与端点共用
`ref_subject_guard.py` 同一实现（包内为自包含副本，改算法请三处同步）。

## 为什么需要"排他"而不只是"选大脸"

实测陷阱帧（720×1280）YuNet 检出 3 个候选：真脸 48k/0.944、相框小人脸
336/0.692、**手部误检 135k/0.635（比真脸大 3 倍）**。单纯取最大会选中手；
score≥0.8 候选过滤后"最大"才安全。而"裁切框把其他人脸圈进去"是第二坑
——扩展框若不收边，相框人仍在参考图内继续干扰编码器。
