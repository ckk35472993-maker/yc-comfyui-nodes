from __future__ import annotations

import torch

if __package__:
    from .mask_ops import GRID_MODES, H3_PIXEL_CELL, quantize_h3_pixel_mask
    from .media_ops import H3_VIDEO_FPS, floor_h3_frame_count, trim_audio_to_frames
    from .mask_runtime import install_model_wrappers
else:
    from mask_ops import GRID_MODES, H3_PIXEL_CELL, quantize_h3_pixel_mask
    from media_ops import H3_VIDEO_FPS, floor_h3_frame_count, trim_audio_to_frames
    from .mask_runtime import install_model_wrappers


class MiniMaxH3TrimSourceAV:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Source video frame batch. Trailing frames are dropped to the largest valid "
                            "MiniMax H3 17k+5 frame count. Frames are never padded or resized."
                        )
                    },
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional synchronized source audio. It is trimmed to the duration of the "
                            "returned frame batch at H3's fixed 24 fps."
                        )
                    },
                )
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("trimmed_frames", "trimmed_audio", "h3_length", "trim_info")
    FUNCTION = "trim"
    CATEGORY = "YC/VideoKit"
    DESCRIPTION = (
        "Drops trailing source frames to the largest valid MiniMax H3 17k+5 length and, "
        "when connected, trims synchronized audio to the same 24 fps duration."
    )

    def trim(self, frames, audio=None):
        if frames.ndim != 4 or frames.shape[-1] < 3:
            raise ValueError(
                f"H3 AV Trim expects ComfyUI IMAGE frames [T,H,W,C], got {list(frames.shape)}."
            )

        source_count = int(frames.shape[0])
        h3_length = floor_h3_frame_count(source_count)
        trimmed_frames = frames[:h3_length]
        trimmed_audio, target_samples = trim_audio_to_frames(audio, h3_length)

        dropped = source_count - h3_length
        duration = h3_length / H3_VIDEO_FPS
        audio_text = "no audio connected"
        if trimmed_audio is not None:
            actual_samples = int(trimmed_audio["waveform"].shape[-1])
            sample_rate = int(trimmed_audio["sample_rate"])
            audio_text = f"audio {actual_samples}/{target_samples} samples at {sample_rate} Hz"
            if actual_samples < target_samples:
                audio_text += " (source audio was shorter; no silence was added)"
        info = (
            f"H3 source trim: {source_count} -> {h3_length} frames; dropped {dropped}; "
            f"duration {duration:.3f}s at {H3_VIDEO_FPS} fps; {audio_text}."
        )
        return (trimmed_frames, trimmed_audio, h3_length, info)


class MiniMaxH3PerRowMaskPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "MiniMax H3 MODEL to patch for per-row mixed-timestep generation masking. "
                            "Connect this MODEL output to your guider."
                        )
                    },
                )
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("patched_model",)
    FUNCTION = "patch"
    CATEGORY = "YC/VideoKit"
    DESCRIPTION = (
        "Temporarily enables per-row mixed-timestep masking for MiniMax H3 without "
        "editing ComfyUI core files. Only the returned MODEL is affected."
    )

    def patch(self, model):
        return (install_model_wrappers(model),)


class MiniMaxH3SetGenerationMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Joint source video+audio latent. Encode the source media, then combine it with "
                            "Concat AV Latent before connecting it here."
                        )
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Video-space generation mask. It may contain one mask frame or a batch matching "
                            "the source video. The sampler resizes and snaps it to H3's patch grid."
                        )
                    },
                ),
                "mask_meaning": (
                    ["white = generate", "white = preserve"],
                    {"tooltip": "Choose whether white pixels are edited or protected."},
                ),
                "audio_mode": (
                    ["generate audio", "preserve source audio"],
                    {
                        "tooltip": (
                            "Generate audio normally, or pin the complete source-audio latent while editing video."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("masked_av_latent",)
    FUNCTION = "set_mask"
    CATEGORY = "YC/VideoKit"
    DESCRIPTION = (
        "Attaches a MiniMax H3 video generation mask to a joint source AV latent. "
        "White normally means generate; black means preserve."
    )

    def set_mask(self, av_latent, mask, mask_meaning, audio_mode):
        import comfy.nested_tensor

        samples = av_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError(
                "H3 Set Generation Mask needs a joint video+audio LATENT. "
                "Use Concat AV Latent before this node."
            )
        streams = samples.unbind()
        if len(streams) < 2:
            raise ValueError("The connected AV latent does not contain both video and audio streams.")

        video_mask = mask.to(torch.float32).clamp(0.0, 1.0)
        if mask_meaning == "white = preserve":
            video_mask = 1.0 - video_mask
        audio_mask = (
            torch.ones_like(streams[1], dtype=torch.float32)
            if audio_mode == "generate audio"
            else torch.zeros_like(streams[1], dtype=torch.float32)
        )

        output = av_latent.copy()
        output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
        return (output,)


class MiniMaxH3MaskGridPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "The exact resized H3 canvas frames that will be VAE-encoded. Width and height "
                            "must be divisible by 32 so the displayed cells align with H3."
                        )
                    },
                ),
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Connect SAM3 Detect, SAM3 Track to Mask, a painted mask, or any standard "
                            "ComfyUI MASK. White is treated as the requested generation area."
                        )
                    },
                ),
                "cell_selection": (
                    list(GRID_MODES),
                    {
                        "tooltip": (
                            "Runtime exact reproduces H3's latent resize and per-row max decision. Other "
                            "modes select cells from direct pixel coverage and can be more or less conservative."
                        )
                    },
                ),
                "cell_adjust": (
                    "INT",
                    {
                        "default": 0,
                        "min": -8,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "Grow with positive values or shrink with negative values, measured in complete "
                            "32x32 H3 cells."
                        ),
                    },
                ),
                "overlay_opacity": (
                    "FLOAT",
                    {
                        "default": 0.38,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Opacity of the orange generated-cell overlay.",
                    },
                ),
                "show_grid": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Draw cyan boundaries for every 32x32 H3 cell.",
                    },
                ),
                "show_source_outline": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Draw the original SAM3 or input-mask boundary in white.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("snapped_mask", "grid_preview", "grid_info")
    FUNCTION = "preview"
    CATEGORY = "YC/VideoKit"
    DESCRIPTION = (
        "Shows exactly which 32x32 MiniMax H3 cells a SAM3 or standard ComfyUI mask will edit, "
        "and outputs the cell-snapped mask for H3 Set Generation Mask."
    )

    def preview(
        self,
        image,
        mask,
        cell_selection,
        cell_adjust,
        overlay_opacity,
        show_grid,
        show_source_outline,
    ):
        import torch.nn.functional as functional

        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError(f"Expected ComfyUI IMAGE [frames,H,W,C], got {list(image.shape)}")
        frames, height, width = image.shape[:3]
        raw, snapped, cells = quantize_h3_pixel_mask(
            mask,
            frames,
            height,
            width,
            mode=cell_selection,
            cell_adjust=cell_adjust,
        )

        preview = image[..., :3].to(torch.float32).clamp(0.0, 1.0).clone()
        selected = snapped.to(device=preview.device, dtype=preview.dtype).unsqueeze(-1)
        orange = torch.tensor([1.0, 0.18, 0.0], device=preview.device, dtype=preview.dtype)
        opacity = float(overlay_opacity)
        preview = preview * (1.0 - selected * opacity) + orange * selected * opacity

        if show_grid:
            grid = torch.zeros((height, width), device=preview.device, dtype=torch.bool)
            grid[::H3_PIXEL_CELL, :] = True
            grid[:, ::H3_PIXEL_CELL] = True
            grid[-1, :] = True
            grid[:, -1] = True
            cyan = torch.tensor([0.0, 0.82, 1.0], device=preview.device, dtype=preview.dtype)
            grid_alpha = 0.72
            preview[:, grid] = preview[:, grid] * (1.0 - grid_alpha) + cyan * grid_alpha

        if show_source_outline:
            source = (raw.to(device=preview.device) >= 0.5).to(preview.dtype)
            dilated = functional.max_pool2d(source, kernel_size=3, stride=1, padding=1)
            eroded = 1.0 - functional.max_pool2d(1.0 - source, kernel_size=3, stride=1, padding=1)
            outline = ((dilated - eroded) > 0).movedim(1, -1)
            preview = torch.where(outline, torch.ones_like(preview), preview)

        counts = cells.sum(dim=(1, 2, 3)).to(torch.int64).cpu().tolist()
        cells_per_frame = cells.shape[-2] * cells.shape[-1]
        if len(counts) <= 8:
            count_text = ", ".join(str(value) for value in counts)
        else:
            count_text = f"min {min(counts)}, max {max(counts)}, total {sum(counts)}"
        grid_info = (
            f"H3 grid: {cells.shape[-1]}x{cells.shape[-2]} cells at {H3_PIXEL_CELL}x{H3_PIXEL_CELL} "
            f"pixels; selected cells per frame: {count_text} / {cells_per_frame}; "
            f"mode: {cell_selection}; cell adjustment: {cell_adjust:+d}."
        )
        return (snapped, preview, grid_info)


class MiniMaxH3TemporalUnionMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": (
                    "MASK",
                    {
                        "tooltip": (
                            "Per-frame video mask. Returns the union across the time axis so every frame "
                            "shares the same generation region, preventing per-frame row snap flicker."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("union_mask",)
    FUNCTION = "union"
    CATEGORY = "YC/VideoKit"
    DESCRIPTION = (
        "Collapses a per-frame video mask into a single temporal union mask, repeated for every frame. "
        "Use before Set Generation Mask to stop mask rows flipping between generate/preserve."
    )

    def union(self, mask):
        m = mask.to(torch.float32).clamp(0.0, 1.0)
        if m.dim() == 2:
            return (m,)
        unioned = m.amax(dim=0, keepdim=True).expand_as(m).contiguous()
        return (unioned,)


class MiniMaxH3SelectObjectMask:
    # 从 SAM3 视频跟踪的多目标结果中选定单一目标，输出逐帧对齐的 mask
    # 解决 obj_masks 输出为 (B, N, H, W) 直接进下游被展平成 B*N 张 mask
    # 导致多个人物的遮罩错位叠加、预览"到处飘"的问题
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "objects": (
                    "EASY_SAM3_OBJECTS_OUTPUT",
                    {"tooltip": "easy sam3VideoSegmentation 的 objects 输出"},
                ),
                "mode": (
                    ["largest", "center", "leftmost", "rightmost", "index"],
                    {
                        "default": "largest",
                        "tooltip": (
                            "目标选择策略: largest=全程累计面积最大(主角通常最大), "
                            "center=质心最接近画面中心, leftmost/rightmost=最左/最右, "
                            "index=按下方的 index 指定(已按首次出现位置从左到右排序)"
                        ),
                    },
                ),
                "index": ("INT", {"default": 0, "min": 0, "max": 999, "tooltip": "mode=index 时生效"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "select"
    CATEGORY = "YC/VideoKit"
    DESCRIPTION = (
        "Pick exactly one tracked object from SAM3 video segmentation output and return "
        "its per-frame mask (B, H, W). Prevents multi-person union/misaligned masks."
    )

    def select(self, objects, mode="largest", index=0):
        import numpy as np

        obj_masks = (objects or {}).get("obj_masks")
        if not isinstance(obj_masks, list) or len(obj_masks) == 0:
            # 无检测结果: 返回单帧空 mask, 避免下游崩溃
            return (torch.zeros((1, 1, 1), dtype=torch.float32),)

        frames = []
        for m in obj_masks:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = np.asarray(m)
            if m.dtype == bool:
                m = m.astype(np.float32)
            frames.append(m)

        n_obj = max(int(f.shape[0]) for f in frames)
        H, W = frames[0].shape[-2], frames[0].shape[-1]

        # 逐目标统计: 累计面积 + 面积加权平均质心
        area_sum = np.zeros(n_obj, dtype=np.float64)
        cx_sum = np.zeros(n_obj, dtype=np.float64)
        cy_sum = np.zeros(n_obj, dtype=np.float64)
        ys, xs = np.mgrid[0:H, 0:W]
        for f in frames:
            for j in range(f.shape[0]):
                a = float(f[j].sum())
                if a <= 0:
                    continue
                area_sum[j] += a
                cx_sum[j] += float((f[j] * xs).sum())
                cy_sum[j] += float((f[j] * ys).sum())

        valid = area_sum > 0
        if not valid.any():
            return (torch.zeros((len(frames), H, W), dtype=torch.float32),)

        mean_cx = np.divide(cx_sum, area_sum, out=np.full(n_obj, W / 2.0), where=valid)
        mean_cy = np.divide(cy_sum, area_sum, out=np.full(n_obj, H / 2.0), where=valid)

        if mode == "index":
            pick = max(0, min(int(index), n_obj - 1))
        elif mode == "leftmost":
            pick = int(np.argmin(np.where(valid, mean_cx, W)))
        elif mode == "rightmost":
            pick = int(np.argmax(np.where(valid, mean_cx, -1.0)))
        elif mode == "center":
            dist = np.hypot(mean_cx - W / 2.0, mean_cy - H / 2.0)
            pick = int(np.argmin(np.where(valid, dist, np.inf)))
        else:  # largest
            pick = int(np.argmax(np.where(valid, area_sum, -1.0)))

        print(f"[MiniMaxH3SelectObjectMask] objects={n_obj} mode={mode} pick={pick} "
              f"areas={area_sum.astype(int).tolist()}")

        out = torch.zeros((len(frames), H, W), dtype=torch.float32)
        for i, f in enumerate(frames):
            if pick < f.shape[0]:
                out[i] = torch.from_numpy(np.asarray(f[pick], dtype=np.float32))
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TrimSourceAV": MiniMaxH3TrimSourceAV,
    "MiniMaxH3PerRowMaskPatch": MiniMaxH3PerRowMaskPatch,
    "MiniMaxH3SetGenerationMask": MiniMaxH3SetGenerationMask,
    "MiniMaxH3MaskGridPreview": MiniMaxH3MaskGridPreview,
    "MiniMaxH3TemporalUnionMask": MiniMaxH3TemporalUnionMask,
    "MiniMaxH3SelectObjectMask": MiniMaxH3SelectObjectMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TrimSourceAV": "YC 媒体·源片裁剪",
    "MiniMaxH3PerRowMaskPatch": "YC 遮罩·分行补丁",
    "MiniMaxH3SetGenerationMask": "YC 遮罩·生成区域",
    "MiniMaxH3MaskGridPreview": "YC 遮罩·网格预览",
    "MiniMaxH3TemporalUnionMask": "YC 遮罩·时序并集",
    "MiniMaxH3SelectObjectMask": "YC 遮罩·目标选取",
}
