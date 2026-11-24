from __future__ import annotations

import threading
import types
from typing import Any

import torch

if __package__:
    from .mask_ops import mask_row_targets, mixed_mod_row, snap_audio_mask, snap_video_mask
else:
    from mask_ops import mask_row_targets, mixed_mod_row, snap_audio_mask, snap_video_mask


STATE_KEY = "minimax_h3_per_row_masking_state"
WRAPPER_KEY = "minimax_h3_per_row_masking"
_SAMPLE_LOCK = threading.RLock()


def _scale_audio_velocity_for_core(
    h3: Any,
    audio_out: torch.Tensor,
    sigma_v: torch.Tensor,
    shift_v: float,
    shift_a: float,
) -> torch.Tensor:
    """Match the audio-velocity contract of the installed H3 core.

    Older ComfyUI builds expected ``_forward`` to convert the audio velocity
    onto the video sigma schedule. Newer ``ModelSamplingAV`` builds perform
    that conversion in ``MiniMaxH3Model.forward`` around diffusion-model
    wrappers and no longer expose ``time_shift_slope``.
    """
    time_shift_slope = getattr(h3, "time_shift_slope", None)
    if time_shift_slope is None:
        return audio_out
    return time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype) * audio_out


def validate_h3_model(model_patcher: Any) -> None:
    base_model = getattr(model_patcher, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None or diffusion_model.__class__.__name__ != "MiniMaxH3Model":
        actual = "None" if diffusion_model is None else diffusion_model.__class__.__name__
        raise ValueError(
            "H3 Per-Row Mask Patch requires a MiniMax H3 MODEL. "
            f"The connected diffusion model is {actual}."
        )

    required = (
        "blocks",
        "final_layer",
        "patch_size",
        "video_patch_proj",
        "audio_patch_proj",
        "_cond_video_rows",
        "_cond_audio_rows",
    )
    missing = [name for name in required if not hasattr(diffusion_model, name)]
    if missing:
        raise RuntimeError(
            "This ComfyUI MiniMax H3 implementation is incompatible with the temporary per-row patch; "
            f"missing: {', '.join(missing)}. Update this node or remove it after native support lands."
        )


def _mod_scale_shift(h: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, segments) -> torch.Tensor:
    for start, stop, row in segments:
        h[start:stop].mul_(1.0 + mixed_mod_row(scale, row, h.dtype)).add_(
            mixed_mod_row(shift, row, h.dtype)
        )
    return h


def _mod_gate(x: torch.Tensor, gate: torch.Tensor, other: torch.Tensor, segments) -> torch.Tensor:
    for start, stop, row in segments:
        x[start:stop].addcmul_(other[start:stop], mixed_mod_row(gate, row, x.dtype))
    return x


def _run_block(block, x, t_emb, mod_segments, rope_freqs, transformer_options):
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
    h = _mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
    x = _mod_gate(
        x,
        gate_msa,
        block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options),
        mod_segments,
    )
    h = _mod_scale_shift(block.norm2(x), shift_mlp, scale_mlp, mod_segments)
    return _mod_gate(x, gate_mlp, block.mlp(h), mod_segments)


def _run_blocks(model, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
    import comfy.model_prefetch

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    end = len(model.blocks) if end is None else end
    queue = comfy.model_prefetch.make_prefetch_queue(list(model.blocks[start:end]), h.device, transformer_options)
    for index in range(start, end):
        block = model.blocks[index]
        comfy.model_prefetch.prefetch_queue_pop(queue, h.device, block)
        if ("double_block", index) in blocks_replace:
            def block_wrap(args, _block=block):
                return {
                    "img": _run_block(
                        _block,
                        args["img"],
                        args["t_emb"],
                        args["mod_segments"],
                        args["rope_freqs"],
                        args["transformer_options"],
                    )
                }

            h = blocks_replace[("double_block", index)](
                {
                    "img": h,
                    "t_emb": t_emb,
                    "mod_segments": mod_segments,
                    "rope_freqs": rope_freqs,
                    "transformer_options": transformer_options,
                },
                {"original_block": block_wrap},
            )["img"]
        else:
            h = _run_block(block, h, t_emb, mod_segments, rope_freqs, transformer_options)
    if queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(queue, h.device, None)
    return h


def _final_layer(model, h, t_emb, video_seg, audio_seg):
    layer = model.final_layer
    shift, scale = layer.adaln_proj(t_emb)

    def modulate(segment):
        start, stop, row = segment
        normalized = layer.norm(h[start:stop])
        return (
            normalized * (1.0 + mixed_mod_row(scale, row, scale.dtype))
            + mixed_mod_row(shift, row, shift.dtype)
        ).to(torch.float32)

    return layer.video_out(modulate(video_seg)), layer.audio_out(modulate(audio_seg))


def _masked_h3_forward(
    model,
    x,
    timestep,
    context,
    transformer_options=None,
    minimax_payload=None,
    *,
    state,
    **kwargs,
):
    import comfy.ldm.common_dit
    import comfy.ldm.minimax.model as h3
    import comfy.model_management

    transformer_options = transformer_options or {}
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, model.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 per-row masking supports batch size 1.")

    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype
    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = h3.PackedLayout(
            text_len,
            latent_t,
            lat_h,
            lat_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", model.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", model.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - h3.time_shift_sigma(sigma_v, shift_v, shift_a))

    visual_aug = float(payload.get("visual_cond_noise_aug", h3.VISUAL_COND_TIMESTEP))
    audio_aug = float(payload.get("audio_cond_noise_aug", h3.AUDIO_COND_TIMESTEP))
    segment_t = {
        "text": t_v,
        "video": t_v,
        "audio": t_a,
        "cond": max(t_v, visual_aug),
        "ref_img": max(t_v, visual_aug),
        "ref_audio": max(t_a, audio_aug),
    }
    pin_video_t = max(t_v, h3.VISUAL_COND_TIMESTEP)
    pin_audio_t = max(t_a, h3.AUDIO_COND_TIMESTEP)

    video_weights = None
    audio_weights = None
    video_mask = state.get("video_mask")
    if video_mask is not None:
        targets = mask_row_targets(
            video_mask[0, 0].to(torch.float32),
            latent_t,
            lat_h,
            lat_w,
            tuple(model.patch_size[1:]),
        )
        if targets is not None:
            if bool(targets.any()):
                video_weights = targets.to(torch.float32).unsqueeze(1)
            else:
                segment_t["video"] = pin_video_t

    audio_mask = state.get("audio_mask")
    if audio_mask is not None:
        targets = audio_mask[0, 0].to(torch.float32).reshape(-1) >= 0.5
        if not bool(targets.all()):
            if bool(targets.any()):
                audio_weights = targets.to(torch.float32).unsqueeze(1)
            else:
                segment_t["audio"] = pin_audio_t

    unique_t = sorted(
        {t_v, t_a}
        | {segment_t[kind] for _, _, kind in layout.segments}
        | ({pin_video_t} if video_weights is not None else set())
        | ({pin_audio_t} if audio_weights is not None else set())
    )
    timestep_row = {value: index for index, value in enumerate(unique_t)}
    segment_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, stop, kind in layout.segments:
        row_base = timestep_row[segment_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for index in range(1, stop - start + 1):
                if index == stop - start or tags[index] != tags[run_start]:
                    mod_segments.append((start + run_start, start + index, row_base + int(tags[run_start])))
                    run_start = index
        elif kind == "video" and video_weights is not None:
            mod_segments.append(
                (
                    start,
                    stop,
                    (
                        row_base + segment_tag[kind],
                        timestep_row[pin_video_t] * 3 + segment_tag[kind],
                        video_weights,
                    ),
                )
            )
        elif kind == "audio" and audio_weights is not None:
            mod_segments.append(
                (
                    start,
                    stop,
                    (
                        row_base + segment_tag[kind],
                        timestep_row[pin_audio_t] * 3 + segment_tag[kind],
                        audio_weights,
                    ),
                )
            )
        else:
            mod_segments.append((start, stop, row_base + segment_tag[kind]))

    image_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = h3.patchify_video(video_x.to(torch.float32), model.patch_size)
    audio_rows = h3.pack_audio(audio_x.to(torch.float32))
    cond_video_rows = model._cond_video_rows(payload, device)
    cond_audio_rows = model._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(
            image_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device
        )
        all_video_rows[~image_update] = cond_video_rows
        all_video_rows[image_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(
            audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device
        )
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = model.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = model.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != model.hidden_size:
        text_states = model.token_refiner(
            model.condition_proj(text_states), transformer_options=transformer_options
        )

    hidden = torch.empty(layout.seq_len, model.hidden_size, dtype=dtype, device=device)
    video_offset = 0
    audio_offset = 0
    for start, stop, kind in layout.segments:
        count = stop - start
        if kind == "text":
            hidden[start:stop] = text_states
        elif kind in ("cond", "ref_img", "video"):
            hidden[start:stop] = video_embed[video_offset:video_offset + count]
            video_offset += count
        else:
            hidden[start:stop] = audio_embed[audio_offset:audio_offset + count]
            audio_offset += count

    timestep_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if model.use_adaln_curves:
        table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
        position = timestep_values.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        t_emb = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    else:
        t_emb = model.time_embedder(timestep_values).to(dtype)

    rope_freqs = h3.rope_rotation_table(model.rope_freqs(layout.position_ids, device), dtype)
    blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
    if ("block_loop", 0) in blocks_replace:
        cache_ranges = [(start, stop) for start, stop, kind in layout.segments if kind in ("audio", "video")]

        def block_loop_wrap(args):
            return {
                "img": _run_blocks(
                    model,
                    args["img"],
                    args["t_emb"],
                    args["mod_segments"],
                    args["rope_freqs"],
                    args["transformer_options"],
                    args.get("start", 0),
                    args.get("end"),
                )
            }

        hidden = blocks_replace[("block_loop", 0)](
            {
                "img": hidden,
                "t_emb": t_emb,
                "mod_segments": mod_segments,
                "rope_freqs": rope_freqs,
                "transformer_options": transformer_options,
                "cache_ranges": cache_ranges,
                "block_count": len(model.blocks),
            },
            {"original_block": block_loop_wrap},
        )["img"]
    else:
        hidden = _run_blocks(model, hidden, t_emb, mod_segments, rope_freqs, transformer_options)

    video_start, video_stop, _ = next(segment for segment in layout.segments if segment[2] == "video")
    audio_start, audio_stop, _ = next(segment for segment in layout.segments if segment[2] == "audio")
    video_seg = (
        video_start,
        video_stop,
        (timestep_row[segment_t["video"]], timestep_row[pin_video_t], video_weights)
        if video_weights is not None
        else timestep_row[segment_t["video"]],
    )
    audio_seg = (
        audio_start,
        audio_stop,
        (timestep_row[segment_t["audio"]], timestep_row[pin_audio_t], audio_weights)
        if audio_weights is not None
        else timestep_row[segment_t["audio"]],
    )
    video_out_rows, audio_out_rows = _final_layer(model, hidden, t_emb, video_seg, audio_seg)
    video_out = h3.unpatchify_video(
        video_out_rows,
        latent_t,
        lat_h // model.patch_size[1],
        lat_w // model.patch_size[2],
        model.latents_dim,
        model.patch_size,
    )
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = h3.unpack_audio(audio_out_rows)
    audio_out = _scale_audio_velocity_for_core(h3, audio_out, sigma_v, shift_v, shift_a)
    return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]


def diffusion_model_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None and len(args) > 3:
        transformer_options = args[3]
    state = (transformer_options or {}).get(STATE_KEY)
    if state is None:
        return executor(*args, **kwargs)

    model = executor.class_obj

    def replacement(*inner_args, **inner_kwargs):
        return _masked_h3_forward(model, *inner_args, state=state, **inner_kwargs)

    # Preserve wrappers registered after this one: replace only the eventual original.
    executor.original = replacement
    return executor(*args, **kwargs)


def _h3_scale_latent_inpaint(latent_shapes):
    def scale(self, sigma, noise, latent_image, **kwargs):
        import comfy.ldm.minimax.model as h3
        import comfy.utils

        clean_streams = comfy.utils.unpack_latents(latent_image, latent_shapes)
        noise_streams = comfy.utils.unpack_latents(noise, latent_shapes)
        augmentation = h3.VISUAL_COND_TIMESTEP
        clean_streams[0] = (
            augmentation * clean_streams[0]
            + (1.0 - augmentation) * noise_streams[0]
        )
        return comfy.utils.pack_latents(clean_streams)[0]

    return scale


def outer_sample_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    if denoise_mask is None:
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    if latent_shapes is None or len(latent_shapes) < 2:
        raise ValueError(
            "H3 per-row masking requires a joint MiniMax H3 video+audio latent. "
            "Connect the output of Concat AV Latent."
        )

    import comfy.utils

    masks = list(comfy.utils.unpack_latents(denoise_mask, latent_shapes))
    if len(masks) < 2:
        raise ValueError("The H3 generation mask is missing its audio mask component.")
    masks[0] = snap_video_mask(masks[0])
    masks[1] = snap_audio_mask(masks[1])
    denoise_mask, _ = comfy.utils.pack_latents(masks)

    if bool((masks[0] >= 0.5).all()) and bool((masks[1] >= 0.5).all()):
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )

    sampler_object = executor.class_obj
    transformer_options = sampler_object.model_options.setdefault("transformer_options", {})
    state = {"video_mask": masks[0], "audio_mask": masks[1]}
    base_model = sampler_object.model_patcher.model
    had_instance_override = "scale_latent_inpaint" in base_model.__dict__
    previous_override = base_model.__dict__.get("scale_latent_inpaint")

    with _SAMPLE_LOCK:
        transformer_options[STATE_KEY] = state
        base_model.scale_latent_inpaint = types.MethodType(
            _h3_scale_latent_inpaint(latent_shapes), base_model
        )
        try:
            return executor(
                noise,
                latent_image,
                sampler,
                sigmas,
                denoise_mask,
                callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        finally:
            transformer_options.pop(STATE_KEY, None)
            if had_instance_override:
                base_model.scale_latent_inpaint = previous_override
            else:
                delattr(base_model, "scale_latent_inpaint")


def install_model_wrappers(model_patcher):
    import comfy.patcher_extension

    validate_h3_model(model_patcher)
    patched = model_patcher.clone()
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        WRAPPER_KEY,
        outer_sample_wrapper,
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        diffusion_model_wrapper,
    )
    return patched
