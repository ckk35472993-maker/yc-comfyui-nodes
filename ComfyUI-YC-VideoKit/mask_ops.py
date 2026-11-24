from __future__ import annotations

import torch
import torch.nn.functional as F


H3_VAE_SCALE = 16
H3_PIXEL_CELL = 32
GRID_MODES = (
    "runtime exact (latent max)",
    "any pixel coverage",
    "50% pixel coverage",
    "full pixel coverage",
)


def _pad_spatial_replicate(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crop or replicate-pad the last two dimensions to the requested size."""
    mask = mask[..., :height, :width]
    pad_h = max(0, height - mask.shape[-2])
    pad_w = max(0, width - mask.shape[-1])
    if pad_h or pad_w:
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="replicate")
    return mask


def snap_video_mask(mask: torch.Tensor, patch_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
    """Snap a prepared [B,C,T,H,W] denoise mask to H3's spatial DiT rows."""
    if mask.ndim != 5:
        raise ValueError(f"Expected a prepared video mask [B,C,T,H,W], got {list(mask.shape)}")

    ph, pw = patch_size
    if ph < 1 or pw < 1:
        raise ValueError("H3 patch dimensions must be positive.")

    batch, channels, frames, height, width = mask.shape
    work = mask[:, :1].to(torch.float32)
    pad_h = (-height) % ph
    pad_w = (-width) % pw
    if pad_h or pad_w:
        work = F.pad(work, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")

    padded_h, padded_w = work.shape[-2:]
    work = work.reshape(batch, 1, frames, padded_h // ph, ph, padded_w // pw, pw)
    work = (work.amax(dim=(-3, -1)) >= 0.5).to(mask.dtype)
    work = work.repeat_interleave(ph, dim=-2).repeat_interleave(pw, dim=-1)
    work = work[..., :height, :width]
    return work.expand(batch, channels, frames, height, width).contiguous()


def snap_audio_mask(mask: torch.Tensor) -> torch.Tensor:
    """Make a prepared [B,C,stereo,T] audio denoise mask binary per audio row."""
    if mask.ndim != 4:
        raise ValueError(f"Expected a prepared audio mask [B,C,channels,T], got {list(mask.shape)}")
    row_mask = (mask.to(torch.float32).amax(dim=1, keepdim=True) >= 0.5).to(mask.dtype)
    return row_mask.expand_as(mask).contiguous()


def mask_row_targets(
    mask: torch.Tensor,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    patch_size: tuple[int, int] = (2, 2),
) -> torch.Tensor | None:
    """Convert [T,H,W] (1=generate) to one boolean target per H3 video row."""
    if mask.ndim != 3:
        raise ValueError(f"Expected [T,H,W] video mask, got {list(mask.shape)}")
    if mask.shape[0] != latent_t:
        mask = F.interpolate(
            mask[None, None].to(torch.float32),
            size=(latent_t, mask.shape[-2], mask.shape[-1]),
            mode="nearest",
        )[0, 0]

    ph, pw = patch_size
    mask = _pad_spatial_replicate(mask, latent_h, latent_w)
    padded_h = ((latent_h + ph - 1) // ph) * ph
    padded_w = ((latent_w + pw - 1) // pw) * pw
    mask = _pad_spatial_replicate(mask, padded_h, padded_w)
    rows = mask.reshape(latent_t, padded_h // ph, ph, padded_w // pw, pw).amax(dim=(2, 4))
    targets = rows.reshape(-1) >= 0.5
    return None if bool(targets.all()) else targets


def mixed_mod_row(values: torch.Tensor, row, dtype: torch.dtype) -> torch.Tensor:
    """Select a modulation row or blend generated/pinned rows per token."""
    if isinstance(row, tuple):
        target_row, pinned_row, weight = row
        return torch.lerp(values[pinned_row], values[target_row], weight.to(values.dtype)).to(dtype)
    return values[row].to(dtype)


def resize_mask_to_canvas(
    mask: torch.Tensor,
    frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize a Comfy [frames,H,W] mask to [frames,1,height,width]."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 3:
        raise ValueError(f"Expected a ComfyUI MASK [frames,H,W], got {list(mask.shape)}")
    if frames < 1 or height < 1 or width < 1:
        raise ValueError("Mask preview canvas dimensions must be positive.")

    work = mask.to(torch.float32).clamp(0.0, 1.0)[None, None]
    work = F.interpolate(
        work,
        size=(frames, height, width),
        mode="trilinear",
        align_corners=False,
    )
    return work[0].movedim(0, 1).contiguous()


def _grow_or_shrink_cells(cells: torch.Tensor, amount: int) -> torch.Tensor:
    if amount == 0:
        return cells
    radius = abs(int(amount))
    kernel = radius * 2 + 1
    if amount > 0:
        return F.max_pool2d(cells, kernel_size=kernel, stride=1, padding=radius)

    # Treat the area beyond the canvas as unselected so edge cells shrink too.
    inverse = 1.0 - cells
    inverse = F.pad(inverse, (radius, radius, radius, radius), mode="constant", value=1.0)
    return 1.0 - F.max_pool2d(inverse, kernel_size=kernel, stride=1)


def quantize_h3_pixel_mask(
    mask: torch.Tensor,
    frames: int,
    height: int,
    width: int,
    mode: str = GRID_MODES[0],
    cell_adjust: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return resized raw mask, snapped pixel mask, and [B,1,gh,gw] cell map."""
    if mode not in GRID_MODES:
        raise ValueError(f"Unknown H3 mask grid mode: {mode}")
    if height % H3_PIXEL_CELL or width % H3_PIXEL_CELL:
        raise ValueError(
            "Accurate H3 grid preview requires the image canvas width and height to be divisible "
            f"by {H3_PIXEL_CELL}. Resize the source frames to the exact H3 canvas first; "
            f"received {width}x{height}."
        )

    raw = resize_mask_to_canvas(mask, frames, height, width)
    grid_h = height // H3_PIXEL_CELL
    grid_w = width // H3_PIXEL_CELL

    if mode == "runtime exact (latent max)":
        latent = F.interpolate(
            raw,
            size=(height // H3_VAE_SCALE, width // H3_VAE_SCALE),
            mode="bilinear",
            align_corners=False,
        )
        cells = latent.reshape(
            frames,
            1,
            grid_h,
            H3_PIXEL_CELL // H3_VAE_SCALE,
            grid_w,
            H3_PIXEL_CELL // H3_VAE_SCALE,
        ).amax(dim=(-3, -1))
        cells = (cells >= 0.5).to(raw.dtype)
    else:
        pixels = raw.reshape(
            frames,
            1,
            grid_h,
            H3_PIXEL_CELL,
            grid_w,
            H3_PIXEL_CELL,
        )
        if mode == "any pixel coverage":
            cells = (pixels.amax(dim=(-3, -1)) >= 0.5).to(raw.dtype)
        elif mode == "50% pixel coverage":
            cells = (pixels.mean(dim=(-3, -1)) >= 0.5).to(raw.dtype)
        else:
            cells = (pixels.amin(dim=(-3, -1)) >= 0.5).to(raw.dtype)

    cells = _grow_or_shrink_cells(cells, cell_adjust)
    snapped = cells.repeat_interleave(H3_PIXEL_CELL, dim=-2).repeat_interleave(
        H3_PIXEL_CELL, dim=-1
    )
    return raw, snapped[:, 0].contiguous(), cells
