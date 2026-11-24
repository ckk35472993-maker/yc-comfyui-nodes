from __future__ import annotations

from typing import Any


H3_FRAME_OFFSET = 5
H3_FRAME_STRIDE = 17
H3_VIDEO_FPS = 24


def floor_h3_frame_count(frame_count: int) -> int:
    """Return the largest 17k+5 frame count that does not exceed the input."""
    count = int(frame_count)
    if count < H3_FRAME_OFFSET:
        raise ValueError(
            f"MiniMax H3 needs at least {H3_FRAME_OFFSET} video frames; received {count}."
        )
    return H3_FRAME_OFFSET + (
        (count - H3_FRAME_OFFSET) // H3_FRAME_STRIDE
    ) * H3_FRAME_STRIDE


def trim_audio_to_frames(audio: Any, frame_count: int, fps: int = H3_VIDEO_FPS):
    """Trim a ComfyUI AUDIO value to the duration of a video frame batch."""
    if audio is None:
        return None, None
    if not isinstance(audio, dict):
        raise ValueError("H3 AV Trim needs a ComfyUI AUDIO value when audio is connected.")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or not hasattr(waveform, "shape") or waveform.ndim < 2:
        raise ValueError("Connected AUDIO has no valid waveform tensor.")
    try:
        resolved_rate = int(sample_rate)
    except (TypeError, ValueError) as error:
        raise ValueError("Connected AUDIO has no valid sample_rate.") from error
    if resolved_rate < 1:
        raise ValueError("Connected AUDIO sample_rate must be positive.")

    target_samples = round(int(frame_count) * resolved_rate / int(fps))
    available_samples = int(waveform.shape[-1])
    trimmed_samples = min(available_samples, target_samples)
    output = audio.copy()
    output["waveform"] = waveform[..., :trimmed_samples]
    return output, target_samples
