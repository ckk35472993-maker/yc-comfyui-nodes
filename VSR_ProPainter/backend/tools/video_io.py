import logging
import os
import queue
import subprocess
import threading

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI

logger = logging.getLogger(__name__)

# ── NVENC 可用性缓存（进程级，只探测一次）──────────────────────────────────
_NVENC_CACHE: dict = {}
_NVENC_LOCK = threading.Lock()


def _probe_nvenc(ffmpeg_path: str) -> bool:
    """检测当前 FFmpeg 是否支持 h264_nvenc（结果缓存）。"""
    with _NVENC_LOCK:
        if ffmpeg_path in _NVENC_CACHE:
            return _NVENC_CACHE[ffmpeg_path]
        available = False
        try:
            r = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=8,
            )
            available = "h264_nvenc" in r.stdout
        except Exception as e:
            logger.debug("[video_io] NVENC probe error: %s", e)
        _NVENC_CACHE[ffmpeg_path] = available
        logger.info("[video_io] h264_nvenc available=%s (%s)", available, ffmpeg_path)
        return available


def _build_encoder_args(ffmpeg_path: str) -> list:
    """返回最优视频编码参数：优先 NVENC，降级 libx264。"""
    if _probe_nvenc(ffmpeg_path):
        # h264_nvenc: 硬件编码，RTX 5090 下帧率可达数千 fps
        # -preset p4: 平衡速度/质量；-cq 18: 恒定质量（类 CRF）；-b:v 0: CQ 模式必须
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18", "-b:v", "0"]
    # 降级：CPU libx264，-preset fast -crf 18
    return ["-c:v", "libx264", "-crf", "18", "-preset", "fast"]


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self._buffer.put((ret, frame))
            if not ret:
                break

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。"""
        return self._buffer.get()

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧，优先使用 NVENC 硬件编码，降级 libx264。
    接口兼容 cv2.VideoWriter（write/release）。

    修复要点：
      1. 优先 h264_nvenc（RTX 5090 编码 241 帧 < 1s，彻底消除 encode/mux 阻塞）
      2. stderr=PIPE + 后台 drain 线程：防止 FFmpeg 日志塞满管道导致 SIGSTOP
      3. release() 超时从 600s → 120s：足够 CPU libx264 fallback，不再无限死等
    """

    def __init__(self, output_path, fps, size):
        w, h = size
        ffmpeg = FFmpegCLI.instance().ffmpeg_path
        encoder_args = _build_encoder_args(ffmpeg)
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "-",
        ] + encoder_args + [
            "-pix_fmt", "yuv420p",
            "-loglevel", "warning",   # warning 级别保留必要诊断，不产生大量噪音
            output_path,
        ]
        logger.info("[FFmpegVideoWriter] cmd: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # PIPE + 后台 drain 防止管道阻塞
        )
        # 后台线程消费 stderr，防止 64KB 管道缓冲区塞满导致 FFmpeg SIGSTOP
        self._stderr_lines: list = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="ffmpeg-stderr-drain"
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        try:
            for raw in self._process.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
                    logger.debug("[FFmpegVideoWriter] ffmpeg: %s", line)
        except Exception:
            pass

    def write(self, frame: np.ndarray) -> None:
        """写入一帧（numpy BGR uint8 数组）。"""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self) -> None:
        """关闭管道并等待 FFmpeg 编码完成（最多 120s，超时强制终止）。"""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        # 等待 stderr drain 线程结束（最多 3s）
        self._stderr_thread.join(timeout=3)
        try:
            # NVENC: < 2s；libx264 fast 10s 视频: < 30s；120s 绰绰有余
            self._process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            logger.error(
                "[FFmpegVideoWriter] FFmpeg encode 超时 120s，强制终止。"
                " stderr 最后 5 行: %s",
                self._stderr_lines[-5:],
            )
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
