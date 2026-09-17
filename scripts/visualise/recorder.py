"""
@file recorder.py
@brief Record Pygame surfaces to an MP4 by piping raw frames to ffmpeg.

The visualiser renders only the newest frame at an uncapped rate, so
FrameRecorder emits frames on a wall-clock accumulator instead, repeating
the last surface when the stream is dry, matching playback to real time.
"""

import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pygame

# Output encoding. CRF 18 is visually lossless for flat vector-style graphics
# while keeping the file small; yuv420p is the pixel format every player and
# browser accepts (the default yuv444p is rejected by Safari/QuickTime).
_CRF = "18"
_PIX_FMT_OUT = "yuv420p"
_PRESET = "medium"

# libx264 requires even frame dimensions. The map area is sized from the lot
# aspect ratio at runtime (see LiveVisualiser._compute_viewport), so the window
# height is not known in advance and can be odd - pad rather than constrain it.
_PAD_FILTER = "pad=ceil(iw/2)*2:ceil(ih/2)*2"

_DEFAULT_FPS = 30
# A recorder started before the first surface arrives has nothing to repeat, so
# frame emission simply waits for the first capture() call.
_FFMPEG_MISSING_MSG = (
    "ffmpeg not found on PATH. Recording needs the ffmpeg binary - install it "
    "with 'sudo apt-get install ffmpeg' and verify with 'make check-host-deps'."
)


def check_ffmpeg() -> Optional[str]:
    """
    @brief Locate the ffmpeg binary.
    @return Absolute path to ffmpeg, or None when it is not on PATH.
    """
    return shutil.which("ffmpeg")


def surface_to_rgb_array(surface: pygame.Surface) -> np.ndarray:
    """
    @brief Convert a Pygame surface to an (H, W, 3) uint8 RGB array.

    pygame.surfarray returns column-major (W, H, 3) data, so the first two axes
    are swapped to give the row-major layout ffmpeg's rawvideo demuxer expects.

    @param surface: Surface to convert.
    @return Array of shape (height, width, 3), dtype uint8.
    """
    return np.asarray(pygame.surfarray.array3d(surface)).swapaxes(0, 1)


class FrameRecorder:
    """
    @class FrameRecorder
    @brief Encode Pygame surfaces to an MP4 via an ffmpeg subprocess pipe.

    Usage:
        rec = FrameRecorder(Path("out.mp4"), width, height)
        rec.start()
        while running:
            rec.capture(screen)   # call every draw; emits at the fixed rate
        rec.stop()

    @note The frame geometry is fixed when ffmpeg starts. Surfaces of a
          different size are letterboxed into that geometry rather than
          restarting the encoder, so a mid-recording window resize yields one
          continuous file instead of a corrupt stream.
    @warning A dead encoder is reported once and then ignored; recording is
             never allowed to take the viewer down with it.
    """

    def __init__(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: int = _DEFAULT_FPS,
    ) -> None:
        """
        @brief Configure a recorder (does not start ffmpeg).
        @param output_path: Destination .mp4 path; parent dirs are created.
        @param width: Frame width in pixels.
        @param height: Frame height in pixels.
        @param fps: Output frame rate.
        """
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = max(1, int(fps))
        self._frame_interval = 1.0 / float(self._fps)

        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._last_frame: Optional[np.ndarray] = None
        self._next_frame_time: float = 0.0
        self._start_time: float = 0.0
        self._frames_written: int = 0
        self._failed: bool = False

    @property
    def is_recording(self) -> bool:
        """@brief True while the encoder is running and healthy."""
        return self._proc is not None and not self._failed

    @property
    def output_path(self) -> Path:
        """@brief Destination path for the encoded video."""
        return self._output_path

    @property
    def frames_written(self) -> int:
        """@brief Number of frames emitted to the encoder so far."""
        return self._frames_written

    def elapsed(self) -> float:
        """@brief Seconds of wall-clock time since start(), 0.0 when stopped."""
        return time.monotonic() - self._start_time if self._proc is not None else 0.0

    def start(self) -> None:
        """
        @brief Spawn ffmpeg and begin accepting frames.
        @throws RuntimeError if ffmpeg is not installed.
        """
        if self._proc is not None:
            return

        ffmpeg = check_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError(_FFMPEG_MISSING_MSG)

        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._proc = subprocess.Popen(
            self._build_command(ffmpeg),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        now = time.monotonic()
        self._start_time = now
        self._next_frame_time = now
        self._frames_written = 0
        self._failed = False
        self._last_frame = None

    def _build_command(self, ffmpeg: str) -> List[str]:
        """
        @brief Build the ffmpeg argument vector for a rawvideo stdin pipe.
        @param ffmpeg: Path to the ffmpeg binary.
        @return Argument list for subprocess.Popen.
        """
        return [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Input: raw frames arriving on stdin at the fixed output rate.
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._width}x{self._height}",
            "-r",
            str(self._fps),
            "-i",
            "-",
            "-vf",
            _PAD_FILTER,
            "-c:v",
            "libx264",
            "-preset",
            _PRESET,
            "-crf",
            _CRF,
            "-pix_fmt",
            _PIX_FMT_OUT,
            str(self._output_path),
        ]

    def capture(self, surface: pygame.Surface) -> None:
        """
        @brief Offer a freshly drawn surface to the encoder.

        Call once per draw. The surface is stored and written only when the
        wall-clock accumulator says a frame is due, so the output rate stays
        fixed no matter how fast or slow the caller loops. When several frame
        intervals have elapsed since the last call, the surface is repeated to
        fill them, keeping video time aligned with real time.

        @param surface: The surface to record.
        """
        if not self.is_recording:
            return

        self._last_frame = self._fit_to_geometry(surface_to_rgb_array(surface))

        now = time.monotonic()
        if now < self._next_frame_time:
            return
        # Emit one frame per elapsed interval so a stalled viewer does not
        # compress real time, but cap the catch-up so a long pause cannot emit
        # thousands of frames in one call.
        due = int((now - self._next_frame_time) / self._frame_interval) + 1
        for _ in range(min(due, self._fps)):
            self._write_frame(self._last_frame)
        self._next_frame_time = now + self._frame_interval

    def _fit_to_geometry(self, frame: np.ndarray) -> np.ndarray:
        """
        @brief Letterbox a frame into the encoder's fixed geometry.

        The window can be resized mid-recording (the map area is re-sized to the
        lot aspect ratio on the first frame of an episode), but ffmpeg's input
        geometry is fixed at spawn. Oversized frames are cropped and undersized
        ones centred on black, so a resize never corrupts the stream.

        @param frame: Array of shape (H, W, 3).
        @return Array of shape (self._height, self._width, 3).
        """
        height, width = frame.shape[0], frame.shape[1]
        if height == self._height and width == self._width:
            return frame

        fitted = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        copy_h = min(height, self._height)
        copy_w = min(width, self._width)
        off_y = (self._height - copy_h) // 2
        off_x = (self._width - copy_w) // 2
        fitted[off_y : off_y + copy_h, off_x : off_x + copy_w] = frame[:copy_h, :copy_w]
        return fitted

    def _write_frame(self, frame: np.ndarray) -> None:
        """
        @brief Write one frame to the encoder, disabling recording on failure.
        @param frame: Array of shape (height, width, 3), dtype uint8.
        """
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(frame.tobytes())
            self._frames_written += 1
        except (BrokenPipeError, OSError):
            # The encoder died (disk full, bad codec args). Stop feeding it, but
            # let the viewer carry on drawing.
            self._failed = True

    def stop(self) -> Optional[Path]:
        """
        @brief Close the pipe and wait for ffmpeg to finalise the file.

        Safe to call when not recording, and safe to call twice.

        @return The output path if frames were written, else None.
        """
        proc, self._proc = self._proc, None
        if proc is None:
            return None

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        self._last_frame = None
        wrote_frames = self._frames_written > 0 and not self._failed
        return self._output_path if wrote_frames else None
