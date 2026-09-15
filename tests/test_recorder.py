"""
@file test_recorder.py
@brief Unit tests for the visualiser's MP4 FrameRecorder.

CPU-only. No CARLA, ROS 2, or GPU required. Tests that actually encode are
skipped when the ffmpeg binary is absent (CI does not guarantee it), so the
pure-Python geometry and lifecycle logic stays covered everywhere.
"""

import os
import subprocess
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import pygame
import pytest

from scripts.visualise.recorder import FrameRecorder, check_ffmpeg, surface_to_rgb_array

_HAS_FFMPEG = check_ffmpeg() is not None
_needs_ffmpeg = pytest.mark.skipif(
    not _HAS_FFMPEG, reason="ffmpeg not installed on this host"
)


@pytest.fixture(scope="module", autouse=True)
def _headless_pygame() -> Iterator[None]:
    """
    @brief Initialise Pygame without a display so surfaces can be created.
    """
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def _filled_surface(width: int, height: int, value: int) -> pygame.Surface:
    """
    @brief Build a uniformly coloured surface.
    @param width: Surface width in pixels.
    @param height: Surface height in pixels.
    @param value: Grey level applied to all three channels.
    @return The filled surface.
    """
    surf = pygame.Surface((width, height))
    surf.fill((value, value, value))
    return surf


def _probe_dimensions(path: Path) -> Optional[Tuple[int, int]]:
    """
    @brief Read a video's pixel dimensions with ffprobe.
    @param path: Video file to inspect.
    @return (width, height), or None if ffprobe is unavailable or fails.
    """
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    width, height = out.stdout.strip().split(",")[:2]
    return int(width), int(height)


# ---------------------------------------------------------------------------
# TestSurfaceConversion
# ---------------------------------------------------------------------------


class TestSurfaceConversion:
    """
    @class TestSurfaceConversion
    @brief Tests for surface_to_rgb_array.
    """

    def test_returns_row_major_rgb(self) -> None:
        """
        @brief Output must be (height, width, 3) - not Pygame's (W, H, 3).
        """
        arr = surface_to_rgb_array(_filled_surface(64, 32, 200))
        assert arr.shape == (32, 64, 3)
        assert arr.dtype == np.uint8

    def test_preserves_colour(self) -> None:
        """
        @brief A uniform fill must survive the axis swap unchanged.
        """
        arr = surface_to_rgb_array(_filled_surface(8, 4, 123))
        assert np.all(arr == 123)


# ---------------------------------------------------------------------------
# TestFrameRecorderLifecycle
# ---------------------------------------------------------------------------


class TestFrameRecorderLifecycle:
    """
    @class TestFrameRecorderLifecycle
    @brief Start/stop behaviour that must hold with or without ffmpeg.
    """

    def test_not_recording_before_start(self, tmp_path: Path) -> None:
        """
        @brief A fresh recorder is idle and reports zero elapsed time.
        """
        rec = FrameRecorder(tmp_path / "out.mp4", 32, 16)
        assert not rec.is_recording
        assert rec.frames_written == 0
        assert rec.elapsed() == 0.0

    def test_stop_without_start_returns_none(self, tmp_path: Path) -> None:
        """
        @brief stop() on an unstarted recorder is a no-op, not an error.
        """
        assert FrameRecorder(tmp_path / "out.mp4", 32, 16).stop() is None

    def test_capture_before_start_is_ignored(self, tmp_path: Path) -> None:
        """
        @brief Frames offered before start() must be dropped silently.
        """
        rec = FrameRecorder(tmp_path / "out.mp4", 32, 16)
        rec.capture(_filled_surface(32, 16, 10))
        assert rec.frames_written == 0

    @_needs_ffmpeg
    def test_double_stop_is_safe(self, tmp_path: Path) -> None:
        """
        @brief stop() must be idempotent so cleanup paths can call it freely.
        """
        rec = FrameRecorder(tmp_path / "out.mp4", 32, 16, fps=5)
        rec.start()
        rec.capture(_filled_surface(32, 16, 60))
        rec.stop()
        assert rec.stop() is None
        assert not rec.is_recording


# ---------------------------------------------------------------------------
# TestFrameRecorderGeometry
# ---------------------------------------------------------------------------


class TestFrameRecorderGeometry:
    """
    @class TestFrameRecorderGeometry
    @brief Letterboxing, which protects the stream from a window resize.
    """

    def test_matching_size_passes_through(self, tmp_path: Path) -> None:
        """
        @brief A correctly sized frame must not be copied or altered.
        """
        rec = FrameRecorder(tmp_path / "out.mp4", 20, 10)
        frame = np.full((10, 20, 3), 42, dtype=np.uint8)
        assert rec._fit_to_geometry(frame) is frame

    def test_smaller_frame_is_centred(self, tmp_path: Path) -> None:
        """
        @brief An undersized frame is centred on black, preserving its content.
        """
        rec = FrameRecorder(tmp_path / "out.mp4", 20, 10)
        fitted = rec._fit_to_geometry(np.full((6, 10, 3), 99, dtype=np.uint8))
        assert fitted.shape == (10, 20, 3)
        assert np.all(fitted[2:8, 5:15] == 99)
        assert np.all(fitted[0, 0] == 0)

    def test_larger_frame_is_cropped(self, tmp_path: Path) -> None:
        """
        @brief An oversized frame is cropped to the encoder's fixed geometry.
        """
        rec = FrameRecorder(tmp_path / "out.mp4", 20, 10)
        fitted = rec._fit_to_geometry(np.full((40, 60, 3), 7, dtype=np.uint8))
        assert fitted.shape == (10, 20, 3)
        assert np.all(fitted == 7)


# ---------------------------------------------------------------------------
# TestFrameRecorderEncoding
# ---------------------------------------------------------------------------


@_needs_ffmpeg
class TestFrameRecorderEncoding:
    """
    @class TestFrameRecorderEncoding
    @brief End-to-end encoding through a real ffmpeg subprocess.
    """

    def test_encodes_playable_mp4(self, tmp_path: Path) -> None:
        """
        @brief A short capture must yield a non-empty MP4 of the right size.
        """
        out = tmp_path / "clip.mp4"
        rec = FrameRecorder(out, 48, 32, fps=10)
        rec.start()
        assert rec.is_recording
        for i in range(12):
            rec.capture(_filled_surface(48, 32, i * 20 % 255))
        assert rec.stop() == out

        assert out.exists() and out.stat().st_size > 0
        assert _probe_dimensions(out) == (48, 32)

    def test_odd_dimensions_are_padded_even(self, tmp_path: Path) -> None:
        """
        @brief Odd geometry must be padded, since libx264 rejects odd sizes.

        The map area is sized from the lot aspect ratio at runtime, so an odd
        window height is entirely reachable in normal use.
        """
        out = tmp_path / "odd.mp4"
        rec = FrameRecorder(out, 45, 31, fps=10)
        rec.start()
        for _ in range(6):
            rec.capture(_filled_surface(45, 31, 128))
        assert rec.stop() == out
        assert _probe_dimensions(out) == (46, 32)

    def test_resize_mid_recording_keeps_one_file(self, tmp_path: Path) -> None:
        """
        @brief A surface size change must letterbox, not corrupt the stream.
        """
        out = tmp_path / "resize.mp4"
        rec = FrameRecorder(out, 40, 24, fps=10)
        rec.start()
        for _ in range(4):
            rec.capture(_filled_surface(40, 24, 90))
        for _ in range(4):
            rec.capture(_filled_surface(30, 18, 160))
        assert rec.stop() == out
        assert _probe_dimensions(out) == (40, 24)

    def test_frame_rate_is_wall_clock_paced(self, tmp_path: Path) -> None:
        """
        @brief Many rapid captures must not emit one frame each.

        The viewer loops far faster than the output rate, so the accumulator
        has to throttle captures down or playback runs in fast-forward.
        """
        rec = FrameRecorder(tmp_path / "paced.mp4", 32, 16, fps=5)
        rec.start()
        for _ in range(200):
            rec.capture(_filled_surface(32, 16, 77))
        written = rec.frames_written
        rec.stop()
        # 200 immediate captures span far less than a second, so at 5 fps only
        # a couple of frames can legitimately be due.
        assert 1 <= written <= 5
