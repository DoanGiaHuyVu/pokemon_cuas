"""Capture only a selected melonDS window; never send mouse or keyboard input."""

import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image
from pydantic import ValidationError

from computer.geometry import Calibration, GeometryError, PixelRect, Rect, WindowInfo
from harness.journal import atomic_json


class CaptureError(Exception):
    pass


class MacWindowBackend:
    def list_windows(self) -> list[WindowInfo]:
        if sys.platform != "darwin":
            raise CaptureError("This capture backend requires macOS.")
        import Quartz

        records = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        if records is None:
            raise CaptureError(
                "macOS window discovery is unavailable. Run from your local terminal; "
                "the Codex shell sandbox can block screen access."
            )
        if not Quartz.CGPreflightScreenCaptureAccess():
            raise CaptureError(
                "Screen Recording permission is required for the app running Python. "
                "Enable it in macOS System Settings, then rerun."
            )
        windows = []
        hidden_main = False
        small_main = []
        for record in records:
            owner = str(record.get("kCGWindowOwnerName", ""))
            title = str(record.get("kCGWindowName", ""))
            bounds = record.get("kCGWindowBounds", {})
            if (
                owner.lower() != "melonds"
                or not re.search(r"\bmelonDS\s+\d", title)
                or record.get("kCGWindowLayer") != 0
            ):
                continue
            if not record.get("kCGWindowIsOnscreen", False):
                hidden_main = True
                continue
            width, height = float(bounds.get("Width", 0)), float(bounds.get("Height", 0))
            if width < 200 or height < 200:
                small_main.append((width, height))
                continue
            windows.append(WindowInfo(
                window_id=int(record["kCGWindowNumber"]), pid=int(record["kCGWindowOwnerPID"]),
                owner=owner, title=title,
                bounds=Rect(left=bounds["X"], top=bounds["Y"], width=bounds["Width"], height=bounds["Height"]),
            ))
        if not windows and small_main:
            width, height = small_main[0]
            raise CaptureError(
                f"macOS reports melonDS only as a small {width:g}×{height:g}-point window "
                "(possibly a Stage Manager preview). Bring the full game window into view. "
                "If switching to Terminal hides it, delay the command with sleep 5 and switch "
                "back to melonDS before capture. Do not resize the game window."
            )
        if not windows and hidden_main:
            raise CaptureError(
                "The melonDS game window exists but macOS reports it offscreen. "
                "Show it on the current desktop. If switching to Terminal hides it, delay "
                "the command with sleep 5 and switch back to melonDS before capture."
            )
        return windows

    def capture_window(self, window_id: int) -> Image.Image:
        with tempfile.TemporaryDirectory(prefix="pokemon-cua-capture-") as directory:
            path = Path(directory) / "window.png"
            try:
                result = subprocess.run(
                    ["/usr/sbin/screencapture", "-x", "-o", "-l", str(window_id), str(path)],
                    capture_output=True, timeout=10, check=False,
                )
                if result.returncode != 0 or not path.is_file():
                    raise CaptureError("Window capture failed. Check Screen Recording permission and visibility.")
                with Image.open(path) as image:
                    return image.convert("RGB")
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CaptureError("Could not capture the emulator window.") from exc


def select_window(backend, window_id: int | None = None) -> WindowInfo:
    windows = backend.list_windows()
    if window_id is not None:
        if windows and not any(window.window_id == window_id for window in windows):
            raise CaptureError("The selected melonDS window is unavailable or was restarted. Run --calibrate and review the new crop.")
        windows = [window for window in windows if window.window_id == window_id]
    if not windows:
        raise CaptureError("No visible main melonDS window found. Open the game and bring melonDS into view.")
    if len(windows) != 1:
        raise CaptureError("Multiple melonDS game windows found. Select one with --window-id.")
    return windows[0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_calibration(
    window: WindowInfo, frame_size: tuple[int, int],
    insets: tuple[float, float, float, float] | None = None, gap: float = 0,
) -> Calibration:
    width, height = frame_size
    sx, sy = width / window.bounds.width, height / window.bounds.height
    if min(sx, sy) <= 0 or abs(sx - sy) / max(sx, sy) > 0.02:
        raise GeometryError("Window image has inconsistent scaling; calibrate a normal, unrotated window.")
    # Standard macOS title bar: 28 desktop points, plus one captured border pixel.
    # Always inspect the preview; custom window chrome can use explicit --insets.
    left, top, right, bottom = insets if insets is not None else (1 / sx, 28 + 1 / sy, 1 / sx, 1 / sy)
    import math
    if any(not math.isfinite(value) or value < 0 for value in (left, top, right, bottom, gap)):
        raise GeometryError("Insets and screen gap must be finite, nonnegative desktop-point values.")
    x0, y0, x1, y1 = round(left * sx), round(top * sy), width - round(right * sx), height - round(bottom * sy)
    if x1 <= x0 or y1 <= y0:
        raise GeometryError("Insets leave no game image.")
    crop_width, crop_height = x1 - x0, y1 - y0
    gap_pixels = round(gap * sy)
    if crop_height - gap_pixels < 2:
        raise GeometryError("The screen gap leaves no DS screens.")
    if abs((crop_height - gap_pixels) / crop_width - 1.5) > 0.03:
        raise GeometryError("Expected two equally sized DS screens stacked vertically. Check layout or --insets.")
    upper_height = (crop_height - gap_pixels) // 2
    return Calibration(
        window=window, frame_width=width, frame_height=height,
        game_crop=PixelRect(left=x0, top=y0, width=crop_width, height=crop_height),
        top_screen=PixelRect(left=0, top=0, width=crop_width, height=upper_height),
        touchscreen=PixelRect(left=0, top=upper_height + gap_pixels, width=crop_width,
                              height=crop_height - upper_height - gap_pixels),
        created_at=utc_now(),
    )


def load_calibration(path: Path) -> Calibration:
    try:
        return Calibration.model_validate_json(path.read_text())
    except (OSError, ValueError, ValidationError) as exc:
        raise CaptureError("No valid calibration found. Run --calibrate and review the preview first.") from exc


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(calibration.model_dump_json(indent=2))
    temporary.replace(path)


def calibrate(backend, profile: Path, images: Path, *, window_id=None, insets=None, gap=0) -> tuple[Calibration, Path]:
    before = select_window(backend, window_id)
    frame = backend.capture_window(before.window_id)
    calibration = build_calibration(before, frame.size, insets, gap)
    calibration.check_window(select_window(backend, before.window_id))
    images.mkdir(parents=True, exist_ok=True)
    preview = images / f"calibration-{uuid4().hex}.png"
    frame.crop(calibration.game_crop.box).save(preview)
    save_calibration(profile, calibration)
    return calibration, preview


def confirm_calibration(backend, profile: Path) -> Calibration:
    calibration = load_calibration(profile)
    calibration.check_window(select_window(backend, calibration.window.window_id))
    calibration.check_frame(backend.capture_window(calibration.window.window_id).size)
    confirmed = calibration.model_copy(update={"verified": True})
    save_calibration(profile, confirmed)
    return confirmed


def capture_observation(backend, profile: Path, images: Path, runs: Path, prompt: str) -> dict:
    calibration = load_calibration(profile)
    if not calibration.verified:
        raise CaptureError("Calibration preview needs review. Then run --confirm-calibration.")
    before = select_window(backend, calibration.window.window_id)
    calibration.check_window(before)
    frame = backend.capture_window(before.window_id)
    calibration.check_frame(frame.size)
    calibration.check_window(select_window(backend, before.window_id))
    identifier = uuid4().hex
    image_path = images / f"{identifier}.png"
    metadata_path = runs / f"{identifier}.json"
    mapping = {
        f"{x},{y}": {
            "pixel": calibration.normalized_to_pixel(x, y),
            "desktop": calibration.normalized_to_desktop(x, y),
            "touch_target": calibration.is_touch_target(x, y),
        }
        for x, y in ((0, 0), (1000, 0), (500, 500), (0, 1000), (1000, 1000))
    }
    observation = {
        "observation_id": identifier, "captured_at": utc_now(), "prompt": prompt,
        "image_path": str(image_path), "metadata_path": str(metadata_path),
        "calibration": calibration.model_dump(mode="json"),
        "game_desktop": calibration.game_desktop.model_dump(),
        "touchscreen_desktop": calibration.touchscreen_desktop.model_dump(),
        "coordinate_samples": mapping, "executed": False, "model_called": False,
    }
    images.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    frame.crop(calibration.game_crop.box).save(image_path)
    atomic_json(metadata_path, observation)
    return observation
