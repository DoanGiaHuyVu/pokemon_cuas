"""Capture correctness and stale-window checks without OS input or screen access."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw

from computer.capture import (
    CaptureError, MacWindowBackend, build_calibration, calibrate, capture_observation, confirm_calibration, select_window,
)
from computer.geometry import GeometryError, Rect, WindowInfo
from harness.cli import main
from pydantic import SecretStr


class FakeBackend:
    def __init__(self):
        self.window = WindowInfo(window_id=123, pid=456, owner="melonDS", title="[60/60] melonDS 1.1",
                                 bounds=Rect(left=470, top=46, width=538, height=833))
        self.frame = Image.new("RGB", (1076, 1666), "magenta")
        draw = ImageDraw.Draw(self.frame)
        draw.rectangle((1, 57, 1074, 860), fill="red")
        draw.rectangle((1, 861, 1074, 1664), fill="blue")
        self.captures = 0
        self.after_capture = None

    def list_windows(self):
        return [self.window]

    def capture_window(self, identifier):
        assert identifier == self.window.window_id
        self.captures += 1
        if self.after_capture:
            self.window = self.after_capture
        return self.frame.copy()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.profile = self.root / "runs/calibration.json"
        self.images = self.root / "screenshots"
        self.runs = self.root / "runs"
        self.backend = FakeBackend()

    def ready(self):
        calibration, preview = calibrate(self.backend, self.profile, self.images)
        confirm_calibration(self.backend, self.profile)
        return calibration, preview

    def test_retina_crop_has_both_screens_and_no_chrome(self):
        calibration, preview = calibrate(self.backend, self.profile, self.images)
        self.assertEqual(calibration.scale_x, 2)
        self.assertEqual(calibration.scale_y, 2)
        with Image.open(preview) as image:
            self.assertEqual(image.size, (1074, 1608))
            self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(image.getpixel((1073, 803)), (255, 0, 0))
            self.assertEqual(image.getpixel((0, 804)), (0, 0, 255))
            self.assertEqual(image.getpixel((1073, 1607)), (0, 0, 255))
        self.assertFalse(calibration.verified)

    def test_normalized_endpoints_and_center_stay_inside_capture(self):
        calibration = build_calibration(self.backend.window, self.backend.frame.size)
        self.assertEqual(calibration.normalized_to_pixel(0, 0), (0, 0))
        self.assertEqual(calibration.normalized_to_pixel(1000, 1000), (1073, 1607))
        self.assertEqual(calibration.normalized_to_pixel(500, 500), (536, 804))
        self.assertEqual(calibration.normalized_to_desktop(0, 0), (470.5, 74.5))
        self.assertEqual(calibration.normalized_to_desktop(1000, 1000), (1007, 878))
        self.assertFalse(calibration.is_touch_target(500, 0))
        self.assertTrue(calibration.is_touch_target(500, 1000))

    def test_invalid_coordinates_are_rejected(self):
        calibration = build_calibration(self.backend.window, self.backend.frame.size)
        for x, y in ((-1, 0), (1001, 0), (0, float("nan")), (float("inf"), 0), (True, 0), ("500", 0)):
            with self.subTest(x=x, y=y), self.assertRaises(GeometryError):
                calibration.normalized_to_desktop(x, y)

    def test_capture_requires_confirmed_preview(self):
        calibrate(self.backend, self.profile, self.images)
        count = self.backend.captures
        with self.assertRaises(CaptureError):
            capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect")
        self.assertEqual(self.backend.captures, count)

    def test_observation_has_prompt_mapping_and_unique_ids(self):
        self.ready()
        first = capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect the game")
        second = capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect again")
        self.assertNotEqual(first["observation_id"], second["observation_id"])
        saved = json.loads(Path(first["metadata_path"]).read_text())
        self.assertEqual(saved["prompt"], "Inspect the game")
        self.assertFalse(saved["executed"])
        self.assertFalse(saved["model_called"])
        self.assertEqual(saved["coordinate_samples"]["1000,1000"]["pixel"], [1073, 1607])
        self.assertEqual(saved["touchscreen_desktop"]["top"], 476.5)

    def test_move_resize_and_restart_are_rejected_before_capture(self):
        self.ready()
        original = self.backend.window
        for updates in (
            {"bounds": original.bounds.model_copy(update={"left": 480})},
            {"bounds": original.bounds.model_copy(update={"width": 600})},
            {"pid": 987},
        ):
            self.backend.window = original.model_copy(update=updates)
            count = self.backend.captures
            with self.subTest(updates=updates), self.assertRaises(GeometryError):
                capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect")
            self.assertEqual(self.backend.captures, count)

    def test_movement_during_capture_does_not_save_an_observation(self):
        self.ready()
        self.backend.after_capture = self.backend.window.model_copy(
            update={"bounds": self.backend.window.bounds.model_copy(update={"top": 90})}
        )
        with self.assertRaises(GeometryError):
            capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect")
        self.assertEqual(list(self.runs.glob("*.json")), [self.profile])

    def test_capture_scale_change_is_rejected(self):
        self.ready()
        self.backend.frame = self.backend.frame.resize((538, 833))
        with self.assertRaises(GeometryError):
            capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect")

    def test_recalibration_after_move_requires_new_confirmation(self):
        self.ready()
        self.backend.window = self.backend.window.model_copy(
            update={"bounds": self.backend.window.bounds.model_copy(update={"left": 100})}
        )
        calibration, _ = calibrate(self.backend, self.profile, self.images)
        self.assertFalse(calibration.verified)
        self.assertEqual(calibration.normalized_to_desktop(0, 0)[0], 100.5)
        confirm_calibration(self.backend, self.profile)
        capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect")

    def test_invalid_layout_insets_and_gap_fail(self):
        for insets, gap in (((-1, 28, 0, 0), 0), ((0, 900, 0, 0), 0), (None, 900), (None, float("nan"))):
            with self.subTest(insets=insets, gap=gap), self.assertRaises(GeometryError):
                build_calibration(self.backend.window, self.backend.frame.size, insets, gap)
        with self.assertRaises(GeometryError):
            build_calibration(self.backend.window, (1076, 900))

    def test_missing_and_ambiguous_windows_fail(self):
        with self.assertRaises(CaptureError):
            select_window(self.backend, 999)
        original = self.backend.window
        self.backend.list_windows = lambda: [original, original.model_copy(update={"window_id": 789})]
        with self.assertRaises(CaptureError):
            select_window(self.backend)
        self.assertEqual(select_window(self.backend, 789).window_id, 789)

    def test_missing_profile_fails(self):
        with self.assertRaises(CaptureError):
            capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect")

    def test_cli_calibration_and_prompt_capture_do_not_call_models(self):
        settings = SimpleNamespace(gemini_api_key=SecretStr("capture-test-secret"))
        output = io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch(
            "harness.cli.load_settings", return_value=settings
        ), patch("computer.capture.MacWindowBackend", return_value=self.backend), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--calibrate"]), 0)
            self.assertEqual(main(["--confirm-calibration"]), 0)
            self.assertEqual(main(["--capture", "--prompt", "Inspect capture-test-secret"]), 0)
        self.assertNotIn("capture-test-secret", output.getvalue())
        observations = [path for path in self.runs.glob("*.json") if path != self.profile]
        self.assertEqual(len(observations), 1)
        saved = json.loads(observations[0].read_text())
        self.assertEqual(saved["prompt"], "Inspect [hidden]")
        self.assertFalse(saved["model_called"])

    def test_cli_capture_error_is_actionable_and_has_no_traceback(self):
        settings = SimpleNamespace(gemini_api_key=SecretStr("capture-test-secret"))
        error = io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch(
            "harness.cli.load_settings", return_value=settings
        ), patch("computer.capture.MacWindowBackend", return_value=self.backend), contextlib.redirect_stderr(error):
            self.assertEqual(main(["--capture", "--prompt", "Inspect"]), 3)
        self.assertIn("Run --calibrate", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())

    def test_mac_discovery_excludes_dialogs_helpers_and_hidden_windows(self):
        base = {
            "kCGWindowOwnerName": "melonDS", "kCGWindowName": "[60/60] melonDS 1.1",
            "kCGWindowNumber": 123, "kCGWindowOwnerPID": 456, "kCGWindowLayer": 0,
            "kCGWindowIsOnscreen": True,
            "kCGWindowBounds": {"X": 470, "Y": 46, "Width": 538, "Height": 833},
        }
        records = [base, {**base, "kCGWindowName": "Open DS ROM"},
                   {**base, "kCGWindowName": "Window"}, {**base, "kCGWindowIsOnscreen": False},
                   {**base, "kCGWindowOwnerName": "Another app"}]
        quartz = SimpleNamespace(
            kCGWindowListOptionAll=0, kCGWindowListExcludeDesktopElements=16, kCGNullWindowID=0,
            CGWindowListCopyWindowInfo=lambda *_: records, CGPreflightScreenCaptureAccess=lambda: True,
        )
        with patch.dict(sys.modules, {"Quartz": quartz}), patch("computer.capture.sys.platform", "darwin"):
            windows = MacWindowBackend().list_windows()
        self.assertEqual(windows, [self.backend.window])

    def test_mac_discovery_reports_sandbox_and_permission_failures(self):
        for records, permission, message in ((None, False, "sandbox"), ([], False, "Screen Recording")):
            quartz = SimpleNamespace(
                kCGWindowListOptionAll=0, kCGWindowListExcludeDesktopElements=16, kCGNullWindowID=0,
                CGWindowListCopyWindowInfo=lambda *_: records,
                CGPreflightScreenCaptureAccess=lambda: permission,
            )
            with self.subTest(message=message), patch.dict(sys.modules, {"Quartz": quartz}), patch(
                "computer.capture.sys.platform", "darwin"
            ), self.assertRaises(CaptureError) as caught:
                MacWindowBackend().list_windows()
            self.assertIn(message, str(caught.exception))

    def test_mac_discovery_identifies_preview_and_offscreen_main_window(self):
        base = {
            "kCGWindowOwnerName": "melonDS", "kCGWindowName": "[60/60] melonDS 1.1",
            "kCGWindowNumber": 123, "kCGWindowOwnerPID": 456, "kCGWindowLayer": 0,
            "kCGWindowIsOnscreen": True,
            "kCGWindowBounds": {"X": 15, "Y": 215, "Width": 63, "Height": 147},
        }
        for record, message in ((base, "63×147"), ({**base, "kCGWindowIsOnscreen": False}, "offscreen")):
            quartz = SimpleNamespace(
                kCGWindowListOptionAll=0, kCGWindowListExcludeDesktopElements=16, kCGNullWindowID=0,
                CGWindowListCopyWindowInfo=lambda *_: [record], CGPreflightScreenCaptureAccess=lambda: True,
            )
            with self.subTest(message=message), patch.dict(sys.modules, {"Quartz": quartz}), patch(
                "computer.capture.sys.platform", "darwin"
            ), self.assertRaises(CaptureError) as caught:
                MacWindowBackend().list_windows()
            self.assertIn(message, str(caught.exception))
            self.assertIn("sleep 5", str(caught.exception))

    def test_restarted_window_reports_recalibration_instead_of_missing_game(self):
        self.backend.window = self.backend.window.model_copy(update={"window_id": 789, "pid": 987})
        with self.assertRaises(CaptureError) as caught:
            select_window(self.backend, 123)
        self.assertIn("restarted", str(caught.exception))
        self.assertIn("--calibrate", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
