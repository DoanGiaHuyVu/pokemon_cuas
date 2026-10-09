"""Meaningful input-boundary and interruption checks; no desktop input in tests."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from computer.capture import build_calibration, save_calibration
from computer.executor import ActionExecutor, ExecutionError, MacExecutionGuard, Stopped
from computer.geometry import GeometryError, Rect, WindowInfo
from computer.input_test import settle_owned_window
from harness.actions import ActionError, parse_action, validate_game_action
from harness.cli import main
from pydantic import SecretStr


class FakeClock:
    def __init__(self):
        self.now = 0
        self.on_sleep = lambda: None

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()


class FakeGuard:
    typing_allowed = False

    def __init__(self):
        self.failure = None

    def check(self):
        if self.failure:
            raise self.failure

    def prepare(self):
        self.check()


class SpyDriver:
    def __init__(self):
        self.events = []
        self.cursor = (50, 50)
        self.fail_safe = False

    def position(self):
        return self.cursor

    def move_to(self, x, y):
        self.cursor = (x, y)
        self.events.append(("move", x, y))

    def key_down(self, key):
        self.events.append(("down", key))

    def key_up(self, key):
        self.events.append(("up", key))

    def mouse_down(self):
        self.events.append(("mouse_down",))

    def mouse_up(self):
        self.events.append(("mouse_up",))

    def check_failsafe(self):
        if self.fail_safe:
            raise ExecutionError("Fail-safe corner reached")


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.stop = self.root / "STOP"
        self.clock, self.guard, self.driver = FakeClock(), FakeGuard(), SpyDriver()
        self.executor = ActionExecutor(self.guard, self.driver, self.stop, clock=self.clock.time, sleep=self.clock.sleep)
        window = WindowInfo(window_id=123, pid=456, owner="melonDS", title="melonDS 1.1",
                            bounds=Rect(left=470, top=46, width=538, height=833))
        self.calibration = build_calibration(window, (1076, 1666)).model_copy(update={"verified": True})

    def action(self, **values):
        return parse_action(json.dumps(values))

    def test_schema_rejects_malformed_missing_extra_and_unsafe_fields(self):
        invalid = ["not JSON", "[]", '[{"tool":"WAIT","seconds":1}]', '{}',
                   '{"tool":"CLICK"}', '{"tool":"PRESS","key":"command"}',
                   '{"tool":"PRESS","key":"x","seconds":true}',
                   '{"tool":"PRESS","key":"x","seconds":1.01}',
                   '{"tool":"MOVE","x":true,"y":500}',
                   '{"tool":"MOVE","x":-1,"y":500}',
                   '{"tool":"MOVE","x":1001,"y":500}',
                   '{"tool":"MOVE","x":"500","y":500}',
                   '{"tool":"WAIT","seconds":3}', '{"tool":"WAIT","seconds":0}',
                   '{"tool":"TYPE","text":"line\\nfeed"}',
                   '{"tool":"TYPE","text":"é"}', '{"tool":"TYPE","text":""}',
                   json.dumps({"tool": "TYPE", "text": "x" * 33}),
                   '{"tool":"PRESS","key":"x","shell":"rm"}']
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ActionError):
                parse_action(raw)
        self.assertEqual(self.driver.events, [])

    def test_action_validation_error_does_not_echo_input(self):
        secret = "test-secret-never-echo"
        with self.assertRaises(ActionError) as caught:
            parse_action(json.dumps({"tool": "PRESS", "key": secret}))
        self.assertNotIn(secret, str(caught.exception))

    def test_game_typing_and_upper_screen_clicks_emit_zero_input(self):
        for action in (self.action(tool="TYPE", text="HUY"), self.action(tool="CLICK", x=500, y=100)):
            with self.subTest(action=action), self.assertRaises(ActionError):
                self.executor.execute(action, self.calibration)
        self.assertEqual(self.driver.events, [])

    def test_unverified_calibration_emits_zero_input(self):
        with self.assertRaises(ActionError):
            self.executor.execute(self.action(tool="PRESS", key="enter"), self.calibration.model_copy(update={"verified": False}))
        self.assertEqual(self.driver.events, [])

    def test_focus_permission_or_geometry_failures_emit_zero_input(self):
        for failure in (ExecutionError("lost focus"), ExecutionError("permission denied"), GeometryError("stale geometry")):
            self.guard.failure = failure
            with self.subTest(failure=failure), self.assertRaises(type(failure)):
                self.executor.execute(self.action(tool="PRESS", key="enter"), self.calibration)
            self.assertEqual(self.driver.events, [])

    def test_mouse_coordinates_are_quantized_inside_game_and_touchscreen(self):
        self.executor.execute(self.action(tool="MOVE", x=0, y=0), self.calibration)
        self.assertEqual(self.driver.events, [("move", 471, 75)])
        self.driver.events.clear()
        self.executor.execute(self.action(tool="CLICK", x=1000, y=1000), self.calibration)
        self.assertEqual(self.driver.events, [("move", 1007, 878), ("mouse_down",), ("mouse_up",)])
        self.assertFalse(self.executor.mouse_held)

    def test_one_key_press_is_bounded_and_released(self):
        result = self.executor.execute(self.action(tool="PRESS", key="enter", seconds=0.1), self.calibration)
        self.assertEqual(self.driver.events, [("down", "enter"), ("up", "enter")])
        self.assertAlmostEqual(self.clock.now, 0.1)
        self.assertFalse(self.executor.held_keys)
        self.assertTrue(result["executed"])

    def test_failed_cursor_placement_cannot_click_current_location(self):
        self.driver.move_to = lambda x, y: self.driver.events.append(("failed_move", x, y))
        with self.assertRaises(ExecutionError):
            self.executor.execute(self.action(tool="CLICK", x=500, y=750), self.calibration)
        self.assertEqual([event[0] for event in self.driver.events], ["failed_move"])

    def test_release_failure_latches_stop_and_attempts_all_releases(self):
        self.executor.held_keys = {"shiftleft", "h"}
        def failed_release(key):
            self.driver.events.append(("release_attempt", key))
            raise OSError("release failed")
        self.driver.key_up = failed_release
        with self.assertRaises(ExecutionError):
            self.executor.cleanup()
        self.assertEqual({event[1] for event in self.driver.events}, {"shiftleft", "h"})
        self.assertTrue(self.stop.exists())

    def test_wait_emits_no_input(self):
        self.executor.execute(self.action(tool="WAIT", seconds=0.5), self.calibration)
        self.assertEqual(self.driver.events, [])
        self.assertAlmostEqual(self.clock.now, 0.5)

    def test_owned_field_waits_for_native_focus_and_stable_bounds(self):
        def sample():
            width = 398 if self.clock.now < 0.1 else 400
            return {"kCGWindowNumber": 55, "kCGWindowBounds": {"Width": width}}, self.clock.now >= 0.04
        record = settle_owned_window(sample, clock=self.clock.time, sleep=self.clock.sleep)
        self.assertEqual(record["kCGWindowBounds"]["Width"], 400)
        self.assertGreaterEqual(self.clock.now, 0.3)
        self.assertEqual(self.driver.events, [])

    def test_owned_field_missing_focus_or_unstable_geometry_times_out_without_input(self):
        samples = [lambda: (None, False),
                   lambda: ({"kCGWindowNumber": 55, "kCGWindowBounds": {"Width": self.clock.now}}, True)]
        for sample in samples:
            self.clock.now = 0
            with self.subTest(sample=sample), self.assertRaises(ExecutionError):
                settle_owned_window(sample, clock=self.clock.time, sleep=self.clock.sleep)
            self.assertLessEqual(self.clock.now, 2.02)
        self.assertEqual(self.driver.events, [])

    def test_existing_stop_signal_emits_no_input(self):
        self.stop.touch()
        with self.assertRaises(Stopped):
            self.executor.execute(self.action(tool="PRESS", key="x"), self.calibration)
        self.assertEqual(self.driver.events, [])

    def test_stop_during_key_hold_releases_and_stays_latched(self):
        self.clock.on_sleep = self.stop.touch
        with self.assertRaises(Stopped):
            self.executor.execute(self.action(tool="PRESS", key="x", seconds=1), self.calibration)
        self.assertEqual(self.driver.events, [("down", "x"), ("up", "x")])
        self.assertFalse(self.executor.held_keys)
        self.assertTrue(self.stop.exists())
        self.assertLessEqual(self.clock.now, 0.02)

    def test_focus_loss_during_click_releases_mouse(self):
        self.clock.on_sleep = lambda: setattr(self.guard, "failure", ExecutionError("lost focus"))
        with self.assertRaises(ExecutionError):
            self.executor.execute(self.action(tool="CLICK", x=500, y=750), self.calibration)
        self.assertEqual(self.driver.events[-2:], [("mouse_down",), ("mouse_up",)])
        self.assertFalse(self.executor.mouse_held)

    def test_keyboard_interrupt_during_hold_releases_key(self):
        def interrupt():
            raise KeyboardInterrupt
        self.clock.on_sleep = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.executor.execute(self.action(tool="PRESS", key="enter"), self.calibration)
        self.assertEqual(self.driver.events, [("down", "enter"), ("up", "enter")])

    def test_fail_safe_during_hold_still_releases_key(self):
        self.clock.on_sleep = lambda: setattr(self.driver, "fail_safe", True)
        with self.assertRaises(ExecutionError):
            self.executor.execute(self.action(tool="PRESS", key="x"), self.calibration)
        self.assertEqual(self.driver.events, [("down", "x"), ("up", "x")])

    def test_backend_error_after_key_down_still_releases_key(self):
        original = self.driver.key_down
        def failed_down(key):
            original(key)
            raise OSError("event delivery error")
        self.driver.key_down = failed_down
        with self.assertRaises(OSError):
            self.executor.execute(self.action(tool="PRESS", key="x"), self.calibration)
        self.assertEqual(self.driver.events, [("down", "x"), ("up", "x")])

    def test_typing_tracks_shift_and_releases_on_focus_loss(self):
        self.guard.typing_allowed = True
        self.executor.execute(self.action(tool="TYPE", text="H!"))
        self.assertEqual(self.driver.events, [("down", "shiftleft"), ("down", "h"), ("up", "h"), ("up", "shiftleft"),
                                              ("down", "shiftleft"), ("down", "1"), ("up", "1"), ("up", "shiftleft")])
        self.assertFalse(self.executor.held_keys)
        self.driver.events.clear()
        original = self.driver.key_down
        def focus_loss(key):
            original(key)
            if key == "shiftleft":
                self.guard.failure = ExecutionError("lost focus")
        self.driver.key_down = focus_loss
        with self.assertRaises(ExecutionError):
            self.executor.execute(self.action(tool="TYPE", text="H"))
        self.assertEqual(self.driver.events, [("down", "shiftleft"), ("up", "shiftleft")])

    def test_cli_preview_does_not_import_or_execute_input_driver(self):
        profile = self.root / "runs/calibration.json"
        save_calibration(profile, self.calibration)
        settings = SimpleNamespace(gemini_api_key=SecretStr("test-secret"))
        output = io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch("harness.cli.load_settings", return_value=settings), patch(
            "computer.executor.execute_game_action", side_effect=AssertionError("preview executed")
        ), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--action", '{"tool":"PRESS","key":"enter"}', "--prompt", "Start once"]), 0)
        self.assertIn("Preview only", output.getvalue())

    def test_cli_stop_does_not_require_credentials_or_resume_execution(self):
        with patch("harness.cli.PROJECT_ROOT", self.root), patch("harness.cli.load_settings", side_effect=AssertionError("stop loaded settings")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--stop"]), 0)
            self.assertTrue((self.root / "runs/STOP").exists())
            self.assertEqual(main(["--clear-stop"]), 0)
            self.assertFalse((self.root / "runs/STOP").exists())

    def native_modules(self, *, pid=456, records=None, permission=True, flags=0):
        window = self.calibration.window
        if records is None:
            records = [{"kCGWindowOwnerPID": window.pid, "kCGWindowNumber": window.window_id,
                        "kCGWindowBounds": {"X": 470, "Y": 46, "Width": 538, "Height": 833}}]
        workspace = SimpleNamespace(frontmostApplication=lambda: SimpleNamespace(processIdentifier=lambda: pid))
        appkit = SimpleNamespace(NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: workspace))
        quartz = SimpleNamespace(
            kCGWindowListOptionOnScreenOnly=1, kCGWindowListExcludeDesktopElements=16, kCGNullWindowID=0,
            CGWindowListCopyWindowInfo=lambda *_: records, CGPreflightPostEventAccess=lambda: permission,
            CGEventSourceFlagsState=lambda *_: flags, kCGEventSourceStateCombinedSessionState=0,
            kCGEventFlagMaskShift=1, kCGEventFlagMaskControl=2, kCGEventFlagMaskAlternate=4,
            kCGEventFlagMaskCommand=8, kCGEventFlagMaskAlphaShift=16,
        )
        return {"AppKit": appkit, "Quartz": quartz}

    def test_native_guard_rejects_wrong_foreground_app_and_modal_window(self):
        guard = MacExecutionGuard(self.calibration.window, None)
        with patch.dict(sys.modules, self.native_modules(pid=999)), self.assertRaises(ExecutionError):
            guard.check()
        dialog = {"kCGWindowOwnerPID": 456, "kCGWindowNumber": 789,
                  "kCGWindowBounds": {"X": 500, "Y": 100, "Width": 300, "Height": 200}}
        main = {"kCGWindowOwnerPID": 456, "kCGWindowNumber": 123,
                "kCGWindowBounds": {"X": 470, "Y": 46, "Width": 538, "Height": 833}}
        with patch.dict(sys.modules, self.native_modules(records=[dialog, main])), self.assertRaises(ExecutionError):
            guard.check()

    def test_native_guard_rejects_permission_modifiers_and_changed_geometry(self):
        guard = MacExecutionGuard(self.calibration.window, None)
        for modules in (self.native_modules(permission=False), self.native_modules(flags=8)):
            with patch.dict(sys.modules, modules), self.assertRaises(ExecutionError):
                guard.prepare()
        resized = [{"kCGWindowOwnerPID": 456, "kCGWindowNumber": 123,
                    "kCGWindowBounds": {"X": 470, "Y": 46, "Width": 600, "Height": 833}}]
        with patch.dict(sys.modules, self.native_modules(records=resized)), self.assertRaises(GeometryError):
            guard.check()

    def test_native_guard_accepts_main_window_with_small_titlebar_helper(self):
        records = [{"kCGWindowOwnerPID": 456, "kCGWindowNumber": 999,
                    "kCGWindowBounds": {"X": 480, "Y": 50, "Width": 66, "Height": 20}},
                   {"kCGWindowOwnerPID": 456, "kCGWindowNumber": 123,
                    "kCGWindowBounds": {"X": 470, "Y": 46, "Width": 538, "Height": 833}}]
        with patch.dict(sys.modules, self.native_modules(records=records)):
            MacExecutionGuard(self.calibration.window, None).prepare()


if __name__ == "__main__":
    unittest.main()
