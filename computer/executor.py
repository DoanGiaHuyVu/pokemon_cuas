"""Bounded manual input with focus checks and release-on-interruption cleanup."""

import math
import time
from pathlib import Path

from computer.capture import CaptureError, MacWindowBackend, select_window
from computer.geometry import Calibration, GeometryError, WindowInfo
from harness.actions import Click, Move, Press, TypeText, Wait, validate_game_action


class ExecutionError(Exception):
    pass


class Stopped(ExecutionError):
    pass


class DeadlineGuard:
    """Checks the job clock during input; owned key/button releases still run after expiry."""
    def __init__(self, guard, deadline, *, clock=time.monotonic):
        if not math.isfinite(deadline): raise ExecutionError("A finite input deadline is required.")
        self.guard, self.deadline, self.clock = guard, deadline, clock
        self.typing_allowed = guard.typing_allowed

    def check(self):
        if self.clock() >= self.deadline: raise Stopped("Job time limit reached; no further input.")
        self.guard.check()
        if self.clock() >= self.deadline: raise Stopped("Job time limit reached; no further input.")

    def prepare(self):
        self.check()
        self.guard.prepare()
        self.check()


class PyAutoGUIDriver:
    def __init__(self):
        import pyautogui
        self.gui = pyautogui
        self.gui.FAILSAFE = True
        self.gui.PAUSE = 0

    def position(self):
        return tuple(self.gui.position())

    def move_to(self, x, y):
        self.gui.moveTo(x, y, duration=0)

    def key_down(self, key):
        self.gui.keyDown(key)

    def key_up(self, key):
        # Cleanup must still release our keys if the cursor reached a fail-safe corner.
        previous = self.gui.FAILSAFE
        try:
            self.gui.FAILSAFE = False
            self.gui.keyUp(key)
        finally:
            self.gui.FAILSAFE = previous

    def mouse_down(self):
        self.gui.mouseDown(button="left")

    def mouse_up(self):
        previous = self.gui.FAILSAFE
        try:
            self.gui.FAILSAFE = False
            self.gui.mouseUp(button="left")
        finally:
            self.gui.FAILSAFE = previous

    def check_failsafe(self):
        self.gui.failSafeCheck()


class MacExecutionGuard:
    """Never activates, moves, or resizes a target window."""

    def __init__(self, window: WindowInfo, backend: MacWindowBackend, calibration: Calibration | None = None):
        self.window, self.backend, self.calibration = window, backend, calibration
        self.typing_allowed = False

    def check(self):
        import AppKit
        import Quartz

        active = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if active is None or active.processIdentifier() != self.window.pid:
            raise ExecutionError("Target lost focus. Bring the intended window forward and submit a new request.")
        records = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        if records is None:
            raise ExecutionError("Cannot verify the focused window; no input sent.")
        # Quartz returns windows front to back. A sheet, settings dialog, or menu
        # above the main window must block global input even within the same app.
        top = next((record for record in records if record.get("kCGWindowOwnerPID") == self.window.pid
                    and float(record.get("kCGWindowBounds", {}).get("Height", 0)) > 40
                    and float(record.get("kCGWindowBounds", {}).get("Width", 0)) > 40), None)
        if top is None or top.get("kCGWindowNumber") != self.window.window_id:
            raise ExecutionError("A different window or dialog has focus. Close it before requesting input.")
        bounds = top["kCGWindowBounds"]
        actual = (float(bounds["X"]), float(bounds["Y"]), float(bounds["Width"]), float(bounds["Height"]))
        expected = self.window.bounds
        if actual != (expected.left, expected.top, expected.width, expected.height):
            raise GeometryError("Window geometry changed. Recalibrate before requesting input.")

    def prepare(self):
        import Quartz
        if not Quartz.CGPreflightPostEventAccess():
            raise ExecutionError("Accessibility permission is required for the app running Python.")
        flags = Quartz.CGEventSourceFlagsState(Quartz.kCGEventSourceStateCombinedSessionState)
        masks = (Quartz.kCGEventFlagMaskShift | Quartz.kCGEventFlagMaskControl |
                 Quartz.kCGEventFlagMaskAlternate | Quartz.kCGEventFlagMaskCommand |
                 Quartz.kCGEventFlagMaskAlphaShift)
        if flags & masks:
            raise ExecutionError("Release modifier keys and disable Caps Lock before input.")
        self.check()
        if self.calibration is not None:
            self.calibration.check_window(select_window(self.backend, self.window.window_id))
            self.calibration.check_frame(self.backend.capture_window(self.window.window_id).size)
        self.check()


SHIFTED_KEYS = dict(zip('~!@#$%^&*()_+{}|:"<>?', '`1234567890-=[]\\;\',./'))


class ActionExecutor:
    def __init__(self, guard, driver, stop_file: Path, *, clock=time.monotonic, sleep=time.sleep):
        self.guard, self.driver, self.stop_file = guard, driver, stop_file
        self.clock, self.sleep = clock, sleep
        self.held_keys = set()
        self.mouse_held = False

    def check(self):
        if self.stop_file.exists():
            raise Stopped("Stop signal is set. No further input will execute until --clear-stop.")
        self.guard.check()
        self.driver.check_failsafe()

    def wait(self, seconds):
        deadline = self.clock() + seconds
        while self.clock() < deadline:
            self.check()
            self.sleep(min(0.02, max(0, deadline - self.clock())))
        self.check()

    def down(self, key):
        self.check()
        self.held_keys.add(key)  # Also release if a backend fails after posting key-down.
        self.driver.key_down(key)

    def up(self, key):
        self.driver.key_up(key)
        self.held_keys.discard(key)

    def cleanup(self):
        errors = []
        if self.mouse_held:
            try:
                self.driver.mouse_up()
                self.mouse_held = False
            except Exception as exc:
                errors.append(exc)
        for key in tuple(self.held_keys):
            try:
                self.up(key)
            except Exception as exc:
                errors.append(exc)
        if errors:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.touch()
            raise ExecutionError("Input release failed. Release held controls manually before continuing.")

    def execute(self, action, calibration: Calibration | None = None):
        if calibration is not None:
            validate_game_action(action, calibration)
        if isinstance(action, TypeText) and not self.guard.typing_allowed:
            raise ExecutionError("Typing is allowed only in the disposable input-test field.")
        if self.stop_file.exists():
            raise Stopped("Stop signal is set. Use --clear-stop before a new request.")
        self.guard.prepare()
        self.check()
        try:
            if isinstance(action, (Move, Click)):
                if calibration is None:
                    raise ExecutionError("Mouse actions require verified game calibration.")
                x, y = calibration.normalized_to_desktop(action.x, action.y)
                rect = calibration.touchscreen_desktop if isinstance(action, Click) else calibration.game_desktop
                # PyAutoGUI sends integer desktop coordinates. Quantize inside the
                # target rectangle so Retina half-points cannot hit a border.
                x = min(max(round(x), math.ceil(rect.left)), math.ceil(rect.left + rect.width) - 1)
                y = min(max(round(y), math.ceil(rect.top)), math.ceil(rect.top + rect.height) - 1)
                self.driver.move_to(x, y)
                self.check()
                if tuple(self.driver.position()) != (x, y):
                    raise ExecutionError("Cursor did not reach the requested target. No click sent.")
                if isinstance(action, Click):
                    self.mouse_held = True
                    self.driver.mouse_down()
                    self.wait(0.08)
                    self.driver.mouse_up()
                    self.mouse_held = False
            elif isinstance(action, Press):
                self.down(action.key)
                self.wait(action.seconds)
                self.up(action.key)
            elif isinstance(action, TypeText):
                for character in action.text:
                    shifted = character.isupper() or character in SHIFTED_KEYS
                    key = SHIFTED_KEYS.get(character, character.lower())
                    if shifted:
                        self.down("shiftleft")
                    self.down(key)
                    self.up(key)
                    if shifted:
                        self.up("shiftleft")
                    self.wait(0.04)
            elif isinstance(action, Wait):
                self.wait(action.seconds)
            else:
                raise ExecutionError("Unsupported action type.")
            self.check()
        finally:
            self.cleanup()
        return {"action": action.model_dump(), "executed": True, "model_called": False,
                "cursor": list(self.driver.position())}


def execute_game_action(action, profile: Path, stop_file: Path, *, deadline=None):
    from computer.capture import load_calibration
    calibration = load_calibration(profile)
    validate_game_action(action, calibration)
    backend = MacWindowBackend()
    window = select_window(backend, calibration.window.window_id)
    calibration.check_window(window)
    guard = MacExecutionGuard(window, backend, calibration)
    if deadline is not None: guard = DeadlineGuard(guard, deadline)
    # Importing the driver sends no input. The executor checks access, focus,
    # geometry, and capture scale before the first event, with one frame capture.
    driver = PyAutoGUIDriver()
    return ActionExecutor(guard, driver, stop_file).execute(action, calibration)


def focus_game_window(profile: Path, *, backend=None):
    """Optional startup focus only. Never re-focus after input has begun."""
    import AppKit
    from computer.capture import load_calibration
    calibration = load_calibration(profile)
    if not calibration.verified:
        raise ExecutionError("Game calibration must be verified before focusing.")
    backend = backend or MacWindowBackend()
    # Stage Manager can hide/shrink a background window. Restore the already
    # calibrated process once, then verify its exact window and geometry.
    window = calibration.window
    app = AppKit.NSRunningApplication.runningApplicationWithProcessIdentifier_(window.pid)
    if app is None or str(app.localizedName()).casefold() != "melonds":
        raise ExecutionError("The selected process is not the intended melonDS app.")
    if not app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps):
        raise ExecutionError("Could not focus melonDS; no game input sent.")
    limit = time.monotonic() + 2
    last_window_error = None
    while time.monotonic() < limit:
        active = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if active is not None and active.processIdentifier() == window.pid:
            try:
                calibration.check_window(select_window(backend, window.window_id))
                return
            except (CaptureError, GeometryError) as exc:
                # Stage Manager can expose intermediate geometry while restoring.
                # Input remains blocked until exact calibrated geometry returns.
                last_window_error = exc
        time.sleep(.02)
    if last_window_error is not None:
        raise last_window_error
    raise ExecutionError("macOS did not focus melonDS; no game input sent.")
