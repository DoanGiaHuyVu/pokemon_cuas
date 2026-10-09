"""Disposable, self-owned native field for literal typing and stop/release checks."""

import os
import time
from pathlib import Path
from uuid import uuid4

from computer.capture import MacWindowBackend
from computer.executor import ActionExecutor, ExecutionError, MacExecutionGuard, PyAutoGUIDriver, Stopped
from computer.geometry import Rect, WindowInfo
from harness.actions import Press, TypeText


def settle_owned_window(sample, *, clock=time.monotonic, sleep=time.sleep):
    """Wait for the owned field's native focus and bounds to settle, at most 2s."""
    deadline = clock() + 2
    previous, stable_since = None, clock()
    while clock() < deadline:
        record, ready = sample()
        signature = (record.get("kCGWindowNumber"), dict(record["kCGWindowBounds"])) if record and ready else None
        if signature is None or signature != previous:
            previous, stable_since = signature, clock()
        elif clock() - stable_since >= 0.2:
            return record
        sleep(0.02)
    raise ExecutionError("Disposable test field did not acquire stable native focus. No input sent.")


def run_input_test(stop_file: Path) -> dict:
    import AppKit
    import Foundation
    import Quartz

    if stop_file.exists():
        raise Stopped("Stop signal is set. Clear it explicitly before starting the input test.")
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    previous_app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
    app.finishLaunching()
    title = f"Pokémon CUA disposable input test {uuid4().hex[:8]}"
    private_stop = stop_file.parent / f"input-test-stop-{uuid4().hex}"
    native_window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(0, 0, 440, 160), AppKit.NSWindowStyleMaskTitled | AppKit.NSWindowStyleMaskClosable,
        AppKit.NSBackingStoreBuffered, False,
    )
    native_window.setReleasedWhenClosed_(False)
    native_window.setTitle_(title)
    field = AppKit.NSTextField.alloc().initWithFrame_(AppKit.NSMakeRect(30, 65, 380, 35))
    field.setFont_(AppKit.NSFont.systemFontOfSize_(20))
    field.setEditable_(True)
    field.setSelectable_(True)
    native_window.contentView().addSubview_(field)
    label = AppKit.NSTextField.labelWithString_("Disposable typing test — no game input")
    label.setFrame_(AppKit.NSMakeRect(30, 110, 380, 25))
    native_window.contentView().addSubview_(label)
    stop_deadline = None

    def pump():
        # Deliver real posted keyboard events through the owned native event loop.
        for _ in range(100):
            event = app.nextEventMatchingMask_untilDate_inMode_dequeue_(
                AppKit.NSEventMaskAny, Foundation.NSDate.distantPast(), Foundation.NSDefaultRunLoopMode, True,
            )
            if event is None:
                break
            app.sendEvent_(event)
        app.updateWindows()
        if stop_file.exists():
            raise Stopped("User stop signal is set.")

    def field_focused():
        editor = field.currentEditor()
        return native_window.isKeyWindow() and editor is not None and native_window.firstResponder() == editor

    try:
        native_window.center()
        native_window.makeKeyAndOrderFront_(None)
        app.activateIgnoringOtherApps_(True)
        native_window.makeFirstResponder_(field)

        def sample():
            pump()
            records = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID)
            record = next((item for item in records or [] if item.get("kCGWindowOwnerPID") == os.getpid()
                           and item.get("kCGWindowNumber") == native_window.windowNumber()), None)
            return record, field_focused()

        record = settle_owned_window(sample)
        bounds = record["kCGWindowBounds"]
        window = WindowInfo(window_id=int(record["kCGWindowNumber"]), pid=os.getpid(), owner="Python", title=title,
                            bounds=Rect(left=bounds["X"], top=bounds["Y"], width=bounds["Width"], height=bounds["Height"]))

        class FieldGuard(MacExecutionGuard):
            def __init__(self):
                super().__init__(window, MacWindowBackend())
                self.typing_allowed = True

            def check(self):
                pump()
                if stop_deadline is not None and time.monotonic() >= stop_deadline:
                    private_stop.touch()
                    raise Stopped("Owned test stop signal is set.")
                if not field_focused():
                    raise ExecutionError("Disposable test field lost native keyboard focus.")
                super().check()

        driver = PyAutoGUIDriver()
        executor = ActionExecutor(FieldGuard(), driver, private_stop)
        expected = "HUY 42!"
        executor.execute(TypeText(tool="TYPE", text=expected))
        pump()
        actual = str(field.currentEditor().string())
        if actual != expected:
            raise ExecutionError(f"Literal typing mismatch in the owned field: expected {expected!r}, got {actual!r}.")

        # Set the stop flag only after a real key-down was sent to the owned field.
        original_down = driver.key_down
        def stop_after_down(key):
            nonlocal stop_deadline
            original_down(key)
            stop_deadline = time.monotonic() + 0.08
        driver.key_down = stop_after_down
        stopped = False
        try:
            executor.execute(Press(tool="PRESS", key="shiftleft", seconds=1))
        except Stopped:
            stopped = True
        released = not Quartz.CGEventSourceKeyState(Quartz.kCGEventSourceStateCombinedSessionState, 0x38)
        if not stopped or not released or executor.held_keys:
            raise ExecutionError("Stop/release check failed. Release Shift before continuing.")
        return {"literal_typing": actual, "stop_detected": stopped, "held_key_released": released,
                "game_input_sent": False}
    finally:
        private_stop.unlink(missing_ok=True)
        native_window.close()
        if previous_app is not None:
            previous_app.activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)
