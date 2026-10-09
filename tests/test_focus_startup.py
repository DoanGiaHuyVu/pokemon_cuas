"""Startup activation can restore Stage Manager previews, never bypass geometry."""
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from computer.capture import CaptureError, calibrate, confirm_calibration
from computer.executor import ExecutionError, focus_game_window
from computer.geometry import GeometryError
from test_capture import FakeBackend
from test_executor import FakeClock
from pydantic import SecretStr


class StartupFocusTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.profile=Path(self.tmp.name)/'calibration.json';self.backend=FakeBackend()
        calibrate(self.backend,self.profile,Path(self.tmp.name)/'screenshots');confirm_calibration(self.backend,self.profile)
        self.activated=False;self.activations=0;self.name='melonDS';self.front_pid=456
        self.app=SimpleNamespace(localizedName=lambda:self.name,activateWithOptions_=self.activate,
                                 processIdentifier=lambda:456)
        self.workspace=SimpleNamespace(frontmostApplication=lambda:SimpleNamespace(processIdentifier=lambda:self.front_pid))
        self.kit=SimpleNamespace(NSRunningApplication=SimpleNamespace(runningApplicationWithProcessIdentifier_=lambda pid:self.app if pid==456 else None),
            NSApplicationActivateIgnoringOtherApps=1,NSWorkspace=SimpleNamespace(sharedWorkspace=lambda:self.workspace))

    def activate(self,options):self.activated=True;self.activations+=1;return True

    def focus(self):
        with patch.dict('sys.modules',{'AppKit':self.kit}):focus_game_window(self.profile,backend=self.backend)

    def test_activation_precedes_window_discovery_for_stage_manager(self):
        original=self.backend.list_windows
        def windows():
            if not self.activated:raise CaptureError('Stage Manager preview')
            return original()
        self.backend.list_windows=windows;self.focus()
        self.assertEqual(self.activations,1)

    def test_wrong_process_is_never_activated(self):
        self.name='OtherApp'
        with self.assertRaises(ExecutionError):self.focus()
        self.assertEqual(self.activations,0)

    def test_restored_geometry_still_must_match(self):
        bounds=self.backend.window.bounds.model_copy(update={'left':471.})
        self.backend.window=self.backend.window.model_copy(update={'bounds':bounds})
        clock=FakeClock()
        with patch('computer.executor.time.monotonic',clock.time),patch('computer.executor.time.sleep',clock.sleep):
            with self.assertRaises(GeometryError):self.focus()
        self.assertEqual(self.activations,1)

    def test_transient_restore_geometry_can_settle_before_exact_match(self):
        window=self.backend.window;calls=0
        def restoring():
            nonlocal calls
            calls+=1
            if calls<3:
                return [window.model_copy(update={'bounds':window.bounds.model_copy(update={'width':300.})})]
            return [window]
        self.backend.list_windows=restoring;clock=FakeClock()
        with patch('computer.executor.time.monotonic',clock.time),patch('computer.executor.time.sleep',clock.sleep):self.focus()
        self.assertEqual(self.activations,1);self.assertEqual(calls,3)

    def test_focus_timeout_has_only_one_activation_and_bounded_wait(self):
        self.front_pid=999;clock=FakeClock()
        with patch('computer.executor.time.monotonic',clock.time),patch('computer.executor.time.sleep',clock.sleep):
            with self.assertRaises(ExecutionError):self.focus()
        self.assertEqual(self.activations,1);self.assertLess(clock.now,2.1)

    def test_cli_explains_changed_geometry_before_starting_models_or_input(self):
        from harness.cli import main
        settings=SimpleNamespace(gemini_api_key=SecretStr('never-print'))
        output=io.StringIO()
        with patch('harness.cli.load_settings',return_value=settings),patch('harness.cli.time.sleep'),patch('computer.executor.focus_game_window',side_effect=GeometryError('The emulator moved, resized, or restarted. Run --calibrate again.')),patch('harness.live_gameplay.run_live_gameplay') as run,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(output):
            self.assertEqual(main(['--play','--focus-game','--prompt','Inspect my Pokemon party']),3)
        run.assert_not_called()
        self.assertIn('emulator moved',output.getvalue());self.assertIn('--confirm-calibration',output.getvalue())
