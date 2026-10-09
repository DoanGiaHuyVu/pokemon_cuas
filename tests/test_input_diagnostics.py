"""Native readiness diagnostics must report the actual barrier without dispatch."""
import contextlib
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pydantic import SecretStr
from computer.diagnostics import check_game_input
from computer.executor import ExecutionError
from harness.cli import main


class InputDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.quartz=SimpleNamespace(CGPreflightScreenCaptureAccess=lambda:True,CGPreflightPostEventAccess=lambda:True,
            CGEventSourceFlagsState=lambda _:0,kCGEventSourceStateCombinedSessionState=0,
            kCGEventFlagMaskShift=1,kCGEventFlagMaskControl=2,kCGEventFlagMaskAlternate=4,
            kCGEventFlagMaskCommand=8,kCGEventFlagMaskAlphaShift=16)
        active=SimpleNamespace(localizedName=lambda:'melonDS')
        self.kit=SimpleNamespace(NSWorkspace=SimpleNamespace(sharedWorkspace=lambda:SimpleNamespace(frontmostApplication=lambda:active)))
        self.guard=Mock();self.cal=SimpleNamespace(verified=True,window=object())

    def report(self,focus=False):
        with patch.dict('sys.modules',{'Quartz':self.quartz,'AppKit':self.kit}),patch('computer.diagnostics.load_calibration',return_value=self.cal),patch('computer.diagnostics.MacExecutionGuard',return_value=self.guard),patch('computer.diagnostics.focus_game_window') as activate:
            result=check_game_input(Path('fixture.json'),focus=focus,backend=object(),sleep=lambda _:None)
            return result,activate.call_count

    def test_missing_access_reports_terminal_guidance_without_focusing_or_input(self):
        self.quartz.CGPreflightPostEventAccess=lambda:False
        report,calls=self.report(focus=True)
        self.assertFalse(report['ready']);self.assertFalse(report['accessibility_granted'])
        self.assertIn('enable Terminal',report['reason']);self.assertEqual(calls,0)
        self.guard.prepare.assert_not_called()
        self.assertEqual(report['game_inputs'],0);self.assertEqual(report['model_calls'],0)

    def test_success_runs_the_same_guard_and_one_optional_activation(self):
        report,calls=self.report(focus=True)
        self.assertTrue(report['ready']);self.assertEqual(calls,1)
        self.guard.prepare.assert_called_once();self.assertEqual(report['game_inputs'],0)

    def test_caps_lock_reason_and_flags_are_preserved(self):
        self.quartz.CGEventSourceFlagsState=lambda _:16
        self.guard.prepare.side_effect=ExecutionError('Release modifier keys and disable Caps Lock before input.')
        report,_=self.report()
        self.assertFalse(report['ready']);self.assertEqual(report['held_modifiers'],['Caps Lock'])
        self.assertIn('Caps Lock',report['reason'])

    def test_cli_check_needs_no_prompt_or_model_and_returns_blocked_status(self):
        report={'ready':False,'reason':'Accessibility permission is required.','game_inputs':0,'model_calls':0}
        output=io.StringIO()
        with patch('harness.cli.load_settings',return_value=SimpleNamespace(gemini_api_key=SecretStr('secret'))),patch('computer.diagnostics.check_game_input',return_value=report) as check,contextlib.redirect_stdout(output):
            self.assertEqual(main(['--input-check','--focus-game','--start-delay','0']),3)
        self.assertIn('Accessibility permission',output.getvalue())
        self.assertEqual(check.call_args.kwargs,{'focus':True,'start_delay':0.})
