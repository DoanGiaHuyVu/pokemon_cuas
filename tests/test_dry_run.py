"""Step 6 durable traces and zero-dispatch checks with fake screen/API evidence."""

import builtins
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import SecretStr

from computer.capture import CaptureError, calibrate, capture_observation, confirm_calibration
from harness.cli import main
from harness.dry_run import TraceWriteError, run_dry_run
from harness.journal import atomic_json
from test_capture import FakeBackend
from test_perception import APIError, FakeClient, response
from test_proposals import proposal


class DryRunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs, self.images = self.root / 'runs', self.root / 'screenshots'
        self.profile = self.runs / 'calibration.json'
        self.backend = FakeBackend()
        calibrate(self.backend, self.profile, self.images)
        confirm_calibration(self.backend, self.profile)
        self.settings = SimpleNamespace(gemini_api_key=SecretStr('test-secret'),
                                        gemma_model='gemma-4-26b-a4b-it', laya_model_path=Path('/fake/checkpoint'))
        self.captured = None
        self.snapshots = []
        self.kind, self.pause, self.invalid, self.stale = 'title', False, False, False
        self.client = FakeClient([])
        self.capture_function = capture_observation
        self.writer = atomic_json
        self.executor = Mock(side_effect=AssertionError('Input execution leaked into dry run'))

    def capture(self, *args, **kwargs):
        self.captured = self.capture_function(*args, **kwargs)
        value = proposal(self.captured['observation_id'], kind=self.kind, pause=self.pause)
        if self.kind == 'main_menu':
            value['visible_observation']['upper_screen']['visible_text'] = ['New Game', 'Continue']
        if self.stale:
            value['observation_id'] = 'old-observation'
        if not self.client.outcomes:
            self.client.outcomes.append(response('bad JSON test-secret' if self.invalid else json.dumps(value)))
        return self.captured

    def write(self, path, record, **kwargs):
        self.writer(path, record, **kwargs)
        if path.name.endswith('.trace.json'):
            self.snapshots.append(json.loads(path.read_text()))

    def run_pipeline(self):
        with patch('harness.dry_run.capture_observation', side_effect=self.capture), patch(
            'harness.dry_run.atomic_json', side_effect=self.write
        ), patch('computer.executor.execute_game_action', self.executor):
            result = run_dry_run(self.settings, 'Reach the menu test-secret', self.backend,
                                 self.profile, self.images, self.runs, client_factory=lambda **kwargs: self.client)
        self.executor.assert_not_called()
        return result

    def saved(self):
        paths = list(self.runs.glob('*.trace.json'))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def assert_no_outcome(self, trace):
        self.assertFalse(trace['executed'] or trace['task_complete'] or trace['laya']['called'])
        self.assertEqual(trace['execution'], {'enabled': False, 'status': 'disabled', 'call_count': 0, 'action': None})
        self.assertEqual(trace['task_outcome'], 'not_assessed')
        self.assertEqual(trace['observed_effect']['status'], 'not_observed')
        self.assertIsNone(trace['observed_effect']['observation_id'])

    def test_full_trace_links_artifacts_hash_and_prediction_without_actual_effect(self):
        trace, path = self.run_pipeline()
        self.assertEqual(trace['status'], 'proposed')
        self.assertEqual(trace['validation'], 'passed')
        self.assertNotEqual(trace['run_id'], trace['observation']['observation_id'])
        self.assertEqual(trace['proposal']['observation_id'], trace['observation']['observation_id'])
        for value in trace['artifacts'].values():
            self.assertTrue(Path(value).is_file())
        self.assertEqual(trace['observation']['image_sha256'], hashlib.sha256(Path(trace['artifacts']['image']).read_bytes()).hexdigest())
        self.assertEqual(trace['model']['attempts'], 1)
        self.assertEqual(trace['model']['usage']['total_token_count'], 30)
        self.assertEqual(trace['expected_effect']['source'], 'model_prediction')
        self.assertEqual(trace['expected_effect']['description'], trace['proposal']['expected_result'])
        self.assertEqual(trace['model_claims'], trace['proposal']['visible_observation'])
        self.assertEqual(trace['model']['max_output_tokens'], 1200)
        self.assertEqual(trace['model']['schema_enforcement'], 'local')
        self.assertIsNotNone(trace['finished_at'])
        self.assertTrue(all(value >= 0 for value in trace['latency_seconds'].values()))
        self.assertEqual(json.loads(path.read_text()), trace)
        self.assertNotIn('test-secret', path.read_text())
        self.assert_no_outcome(trace)

    def test_intermediate_snapshots_are_readable_before_side_effects(self):
        trace, _ = self.run_pipeline()
        self.assertEqual([snapshot['events'][-1]['event'] for snapshot in self.snapshots],
                         ['run_started', 'capture_started', 'capture_recorded', 'planning_started', 'proposal_validated', 'run_finished'])
        self.assertIsNone(self.snapshots[1]['observation'])
        self.assertEqual(self.snapshots[1]['model']['attempts'], 0)
        self.assertIsNone(self.snapshots[3]['model']['attempts'])
        self.assertEqual(self.snapshots[3]['status'], 'in_progress')
        for snapshot in self.snapshots:
            self.assert_no_outcome(snapshot)
            self.assertEqual([event['sequence'] for event in snapshot['events']], list(range(1, len(snapshot['events']) + 1)))
        self.assertEqual(trace['status'], 'proposed')

    def test_menu_pause_never_counts_as_success_or_execution(self):
        self.kind, self.pause = 'main_menu', True
        trace, _ = self.run_pipeline()
        self.assertEqual(trace['status'], 'paused')
        self.assertIsNone(trace['proposal']['action'])
        self.assert_no_outcome(trace)

    def test_invalid_and_stale_outputs_preserved_and_blocked(self):
        for field in ('invalid', 'stale'):
            with self.subTest(field=field):
                setattr(self, field, True)
                trace, _ = self.run_pipeline()
                self.assertEqual(trace['status'], 'blocked')
                self.assertEqual(trace['validation'], 'failed')
                self.assertEqual(trace['error']['category'], 'proposal_validation')
                self.assertIsNone(trace['proposal'])
                record = json.loads(Path(trace['artifacts']['proposal']).read_text())
                self.assertIn('raw_response', record)
                self.assertEqual(record['status'], 'rejected')
                self.assertNotIn('test-secret', json.dumps(record))
                self.assert_no_outcome(trace)
                setattr(self, field, False)

    def test_capture_failure_is_durable_and_calls_no_model_or_executor(self):
        original = self.backend.window
        self.backend.window = original.model_copy(update={'bounds': original.bounds.model_copy(update={'left': 900})})
        count = self.backend.captures
        trace, _ = self.run_pipeline()
        self.assertEqual(trace['status'], 'blocked')
        self.assertEqual(trace['error']['category'], 'capture')
        self.assertEqual(self.backend.captures, count)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(trace['model']['attempts'], 0)
        self.assertIsNone(trace['observation'])
        self.assert_no_outcome(trace)

    def test_api_failure_logs_attempts_latency_and_sanitized_error(self):
        self.client.outcomes = [APIError(403)]
        trace, _ = self.run_pipeline()
        self.assertEqual(trace['status'], 'blocked')
        self.assertEqual(trace['error']['category'], 'model_request')
        self.assertEqual(trace['model']['attempts'], 1)
        self.assertTrue(trace['model']['called'])
        self.assertIsNotNone(trace['latency_seconds']['planning_and_validation'])
        self.assertNotIn('test-secret', json.dumps(trace))
        self.assert_no_outcome(trace)

    def test_transient_retry_remains_one_run_and_zero_execution(self):
        self.client.outcomes = [APIError(503)]
        original_capture = self.capture
        def retry_capture(*args, **kwargs):
            observation = original_capture(*args, **kwargs)
            self.client.outcomes.append(response(json.dumps(proposal(observation['observation_id']))))
            return observation
        with patch.object(self, 'capture', side_effect=retry_capture):
            trace, _ = self.run_pipeline()
        self.assertEqual(trace['model']['attempts'], 2)
        self.assertEqual(len(self.client.calls), 2)
        self.assertEqual(len(list(self.runs.glob('*.trace.json'))), 1)
        self.assert_no_outcome(trace)

    def test_stop_before_capture_and_after_planning_block_further_work(self):
        stop = self.runs / 'STOP'
        stop.touch()
        trace, _ = self.run_pipeline()
        self.assertEqual(trace['error']['category'], 'stop_signal')
        self.assertEqual(self.client.calls, [])
        self.assertIsNone(self.captured)
        stop.unlink()
        async def request_and_stop(**kwargs):
            self.client.calls.append(kwargs)
            stop.touch()
            return self.client.outcomes.pop(0)
        self.client.aio.models.generate_content = request_and_stop
        trace, _ = self.run_pipeline()
        self.assertEqual(trace['status'], 'blocked')
        self.assertEqual(trace['error']['category'], 'stop_signal')
        self.assertEqual(trace['validation'], 'passed')
        self.assert_no_outcome(trace)

    def test_capture_and_planning_cancellation_persist_trace(self):
        with patch('harness.dry_run.capture_observation', side_effect=KeyboardInterrupt):
            trace, _ = run_dry_run(self.settings, 'Inspect', self.backend, self.profile, self.images, self.runs)
        self.assertEqual(trace['status'], 'cancelled')
        self.assertFalse(trace['model']['called'])
        self.assert_no_outcome(trace)
        with patch('models.planner.propose_images', new=lambda *args, **kwargs: None), patch(
            'models.planner.asyncio.run', side_effect=KeyboardInterrupt
        ):
            trace, _ = self.run_pipeline()
        self.assertEqual(trace['status'], 'cancelled')
        self.assertIsNone(trace['model']['attempts'])
        self.assertIsNone(trace['model']['called'])
        self.assertEqual(trace['exit_code'], 130)
        self.assert_no_outcome(trace)

    def test_stop_after_capture_prevents_api_request(self):
        original_capture = self.capture
        def capture_then_stop(*args, **kwargs):
            observation = original_capture(*args, **kwargs)
            (self.runs / 'STOP').touch()
            return observation
        with patch.object(self, 'capture', side_effect=capture_then_stop):
            trace, _ = self.run_pipeline()
        self.assertEqual(trace['status'], 'blocked')
        self.assertEqual(trace['error']['category'], 'stop_signal')
        self.assertIsNotNone(trace['observation'])
        self.assertEqual(trace['model']['attempts'], 0)
        self.assertEqual(self.client.calls, [])
        self.assert_no_outcome(trace)

    def test_unexpected_exception_sanitized_in_trace(self):
        with patch('harness.dry_run.capture_observation', side_effect=RuntimeError('test-secret')):
            trace, _ = run_dry_run(self.settings, 'Inspect', self.backend, self.profile, self.images, self.runs)
        self.assertEqual(trace['status'], 'failed')
        self.assertEqual(trace['error']['category'], 'internal_error')
        self.assertNotIn('test-secret', json.dumps(trace))
        self.assert_no_outcome(trace)

    def test_failed_initial_trace_write_stops_before_capture_or_model(self):
        with patch('harness.dry_run.atomic_json', side_effect=OSError('test-secret')), patch(
            'harness.dry_run.capture_observation'
        ) as capture, self.assertRaises(TraceWriteError) as caught:
            run_dry_run(self.settings, 'Inspect', self.backend, self.profile, self.images, self.runs,
                        client_factory=lambda **kwargs: self.client)
        capture.assert_not_called()
        self.assertEqual(self.client.calls, [])
        self.assertNotIn('test-secret', str(caught.exception))

    def test_later_trace_failure_leaves_last_complete_snapshot_and_stops_before_model(self):
        count = 0
        def failing_write(path, record, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise OSError('test-secret')
            self.writer(path, record, **kwargs)
        with patch('harness.dry_run.atomic_json', side_effect=failing_write), self.assertRaises(TraceWriteError):
            run_dry_run(self.settings, 'Inspect', self.backend, self.profile, self.images, self.runs,
                        client_factory=lambda **kwargs: self.client)
        saved = self.saved()
        self.assertEqual(saved['events'][-1]['event'], 'capture_started')
        self.assertEqual(saved['status'], 'in_progress')
        self.assertIsNone(saved['finished_at'])
        self.assertEqual(self.client.calls, [])
        self.assert_no_outcome(saved)

    def test_each_request_has_unique_run_and_fresh_observation(self):
        first, _ = self.run_pipeline()
        second, _ = self.run_pipeline()
        self.assertNotEqual(first['run_id'], second['run_id'])
        self.assertNotEqual(first['observation']['observation_id'], second['observation']['observation_id'])
        self.assertEqual(len(self.client.calls), 2)

    def test_pipeline_revalidates_raw_output_and_ignores_adapter_execution_claims(self):
        observation = capture_observation(self.backend, self.profile, self.images, self.runs, 'Inspect')
        fake = {'raw_response': json.dumps(proposal(observation['observation_id'])),
                'executed': True, 'task_complete': True, 'attempts': 1}
        with patch('harness.dry_run.capture_observation', return_value=observation), patch(
            'harness.dry_run.propose_observation', return_value=(fake, Path(observation['metadata_path']).with_suffix('.proposal.json'))
        ):
            trace, _ = run_dry_run(self.settings, 'Inspect', self.backend, self.profile, self.images, self.runs)
        self.assert_no_outcome(trace)
        fake['raw_response'] = fake['raw_response'].replace('"task_complete": false', '"task_complete": true')
        with patch('harness.dry_run.capture_observation', return_value=observation), patch(
            'harness.dry_run.propose_observation', return_value=(fake, self.runs / 'unused.json')
        ):
            trace, _ = run_dry_run(self.settings, 'Inspect', self.backend, self.profile, self.images, self.runs)
        self.assertEqual(trace['status'], 'blocked')
        self.assert_no_outcome(trace)

    def invoke_cli(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'pyautogui', 'laya', 'torch', 'transformers'} or name == 'computer.executor':
                raise AssertionError('Dry run imported input/local model runtime')
            return original_import(name, *args, **kwargs)
        output, error = io.StringIO(), io.StringIO()
        with patch('harness.cli.PROJECT_ROOT', self.root), patch('harness.cli.load_settings', return_value=self.settings), patch(
            'computer.capture.MacWindowBackend', return_value=self.backend
        ), patch('harness.dry_run.capture_observation', side_effect=self.capture), patch(
            'google.genai.Client', return_value=self.client
        ), patch('computer.executor.execute_game_action', self.executor), patch(
            'builtins.__import__', side_effect=guarded_import
        ), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = main(['--dry-run', '--prompt', 'Reach the menu test-secret'])
        self.executor.assert_not_called()
        return status, output.getvalue(), error.getvalue()

    def test_cli_imports_no_executor_or_local_model_and_reports_trace(self):
        status, output, error = self.invoke_cli()
        self.assertEqual(status, 0)
        self.assertEqual(error, '')
        self.assertIn('Dry-run status: proposed', output)
        self.assertIn('Trace:', output)
        self.assertIn('Execution calls: 0', output)
        self.assertNotIn('test-secret', output)
        self.assert_no_outcome(self.saved())

    def test_cli_block_and_trace_write_failure_are_clean_errors(self):
        self.invalid = True
        status, output, error = self.invoke_cli()
        self.assertEqual(status, 4)
        self.assertIn('Trace:', output)
        self.assertNotIn('Traceback', error)
        with patch('harness.dry_run.atomic_json', side_effect=OSError('test-secret')):
            status, output, error = self.invoke_cli()
        self.assertEqual(status, 5)
        self.assertIn('Trace target', output)
        self.assertNotIn('test-secret', output + error)

    def test_cli_cancellation_reports_durable_trace_and_returns_130(self):
        with patch('models.planner.propose_images', new=lambda *args, **kwargs: None), patch(
            'models.planner.asyncio.run', side_effect=KeyboardInterrupt
        ):
            status, output, error = self.invoke_cli()
        self.assertEqual(status, 130)
        self.assertIn('Trace:', output)
        self.assertNotIn('Traceback', error)
        self.assert_no_outcome(self.saved())

    def test_cli_rejects_execute_action_preflight_and_other_operations(self):
        for extra in (['--execute'], ['--action', '{}'], ['--mode', 'preflight'], ['--propose'], ['--capture']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--dry-run', '--prompt', 'Inspect', *extra])


class JournalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'runs/trace.json'

    def test_unicode_redaction_and_only_complete_document_replaces_old_snapshot(self):
        atomic_json(self.path, {'message': 'Pokémon test-secret'}, secret='test-secret')
        self.assertEqual(json.loads(self.path.read_text()), {'message': 'Pokémon [hidden]'})
        old = self.path.read_bytes()
        with patch('harness.journal.os.replace', side_effect=OSError('disk error')), self.assertRaises(OSError):
            atomic_json(self.path, {'message': 'replacement'})
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_flush_failure_preserves_old_snapshot_and_cleans_temporary_file(self):
        atomic_json(self.path, {'version': 1})
        with patch('harness.journal.os.fsync', side_effect=OSError('disk error')), self.assertRaises(OSError):
            atomic_json(self.path, {'version': 2})
        self.assertEqual(json.loads(self.path.read_text()), {'version': 1})
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_nonfinite_json_rejected_before_replacing_any_file(self):
        atomic_json(self.path, {'version': 1})
        with self.assertRaises(ValueError):
            atomic_json(self.path, {'latency': float('nan')})
        self.assertEqual(json.loads(self.path.read_text()), {'version': 1})

    def test_nested_secret_is_redacted_before_json_escaping(self):
        secret = 'test-"\\-secret'
        record = {'items': [secret, {'message': 'prefix ' + secret}]}
        atomic_json(self.path, record, secret=secret)
        self.assertEqual(json.loads(self.path.read_text()), {'items': ['[hidden]', {'message': 'prefix [hidden]'}]})
        self.assertEqual(record['items'][0], secret)


if __name__ == '__main__':
    unittest.main()
