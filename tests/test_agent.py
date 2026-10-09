"""Autonomous routing and failure boundaries with no desktop, network, or weights."""
import builtins
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr
from harness.agent import AgentRecordError, GoalAssessment, GuardResult, run_agent
from harness.agent_job import job_from_prompt
from harness.agent_proposals import parse_agent_proposal
from harness.agent_simulation import SimClock, SimulatedPorts, fixture_prediction
from harness.cli import main
from harness.proposals import ProposalError
from models.laya_local import ARTIFACTS
from test_laya_local import questions_fixture
from test_executor import FakeClock, FakeGuard, SpyDriver
from computer.executor import ActionExecutor, DeadlineGuard, Stopped
from harness.actions import parse_action


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'checkpoint'
        for name in ARTIFACTS:
            p = self.checkpoint / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('{}')
        self.questions = questions_fixture()
        (self.checkpoint / 'questions.json').write_text(json.dumps(self.questions))
        self.clock = SimClock()
        self.ports = SimulatedPorts(self.questions, self.clock)
        self.job = job_from_prompt('Enter the game and get your first Pokemon')

    def run_job(self, ports=None, job=None):
        ports = ports or self.ports
        record, path = run_agent(job or self.job, ports, self.root / 'runs', self.checkpoint,
                                 simulation=True, clock=self.clock)
        self.assertTrue(ports.closed)
        self.assertEqual(json.loads(path.read_text()), record)
        self.assertEqual(record['human_review_requests'], 0)
        self.assertEqual(record['real_game_input_attempts'], 0)
        return record

    def test_objective_autorechecks_review_and_verifies_different_fresh_frame(self):
        r = self.run_job()
        self.assertEqual(r['status'], 'simulation_complete')
        self.assertEqual(r['completed_inputs'], 10)
        self.assertEqual(r['recoveries'], 10)
        routes = [d['route'] for d in r['decisions']]
        self.assertEqual(routes.count('automatic_fresh_replan'), 10)
        self.assertEqual(routes.count('automatic_check_cleared; no_human_required'), 10)
        for i in range(0, len(r['decisions']), 2):
            self.assertNotEqual(r['decisions'][i]['observation_id'], r['decisions'][i+1]['observation_id'])
        self.assertNotEqual(r['proposals'][-1]['observation_id'], r['goal_assessments'][-1]['observation_id'])
        self.assertFalse(r['task_complete'])
        self.assertFalse(r['laya_semantic_reliability_validated'])
        self.assertFalse(r['real_model_inference'])
        self.assertEqual(len({x['id'] for x in self.ports.inputs}), 10)

    def test_two_minute_job_never_dispatches_after_deadline_or_finishes_at_starter(self):
        r = self.run_job(job=job_from_prompt('play the game for 2 minutes'))
        self.assertEqual((r['status'], r['reason']), ('time_limit', 'time_limit'))
        self.assertGreater(r['completed_inputs'], 10)
        self.assertEqual(r['goal_assessments'], [])
        self.assertTrue(all(x['started_monotonic'] < 120 for x in self.ports.inputs))
        self.assertEqual(r['task_outcome'], 'not_assessed')

    def test_menu_objective_stops_before_new_game(self):
        r = self.run_job(job=job_from_prompt('Reach the New Game or Continue menu and pause there.'))
        self.assertEqual(r['status'], 'simulation_complete')
        self.assertEqual(r['completed_inputs'], 1)

    def test_continue_observe_and_stop_native_routes(self):
        for label, reason in [('continue', 'action_limit'), ('observe', 'action_limit'), ('stop', 'laya_stop')]:
            with self.subTest(label=label):
                p = SimulatedPorts(self.questions, self.clock, label=label)
                r = self.run_job(p, self.job.model_copy(update={'max_actions': 1}))
                self.assertEqual(r['reason'], reason)
                self.assertEqual(len(p.inputs), 0 if label == 'stop' else 1)
                self.assertEqual(r['sampling_flags'], 1 if label == 'observe' else 0)

    def test_review_recheck_cannot_override_later_stop(self):
        labels = iter(['human_review', 'stop'])
        self.ports.evaluate = lambda text, deadline: fixture_prediction(self.questions, next(labels))
        r = self.run_job()
        self.assertEqual(r['reason'], 'laya_stop')
        self.assertEqual(self.ports.inputs, [])

    def test_changed_action_needs_another_fresh_review(self):
        self.ports.stage = 2
        original = self.ports.plan
        count = 0
        def plan(*args):
            nonlocal count
            raw = json.loads(original(*args)); count += 1
            if count == 2: raw['action']['key'] = 'z'
            return json.dumps(raw)
        self.ports.plan = plan
        r = self.run_job(job=self.job.model_copy(update={'max_actions': 1}))
        self.assertEqual(r['recoveries'], 3)
        self.assertEqual(self.ports.inputs[0]['observation_id'], 'sim-frame-4')

    def test_hard_focus_block_even_for_continue(self):
        self.ports.label = 'continue'
        self.ports.guard = lambda obs, deadline: GuardResult(status='blocked', current_observation_id=obs.observation_id, reason='lost_focus')
        r = self.run_job()
        self.assertEqual(r['reason'], 'hard_guard_blocked')
        self.assertEqual(self.ports.inputs, [])

    def test_changed_frame_forces_new_plan_and_cannot_dispatch_old_action(self):
        self.ports.label = 'continue'
        original = self.ports.guard
        def guard(obs, deadline):
            if self.ports.captures == 1: return GuardResult(status='changed', current_observation_id='new-frame')
            return original(obs, deadline)
        self.ports.guard = guard
        r = self.run_job(job=self.job.model_copy(update={'max_actions': 1}))
        self.assertEqual(r['recoveries'], 1)
        self.assertEqual(self.ports.inputs[0]['observation_id'], 'sim-frame-2')

    def test_stale_or_wrong_source_observations_never_execute(self):
        original = self.ports.capture
        for mutation, reason in [({'captured_monotonic': -100.0}, 'recovery_limit'), ({'source': 'live'}, 'invalid_observation_source')]:
            with self.subTest(mutation=mutation):
                self.ports.capture = lambda *args: original(*args).model_copy(update=mutation)
                r = self.run_job()
                self.assertEqual(r['reason'], reason)
                self.assertEqual(self.ports.inputs, [])

    def test_stale_proposal_after_model_work_is_discarded(self):
        original = self.ports.evaluate
        def slow(*args):
            result = original(*args); self.clock.advance(41); return result
        self.ports.evaluate = slow
        r = self.run_job()
        self.assertEqual(r['reason'], 'recovery_limit')
        self.assertEqual(self.ports.inputs, [])

    def test_geometry_change_or_reused_observation_is_terminal(self):
        original = self.ports.capture
        def changed(*args):
            obs = original(*args)
            if self.ports.captures > 1:
                return obs.model_copy(update={'calibration': obs.calibration.model_copy(update={'created_at': 'changed'})})
            return obs
        self.ports.capture = changed
        r = self.run_job()
        self.assertEqual(r['reason'], 'geometry_changed')
        self.assertEqual(self.ports.inputs, [])

    def test_persistent_stop_before_run_and_during_model(self):
        stop = self.root / 'runs/STOP'; stop.parent.mkdir(); stop.touch()
        r = self.run_job()
        self.assertEqual(r['model_calls'], 0)
        stop.unlink()
        original = self.ports.evaluate
        def stopping(*args):
            result = original(*args); stop.touch(); return result
        self.ports.evaluate = stopping
        r = self.run_job()
        self.assertEqual(r['reason'], 'persistent_stop')
        self.assertEqual(self.ports.inputs, [])

    def test_late_model_response_has_no_input(self):
        original = self.ports.plan
        def late(*args):
            raw = original(*args); self.clock.advance(20); return raw
        self.ports.plan = late
        r = self.run_job(job=self.job.model_copy(update={'seconds': 10.0}))
        self.assertEqual(r['status'], 'time_limit')
        self.assertEqual(r['proposals'], [])
        self.assertEqual(self.ports.inputs, [])

    def test_tied_malformed_or_truncated_laya_recovers_then_stops(self):
        for kind in ('tie', 'malformed', 'truncated', 'collapsed'):
            with self.subTest(kind=kind):
                def evaluate(text, deadline):
                    prediction = fixture_prediction(self.questions)
                    if kind == 'tie':
                        answer = prediction['answers']['action']
                        answer.update(probabilities={k: .25 for k in answer['probabilities']}, answer_confidence=.25)
                    if kind == 'malformed': prediction['answers']['action']['choice'] = 'allow'
                    if kind == 'truncated': prediction['usage'].update(truncated=True, state_tokens_dropped=1, truncated_questions=['risk'])
                    if kind == 'collapsed': prediction['usage']['options'] = {'risk': {'total': 4, 'distinct': 2}}
                    return prediction
                self.ports.evaluate = evaluate
                r = self.run_job()
                self.assertEqual(r['reason'], 'recovery_limit')
                self.assertEqual(self.ports.inputs, [])

    def test_api_failures_bounded_without_input(self):
        def failing(*args): raise RuntimeError('provider secret')
        self.ports.plan = failing
        r = self.run_job()
        self.assertEqual(r['model_calls'], 4)
        self.assertEqual(r['reason'], 'recovery_limit')
        self.assertNotIn('provider secret', json.dumps(r))

    def test_no_progress_stops_even_with_animation_pixels_changing(self):
        self.ports.label = 'continue'
        original = self.ports.execute
        def no_progress(*args):
            result = original(*args); self.ports.stage = 0; return result
        self.ports.execute = no_progress
        original_capture = self.ports.capture
        def animated(*args):
            obs = original_capture(*args)
            return obs.model_copy(update={'image_sha256': f'{self.ports.captures:064x}'})
        self.ports.capture = animated
        r = self.run_job()
        self.assertEqual(r['reason'], 'no_progress_limit')
        self.assertEqual(len(self.ports.inputs), 3)

    def test_model_call_and_action_caps(self):
        self.ports.label = 'continue'
        r = self.run_job(job=self.job.model_copy(update={'max_model_calls': 1}))
        self.assertEqual(r['reason'], 'model_call_limit')
        self.assertEqual(self.ports.inputs, [])

    def test_partial_input_error_has_one_attempt_and_is_never_retried(self):
        self.ports.label = 'continue'
        original = self.ports.execute
        def partial(*args):
            original(*args); raise RuntimeError('delivery unknown')
        self.ports.execute = partial
        r = self.run_job()
        self.assertEqual(r['reason'], 'execution_uncertain_no_retry')
        self.assertEqual(len(self.ports.inputs), 1)
        self.assertEqual(r['executions'][0]['status'], 'delivery_unknown_no_retry')
        self.assertTrue(r['executions'][0]['may_have_executed'])

    def test_wrong_executor_receipt_is_not_replayed_or_recorded_complete(self):
        self.ports.label = 'continue'
        self.ports.execute = lambda *args: {'executed': True, 'action': {'tool': 'PRESS', 'key': 'z'}, 'simulated': True}
        r = self.run_job()
        self.assertEqual(r['reason'], 'execution_uncertain_no_retry')
        self.assertEqual(r['completed_inputs'], 0)

    def test_post_input_capture_failure_does_not_replay_input(self):
        self.ports.label = 'continue'
        original = self.ports.capture
        def capture(*args):
            if self.ports.inputs: raise RuntimeError('capture unavailable')
            return original(*args)
        self.ports.capture = capture
        r = self.run_job()
        self.assertEqual(r['reason'], 'recovery_limit')
        self.assertEqual(len(self.ports.inputs), 1)
        self.assertEqual(r['completed_inputs'], 1)
        self.assertIsNone(r['executions'][0]['post_observation_id'])

    def test_write_failure_after_input_keeps_durable_intent_and_stops(self):
        from harness.journal import atomic_json
        self.ports.label = 'continue'
        def save(path, record, **kw):
            if record['completed_inputs']: raise OSError('disk full')
            atomic_json(path, record, **kw)
        with patch('harness.agent.atomic_json', side_effect=save), self.assertRaises(AgentRecordError) as error:
            self.run_job()
        self.assertTrue(self.ports.closed)
        self.assertEqual(len(self.ports.inputs), 1)
        persisted = json.loads(error.exception.path.read_text())
        self.assertEqual(persisted['executions'][-1]['status'], 'dispatch_intent')
        self.assertTrue(persisted['executions'][-1]['may_have_executed'])

    def test_reused_frame_identity_is_terminal(self):
        original = self.ports.capture
        self.ports.capture = lambda *args: original(*args).model_copy(update={'observation_id': 'always-same'})
        r = self.run_job()
        self.assertEqual(r['reason'], 'reused_observation_or_unverified_geometry')
        self.assertEqual(self.ports.inputs, [])

    def test_invalid_checkpoint_questions_produce_durable_failure(self):
        (self.checkpoint / 'questions.json').write_text('{}')
        r = self.run_job()
        self.assertEqual(r['status'], 'failed')
        self.assertEqual(r['model_calls'], 0)
        self.assertEqual(self.ports.inputs, [])

    def test_persistence_failure_initial_and_before_dispatch_prevents_input(self):
        from harness.journal import atomic_json
        for fail_at in ('initial', 'intent'):
            with self.subTest(fail_at=fail_at):
                self.ports.label = 'continue'
                def save(path, record, **kw):
                    if fail_at == 'initial' or record['state'] == 'execute': raise OSError('disk full')
                    atomic_json(path, record, **kw)
                with patch('harness.agent.atomic_json', side_effect=save), self.assertRaises(AgentRecordError): self.run_job()
                self.assertTrue(self.ports.closed)
                self.assertEqual(self.ports.inputs, [])

    def test_cleanup_failure_creates_stop_file(self):
        self.ports.label = 'stop'
        def close():
            self.ports.closed = True; raise RuntimeError('release failed')
        self.ports.close = close
        r = self.run_job()
        self.assertEqual(r['reason'], 'cleanup_failed_persistent_stop')
        self.assertTrue((self.root / 'runs/STOP').exists())

    def test_goal_identity_and_uncertainty_do_not_finish_objective(self):
        self.ports.stage = 10
        for verdict, identity in [('confirmed', 'wrong-frame'), ('uncertain', None)]:
            with self.subTest(verdict=verdict):
                self.ports.assess_goal = lambda obs, *args: GoalAssessment(observation_id=identity or obs.observation_id, verdict=verdict, evidence='fixture', source='simulation')
                r = self.run_job()
                self.assertEqual(r['reason'], 'recovery_limit')
                self.assertFalse(r['task_complete'])
                self.assertEqual(self.ports.inputs, [])

    def test_outcome_success_cannot_finish_goal_or_override_stop(self):
        def success(*args):
            result = fixture_prediction(self.questions, 'stop')
            a = result['answers']['outcome']; a.update(choice='success', probabilities={k: .7 if k == 'success' else .1 for k in a['probabilities']})
            return result
        self.ports.evaluate = success
        r = self.run_job()
        self.assertEqual(r['reason'], 'laya_stop')
        self.assertEqual(r['goal_assessments'], [])
        self.assertFalse(r['task_complete'])

    def test_autonomous_context_keeps_facts_claims_and_no_manual_approval(self):
        self.ports.label = 'continue'
        self.run_job(job=self.job.model_copy(update={'max_actions': 1}))
        packed = json.dumps(self.ports.contexts[0])
        self.assertIn('first Pokemon', packed)
        self.assertIn('automatic', packed)
        self.assertIn('model_claims', packed)
        self.assertNotIn('manual approval', packed)

    def test_long_goal_shortening_prevents_laya_and_input(self):
        r = self.run_job(job=self.job.model_copy(update={'prompt': 'x' * 700}))
        self.assertEqual(r['reason'], 'recovery_limit')
        self.assertEqual(self.ports.contexts, [])
        self.assertEqual(self.ports.inputs, [])

    def test_prompt_duration_parsing_and_conflicts(self):
        for prompt in ('play the game for 2 minutes', 'play for two minutes'):
            job = job_from_prompt(prompt)
            self.assertEqual((job.mode, job.seconds), ('timed', 120.0))
        self.assertEqual(job_from_prompt('get first Pokemon within 2 minutes').mode, 'objective')
        self.assertEqual(job_from_prompt('play the game', seconds=120).mode, 'timed')
        self.assertEqual(job_from_prompt('Reach the New Game button screen and pause there').goal_kind, 'menu')
        for prompt, seconds in [('play for 2 minutes', 100), ('play for 0 seconds', None), ('play for 31 minutes', None), ('play for 2 minutes and wait for 3 seconds', None)]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError): job_from_prompt(prompt, seconds=seconds)

    def test_invalid_actions_unknown_screen_and_save_overwrite(self):
        self.ports.stage = 1
        obs = self.ports.capture(self.job, 600)
        base = json.loads(self.ports.plan(obs, self.job, [], 600))
        variants = []
        for action in ({'tool': 'TYPE', 'text': 'HUY'}, {'tool': 'SHELL', 'text': 'anything'}, {'tool': 'CLICK', 'x': 500, 'y': 100}):
            raw = json.loads(json.dumps(base)); raw.update(action=action, target_label='New Game' if action['tool'] == 'CLICK' else None); variants.append(raw)
        raw = json.loads(json.dumps(base)); raw['visible_observation']['screen_kind'] = 'unknown'; variants.append(raw)
        raw = json.loads(json.dumps(base)); raw.update(goal_status='candidate_complete', goal_evidence='got it'); variants.append(raw)
        raw = json.loads(json.dumps(base)); raw['visible_observation']['lower_touchscreen']['visible_text'].append('Continue'); variants.append(raw)
        raw = json.loads(json.dumps(base)); raw.update(action={'tool': 'CLICK', 'x': 500, 'y': 750}, target_label='Snivy'); variants.append(raw)
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ProposalError): parse_agent_proposal(json.dumps(raw), obs.observation_id, obs.calibration, self.job)
        allowed = json.loads(json.dumps(base)); allowed['visible_observation']['lower_touchscreen']['visible_text'].append('Continue'); allowed['selected_menu_label'] = 'Continue'
        self.assertEqual(parse_agent_proposal(json.dumps(allowed), obs.observation_id, obs.calibration, self.job).action.key, 'x')

    def test_cli_simulation_never_imports_native_or_model_runtime(self):
        settings = SimpleNamespace(laya_model_path=self.checkpoint, gemini_api_key=SecretStr('hidden'))
        original = builtins.__import__
        def guarded(name, *args, **kw):
            if name.startswith(('pyautogui', 'torch', 'transformers', 'google.genai', 'models.agent_planner')): raise AssertionError(name)
            return original(name, *args, **kw)
        with patch('harness.cli.PROJECT_ROOT', self.root), patch('harness.cli.load_settings', return_value=settings), patch('builtins.__import__', side_effect=guarded), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--agent-test', '--prompt', 'play the game for 2 minutes']), 0)
            self.assertEqual(main(['--agent-plan', '--prompt', 'get your first Pokemon']), 0)

    def test_cli_rejects_execution_or_mixed_modes(self):
        for args in (['--agent-test', '--execute'], ['--agent-test', '--dry-run'], ['--dry-run', '--seconds', '10'], ['--agent-test', '--laya-device', 'mps']):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit): main(args)

    def test_cli_corrupt_questions_returns_failure_without_traceback(self):
        (self.checkpoint / 'questions.json').write_text('{}')
        settings = SimpleNamespace(laya_model_path=self.checkpoint, gemini_api_key=SecretStr('hidden'))
        with patch('harness.cli.PROJECT_ROOT', self.root), patch('harness.cli.load_settings', return_value=settings), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--agent-test', '--prompt', 'get your first Pokemon']), 3)


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.clock, self.guard, self.driver = FakeClock(), FakeGuard(), SpyDriver()
        self.profile = SimulatedPorts(questions_fixture(), SimClock()).calibration

    def executor(self, deadline):
        guard = DeadlineGuard(self.guard, deadline, clock=self.clock.time)
        return ActionExecutor(guard, self.driver, Path(self.temp.name)/'STOP', clock=self.clock.time, sleep=self.clock.sleep)

    def test_expired_deadline_prevents_key_down(self):
        with self.assertRaises(Stopped): self.executor(0).execute(parse_action('{"tool":"PRESS","key":"x","seconds":0.1}'), self.profile)
        self.assertEqual(self.driver.events, [])

    def test_key_and_mouse_release_on_deadline(self):
        for raw, release in [('{"tool":"PRESS","key":"x","seconds":0.1}', ('up', 'x')), ('{"tool":"CLICK","x":500,"y":750}', ('mouse_up',))]:
            with self.subTest(raw=raw):
                self.clock.now = 0; self.driver.events.clear()
                executor = self.executor(.01)
                with self.assertRaises(Stopped): executor.execute(parse_action(raw), self.profile)
                self.assertIn(release, self.driver.events)
                self.assertEqual(executor.held_keys, set())
                self.assertFalse(executor.mouse_held)

    def test_permission_check_that_uses_budget_prevents_dispatch(self):
        self.guard.prepare = lambda: self.clock.sleep(1)
        with self.assertRaises(Stopped): self.executor(.5).execute(parse_action('{"tool":"PRESS","key":"x","seconds":0.1}'), self.profile)
        self.assertEqual(self.driver.events, [])

if __name__ == '__main__': unittest.main()
