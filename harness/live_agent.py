"""Step 10: live menu arrival only, with no menu confirmation or mouse input."""
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from computer.capture import MacWindowBackend, capture_observation
from computer.executor import ActionExecutor, DeadlineGuard, ExecutionError, MacExecutionGuard, PyAutoGUIDriver
from computer.geometry import Calibration
from harness.actions import Press
from harness.agent import GoalAssessment, GuardResult, HardStop, Observation, run_agent, state_key
from harness.agent_job import live_menu_job
from harness.agent_proposals import parse_agent_proposal
from harness.journal import atomic_json
from models.agent_planner import plan_agent_observation
from models.curl_transport import CurlGemmaClient
from models.laya_local import LayaError, load_questions, validate_prediction
from models.laya_session import LayaSession
from models.menu_check import check_menu_observation
from models.perception import PerceptionError


class LiveMenuPorts:
    step = 10
    guard_uses_model = True

    def __init__(self, settings, root, *, device='auto', clock=time.monotonic, backend=None,
                 planner=None, checker=None, laya_runner=None, guard_factory=None, driver_factory=None,
                 sleep=time.sleep, client_factory=None):
        self.settings, self.root, self.device, self.clock = settings, Path(root), device, clock
        self.sleep = sleep
        self.client_factory = client_factory or CurlGemmaClient
        self.backend = backend or MacWindowBackend()
        self.planner = planner or plan_agent_observation
        self.checker = checker or check_menu_observation
        self.laya_runner = laya_runner
        self.laya_session = None
        self.guard_factory = guard_factory or MacExecutionGuard
        self.driver_factory = driver_factory or PyAutoGUIDriver
        self.profile = self.root / 'runs/calibration.json'
        self.stop_file = self.root / 'runs/STOP'
        self.secret = settings.gemini_api_key.get_secret_value()
        self.observations = {}
        self.proposals = {}
        self.executors = []
        self.dispatched = set()
        self.ticket = None
        self.menu_seen = False
        self.audit_path = self.root / 'runs' / f'{uuid4().hex}.live-menu.json'
        self.audit = {'step':10, 'scope':'reach_new_game_menu_without_selecting', 'events':[],
                      'gemma_transport':'curl_http2_single_attempt',
                      'screen_captures':0, 'hosted_attempts':0, 'laya_worker_records':[], 'input_dispatches':0,
                      'model_claims_are_unverified':True, 'closed':False}

    def save(self, kind, **fields):
        self.audit['events'].append({'kind':kind, **fields})
        atomic_json(self.audit_path, self.audit, secret=self.secret)

    def check(self, deadline):
        if self.stop_file.exists(): raise HardStop('persistent_stop')
        if self.clock() >= deadline: raise HardStop('time_limit')

    def native_guard(self, observation, deadline):
        calibration = observation.calibration
        native = self.guard_factory(calibration.window, self.backend, calibration)
        return DeadlineGuard(native, deadline, clock=self.clock)

    def capture(self, job, deadline):
        self.check(deadline)
        captured = self.clock()  # Conservative age includes the capture/write time.
        raw = capture_observation(self.backend, self.profile, self.root/'screenshots', self.root/'runs', job.prompt)
        self.check(deadline)
        obs = Observation(observation_id=raw['observation_id'], captured_monotonic=float(captured),
                          image_sha256=hashlib.sha256(Path(raw['image_path']).read_bytes()).hexdigest(),
                          calibration=Calibration.model_validate(raw['calibration']), source='live',
                          metadata={'image_path':raw['image_path'], 'metadata_path':raw['metadata_path'],
                                    'live_port_record':str(self.audit_path)})
        self.observations[obs.observation_id] = (obs, raw, job)
        self.audit['screen_captures'] += 1
        self.save('capture', observation_id=obs.observation_id, image_path=raw['image_path'], image_sha256=obs.image_sha256)
        return obs

    def raw(self, observation):
        known = self.observations.get(observation.observation_id)
        if known is None or known[0] != observation: raise HardStop('unowned_live_observation')
        if hashlib.sha256(Path(known[1]['image_path']).read_bytes()).hexdigest() != observation.image_sha256:
            raise HardStop('observation_image_changed')
        return known[1]

    def api(self, operation, observation, deadline, *args):
        self.check(deadline)
        raw = self.raw(observation)
        self.save('hosted_request_started', operation=operation, observation_id=observation.observation_id)
        try:
            if operation == 'plan':
                result = self.planner(self.settings, raw, *args, deadline, clock=self.clock,
                                      client_factory=self.client_factory)
                data, metrics = result['text'], result
            else:
                data, metrics = self.checker(self.settings, raw, deadline, clock=self.clock,
                                             client_factory=self.client_factory)
        except BaseException as exc:
            self.audit['hosted_attempts'] += getattr(exc, 'attempts', 1) or 0
            self.save('hosted_request_failed', operation=operation, error_class=type(exc).__name__,
                      cause_class=type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                      error=str(exc) if isinstance(exc, PerceptionError) else 'Error details withheld.',
                      rejected_reply=getattr(exc,'rejected_reply',None))
            raise
        self.audit['hosted_attempts'] += metrics['attempts']
        self.save('hosted_result', operation=operation, observation_id=observation.observation_id,
                  result=data if isinstance(data, str) else data.model_dump(), usage=metrics.get('usage'),
                  response_wrapper=metrics.get('response_wrapper'))
        self.check(deadline)
        return data

    def plan(self, observation, job, history, deadline):
        self.ticket = None
        raw = self.api('plan', observation, deadline, job, [x.model_dump() for x in history])
        proposal = parse_agent_proposal(raw, observation.observation_id, observation.calibration, job)
        visible = proposal.visible_observation
        labels = {x.strip().casefold() for screen in (visible.upper_screen, visible.lower_touchscreen) for x in screen.visible_text}
        if visible.screen_kind == 'main_menu' or labels & {'new game', 'continue'}:
            self.menu_seen = True
        if proposal.action is not None:
            if self.menu_seen: raise HardStop('menu_input_forbidden')
            if visible.screen_kind not in {'opening','title'} or not isinstance(proposal.action, Press) or proposal.action.key != 'x' or proposal.action.seconds > .2:
                raise HardStop('step10_entry_action_only')
        self.proposals[observation.observation_id] = proposal
        return raw

    def evaluate(self, text, deadline):
        self.check(deadline)
        name = 'live-' + uuid4().hex
        self.save('laya_request_started', state_sha256=hashlib.sha256(text.encode()).hexdigest())
        if self.laya_runner is not None:
            worker, path = self.laya_runner(self.settings, self.root/'runs', [{'name':name,'state':text}],
                                           device=self.device, timeout=max(.01, deadline-self.clock()))
        else:
            if self.laya_session is None:
                self.laya_session = LayaSession(self.settings, self.root/'runs', device=self.device,
                                                clock=self.clock, sleep=self.sleep)
            worker, path = self.laya_session.evaluate(name, text, deadline, self.check)
        self.audit['laya_worker_records'].append(str(path))
        self.save('laya_worker_finished', worker_record=str(path), status=worker.get('status'))
        self.check(deadline)
        if worker.get('status') == 'cancelled': raise KeyboardInterrupt()
        result = worker.get('result') or {}
        if worker.get('status') != 'passed' or not result.get('checkpoint_unchanged') or result.get('network_attempts_blocked') != 0:
            raise LayaError('Live local inference did not pass offline/source checks.')
        cases = result.get('cases', [])
        if len(cases) != 1 or cases[0].get('name') != name or cases[0].get('state') != text:
            raise LayaError('Live prediction does not match its text context.')
        validated = validate_prediction(cases[0]['prediction'], load_questions(self.settings.laya_model_path), result['token_budget']['max_len'])
        self.save('laya_decision', prediction=cases[0]['prediction'], structural_validation=validated, runtime=result.get('runtime'))
        return cases[0]['prediction']

    def review_identity(self, proposal):
        # Operator-approved entry gestures are equivalent across title animation
        # and blinking text. A separate fresh screen check still gates dispatch.
        if (not self.menu_seen and proposal.visible_observation.screen_kind in {'opening','title'}
                and isinstance(proposal.action, Press) and proposal.action.key == 'x' and proposal.action.seconds <= .2):
            return 'menu_entry_opening_or_title_x_only'
        return state_key(proposal)

    def review_requires_replan(self, proposal):
        # For the explicitly authorized entry gesture, review means an independent
        # fresh menu/entry check before dispatch. It cannot approve menu input.
        return self.review_identity(proposal) != 'menu_entry_opening_or_title_x_only'

    def guard(self, observation, deadline):
        self.ticket = None
        self.check(deadline)
        try:
            self.native_guard(observation, deadline).prepare()
        except ExecutionError:
            return GuardResult(status='blocked', current_observation_id=observation.observation_id, reason='native_input_guard')
        proposal = self.proposals.get(observation.observation_id)
        if proposal is None or proposal.action is None: raise HardStop('no_live_proposal')
        job = self.observations[observation.observation_id][2]
        fresh = self.capture(job, deadline)
        checked = self.api('pre_input_menu_check', fresh, deadline)
        self.check(deadline)
        if checked.menu_present:
            self.menu_seen = True
            self.save('menu_latched_no_input', observation_id=fresh.observation_id)
            return GuardResult(status='changed', current_observation_id=fresh.observation_id, reason='menu_now_visible')
        if not checked.entry_press_allowed:
            return GuardResult(status='changed', current_observation_id=fresh.observation_id, reason='entry_not_confirmed')
        self.native_guard(fresh, deadline).check()
        self.ticket = {'proposal_observation_id':observation.observation_id, 'fresh_observation_id':fresh.observation_id,
                       'action':proposal.action.model_dump(), 'checked_at':self.clock()}
        self.save('entry_press_ticket', **self.ticket)
        return GuardResult(status='clear', current_observation_id=observation.observation_id, reason='fresh_entry_check_same_x_scope')

    def validate_dispatch_action(self, action):
        if not isinstance(action, Press) or action.key != 'x' or action.seconds > .2:
            raise HardStop('entry_press_only')

    def execute(self, action, observation, action_id, deadline):
        self.check(deadline)
        ticket, self.ticket = self.ticket, None  # Single use even when dispatch fails.
        if self.menu_seen or ticket is None or ticket['proposal_observation_id'] != observation.observation_id or ticket['action'] != action.model_dump():
            raise HardStop('missing_or_invalid_entry_ticket')
        if action_id in self.dispatched or self.clock()-ticket['checked_at'] > 2:
            raise HardStop('duplicate_or_expired_entry_ticket')
        self.validate_dispatch_action(action)
        fresh = self.observations[ticket['fresh_observation_id']][0]
        self.raw(fresh)  # Image integrity, geometry, focus, stop, and clock all remain mandatory.
        executor = ActionExecutor(self.native_guard(fresh, deadline), self.driver_factory(), self.stop_file,
                                  clock=self.clock, sleep=self.sleep)
        self.executors.append(executor)
        self.dispatched.add(action_id)
        self.save('input_dispatch_intent', action_id=action_id, observation_id=observation.observation_id,
                  gate_observation_id=fresh.observation_id, action=action.model_dump())
        self.audit['input_dispatches'] += 1
        result = executor.execute(action, fresh.calibration)
        self.save('input_completed', action_id=action_id, result=result)
        return result

    def assess_goal(self, observation, job, deadline):
        checked = self.api('goal_menu_check', observation, deadline)
        if checked.menu_present: self.menu_seen = True
        return GoalAssessment(observation_id=observation.observation_id,
                              verdict='confirmed' if checked.goal_confirmed else 'not_confirmed',
                              evidence=checked.evidence, source='vision_model')

    def close(self):
        self.ticket = None
        try:
            for executor in self.executors: executor.cleanup()
        finally:
            if self.laya_session is not None: self.laya_session.close()
            self.audit['closed'] = True
            self.save('ports_closed')


def run_live_menu(settings, job, root, *, device='auto', ports=None):
    # Extra inputs are never needed once the menu is seen. No mouse/TYPE/menu input is enabled.
    job = live_menu_job(job)
    ports = ports or LiveMenuPorts(settings, root, device=device)
    return run_agent(job, ports, Path(root)/'runs', settings.laya_model_path,
                     secret=settings.gemini_api_key.get_secret_value(), clock=ports.clock, simulation=False)
