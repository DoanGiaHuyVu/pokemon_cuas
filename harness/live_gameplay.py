"""Bounded existing-save exploration and read-only party inspection."""
from functools import partial
from pathlib import Path

from computer.capture import CaptureError
from computer.executor import ExecutionError
from computer.geometry import GeometryError
from harness.actions import Click, Press
from harness.agent import GoalAssessment, GuardResult, HardStop, run_agent
from harness.agent_job import live_gameplay_job
from harness.agent_proposals import parse_agent_proposal
from harness.live_agent import LiveMenuPorts
from models.gameplay import check_gameplay_observation, check_gameplay_effect, plan_gameplay_observation


class LiveGameplayPorts(LiveMenuPorts):
    step = 'white_gameplay'

    def __init__(self, *args, **kwargs):
        self.effect_checker = kwargs.pop('effect_checker', check_gameplay_effect)
        kwargs.setdefault('planner', plan_gameplay_observation)
        kwargs.setdefault('checker', check_gameplay_observation)
        super().__init__(*args, **kwargs)
        self.audit.update(step=self.step, scope='existing_save_walking_and_read_only_inspection')
        self.party_seen = []
        self.settle_pending = False
        self.input_preflight_done = False

    def capture(self, job, deadline):
        if self.settle_pending:
            self.settle_pending = False
            self.check(deadline)
            self.sleep(min(.4, max(0, deadline-self.clock())))
            self.check(deadline)
        try:
            observed = super().capture(job, deadline)
            if not self.input_preflight_done:
                try:
                    self.native_guard(observed, deadline).prepare()
                except ExecutionError as exc:
                    self.save('native_guard_blocked', phase='startup', reason=str(exc))
                    raise HardStop('native_input_guard: ' + str(exc)) from exc
                self.input_preflight_done = True
            return observed
        except (CaptureError, GeometryError) as exc:
            raise HardStop('game_window_not_visible_or_changed') from exc

    def review_requires_replan(self, proposal):
        return False  # All labels still require an independent fresh scope/action check.

    def plan(self, observation, job, history, deadline):
        self.ticket = None
        context = [e.model_dump() for e in history]
        if self.party_seen: context.append({'independently_observed_party':self.party_seen})
        raw = self.api('plan', observation, deadline, job, context)
        proposal = parse_agent_proposal(raw, observation.observation_id, observation.calibration, job)
        kind = proposal.visible_observation.screen_kind
        if kind not in {'overworld','game_menu','party','party_options','summary','bag','transition','unknown'}:
            raise HardStop('outside_existing_save_gameplay_scope')
        if proposal.goal_status == 'candidate_complete' and (job.goal_kind != 'party' or kind != 'party'):
            raise HardStop('unsupported_gameplay_goal_claim')
        if proposal.action is not None:
            self.validate_dispatch_action(proposal.action)
            if isinstance(proposal.action, Press) and proposal.action.key == 'x' and not proposal.selected_menu_label and kind != 'game_menu':
                raise ValueError('Confirmation requires a visible selected option.')
        self.proposals[observation.observation_id] = proposal
        return raw

    def validate_dispatch_action(self, action):
        if not isinstance(action, (Press, Click)): raise HardStop('gameplay_navigation_only')
        if isinstance(action, Press) and (action.key not in {'up','down','left','right','s','z','x'} or action.seconds > .3):
            raise HardStop('gameplay_navigation_only')

    def remember(self, checked):
        if checked.party_confirmed:
            self.party_seen = [m.model_dump() for m in checked.party_members]
            self.save('party_visually_observed', observation_id=checked.observation_id, members=self.party_seen)

    def guard(self, observation, deadline):
        self.ticket = None
        self.check(deadline)
        try: self.native_guard(observation, deadline).prepare()
        except ExecutionError as exc:
            self.save('native_guard_blocked', phase='pre_input', reason=str(exc))
            return GuardResult(status='blocked',current_observation_id=observation.observation_id,reason='native_input_guard: ' + str(exc))
        proposal = self.proposals.get(observation.observation_id)
        if proposal is None or proposal.action is None: raise HardStop('no_gameplay_proposal')
        fresh = self.capture(self.observations[observation.observation_id][2], deadline)
        checked = self.api('pre_input_gameplay_check', fresh, deadline)
        self.remember(checked)
        if not checked.supported: raise HardStop('outside_existing_save_gameplay_scope')
        if checked.screen_kind != proposal.visible_observation.screen_kind or not checked.permits(proposal.action, proposal.target_label):
            return GuardResult(status='changed',current_observation_id=fresh.observation_id,reason='gameplay_scene_or_permission_changed')
        if isinstance(proposal.action, Press) and proposal.action.key == 'x':
            from models.gameplay import normalized
            if normalized(checked.selected_label) != normalized(proposal.selected_menu_label):
                return GuardResult(status='changed',current_observation_id=fresh.observation_id,reason='selected_option_changed')
        self.native_guard(fresh, deadline).check()
        self.ticket = {'proposal_observation_id':observation.observation_id,'fresh_observation_id':fresh.observation_id,
                       'action':proposal.action.model_dump(),'checked_at':self.clock()}
        self.save('gameplay_ticket', **self.ticket)
        return GuardResult(status='clear',current_observation_id=observation.observation_id,reason='fresh_gameplay_scope_and_selection_check')

    def execute(self, action, observation, action_id, deadline):
        result = super().execute(action, observation, action_id, deadline)
        self.settle_pending = True
        return result

    def assess_effect(self, before, after, action, job, deadline):
        # One sequential request through the same audited transport; no extra retry loop.
        checker = self.checker
        try:
            self.checker = partial(self.effect_checker, previous=self.raw(before), action=action)
            effect = self.api('post_input_gameplay_effect', after, deadline)
        finally:
            self.checker = checker
        return effect.model_dump()

    def assess_goal(self, observation, job, deadline):
        checked = self.api('goal_gameplay_check', observation, deadline)
        self.remember(checked)
        return GoalAssessment(observation_id=observation.observation_id,
            verdict='confirmed' if job.goal_kind == 'party' and checked.party_confirmed else 'not_confirmed',
            evidence=checked.evidence, source='vision_model')


def run_live_gameplay(settings, job, root, *, device='auto', ports=None):
    job = live_gameplay_job(job)
    ports = ports or LiveGameplayPorts(settings, root, device=device)
    return run_agent(job, ports, Path(root)/'runs', settings.laya_model_path,
                     secret=settings.gemini_api_key.get_secret_value(), clock=ports.clock, simulation=False)
