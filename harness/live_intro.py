"""Step 11: one x per fresh introduction observation, stop before name input."""
from pathlib import Path

from computer.executor import ExecutionError
from computer.title_anchor import TitleAnchor
from harness.actions import Press
from harness.agent import GoalAssessment, GuardResult, HardStop, run_agent
from harness.agent_job import live_intro_job
from harness.agent_proposals import parse_agent_proposal
from harness.live_agent import LiveMenuPorts
from models.intro_check import IntroCheck, check_intro_observation


class LiveIntroPorts(LiveMenuPorts):
    step=11

    def __init__(self,*args,**kwargs):
        detector=kwargs.pop('anchor_detector',None)
        kwargs.setdefault('checker',check_intro_observation)
        super().__init__(*args,**kwargs)
        self.name_seen=False
        self.new_game_dispatched=False
        self.boy_choice_dispatched=False
        self.checks_by_observation={}
        self.anchor_detector=detector or TitleAnchor(self.root/'runs/title-anchor.json').matches
        self.pair_remaining=0
        self.local_pair_proposals=set()
        self.audit.update(step=11,scope='introduction_to_boy_name_entry_without_naming',
                          planner_mode='fresh_visual_facts_plus_operator_one_x_policy')

    def local_title(self,observation):
        if not self.anchor_detector(self.raw(observation)): return None
        checked=IntroCheck(observation_id=observation.observation_id,screen_kind='title',
            upper_screen={'description':'Title-logo pixels match the reviewed local reference.',
                          'visible_text':['Pokémon','BLACK','VERSION']},
            lower_touchscreen={'description':'Captured lower display; no text is inferred by the local matcher.','visible_text':[]},
            selected_label=None,new_game_visible=False,continue_visible=False,name_keyboard_visible=False,
            player_gender='unknown',evidence='Reviewed title-logo pixels matched with the fixed tight tolerance.')
        self.save('local_title_match',observation_id=observation.observation_id,
                  source='reviewed_pixel_anchor',image_sha256=observation.image_sha256)
        return checked

    def remember(self,checked):
        self.checks_by_observation[checked.observation_id]=checked
        if checked.at_name_entry:
            self.name_seen=True
            self.save('name_entry_latched_no_input',observation_id=checked.observation_id)
        if checked.existing_save: raise HardStop('existing_save_outside_intro_scope')

    def plan(self,observation,job,history,deadline):
        self.ticket=None
        checked=self.local_title(observation) if self.pair_remaining else None
        if checked is not None:
            self.pair_remaining-=1
            self.local_pair_proposals.add(observation.observation_id)
        else:
            self.pair_remaining=0
            checked=self.api('intro_plan_check',observation,deadline)
        self.remember(checked)
        allowed=checked.x_allowed and not self.name_seen
        if checked.entry_menu and self.new_game_dispatched: raise HardStop('new_game_reentry_forbidden')
        raw={'observation_id':observation.observation_id,
             'visible_observation':{'screen_kind':checked.screen_kind,'upper_screen':checked.upper_screen.model_dump(),
                                    'lower_touchscreen':checked.lower_touchscreen.model_dump()},
             'status':'propose' if allowed else 'pause',
             'action':{'tool':'PRESS','key':'x','seconds':.1} if allowed else None,'target_label':None,
             'expected_result':'The introduction may advance; a fresh observation must verify the effect.',
             'rationale':'Operator rule: one short x from a recognized introduction screen, with fresh checks.',
             'task_complete':False,'goal_status':'candidate_complete' if checked.at_name_entry else 'in_progress',
             'goal_evidence':checked.evidence if checked.at_name_entry else '',
             'selected_menu_label':checked.selected_label}
        import json
        proposal=parse_agent_proposal(json.dumps(raw),observation.observation_id,observation.calibration,job)
        self.proposals[observation.observation_id]=proposal
        self.save('operator_policy_proposal',proposal=raw)
        return json.dumps(raw)

    def review_requires_replan(self,proposal):
        # An independent fresh phase/selection check implements automatic review.
        return False

    def guard(self,observation,deadline):
        self.ticket=None
        self.check(deadline)
        try: self.native_guard(observation,deadline).prepare()
        except ExecutionError:
            return GuardResult(status='blocked',current_observation_id=observation.observation_id,reason='native_input_guard')
        proposal=self.proposals.get(observation.observation_id)
        original=self.checks_by_observation.get(observation.observation_id)
        if proposal is None or proposal.action is None or original is None: raise HardStop('no_intro_proposal')
        fresh=self.capture(self.observations[observation.observation_id][2],deadline)
        checked=self.local_title(fresh) if original.action_phase=='entry' else None
        if checked is None: checked=self.api('pre_input_intro_check',fresh,deadline)
        self.remember(checked)
        if self.name_seen or not checked.x_allowed or checked.action_phase!=original.action_phase:
            return GuardResult(status='changed',current_observation_id=fresh.observation_id,reason='intro_phase_or_selection_changed')
        if checked.entry_menu and self.new_game_dispatched: raise HardStop('new_game_reentry_forbidden')
        self.native_guard(fresh,deadline).check()
        self.ticket={'proposal_observation_id':observation.observation_id,'fresh_observation_id':fresh.observation_id,
                     'action':proposal.action.model_dump(),'checked_at':self.clock(),
                     'intro_phase':checked.screen_kind,'selected_label':checked.selected_label}
        self.save('intro_press_ticket',**self.ticket)
        return GuardResult(status='clear',current_observation_id=observation.observation_id,reason='fresh_intro_x_check')

    def execute(self,action,observation,action_id,deadline):
        if self.name_seen: raise HardStop('name_entry_input_forbidden')
        if not isinstance(action,Press) or action.key!='x' or action.seconds>.2: raise HardStop('intro_x_only')
        ticket=dict(self.ticket or {})
        result=super().execute(action,observation,action_id,deadline)
        if ticket.get('intro_phase') in {'opening','title'} and observation.observation_id not in self.local_pair_proposals:
            self.pair_remaining=1
        if ticket.get('intro_phase')=='main_menu': self.new_game_dispatched=True
        if ticket.get('intro_phase')=='character_select': self.boy_choice_dispatched=True
        self.save('intro_effect_pending',new_game_dispatched=self.new_game_dispatched,
                  boy_choice_dispatched=self.boy_choice_dispatched)
        return result

    def assess_goal(self,observation,job,deadline):
        checked=self.api('goal_intro_check',observation,deadline)
        self.remember(checked)
        return GoalAssessment(observation_id=observation.observation_id,
            verdict='confirmed' if checked.goal_confirmed else 'not_confirmed',
            evidence=checked.evidence,source='vision_model')


def run_live_intro(settings,job,root,*,device='auto',ports=None):
    job=live_intro_job(job)
    ports=ports or LiveIntroPorts(settings,root,device=device)
    return run_agent(job,ports,Path(root)/'runs',settings.laya_model_path,
                     secret=settings.gemini_api_key.get_secret_value(),clock=ports.clock,simulation=False)
