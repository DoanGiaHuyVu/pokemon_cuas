"""Existing-save gameplay wiring with fake inputs, models, and captures."""
import json
import unittest
from unittest.mock import patch

from harness.actions import Click, Press, TypeText
from harness.agent_job import job_from_prompt, live_gameplay_job
from harness.live_gameplay import LiveGameplayPorts, run_live_gameplay
from models.gameplay import GameplayCheck, GameplayEffect, check_gameplay_observation, check_gameplay_effect, plan_gameplay_observation
import test_live_agent as fixtures
from test_perception import FakeClient, response


def facts(identity, kind, selected=None, members=None, targets=None):
    return GameplayCheck(observation_id=identity, screen_kind=kind,
        upper_screen={'description':'Actual world fixture.', 'visible_text':[]},
        lower_touchscreen={'description':'Actual menu fixture.', 'visible_text':['POKÉMON','BAG'] if kind=='game_menu' else ['Oshawott'] if kind=='party' else []},
        selected_label=selected, selection_active=bool(selected) if kind=='game_menu' else None,
        party_members=members or [], touch_targets=targets or [], evidence='Visible fixture evidence.')


MEMBERS=[{'name':'Oshawott','level':7,'hp_text':'25/25'}]


class GameplayPortsTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.LivePortsTests('test_already_at_menu_has_zero_input_and_zero_laya_calls')
        self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.job=job_from_prompt('Inspect the Pokemon party and pause.',seconds=120)
        self.stage=0;self.kind_override=None;self.effect_progress='confirmed'
        self.plans=0;self.checks=0;self.effects=0
        self.ports=LiveGameplayPorts(self.f.settings,self.f.root,clock=self.f.clock.time,backend=self.f.backend,
            planner=self.planner,checker=self.checker,effect_checker=self.effect,laya_runner=self.f.laya,
            guard_factory=lambda *a:self.f.guard,driver_factory=lambda:self.f.driver,sleep=self.f.clock.sleep)
        self.f.driver.key_down=self.key_down

    def key_down(self,key):self.f.driver.events.append(('down',key));self.stage+=1

    def kind(self):return self.kind_override or ['overworld','game_menu','party'][min(self.stage,2)]

    def checker(self,settings,obs,deadline,**kw):
        self.checks+=1;self.f.clock.now+=.1
        kind=self.kind()
        return facts(obs['observation_id'],kind,selected='Pokemon' if kind=='game_menu' else 'Oshawott' if kind=='party' else None,
                     members=MEMBERS if kind=='party' else []),{'attempts':1,'usage':{}}

    def planner(self,settings,obs,job,history,deadline,**kw):
        self.plans+=1;self.f.clock.now+=.1
        c,_=self.checker(settings,obs,deadline)
        done=c.screen_kind=='party' and job.goal_kind=='party'
        raw={'observation_id':obs['observation_id'],'visible_observation':{'screen_kind':c.screen_kind,
             'upper_screen':c.upper_screen.model_dump(),'lower_touchscreen':c.lower_touchscreen.model_dump()},
             'status':'pause' if done else 'propose','action':None if done else {'tool':'PRESS','key':'s' if c.screen_kind=='overworld' else 'x','seconds':.1},
             'target_label':None,'expected_result':'Menu may change.','rationale':'Inspect owned party.', 'task_complete':False,
             'goal_status':'candidate_complete' if done else 'in_progress','goal_evidence':'Oshawott Lv7 25/25 visible' if done else '',
             'selected_menu_label':c.selected_label}
        return {'text':json.dumps(raw),'attempts':1,'usage':{}}

    def effect(self,settings,obs,deadline,*,previous,action,**kw):
        self.effects+=1;self.f.clock.now+=.1
        return GameplayEffect(before_id=previous['observation_id'],after_id=obs['observation_id'],progress=self.effect_progress,
            evidence='The intended panel visibly changed.'),{'attempts':1,'usage':{}}

    def run_job(self):return run_live_gameplay(self.f.settings,self.job,self.f.root,ports=self.ports)

    def test_menu_and_party_use_two_fresh_actions_and_independent_goal(self):
        self.f.label='human_review';r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported');self.assertEqual(r['completed_inputs'],2)
        self.assertEqual(self.f.driver.events,[('down','s'),('up','s'),('down','x'),('up','x')])
        self.assertEqual(self.f.laya_calls,2);self.assertEqual(self.effects,2)
        self.assertEqual(r['human_review_requests'],0);self.assertEqual(self.ports.party_seen,MEMBERS)
        self.assertTrue(self.ports.audit['closed']);self.assertGreaterEqual(self.f.clock.now,.8)
        self.assertTrue(all(e['semantic_effect']['progress']=='confirmed' for e in r['executions']))

    def test_already_at_party_has_zero_input_and_zero_laya(self):
        self.stage=2;r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported');self.assertEqual(self.f.driver.events,[])
        self.assertEqual(self.f.laya_calls,0)

    def test_neutral_menu_arms_then_fresh_highlight_confirms_with_two_x(self):
        self.kind=lambda:['overworld','game_menu','game_menu','party'][min(self.stage,3)]
        original=self.checker
        def inactive(*a,**kw):
            c,m=original(*a,**kw)
            if self.stage==1:
                c=c.model_copy(update={'selected_label':None,'selection_active':False,
                    'lower_touchscreen':c.lower_touchscreen.model_copy(update={'visible_text':['POKÉMON','POKÉDEX','BAG','ADAM','SAVE','OPTIONS']})})
            return c,m
        self.checker=inactive;self.ports.checker=inactive
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported');self.assertEqual(r['completed_inputs'],3)
        self.assertEqual(self.f.driver.events,[('down','s'),('up','s'),('down','x'),('up','x'),('down','x'),('up','x')])
        self.assertEqual(self.f.laya_calls,3)

    def test_laya_stop_has_zero_dispatch(self):
        self.f.label='stop';r,_=self.run_job()
        self.assertEqual(r['reason'],'laya_stop');self.assertEqual(self.f.driver.events,[])

    def test_input_access_failure_is_explained_before_any_model_call(self):
        from computer.executor import ExecutionError
        self.f.guard.failure=ExecutionError('Accessibility permission is required for the app running Python.')
        r,_=self.run_job()
        self.assertIn('Accessibility permission',r['reason']);self.assertEqual(r['model_calls'],0)
        self.assertEqual(self.f.laya_calls,0);self.assertEqual(self.f.driver.events,[])

    def test_later_focus_loss_preserves_the_specific_guard_reason(self):
        from computer.executor import ExecutionError
        original=self.planner
        def lost_focus(*a,**kw):
            result=original(*a,**kw);self.f.guard.failure=ExecutionError('Target lost focus.');return result
        self.ports.planner=lost_focus;r,_=self.run_job()
        self.assertEqual(r['reason'],'hard_guard_blocked')
        self.assertIn('Target lost focus',r['decisions'][-1]['hard_guard']['reason'])
        self.assertEqual(self.f.driver.events,[])

    def test_window_unavailable_stops_without_repeated_capture_recovery(self):
        from computer.capture import CaptureError
        with patch('harness.live_agent.capture_observation',side_effect=CaptureError('Stage Manager preview')):
            r,_=self.run_job()
        self.assertEqual(r['reason'],'game_window_not_visible_or_changed')
        self.assertEqual(r['recoveries'],0);self.assertEqual(self.f.driver.events,[])

    def test_later_dialogue_is_not_treated_as_introduction(self):
        self.kind_override='dialogue';r,_=self.run_job()
        self.assertEqual(r['reason'],'outside_existing_save_gameplay_scope');self.assertEqual(self.f.driver.events,[])

    def test_fresh_battle_appearance_blocks_input(self):
        original=self.checker
        def changed(*a,**kw):
            if self.plans: self.kind_override='battle'
            return original(*a,**kw)
        # Planner gets original walking facts; independent gate sees battle.
        self.ports.checker=changed;r,_=self.run_job()
        self.assertEqual(r['reason'],'outside_existing_save_gameplay_scope');self.assertEqual(self.f.driver.events,[])

    def test_selection_changes_to_save_blocks_x(self):
        self.stage=1;original=self.checker
        def changed(*a,**kw):
            c,m=original(*a,**kw);return c.model_copy(update={'selected_label':'SAVE'}),m
        self.ports.checker=changed;r,_=self.run_job()
        self.assertEqual(r['reason'],'recovery_limit');self.assertEqual(self.f.driver.events,[])

    def test_confirmed_walking_can_continue_without_false_text_no_progress(self):
        self.kind_override='overworld';self.job=job_from_prompt('Wander for 2 minutes');self.job=self.job.model_copy(update={'max_actions':5})
        original=self.planner
        def walking(*a,**kw):
            m=original(*a,**kw);r=json.loads(m['text']);r['action']['key']='right';m['text']=json.dumps(r);return m
        self.ports.planner=walking;r,_=self.run_job()
        self.assertEqual(r['completed_inputs'],5);self.assertEqual(r['reason'],'action_limit')
        self.assertEqual(self.effects,5)

    def test_ineffective_walking_stops_after_three(self):
        self.kind_override='overworld';self.effect_progress='unchanged';self.job=job_from_prompt('Wander for 2 minutes')
        original=self.planner
        def walking(*a,**kw):
            m=original(*a,**kw);r=json.loads(m['text']);r['action']['key']='left';m['text']=json.dumps(r);return m
        self.ports.planner=walking;r,_=self.run_job()
        self.assertEqual(r['completed_inputs'],3);self.assertEqual(r['reason'],'no_progress_limit')

    def test_mismatched_effect_identity_is_terminal_no_replay(self):
        original=self.effect
        def changed(*a,**kw):
            e,m=original(*a,**kw);return e.model_copy(update={'after_id':'wrong'}),m
        self.ports.effect_checker=changed;r,_=self.run_job()
        self.assertEqual(r['reason'],'invalid_effect_assessment');self.assertEqual(r['completed_inputs'],1)

    def test_cli_routes_party_and_wander(self):
        import contextlib,io
        from harness.cli import main
        fake={'status':'goal_reported','reason':'fresh_goal_evidence','completed_inputs':0,'model_calls':2,'elapsed_seconds':1,'exit_code':0,'observations':[]}
        with patch('harness.cli.PROJECT_ROOT',self.f.root),patch('harness.cli.load_settings',return_value=self.f.settings),patch('harness.cli.time.sleep'),patch('harness.live_gameplay.run_live_gameplay',return_value=(fake,self.f.root/'record.json')) as run,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--play','--prompt','Inspect my Pokemon party']),0)
            self.assertEqual(main(['--play','--prompt','Wander for 2 minutes']),0)
            self.assertEqual(run.call_count,2)


class GameplayPolicyTests(unittest.TestCase):
    def test_key_mapping_durations_and_scope(self):
        walk=facts('id','overworld')
        for key in ['up','down','left','right','s']:self.assertTrue(walk.permits(Press(tool='PRESS',key=key,seconds=.1)))
        for key in ['x','z','enter']:self.assertFalse(walk.permits(Press(tool='PRESS',key=key,seconds=.1)))
        menu=facts('id','game_menu','Pokemon')
        self.assertTrue(menu.permits(Press(tool='PRESS',key='x',seconds=.1)))
        self.assertFalse(menu.permits(Press(tool='PRESS',key='x',seconds=.3)))
        for label in ['SAVE','OPTIONS','ADAM','POKEDEX',None]:
            self.assertFalse(menu.model_copy(update={'selected_label':label}).permits(Press(tool='PRESS',key='x',seconds=.1)))
        self.assertFalse(facts('id','dialogue').permits(Press(tool='PRESS',key='x',seconds=.1)))
        self.assertFalse(menu.permits(TypeText(tool='TYPE',text='ABC')))

    def test_party_popup_summary_only_and_bag_read_only(self):
        x=Press(tool='PRESS',key='x',seconds=.1)
        self.assertTrue(facts('id','party','Oshawott',MEMBERS).permits(x))
        self.assertTrue(facts('id','party_options','SUMMARY').permits(x))
        for selected in ['Switch','Item','Release','Give','Take',None]:self.assertFalse(facts('id','party_options',selected).permits(x))
        self.assertFalse(facts('id','bag','Potion').permits(x))
        self.assertTrue(facts('id','bag').permits(Press(tool='PRESS',key='z',seconds=.1)))

    def test_menu_arming_requires_positive_neutral_complete_menu(self):
        x=Press(tool='PRESS',key='x',seconds=.1)
        c=facts('id','game_menu',None)
        self.assertFalse(c.permits(x))
        c=c.model_copy(update={'lower_touchscreen':c.lower_touchscreen.model_copy(update={'visible_text':['POKÉMON','POKÉDEX','BAG','SAVE','OPTIONS']})})
        self.assertTrue(c.permits(x))
        self.assertFalse(c.model_copy(update={'selection_active':None}).permits(x))
        self.assertFalse(c.model_copy(update={'selection_active':True}).permits(x))

    def test_click_requires_fresh_visible_label_and_button_box(self):
        c=facts('id','game_menu','Pokemon',targets=[{'label':'Pokemon','box':(10,550,490,680)}])
        self.assertTrue(c.permits(Click(tool='CLICK',x=200,y=600),'POKÉMON'))
        self.assertFalse(c.permits(Click(tool='CLICK',x=700,y=600),'Pokemon'))
        self.assertFalse(c.permits(Click(tool='CLICK',x=200,y=600),'SAVE'))
        self.assertFalse(c.model_copy(update={'touch_targets':[]}).permits(Click(tool='CLICK',x=200,y=600),'Pokemon'))

    def test_party_goal_needs_owned_names_and_level_or_hp(self):
        self.assertTrue(facts('id','party',members=MEMBERS).party_confirmed)
        self.assertFalse(facts('id','party').party_confirmed)
        self.assertFalse(facts('id','game_menu',members=MEMBERS).party_confirmed)
        self.assertFalse(facts('id','party',members=[{'name':'Oshawott','level':None,'hp_text':None}]).party_confirmed)

    def test_jobs_are_bounded_and_first_pokemon_still_outside_scope(self):
        self.assertEqual(job_from_prompt('Check the Pokemon in the bag').goal_kind,'party')
        self.assertEqual(job_from_prompt('Inspect my Bag').goal_kind,'other')
        self.assertEqual(job_from_prompt('Inspect my Pokemon party through the Pokemon menu and pause on the party list.').goal_kind,'party')
        self.assertEqual(job_from_prompt('Wander and inspect my Pokemon for 2 minutes').goal_kind,'exploration')
        self.assertEqual(live_gameplay_job(job_from_prompt('Wander for 2 minutes')).seconds,120)
        with self.assertRaises(ValueError):live_gameplay_job(job_from_prompt('Get your first Pokemon'))

    def test_real_protocol_two_images_and_four_image_effect(self):
        f=fixtures.LivePortsTests('test_already_at_menu_has_zero_input_and_zero_laya_calls');f.setUp()
        try:
            before=f.ports.capture(f.job,100);after=f.ports.capture(f.job,100)
            b=f.ports.raw(before);a=f.ports.raw(after)
            checked=facts(after.observation_id,'party',members=MEMBERS)
            client=FakeClient([response('```json\n'+checked.model_dump_json()+'\n```')])
            result,metrics=check_gameplay_observation(f.settings,a,100,clock=f.clock.time,client_factory=lambda **kw:client)
            self.assertTrue(result.party_confirmed);self.assertEqual(metrics['attempts'],1)
            effect=GameplayEffect(before_id=before.observation_id,after_id=after.observation_id,progress='confirmed',evidence='Menu opened.')
            client=FakeClient([response(effect.model_dump_json())])
            result,_=check_gameplay_effect(f.settings,a,100,previous=b,action={'tool':'PRESS','key':'s','seconds':.1},clock=f.clock.time,client_factory=lambda **kw:client)
            self.assertEqual(result.progress,'confirmed');self.assertEqual(len(client.calls[0]['contents']),9)
            client=FakeClient([response(effect.model_copy(update={'before_id':'wrong'}).model_dump_json())])
            with self.assertRaises(ValueError):check_gameplay_effect(f.settings,a,100,previous=b,action={},clock=f.clock.time,client_factory=lambda **kw:client)
        finally:f.doCleanups()
