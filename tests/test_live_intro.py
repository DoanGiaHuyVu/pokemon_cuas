"""Introduction wiring and phase boundaries; no real API, weights, or input."""
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.actions import Press
from harness.agent import HardStop
from harness.agent_job import job_from_prompt, live_intro_job
from harness.cli import main
from harness.live_intro import LiveIntroPorts, run_live_intro
from models.intro_check import IntroCheck, IntroResponseError, check_intro_observation
import test_live_agent as live_fixture
from test_perception import FakeClient, response


def facts(identity,kind='dialogue',text='Welcome!',selected=None,gender='unknown'):
    menu=kind=='main_menu'; name=kind=='name_entry'
    return IntroCheck(observation_id=identity,screen_kind=kind,
        upper_screen={'description':'Fixture upper display.','visible_text':['NEW GAME'] if menu else [text]},
        lower_touchscreen={'description':'Fixture lower display.','visible_text':['A','B','C','OK'] if name else []},
        selected_label='NEW GAME' if menu else selected,new_game_visible=menu,continue_visible=False,
        name_keyboard_visible=name,player_gender=gender,evidence='Fixture screen facts.')


class IntroPortsTests(unittest.TestCase):
    def setUp(self):
        self.f=live_fixture.LivePortsTests('test_already_at_menu_has_zero_input_and_zero_laya_calls')
        self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.job=job_from_prompt('Enter New Game, advance the introduction with x, select Boy, and stop at name entry.',seconds=900.)
        self.stage=0
        self.stages=[('main_menu','NEW GAME',None,'unknown'),('dialogue','Hello!',None,'unknown'),
                     ('dialogue','Are you ready?',None,'unknown'),('character_select','Boy or Girl?','Boy','boy'),
                     ('name_entry','Your name?',None,'boy')]
        self.check_calls=0
        self.ports=LiveIntroPorts(self.f.settings,self.f.root,clock=self.f.clock.time,backend=self.f.backend,
            checker=self.checker,laya_runner=self.f.laya,guard_factory=lambda *a:self.f.guard,
            driver_factory=lambda:self.f.driver,sleep=self.f.clock.sleep)
        self.f.driver.key_down=self.key_down

    def key_down(self,key): self.f.driver.events.append(('down',key)); self.stage+=1

    def checker(self,settings,obs,deadline,**kwargs):
        self.check_calls+=1; self.f.clock.now+=.1
        kind,text,selected,gender=self.stages[min(self.stage,len(self.stages)-1)]
        return facts(obs['observation_id'],kind,text,selected,gender),{'attempts':1,'usage':{}}

    def run_job(self): return run_live_intro(self.f.settings,self.job,self.f.root,ports=self.ports)

    def test_full_intro_uses_only_one_x_per_observation_and_stops_before_name_input(self):
        self.f.label='human_review'
        r,path=self.run_job()
        self.assertEqual(r['status'],'goal_reported'); self.assertEqual(r['step'],11)
        self.assertEqual(r['completed_inputs'],4); self.assertEqual(r['model_calls'],14)
        self.assertEqual(self.f.driver.events,[('down','x'),('up','x')]*4)
        self.assertEqual(self.f.laya_calls,4); self.assertEqual(r['human_review_requests'],0)
        self.assertTrue(self.ports.new_game_dispatched); self.assertTrue(self.ports.boy_choice_dispatched)
        self.assertTrue(self.ports.name_seen); self.assertTrue(self.ports.audit['closed'])
        self.assertEqual(json.loads(path.read_text()),r)
        self.assertNotEqual(r['goal_assessments'][0]['observation_id'],r['proposals'][-1]['observation_id'])

    def test_already_at_name_entry_has_no_input_or_laya(self):
        self.stage=4; r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported'); self.assertEqual(self.f.driver.events,[])
        self.assertEqual(self.f.laya_calls,0)

    def test_existing_save_on_initial_or_fresh_menu_blocks_new_game(self):
        original=self.checker
        for at_call in (1,2):
            with self.subTest(at_call=at_call):
                self.check_calls=0
                def checked(*a,**kw):
                    c,m=original(*a,**kw)
                    if self.check_calls==at_call: c=c.model_copy(update={'continue_visible':True})
                    return c,m
                self.ports.checker=checked
                r,_=self.run_job()
                self.assertEqual(r['reason'],'existing_save_outside_intro_scope')
                self.assertEqual(self.f.driver.events,[])

    def test_name_keyboard_appears_during_gate_no_press(self):
        original=self.checker
        def changed(*a,**kw):
            if self.check_calls==1: self.stage=4
            return original(*a,**kw)
        self.ports.checker=changed
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported'); self.assertEqual(self.f.driver.events,[])
        self.assertEqual(r['decisions'][0]['hard_guard']['status'],'changed')

    def test_character_selection_changes_to_girl_no_confirmation(self):
        self.stage=3; original=self.checker
        def changed(*a,**kw):
            if self.check_calls>=1: self.stages[3]=('character_select','Boy or Girl?','Girl','girl')
            return original(*a,**kw)
        self.ports.checker=changed
        r,_=self.run_job()
        self.assertEqual(r['reason'],'recovery_limit'); self.assertEqual(self.f.driver.events,[])

    def test_laya_stop_always_precedes_fresh_gate_and_input(self):
        self.f.label='stop'; r,_=self.run_job()
        self.assertEqual(r['reason'],'laya_stop'); self.assertEqual(self.check_calls,1)
        self.assertEqual(self.f.driver.events,[])

    def test_new_game_reentry_is_not_automatically_confirmed_again(self):
        self.stages=[self.stages[0]]
        r,_=self.run_job()
        self.assertEqual(r['reason'],'new_game_reentry_forbidden'); self.assertEqual(r['completed_inputs'],1)

    def test_ineffective_dialogue_input_stops_at_no_progress_limit(self):
        self.stages=[('dialogue','Unchanging dialogue.',None,'unknown')]
        r,_=self.run_job()
        self.assertEqual(r['reason'],'no_progress_limit'); self.assertEqual(r['completed_inputs'],3)

    def test_unknown_transition_frames_never_receive_input(self):
        self.stages=[('transition','',None,'unknown')]
        # Empty quoted text is prohibited; keep the transition fact text absent.
        original=self.checker
        def no_text(*a,**kw):
            c,m=original(*a,**kw)
            return c.model_copy(update={'upper_screen':c.upper_screen.model_copy(update={'visible_text':[]})}),m
        self.stages=[('transition','Fixture',None,'unknown')]; self.ports.checker=no_text
        r,_=self.run_job()
        self.assertEqual(r['reason'],'recovery_limit'); self.assertEqual(self.f.driver.events,[])

    def test_question_about_name_without_keyboard_does_not_prove_goal(self):
        self.stage=4; original=self.checker
        def no_keyboard(*a,**kw):
            c,m=original(*a,**kw)
            return c.model_copy(update={'name_keyboard_visible':False}),m
        self.ports.checker=no_keyboard
        r,_=self.run_job()
        self.assertEqual(r['reason'],'recovery_limit'); self.assertEqual(self.f.driver.events,[])

    def test_only_short_x_can_use_an_intro_ticket(self):
        obs=self.ports.capture(self.job,100)
        self.ports.plan(obs,self.job,[],100); self.ports.guard(obs,100)
        for action in (Press(tool='PRESS',key='enter',seconds=.1),Press(tool='PRESS',key='x',seconds=.5)):
            with self.assertRaises(HardStop): self.ports.execute(action,obs,'id',100)
        self.assertEqual(self.f.driver.events,[])

    def test_entry_pair_has_separate_fresh_laya_decisions_and_stops_pair_at_menu(self):
        self.stages=[('title','PRESS START',None,'unknown'),('title','PRESS START',None,'unknown')]+self.stages
        self.ports.anchor_detector=lambda obs:self.stage<2
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported'); self.assertEqual(r['completed_inputs'],6)
        self.assertEqual(self.f.laya_calls,6)
        self.assertNotEqual(r['decisions'][0]['text_sha256'],r['decisions'][1]['text_sha256'])
        self.assertEqual(self.f.driver.events,[('down','x'),('up','x')]*6)
        facts_events=[e for e in self.ports.audit['events'] if e['kind']=='hosted_result']
        self.assertEqual(facts_events[0]['result']['screen_kind'],'title')
        self.assertEqual(facts_events[1]['result']['screen_kind'],'main_menu')
        self.assertTrue(any(e['kind']=='local_title_match' for e in self.ports.audit['events']))

    def test_conditional_second_press_is_cancelled_when_name_keyboard_appears(self):
        self.stages=[('title','PRESS START',None,'unknown'),('name_entry','Your name?',None,'boy')]
        self.ports.anchor_detector=lambda obs:self.stage==0
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported'); self.assertEqual(r['completed_inputs'],1)
        self.assertEqual(self.f.laya_calls,1)
        self.assertEqual(self.f.driver.events,[('down','x'),('up','x')])

    def test_cli_dispatches_intro_and_keeps_other_live_scope_disabled(self):
        fake={'status':'goal_reported','reason':'fresh_goal_evidence','completed_inputs':4,'model_calls':14,
              'elapsed_seconds':1,'exit_code':0,'observations':[]}
        import contextlib,io
        with patch('harness.cli.PROJECT_ROOT',self.f.root),patch('harness.cli.load_settings',return_value=self.f.settings),patch('harness.cli.time.sleep'),patch('harness.live_intro.run_live_intro',return_value=(fake,self.f.root/'record.json')) as run,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--play','--prompt',self.job.prompt,'--start-delay','0']),0)
            self.assertEqual(run.call_args.args[1].max_actions,40)
            self.assertEqual(main(['--play','--prompt','Get your first Pokemon']),2)
            self.assertEqual(run.call_count,1)


class IntroCheckTests(unittest.TestCase):
    def test_name_requires_keyboard_letters_and_boy_evidence(self):
        self.assertFalse(facts('id','dialogue','What is your name?',gender='boy').goal_confirmed)
        self.assertFalse(facts('id','name_entry',gender='unknown').goal_confirmed)
        c=facts('id','name_entry',gender='boy'); self.assertTrue(c.goal_confirmed)
        self.assertFalse(c.x_allowed)
        self.assertFalse(c.model_copy(update={'lower_touchscreen':c.lower_touchscreen.model_copy(update={'visible_text':[]})}).goal_confirmed)

    def test_menu_and_gender_selection_must_be_positive_and_not_contradictory(self):
        self.assertTrue(facts('id','main_menu').x_allowed)
        for update in ({'selected_label':'MYSTERY GIFT'},{'continue_visible':True},{'screen_kind':'dialogue'},{'name_keyboard_visible':True}):
            self.assertFalse(facts('id','main_menu').model_copy(update=update).x_allowed)
        self.assertTrue(facts('id','character_select',selected='Boy',gender='boy').x_allowed)
        self.assertFalse(facts('id','character_select',selected='Girl',gender='girl').x_allowed)
        self.assertFalse(facts('id','unknown').x_allowed)

    def test_real_checker_protocol_has_two_images_single_attempt_and_identity_binding(self):
        f=live_fixture.LivePortsTests('test_already_at_menu_has_zero_input_and_zero_laya_calls'); f.setUp()
        try:
            obs=f.ports.capture(f.job,100); raw=f.ports.raw(obs)
            result=facts(obs.observation_id,'name_entry',gender='boy')
            client=FakeClient([response(result.model_dump_json())])
            checked,metrics=check_intro_observation(f.settings,raw,100,clock=f.clock.time,client_factory=lambda **kw:client)
            self.assertTrue(checked.goal_confirmed); self.assertEqual(metrics['attempts'],1)
            self.assertIn('upper DS display ONLY',client.calls[0]['contents'][0])
            self.assertIn('lower DS touchscreen ONLY',client.calls[0]['contents'][2])
            client=FakeClient([response(result.model_copy(update={'observation_id':'wrong'}).model_dump_json())])
            with self.assertRaises(ValueError): check_intro_observation(f.settings,raw,100,clock=f.clock.time,client_factory=lambda **kw:client)
            client=FakeClient([response('```json\n'+result.model_dump_json()+'\n```')])
            checked,metrics=check_intro_observation(f.settings,raw,100,clock=f.clock.time,client_factory=lambda **kw:client)
            self.assertTrue(checked.goal_confirmed)
            self.assertEqual(metrics['response_wrapper'],'single_json_code_fence')
            client=FakeClient([response('```json\n'+result.model_dump_json()+'\n```\nExtra instruction')])
            with self.assertRaises(IntroResponseError) as error:
                check_intro_observation(f.settings,raw,100,clock=f.clock.time,client_factory=lambda **kw:client)
            self.assertTrue(error.exception.rejected_reply.startswith('```'))
        finally: f.doCleanups()

    def test_intro_job_is_scoped_and_bounded(self):
        j=live_intro_job(job_from_prompt('Reach name-entry and pause.'))
        self.assertEqual(j.goal_kind,'introduction'); self.assertEqual(j.max_actions,40)
        with self.assertRaises(ValueError): live_intro_job(job_from_prompt('Play for two minutes.'))
