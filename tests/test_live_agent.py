"""Live-port wiring with fake backend/provider/local worker/input, no OS input."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from pydantic import SecretStr

from computer.capture import calibrate, confirm_calibration
from harness.actions import Press
from harness.agent import HardStop, run_agent
from harness.agent_job import job_from_prompt
from harness.agent_simulation import fixture_prediction
from harness.cli import main
from harness.live_agent import LiveMenuPorts, run_live_menu
from models.menu_check import MenuCheck, check_menu_observation
from models.laya_local import ARTIFACTS
from test_capture import FakeBackend
from test_executor import FakeClock, FakeGuard, SpyDriver
from test_laya_local import questions_fixture
from test_perception import FakeClient, response


def check(obs, kind='title', menu=False):
    return MenuCheck(observation_id=obs['observation_id'], screen_kind=kind,
                     visible_text=['New Game'] if menu else ['PRESS START'], new_game_visible=menu,
                     continue_visible=False, evidence='Only the requested game menu is visible.' if menu else 'Only title is visible.')


class LivePortsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = FakeBackend()
        calibrate(self.backend, self.root/'runs/calibration.json', self.root/'screenshots')
        confirm_calibration(self.backend, self.root/'runs/calibration.json')
        self.clock, self.guard, self.driver = FakeClock(), FakeGuard(), SpyDriver()
        self.checkpoint = self.root/'checkpoint'
        for name in ARTIFACTS:
            p=self.checkpoint/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
        (self.checkpoint/'questions.json').write_text(json.dumps(questions_fixture()))
        self.settings=SimpleNamespace(laya_model_path=self.checkpoint,gemini_api_key=SecretStr('never-print'),gemma_model='fixture')
        self.job=job_from_prompt('Reach the New Game menu and stop without pressing New Game.')
        self.stage=0
        self.label='continue'
        self.plans=0
        self.checks=0
        self.laya_calls=0
        self.ports=LiveMenuPorts(self.settings,self.root,clock=self.clock.time,backend=self.backend,
                                planner=self.planner,checker=self.checker,laya_runner=self.laya,
                                guard_factory=lambda *args:self.guard,driver_factory=lambda:self.driver,
                                sleep=self.clock.sleep)
        self.driver.key_down=self.key_down

    def key_down(self,key):
        self.driver.events.append(('down',key)); self.stage+=1

    def planner(self,settings,obs,job,history,deadline,**kw):
        self.plans+=1; self.clock.now+=.1
        menu=self.stage>0
        raw={'observation_id':obs['observation_id'],'visible_observation':{'screen_kind':'main_menu' if menu else 'title',
             'upper_screen':{'description':'fixture','visible_text':['New Game'] if menu else ['PRESS START']},
             'lower_touchscreen':{'description':'fixture','visible_text':[]}},'status':'pause' if menu else 'propose',
             'action':None if menu else {'tool':'PRESS','key':'x','seconds':.1},'target_label':None,
             'expected_result':'The menu may appear.','rationale':'Follow the menu-only goal.', 'task_complete':False,
             'goal_status':'candidate_complete' if menu else 'in_progress','goal_evidence':'New Game visible' if menu else '',
             'selected_menu_label':None}
        return {'text':json.dumps(raw),'attempts':1,'usage':{}}

    def checker(self,settings,obs,deadline,**kw):
        self.checks+=1; self.clock.now+=.1
        return check(obs,'main_menu' if self.stage>0 else 'title',self.stage>0), {'attempts':1,'usage':{}}

    def laya(self,settings,runs,states,**kw):
        self.laya_calls+=1
        self.assertGreater(kw['timeout'],0)
        item=states[0]
        self.assertIsInstance(item['state'],str)
        result={'checkpoint_unchanged':True,'network_attempts_blocked':0,'token_budget':{'max_len':1024},
                'cases':[{'name':item['name'],'state':item['state'],'prediction':fixture_prediction(questions_fixture(),self.label)}]}
        return {'status':'passed','result':result},self.root/'worker.json'

    def run_job(self):
        return run_live_menu(self.settings,self.job,self.root,ports=self.ports)

    def test_full_live_port_wiring_sends_one_x_then_verifies_menu_without_selection(self):
        r,path=self.run_job()
        self.assertEqual(r['status'],'goal_reported')
        self.assertEqual(r['step'],10)
        self.assertEqual(r['completed_inputs'],1)
        self.assertEqual(self.driver.events,[('down','x'),('up','x')])
        self.assertEqual(r['real_game_input_attempts'],1)
        self.assertEqual(r['model_calls'],5)  # plan + Laya + pre-input check + plan + goal check
        self.assertEqual(self.laya_calls,1)
        self.assertEqual(r['human_review_requests'],0)
        self.assertTrue(self.ports.menu_seen)
        self.assertTrue(self.ports.audit['closed'])
        self.assertEqual(json.loads(path.read_text()),r)
        self.assertFalse(r['task_complete'])

    def test_already_at_menu_has_zero_input_and_zero_laya_calls(self):
        self.stage=1
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported')
        self.assertEqual(self.driver.events,[])
        self.assertEqual(self.laya_calls,0)

    def test_menu_appears_during_gate_never_confirms_it(self):
        original=self.checker
        def changed(*args,**kw): self.stage=1; return original(*args,**kw)
        self.ports.checker=changed
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported')
        self.assertEqual(self.driver.events,[])
        self.assertEqual(r['recoveries'],1)
        self.assertEqual(r['decisions'][0]['hard_guard']['status'],'changed')

    def test_forced_stop_is_a_test_seam_and_emits_zero_input(self):
        self.label='stop'
        r,_=self.run_job()
        self.assertEqual(r['reason'],'laya_stop')
        self.assertEqual(self.driver.events,[])
        self.assertEqual(self.checks,0)

    def test_human_review_gets_automatic_fresh_recheck_then_one_x(self):
        self.label='human_review'
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported')
        self.assertEqual(self.laya_calls,1)
        self.assertEqual(self.driver.events,[('down','x'),('up','x')])
        self.assertEqual(r['recoveries'],0)
        self.assertEqual(self.checks,2)
        self.assertEqual(r['decisions'][0]['route'],'automatic_fresh_entry_check_cleared; no_human_required')

    def test_blinking_title_and_opening_share_only_the_approved_entry_gesture(self):
        self.label='human_review'
        original=self.planner
        def changed(*args,**kw):
            result=original(*args,**kw)
            raw=json.loads(result['text'])
            if self.plans==1:
                raw['visible_observation']['screen_kind']='opening'
                raw['visible_observation']['upper_screen']['visible_text']=[]
                raw['visible_observation']['lower_touchscreen']['visible_text']=['Pokémon']
            result['text']=json.dumps(raw); return result
        self.ports.planner=changed
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported')
        self.assertEqual(r['recoveries'],0)
        self.assertEqual(self.driver.events,[('down','x'),('up','x')])
        self.assertEqual(self.checks,2)  # Only dispatch check and final fresh goal assessment.

    def test_opening_unknowns_do_not_consume_the_entire_live_recheck_budget(self):
        self.label='human_review'
        original=self.planner
        def opening(*args,**kw):
            result=original(*args,**kw)
            if self.plans<=3:
                raw=json.loads(result['text']); raw['visible_observation']['screen_kind']='unknown'
                raw.update(status='pause',action=None); result['text']=json.dumps(raw)
            return result
        self.ports.planner=opening
        r,_=self.run_job()
        self.assertEqual(r['status'],'goal_reported')
        self.assertEqual(r['recoveries'],3)
        self.assertEqual(self.driver.events,[('down','x'),('up','x')])

    def test_scope_refuses_non_menu_live_job_before_any_capture(self):
        with self.assertRaises(ValueError): run_live_menu(self.settings,job_from_prompt('Play for 2 minutes'),self.root,ports=self.ports)
        self.assertEqual(self.ports.audit['screen_captures'],0)

    def test_mismatched_offline_case_is_rejected(self):
        original=self.laya
        def mismatched(*args,**kw):
            worker,path=original(*args,**kw); worker['result']['cases'][0]['state']='wrong-context'; return worker,path
        self.ports.laya_runner=mismatched
        r,_=self.run_job()
        self.assertEqual(r['reason'],'recovery_limit')
        self.assertEqual(self.driver.events,[])

    def test_input_has_single_use_ticket_and_duplicate_id_is_refused(self):
        obs=self.ports.capture(self.job,60)
        self.ports.plan(obs,self.job,[],60)
        self.ports.guard(obs,60)
        self.ports.execute(Press(tool='PRESS',key='x',seconds=.1),obs,'unique-id',60)
        with self.assertRaises(HardStop): self.ports.execute(Press(tool='PRESS',key='x',seconds=.1),obs,'unique-id',60)
        self.assertEqual(self.driver.events,[('down','x'),('up','x')])

    def test_native_focus_failure_blocks_before_hosted_gate_or_input(self):
        from computer.executor import ExecutionError
        self.guard.failure=ExecutionError('lost focus')
        r,_=self.run_job()
        self.assertEqual(r['reason'],'hard_guard_blocked')
        self.assertEqual(self.driver.events,[])
        self.assertEqual(self.checks,0)

    def test_gate_contradictory_flags_cannot_authorize_x(self):
        self.ports.checker=lambda settings,obs,*a,**kw:(MenuCheck(observation_id=obs['observation_id'],screen_kind='title',visible_text=['New Game'],new_game_visible=False,continue_visible=False,evidence='contradictory fixture'),{'attempts':1})
        r,_=self.run_job()
        self.assertEqual(r['reason'],'menu_input_forbidden')
        self.assertEqual(self.driver.events,[])
        self.assertTrue(self.ports.menu_seen)

    def test_guarded_png_integrity_mismatch_never_posts_input(self):
        obs=self.ports.capture(self.job,60)
        Path(obs.metadata['image_path']).write_bytes(b'changed')
        with self.assertRaises(HardStop): self.ports.plan(obs,self.job,[],60)
        self.assertEqual(self.driver.events,[])

    def test_cli_play_scope_validation_and_dispatch(self):
        fake={'status':'goal_reported','reason':'fresh_goal_evidence','completed_inputs':1,'model_calls':5,
              'elapsed_seconds':1,'exit_code':0,'observations':[]}
        import contextlib,io
        with patch('harness.cli.PROJECT_ROOT',self.root),patch('harness.cli.load_settings',return_value=self.settings),patch('harness.cli.time.sleep'),patch('harness.live_agent.run_live_menu',return_value=(fake,self.root/'record.json')) as run,contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--play','--prompt',self.job.prompt,'--laya-device','cpu','--start-delay','0']),0)
            run.assert_called_once()
            self.assertEqual(main(['--play','--prompt','Get your first Pokemon']),2)
            self.assertEqual(run.call_count,1)


class MenuCheckTests(unittest.TestCase):
    def test_title_logo_and_inconsistent_menu_do_not_prove_goal(self):
        for c in (check({'observation_id':'id'}),MenuCheck(observation_id='id',screen_kind='main_menu',visible_text=[],new_game_visible=True,continue_visible=False,evidence='claim')):
            self.assertFalse(c.goal_confirmed)
        self.assertTrue(check({'observation_id':'id'},'main_menu',True).goal_confirmed)

    def test_actual_checker_two_images_identity_and_single_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); backend=FakeBackend()
            calibrate(backend,root/'profile.json',root/'images'); confirm_calibration(backend,root/'profile.json')
            from computer.capture import capture_observation
            obs=capture_observation(backend,root/'profile.json',root/'images',root/'runs','test')
            result=check(obs,'main_menu',True)
            client=FakeClient([response(result.model_dump_json())])
            settings=SimpleNamespace(gemini_api_key=SecretStr('secret'),gemma_model='fixture')
            checked,metrics=check_menu_observation(settings,obs,10,clock=lambda:0,client_factory=lambda **kw:client)
            self.assertTrue(checked.goal_confirmed)
            self.assertEqual(metrics['attempts'],1)
            self.assertEqual(len(client.calls),1)
            self.assertIn('upper DS display ONLY',client.calls[0]['contents'][0])
            self.assertIn('lower DS touchscreen ONLY',client.calls[0]['contents'][2])
            wrong=result.model_copy(update={'observation_id':'wrong-frame'})
            client=FakeClient([response(wrong.model_dump_json())])
            with self.assertRaises(ValueError): check_menu_observation(settings,obs,10,clock=lambda:0,client_factory=lambda **kw:client)

if __name__=='__main__': unittest.main()

class StartupFocusTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.backend=FakeBackend(); self.profile=self.root/'profile.json'
        calibrate(self.backend,self.profile,self.root/'images'); confirm_calibration(self.backend,self.profile)

    def modules(self,name='melonDS',accepted=True):
        self.activated=[]
        target=SimpleNamespace(localizedName=lambda:name,
            activateWithOptions_=lambda option:self.activated.append(option) or accepted)
        appkit=SimpleNamespace(NSRunningApplication=SimpleNamespace(runningApplicationWithProcessIdentifier_=lambda pid:target),
            NSApplicationActivateIgnoringOtherApps=1,
            NSWorkspace=SimpleNamespace(sharedWorkspace=lambda:SimpleNamespace(frontmostApplication=lambda:
                SimpleNamespace(processIdentifier=lambda:self.backend.window.pid))))
        return {'AppKit':appkit}

    def test_focus_targets_verified_process_without_changing_window(self):
        from computer.executor import focus_game_window
        before=self.backend.window.model_dump()
        with patch.dict('sys.modules',self.modules()): focus_game_window(self.profile,backend=self.backend)
        self.assertEqual(self.activated,[1])
        self.assertEqual(self.backend.window.model_dump(),before)

    def test_wrong_process_name_or_refused_activation_fail_without_input(self):
        from computer.executor import ExecutionError,focus_game_window
        for name,accepted in [('Other',True),('melonDS',False)]:
            with self.subTest(name=name),patch.dict('sys.modules',self.modules(name,accepted)),self.assertRaises(ExecutionError):
                focus_game_window(self.profile,backend=self.backend)
            self.assertEqual(self.activated,[] if name=='Other' else [1])
