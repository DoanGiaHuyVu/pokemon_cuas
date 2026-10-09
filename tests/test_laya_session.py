import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from harness.journal import atomic_json
from models.laya_local import ARTIFACTS, LayaError
from models.laya_session import LayaSession
from models.laya_worker import main as worker_main
from test_executor import FakeClock
from test_laya_local import questions_fixture, prediction_fixture


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); self.checkpoint=self.root/'checkpoint'
        for name in ARTIFACTS:
            p=self.checkpoint/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_text('{}')
        (self.checkpoint/'questions.json').write_text(json.dumps(questions_fixture()))
        self.settings=SimpleNamespace(laya_model_path=self.checkpoint,laya_source_path=self.root/'source')
        self.clock=FakeClock(); self.count=0
        self.process=Mock(); self.process.poll.return_value=None
        self.process.stdin.write.side_effect=self.reply
        self.spawn=Mock(side_effect=self.start)
        self.session=LayaSession(self.settings,self.root/'runs',clock=self.clock.time,sleep=self.clock.sleep,spawn=self.spawn)
        self.addCleanup(self.session.close)

    def start(self,args,**kwargs):
        self.assertNotIn('GEMINI_API_KEY',kwargs['env'])
        self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'],'1')
        atomic_json(Path(args[-1]),{'status':'ready'})
        return self.process

    def reply(self,line):
        self.count+=1; item=json.loads(line)
        result={'status':'passed','sequence':self.count,'checkpoint_unchanged':True,'network_attempts_blocked':0,
                'cases':[item|{'prediction':prediction_fixture()}]}
        atomic_json(self.session.workspace/f'query-{self.count}.json',result)

    def test_two_fresh_contexts_share_one_offline_process_and_separate_results(self):
        records=[self.session.evaluate(f'case-{n}',f'fresh-state-{n}',100,lambda deadline:None) for n in range(2)]
        self.assertEqual(self.spawn.call_count,1)
        self.assertNotEqual(records[0][1],records[1][1])
        self.assertEqual(records[1][0]['result']['cases'][0]['state'],'fresh-state-1')
        self.session.close(); self.session.close()
        self.process.stdin.close.assert_called_once()
        with self.assertRaises(LayaError): self.session.evaluate('new','new',100,lambda d:None)

    def test_mismatched_context_checkpoint_or_network_failure_never_authorizes(self):
        for field in ('state','checkpoint','network'):
            def reply(line):
                self.reply(line)
                p=self.session.workspace/f'query-{self.count}.json'; r=json.loads(p.read_text())
                if field=='state': r['cases'][0]['state']='wrong'
                if field=='checkpoint': r['checkpoint_unchanged']=False
                if field=='network': r['network_attempts_blocked']=1
                atomic_json(p,r)
            self.process.stdin.write.side_effect=reply
            with self.assertRaises(LayaError): self.session.evaluate('case','fresh',100,lambda d:None)

    def test_stop_during_wait_and_dead_worker_are_terminal(self):
        self.process.stdin.write.side_effect=lambda line:None
        def stopped(deadline):
            if self.session.sequence: raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt): self.session.evaluate('case','fresh',100,stopped)
        self.process.poll.return_value=1
        with self.assertRaises(LayaError): self.session.evaluate('next','fresh',100,lambda d:None)

    def test_deadline_while_waiting_has_no_hidden_retry(self):
        self.process.stdin.write.side_effect=lambda line:None
        with self.assertRaises(LayaError): self.session.evaluate('case','fresh',.1,lambda d:None)
        self.assertEqual(self.process.stdin.write.call_count,1)
        self.assertEqual(self.spawn.call_count,1)

    def test_worker_live_protocol_loads_once_preserves_fresh_predictions_and_closes(self):
        source=self.settings.laya_source_path; (source/'laya').mkdir(parents=True)
        (source/'laya/__init__.py').write_text('# fixture')
        (self.checkpoint/'rl_agent_config.json').write_text(json.dumps({'encoder':'answerdotai/ModernBERT-large','max_len':1024,'head_max_len':256}))
        workspace=self.root/'worker'; workspace.mkdir()
        request,result=workspace/'request.json',workspace/'result.json'
        atomic_json(request,{'mode':'live','checkpoint':str(self.checkpoint),'source':str(source),'device':'cpu'})
        agent=MagicMock(); agent.device=SimpleNamespace(type='cpu',__str__=lambda self:'cpu')
        agent.temperature=1.; agent.temperature_by_options={}
        agent.cpu_fallback_count=0; agent.predict.side_effect=lambda *a,**kw:prediction_fixture()
        laya=SimpleNamespace(__file__=str(source/'laya/__init__.py'),__version__='test',load=Mock(return_value=agent))
        torch=Mock(); torch.backends.mps.is_available.return_value=False
        import sys
        lines=''.join(json.dumps({'name':f'case-{n}','state':f'fresh-{n}'})+'\n' for n in range(2))
        with patch.dict(sys.modules,{'torch':torch,'laya':laya}),patch('models.laya_worker.importlib.metadata.version',return_value='test'),patch.object(sys,'path',list(sys.path)),patch.object(sys,'stdin',io.StringIO(lines)),patch.dict('os.environ'),patch('socket.socket.connect'),patch('socket.socket.connect_ex'),patch('socket.create_connection'):
            self.assertEqual(worker_main(request,result),0)
        laya.load.assert_called_once(); self.assertEqual(agent.predict.call_count,2)
        agent.__exit__.assert_called_once()
        for n in range(2):
            r=json.loads((workspace/f'query-{n+1}.json').read_text())
            self.assertEqual(r['cases'][0]['state'],f'fresh-{n}')
            self.assertTrue(r['checkpoint_unchanged']); self.assertEqual(r['network_attempts_blocked'],0)
