"""Persistent isolated offline Laya process for fresh live text decisions."""
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from harness.journal import atomic_json
from models.laya_local import LayaError, load_questions, offline_environment


class LayaSession:
    def __init__(self, settings, runs, *, device='auto', clock=time.monotonic,
                 sleep=time.sleep, spawn=subprocess.Popen):
        if device not in {'auto', 'cpu', 'mps'}: raise LayaError('Invalid local device.')
        self.settings, self.device, self.clock, self.sleep, self.spawn = settings, device, clock, sleep, spawn
        self.workspace = Path(runs)/'laya-runtime'/uuid4().hex
        self.workspace.mkdir(parents=True)
        self.request_path = self.workspace/'request.json'
        self.result_path = self.workspace/'result.json'
        self.process = None
        self.sequence = 0
        self.closed = False

    def wait_for(self, path, deadline, check, *, ready=False):
        while True:
            check(deadline)
            if self.clock() >= deadline: raise LayaError('Local inference deadline reached.')
            if path.is_file():
                result = json.loads(path.read_text())
                if result.get('status') == 'failed': raise LayaError('Offline live worker failed.')
                if not ready or result.get('status') == 'ready': return result
            if self.process.poll() is not None: raise LayaError('Offline live worker exited before responding.')
            self.sleep(min(.02, max(0, deadline-self.clock())))

    def evaluate(self, name, text, deadline, check):
        if self.closed or self.sequence >= 48: raise LayaError('Local session is closed or exhausted.')
        if not isinstance(text, str) or not 1 <= len(text) <= 5000: raise LayaError('Bounded text required.')
        check(deadline)
        if self.process is None:
            load_questions(self.settings.laya_model_path)
            atomic_json(self.request_path, {'mode':'live', 'checkpoint':str(self.settings.laya_model_path),
                'source':str(self.settings.laya_source_path), 'device':self.device})
            self.process = self.spawn([sys.executable, '-m', 'models.laya_worker', str(self.request_path), str(self.result_path)],
                cwd=Path(__file__).resolve().parent.parent, env=offline_environment(),
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            self.wait_for(self.result_path, deadline, check, ready=True)
        if self.process.poll() is not None: raise LayaError('Offline live worker is no longer running.')
        self.sequence += 1
        query_path = self.workspace/f'query-{self.sequence}.json'
        self.process.stdin.write(json.dumps({'name':name, 'state':text}, ensure_ascii=True)+'\n')
        self.process.stdin.flush()
        result = self.wait_for(query_path, deadline, check)
        cases = result.get('cases', [])
        if (result.get('status') != 'passed' or result.get('sequence') != self.sequence
                or not result.get('checkpoint_unchanged') or result.get('network_attempts_blocked') != 0
                or len(cases) != 1 or cases[0].get('name') != name or cases[0].get('state') != text):
            raise LayaError('Offline live prediction identity/checks failed.')
        return {'status':'passed', 'result':result}, query_path

    def close(self):
        if self.closed: return
        self.closed = True
        if self.process is not None:
            if self.process.stdin is not None:
                try: self.process.stdin.close()
                except OSError: pass
            try: self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill(); self.process.wait(timeout=1)
