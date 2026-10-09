"""HTTP/2 REST payload, credential handling, and cancellation with process fakes."""
import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from pydantic import SecretStr
from types import SimpleNamespace
from models.curl_transport import CurlGemmaClient
from models.perception import PerceptionError, request_image_text


class CurlTests(unittest.TestCase):
    def setUp(self):
        self.settings=SimpleNamespace(gemini_api_key=SecretStr('fixture.secret+opaque'),gemma_model='gemma-4-26b-a4b-it')
        self.images=[{'rectangle':{'left':0,'top':i*10,'width':10,'height':10},'data':b'png'} for i in range(2)]
        self.payloads=[]; self.arguments=[]; self.headers=[]; self.terminated=[]
        self.http=200; self.exit=0
        self.reply={'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"screen_kind":"title"}'}]}}],
                    'usageMetadata':{'promptTokenCount':20,'candidatesTokenCount':10,'totalTokenCount':30}}

    async def spawn(self,*args,**kwargs):
        self.arguments.append(args)
        body=Path(args[args.index('--data-binary')+1][1:]); self.payloads.append(json.loads(body.read_text()))
        reply=Path(args[args.index('--output')+1]); reply.write_text(json.dumps(self.reply))
        parent=self
        class Process:
            returncode=None
            async def communicate(self,data):
                parent.headers.append(data); self.returncode=parent.exit; return str(parent.http).encode(),b''
            def terminate(self): parent.terminated.append(True); self.returncode=0
            def kill(self): self.returncode=0
            async def wait(self): return self.returncode
        return Process()

    def invoke(self):
        return asyncio.run(request_image_text(self.settings,self.images,'goal',system_instruction='Read both screens.',
            max_output_tokens=700,max_characters=6000,request_prefix='Current frame: ',max_attempts=1,
            client_factory=CurlGemmaClient,timeout_seconds=.1))

    def test_exact_payload_two_pngs_key_only_in_stdin_and_no_redirect_or_tls_bypass(self):
        with patch('models.curl_transport.asyncio.create_subprocess_exec',new=self.spawn): result=self.invoke()
        self.assertEqual(result['attempts'],1)
        self.assertEqual(result['usage']['total_token_count'],30)
        args=self.arguments[0]
        self.assertNotIn('fixture.secret+opaque',repr(args))
        self.assertNotIn('fixture.secret+opaque',json.dumps(self.payloads))
        self.assertIn(b'fixture.secret+opaque',self.headers[0])
        self.assertIn('--http2',args)
        self.assertNotIn('--location',args)
        self.assertNotIn('--insecure',args)
        self.assertEqual(len([p for p in self.payloads[0]['contents'][0]['parts'] if 'inlineData' in p]),2)
        self.assertEqual(self.payloads[0]['generationConfig']['thinkingConfig'],{'thinkingLevel':'MINIMAL'})
        self.assertFalse(Path(args[args.index('--output')+1]).exists())

    def test_http_and_transport_failures_are_single_attempt_and_sanitized(self):
        for http,code in [(401,0),(429,0),(200,56)]:
            self.http,self.exit=http,code
            with self.subTest(http=http,code=code),patch('models.curl_transport.asyncio.create_subprocess_exec',new=self.spawn),self.assertRaises(PerceptionError) as error: self.invoke()
            self.assertNotIn('fixture.secret+opaque',str(error.exception))
            self.assertEqual(error.exception.attempts,1)
        self.assertEqual(len(self.arguments),3)

    def test_function_calls_and_nonstop_results_cannot_become_text_success(self):
        for mutation in ('function','truncated'):
            self.reply['candidates'][0]={'finishReason':'STOP' if mutation=='function' else 'MAX_TOKENS',
                'content':{'parts':[{'text':'answer','functionCall':{'name':'unsafe'}}] if mutation=='function' else [{'text':'answer'}]}}
            with self.subTest(mutation=mutation),patch('models.curl_transport.asyncio.create_subprocess_exec',new=self.spawn),self.assertRaises(PerceptionError): self.invoke()

    def test_cancelled_async_request_terminates_child_and_removes_payload(self):
        original=self.spawn
        async def stalled(*args,**kwargs):
            process=await original(*args,**kwargs)
            async def communicate(data): await asyncio.Event().wait()
            process.communicate=communicate
            return process
        with patch('models.curl_transport.asyncio.create_subprocess_exec',new=stalled),self.assertRaises(PerceptionError): self.invoke()
        self.assertEqual(self.terminated,[True])
        self.assertFalse(Path(self.arguments[0][self.arguments[0].index('--output')+1]).exists())

if __name__=='__main__': unittest.main()
