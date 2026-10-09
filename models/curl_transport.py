"""Single-attempt Gemini REST transport using macOS curl's proven HTTP/2 path.

Credentials go through stdin, never command arguments, payload files, or logs.
No redirects, TLS bypass, SDK retries, tools, or alternate providers are enabled.
"""
import asyncio
import base64
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace


class CurlAPIError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__('Gemini API rejected this request.')


class CurlTransportError(Exception):
    def __init__(self, code):
        self.curl_exit_code = code
        super().__init__('Gemini transport failed; raw diagnostics withheld.')


class CurlGemmaClient:
    """Small genai-compatible response interface; exactly one REST request per call."""
    def __init__(self, *, api_key, http_options, vertexai=False, enterprise=False):
        if vertexai is not False or enterprise is not False:
            raise ValueError('Only the configured Google AI API is allowed.')
        if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096 or not api_key.isascii()
                or any(ord(c) < 32 or ord(c) == 127 for c in api_key)):
            raise ValueError('Invalid API credential format.')
        if http_options.base_url != 'https://generativelanguage.googleapis.com':
            raise ValueError('Only the configured Google API host is allowed.')
        if http_options.retry_options.attempts != 1:
            raise ValueError('Transport requires one attempt.')
        self.key = api_key
        self.timeout = http_options.timeout / 1000
        self.process = None
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate), aclose=self.aclose)

    async def generate(self, *, model, contents, config):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', model):
            raise ValueError('Invalid model identifier.')
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append({'text': item})
            elif item.inline_data is not None and item.inline_data.mime_type == 'image/png' and isinstance(item.inline_data.data, bytes):
                parts.append({'inlineData': {'mimeType':'image/png','data':base64.b64encode(item.inline_data.data).decode('ascii')}})
            else:
                raise ValueError('Only text and game PNGs are permitted.')
        payload = {'contents':[{'role':'user','parts':parts}],
                   'systemInstruction':{'parts':[{'text':config.system_instruction}]},
                   'generationConfig':{'temperature':config.temperature, 'maxOutputTokens':config.max_output_tokens,
                                       'thinkingConfig':{'thinkingLevel':'MINIMAL'}}}
        header_config = ('header = ' + json.dumps('x-goog-api-key: '+self.key) + '\n'
                         'header = "Content-Type: application/json"\n')
        with tempfile.TemporaryDirectory(prefix='pokemon-cua-http2-') as tmp:
            body, reply = Path(tmp)/'request.json', Path(tmp)/'reply.json'
            body.write_text(json.dumps(payload))
            body.chmod(0o600)
            args = ['/usr/bin/curl','--http2','--silent','--show-error','--connect-timeout',str(min(10,self.timeout)),
                    '--max-time',str(self.timeout),'--max-filesize','524288','--config','-',
                    '--data-binary','@'+str(body),'--output',str(reply),'--write-out','%{http_code}',
                    'https://generativelanguage.googleapis.com/v1beta/models/'+model+':generateContent']
            try:
                self.process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
                                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                stdout, _ = await self.process.communicate(header_config.encode())
                if self.process.returncode == 28: raise TimeoutError('Gemini request timed out.')
                if self.process.returncode != 0: raise CurlTransportError(self.process.returncode)
                status = stdout.decode('ascii').strip()
                if not re.fullmatch(r'\d{3}', status): raise ValueError('Missing HTTP status.')
                if int(status) != 200: raise CurlAPIError(int(status))
                if not reply.is_file() or reply.stat().st_size > 524288: raise ValueError('Missing or excessive API reply.')
                data = json.loads(reply.read_text())
                if not isinstance(data, dict) or 'error' in data: raise ValueError('Unexpected API envelope.')
                return response(data)
            finally:
                await self.aclose()

    async def aclose(self):
        process, self.process = self.process, None
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=1)
            except TimeoutError:
                process.kill()
                await asyncio.wait_for(process.wait(), timeout=1)

    def close(self):
        # The async request's finally block owns bounded process cleanup.
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()


def response(data):
    candidates = []
    for candidate in data.get('candidates', []):
        parts = [SimpleNamespace(text=p.get('text'), thought=p.get('thought',False), function_call=p.get('functionCall'))
                 for p in candidate.get('content',{}).get('parts',[])]
        candidates.append(SimpleNamespace(finish_reason=candidate.get('finishReason'),content=SimpleNamespace(parts=parts)))
    usage=data.get('usageMetadata',{})
    return SimpleNamespace(candidates=candidates, usage_metadata=SimpleNamespace(
        prompt_token_count=usage.get('promptTokenCount'),candidates_token_count=usage.get('candidatesTokenCount'),
        total_token_count=usage.get('totalTokenCount')))
