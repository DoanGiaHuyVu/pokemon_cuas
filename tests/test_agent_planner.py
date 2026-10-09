"""Bounded planner payload and API budget tests; fake provider only."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image
from pydantic import SecretStr

from harness.agent_job import job_from_prompt
from harness.agent_simulation import SimClock, simulation_calibration
from models.agent_planner import plan_agent_observation
from models.perception import PerceptionError, request_image_text
from test_perception import FakeClient, APIError, response


class AgentPlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calibration = simulation_calibration()
        image = self.root / 'game.png'; Image.new('RGB', (256, 384)).save(image)
        self.observation = {'observation_id': 'fresh-1', 'calibration': self.calibration.model_dump(), 'image_path': str(image)}
        self.settings = SimpleNamespace(gemini_api_key=SecretStr('secret-never-print'), gemma_model='test-gemma')
        self.clock = SimClock()

    def test_goal_payload_has_both_screens_and_single_attempt(self):
        client = FakeClient([response('{}')])
        constructors = []
        def factory(**kwargs): constructors.append(kwargs); return client
        result = plan_agent_observation(self.settings, self.observation, job_from_prompt('enter the game and get first Pokemon'), [], 120, clock=self.clock, client_factory=factory)
        self.assertEqual(result['text'], '{}')
        self.assertEqual(result['attempts'], 1)
        contents = client.calls[0]['contents']
        self.assertIn('upper DS display ONLY', contents[0])
        self.assertIn('lower DS touchscreen ONLY', contents[2])
        self.assertIn('first Pokemon', contents[4])
        instruction = client.calls[0]['config'].system_instruction
        self.assertIn('Do not stop at New Game/Continue unless', instruction)
        self.assertIn('never TYPE', instruction)
        self.assertEqual(constructors[0]['http_options'].retry_options.attempts, 1)
        self.assertTrue(client.async_closed and client.closed)

    def test_autonomous_transient_failure_never_hides_a_retry(self):
        client = FakeClient([APIError(429), response('{}')])
        with self.assertRaises(PerceptionError) as caught:
            plan_agent_observation(self.settings, self.observation, job_from_prompt('play for 2 minutes'), [], 120, clock=self.clock, client_factory=lambda **kw: client)
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.async_closed and client.closed)

    def test_invalid_attempt_budget_fails_before_client_creation(self):
        for attempts in (0, 3, True):
            with patch('google.genai.Client') as create, self.subTest(attempts=attempts), self.assertRaises(PerceptionError):
                asyncio.run(request_image_text(self.settings, [], 'job', system_instruction='instruction', max_output_tokens=100, max_characters=100, request_prefix='', max_attempts=attempts))
            create.assert_not_called()

    def test_deadline_before_planning_and_late_response_are_discarded(self):
        client = FakeClient([response('{}')])
        with self.assertRaises(TimeoutError):
            plan_agent_observation(self.settings, self.observation, job_from_prompt('play for 2 minutes'), [], 0, clock=self.clock, client_factory=lambda **kw: client)
        self.assertEqual(client.calls, [])
        original = client.generate
        async def late(**kw):
            result = await original(**kw); self.clock.advance(121); return result
        client.aio.models.generate_content = late
        with self.assertRaises(TimeoutError):
            plan_agent_observation(self.settings, self.observation, job_from_prompt('play for 2 minutes'), [], 120, clock=self.clock, client_factory=lambda **kw: client)
        self.assertTrue(client.closed)

if __name__ == '__main__': unittest.main()
