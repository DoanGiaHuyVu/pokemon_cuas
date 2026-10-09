"""Perception boundary checks with fake API responses; no real network or input."""

import asyncio
import builtins
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from pydantic import SecretStr

from computer.capture import calibrate, capture_observation, confirm_calibration
from harness.cli import main
from models.perception import PerceptionError, describe_image, perceive_observation, read_game_png, response_description
from test_capture import FakeBackend


def response(text="Upper screen: Pokémon Black. Lower touchscreen: Reshiram.", *, reason="STOP", thought=False):
    part = SimpleNamespace(text=text, thought=thought, function_call=None)
    return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=reason, content=SimpleNamespace(parts=[part]))],
                           usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=20, total_token_count=30))


class APIError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__("test-secret-provider-error")


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []
        self.closed, self.async_closed = False, False
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate), aclose=self.aclose)

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def aclose(self):
        self.async_closed = True

    def close(self):
        self.closed = True


class PerceptionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = FakeBackend()
        self.profile, self.images, self.runs = self.root / "runs/calibration.json", self.root / "screenshots", self.root / "runs"
        calibrate(self.backend, self.profile, self.images)
        confirm_calibration(self.backend, self.profile)
        self.observation = capture_observation(self.backend, self.profile, self.images, self.runs, "Inspect both screens test-secret")
        self.settings = SimpleNamespace(gemini_api_key=SecretStr("test-secret"), gemma_model="gemma-4-26b-a4b-it")
        self.constructor = None
        self.sleeps = []

    def factory(self, client):
        def create(**kwargs):
            self.constructor = kwargs
            return client
        return create

    async def sleep(self, duration):
        self.sleeps.append(duration)

    def invoke(self, client, **kwargs):
        images = [{"rectangle": {"left": 0, "top": top, "width": 10, "height": 10}, "data": b"image-bytes"}
                  for top in (0, 10)]
        return asyncio.run(describe_image(self.settings, images, "Inspect test-secret", client_factory=self.factory(client), sleep=self.sleep, **kwargs))

    def test_actual_payload_preserves_both_screens_original_resolution_and_no_desktop(self):
        client = FakeClient([response()])
        record, path = perceive_observation(self.settings, self.observation, client_factory=self.factory(client))
        contents = client.calls[0]["contents"]
        self.assertIn("upper DS display ONLY", contents[0])
        self.assertIn("lower DS touchscreen ONLY", contents[2])
        self.assertIn('"top": 804', contents[2])
        for index, color in ((1, (255, 0, 0)), (3, (0, 0, 255))):
            with Image.open(io.BytesIO(contents[index].inline_data.data)) as image:
                self.assertEqual(image.size, (1074, 804))
                self.assertEqual(image.getpixel((0, 0)), color)
                self.assertEqual(image.getpixel((1073, 803)), color)
        config = client.calls[0]["config"]
        self.assertIn("TWO separately labeled images", config.system_instruction)
        self.assertIsNone(config.tools)
        self.assertTrue(config.automatic_function_calling.disable)
        self.assertEqual(config.thinking_config.thinking_level, "MINIMAL")
        self.assertEqual(config.max_output_tokens, 600)
        self.assertEqual(self.constructor["http_options"].retry_options.attempts, 1)
        self.assertTrue(self.constructor["http_options"].client_args["http2"])
        self.assertTrue(self.constructor["http_options"].async_client_args["http2"])
        self.assertEqual(self.constructor["http_options"].timeout, 25000)
        self.assertEqual(self.constructor["http_options"].base_url, "https://generativelanguage.googleapis.com")
        self.assertFalse(record["executed"])
        self.assertFalse(record["laya_called"])
        self.assertTrue(record["model_called"])
        self.assertEqual(record["observation_id"], self.observation["observation_id"])
        self.assertEqual(json.loads(path.read_text())["status"], "described")
        self.assertTrue(client.closed and client.async_closed)

    def test_prompt_and_model_output_redact_key(self):
        client = FakeClient([response("Upper: test-secret")])
        record, path = perceive_observation(self.settings, self.observation, client_factory=self.factory(client))
        self.assertNotIn("test-secret", client.calls[0]["contents"][4])
        self.assertNotIn("test-secret", path.read_text())
        self.assertEqual(record["description"], "Upper: [hidden]")

    def test_missing_credentials_send_zero_api_requests(self):
        self.settings.gemini_api_key = SecretStr(" ")
        client = FakeClient([response()])
        with self.assertRaises(PerceptionError) as caught:
            self.invoke(client)
        self.assertEqual(caught.exception.attempts, 0)
        self.assertIsNone(self.constructor)
        self.assertEqual(client.calls, [])

    def test_auth_model_and_bad_request_errors_do_not_retry_or_echo_provider_text(self):
        for code in (400, 401, 403, 404):
            client = FakeClient([APIError(code)])
            with self.subTest(code=code), self.assertRaises(PerceptionError) as caught:
                self.invoke(client)
            self.assertEqual(len(client.calls), 1)
            self.assertNotIn("test-secret", str(caught.exception))
            self.assertTrue(client.closed and client.async_closed)
        self.assertEqual(self.sleeps, [])

    def test_transient_error_retries_once_then_succeeds(self):
        client = FakeClient([APIError(503), response()])
        self.assertEqual(self.invoke(client)["attempts"], 2)
        self.assertEqual(self.sleeps, [1])
        self.assertEqual(len(client.calls), 2)

    def test_persistent_quota_or_server_error_stops_after_two_requests(self):
        for code in (429, 500):
            client = FakeClient([APIError(code), APIError(code)])
            with self.subTest(code=code), self.assertRaises(PerceptionError) as caught:
                self.invoke(client)
            self.assertEqual(caught.exception.attempts, 2)
            self.assertEqual(len(client.calls), 2)
        self.assertEqual(self.sleeps, [1, 1])

    def test_wall_timeout_cancels_each_pending_request_and_is_bounded(self):
        client = FakeClient([])
        cancelled = []
        async def stalled(**kwargs):
            client.calls.append(kwargs)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(True)
        client.aio.models.generate_content = stalled
        with self.assertRaises(PerceptionError) as caught:
            self.invoke(client, timeout_seconds=0.01)
        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(cancelled, [True, True])
        self.assertEqual(self.sleeps, [1])
        self.assertTrue(client.async_closed)

    def test_empty_blocked_truncated_and_function_responses_are_not_success_or_retried(self):
        function_response = response()
        function_response.candidates[0].content.parts[0].function_call = object()
        for outcome in (response(""), response("   "), response("secret thought", thought=True),
                        response(reason="MAX_TOKENS"), response(reason="SAFETY"),
                        SimpleNamespace(candidates=[]), response("x" * 6001), function_response):
            client = FakeClient([outcome])
            with self.subTest(outcome=outcome), self.assertRaises(PerceptionError):
                self.invoke(client)
            self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.sleeps, [])

    def test_thought_parts_are_excluded_from_description(self):
        outcome = response("Hidden reasoning", thought=True)
        outcome.candidates[0].content.parts.append(response("Visible text").candidates[0].content.parts[0])
        self.assertEqual(response_description(outcome), "Visible text")

    def test_unexpected_client_exception_is_sanitized(self):
        client = FakeClient([ValueError("test-secret")])
        with self.assertRaises(PerceptionError) as caught:
            self.invoke(client)
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertEqual(len(client.calls), 1)

    def test_invalid_missing_wrong_size_and_non_png_images_send_no_request(self):
        path = Path(self.observation["image_path"])
        for kind in ("missing", "corrupt", "wrong_size", "jpeg"):
            if kind == "missing":
                path.unlink()
            elif kind == "corrupt":
                path.write_bytes(b"invalid PNG")
            else:
                Image.new("RGB", (10, 10)).save(path, format="JPEG" if kind == "jpeg" else "PNG")
            client = FakeClient([response()])
            with self.subTest(kind=kind), self.assertRaises(PerceptionError):
                perceive_observation(self.settings, self.observation, client_factory=self.factory(client))
            self.assertEqual(client.calls, [])
            saved = json.loads(Path(self.observation["metadata_path"]).with_suffix(".perception.json").read_text())
            self.assertFalse(saved["model_called"])
            self.assertEqual(saved["status"], "failed")

    def test_failed_api_record_does_not_claim_execution_or_description(self):
        client = FakeClient([APIError(403)])
        with self.assertRaises(PerceptionError):
            perceive_observation(self.settings, self.observation, client_factory=self.factory(client))
        saved = json.loads(Path(self.observation["metadata_path"]).with_suffix(".perception.json").read_text())
        self.assertTrue(saved["model_called"])
        self.assertFalse(saved["executed"])
        self.assertNotIn("description", saved)
        self.assertNotIn("test-secret", json.dumps(saved))

    def test_cancelled_perception_records_unknown_request_count_and_no_execution(self):
        with patch("models.perception.describe_image", new=lambda *args, **kwargs: None), patch(
            "models.perception.asyncio.run", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            perceive_observation(self.settings, self.observation)
        saved = json.loads(Path(self.observation["metadata_path"]).with_suffix(".perception.json").read_text())
        self.assertEqual(saved["status"], "cancelled")
        self.assertIsNone(saved["model_called"])
        self.assertIsNone(saved["attempts"])
        self.assertFalse(saved["executed"])

    def test_cli_perception_calls_no_input_or_laya_and_saves_real_fake_api_result(self):
        client = FakeClient([response()])
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"pyautogui", "laya", "torch", "transformers"} or name == "computer.executor":
                raise AssertionError("Perception imported an input/local model runtime")
            return original_import(name, *args, **kwargs)
        output = io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch("harness.cli.load_settings", return_value=self.settings), patch(
            "computer.capture.MacWindowBackend", return_value=self.backend
        ), patch("google.genai.Client", side_effect=self.factory(client)), patch(
            "builtins.__import__", side_effect=guarded_import
        ), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--perceive", "--prompt", "Inspect both DS screens"]), 0)
        self.assertIn("Perception only", output.getvalue())
        self.assertEqual(len(client.calls), 1)

    def test_cli_perception_cannot_be_combined_with_execution_and_reports_errors_cleanly(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--perceive", "--execute", "--prompt", "Inspect"])
        client = FakeClient([APIError(403)])
        output, error = io.StringIO(), io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch("harness.cli.load_settings", return_value=self.settings), patch(
            "computer.capture.MacWindowBackend", return_value=self.backend
        ), patch("google.genai.Client", side_effect=self.factory(client)), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            self.assertEqual(main(["--perceive", "--prompt", "Inspect"]), 4)
        self.assertNotIn("Traceback", error.getvalue())
        self.assertNotIn("test-secret", error.getvalue())
        self.assertIn("Perception record:", output.getvalue())

    def test_cli_cancelled_perception_returns_130_without_traceback_or_input(self):
        output = io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch("harness.cli.load_settings", return_value=self.settings), patch(
            "computer.capture.MacWindowBackend", return_value=self.backend
        ), patch("models.perception.perceive_observation", side_effect=KeyboardInterrupt), contextlib.redirect_stdout(output):
            self.assertEqual(main(["--perceive", "--prompt", "Inspect"]), 130)
        self.assertIn("no game input executed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
