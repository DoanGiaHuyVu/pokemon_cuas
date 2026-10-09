"""Offline model boundaries and typed-output validation; no real weights in unit tests."""

import builtins
import contextlib
import io
import json
import subprocess
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness.cli import main
from harness.journal import atomic_json
from harness.settings import Settings
from models.laya_local import ARTIFACTS, CHOICES, LayaError, load_questions, offline_environment, run_laya_test, validate_prediction
from models.laya_worker import main as worker_main, original_examples


def questions_fixture():
    return {key: {"type": "choice", "instructions": key, "criteria": {label: label for label in sorted(labels)}}
            for key, labels in CHOICES.items()} | {
        "needs_review": {"type": "noul", "instructions": "Review?", "criteria": {"false": "No", "true": "Yes"}},
        "risk": {"type": "score", "instructions": "Risk?", "criteria": ["zero", "one", "two", "three"]},
        "urgency": {"type": "score", "instructions": "Urgency?", "criteria": ["zero", "one", "two", "three"]},
    }


def prediction_fixture():
    answers = {}
    for key, q in questions_fixture().items():
        answer = {"type": q["type"], "confidence": 0.2, "answer_confidence": 0.25, "action": {"act_probability": 0.8}}
        if q["type"] == "choice":
            answer.update(choice=next(iter(q["criteria"])), probabilities={k: 0.25 for k in q["criteria"]})
        elif q["type"] == "noul":
            answer.update(noul=0.3, confidence=0.7, answer_confidence=0.7)
        else:
            answer.update(score=1.5, legend={str(i): v for i, v in enumerate(q["criteria"])},
                          probabilities={str(i): 0.25 for i in range(4)})
        answers[key] = answer
    return {"model": "laya-rl-agent", "answers": answers,
            "usage": {"input_tokens": 500, "output_tokens": 0, "state_tokens": 20,
                      "state_tokens_dropped": 0, "truncated": False, "truncated_questions": []}}


class TypedLayaTests(unittest.TestCase):
    def validate(self, result):
        return validate_prediction(result, questions_fixture(), 1024)

    def test_preserves_choice_boolean_probability_and_expected_score_semantics(self):
        result = prediction_fixture()
        checked = self.validate(result)
        self.assertEqual(checked["modal_labels"]["needs_review"], "false")
        self.assertEqual(result["answers"]["risk"]["score"], 1.5)
        self.assertTrue(checked["structurally_eligible"])
        self.assertFalse(checked["semantic_guardrail_validated"])
        self.assertNotEqual(result["answers"]["action"]["confidence"], result["answers"]["action"]["answer_confidence"])

    def test_rejects_missing_unknown_answers_and_extra_fields(self):
        variants = []
        missing = prediction_fixture(); missing["answers"].pop("risk"); variants.append(missing)
        extra = prediction_fixture(); extra["answers"]["screen"] = {}; variants.append(extra)
        field = prediction_fixture(); field["answers"]["action"]["execute"] = True; variants.append(field)
        wrong = prediction_fixture(); wrong["answers"]["risk"]["type"] = "choice"; variants.append(wrong)
        for result in variants:
            with self.subTest(result=result), self.assertRaises(LayaError):
                self.validate(result)

    def test_rejects_nonfinite_out_of_range_bool_and_string_probabilities(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            for field in ("confidence", "answer_confidence", "noul"):
                result = prediction_fixture()
                result["answers"]["needs_review"][field] = value
                with self.subTest(value=value, field=field), self.assertRaises(LayaError):
                    self.validate(result)

    def test_rejects_bad_labels_sum_and_choice_disagreement(self):
        for mutation in ("label", "prob_label", "sum", "choice", "aux"):
            result = prediction_fixture(); a = result["answers"]["action"]
            if mutation == "label": a["choice"] = "allow"
            if mutation == "prob_label": a["probabilities"]["allow"] = a["probabilities"].pop("stop")
            if mutation == "sum": a["probabilities"]["stop"] = 1
            if mutation == "choice": a["probabilities"] = {key: float(key == "stop") for key in CHOICES["action"]}
            if mutation == "aux": a["action"]["act_probability"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(LayaError): self.validate(result)

    def test_rejects_inconsistent_score_or_checkpoint_legend(self):
        for value in (float("nan"), True, 4, -1, "1.5", 2):
            result = prediction_fixture(); result["answers"]["risk"]["score"] = value
            with self.subTest(value=value), self.assertRaises(LayaError): self.validate(result)
        result = prediction_fixture(); result["answers"]["risk"]["legend"]["0"] = "High risk"
        with self.assertRaises(LayaError): self.validate(result)

    def test_token_overflow_and_collapsed_options_are_ineligible(self):
        result = prediction_fixture()
        result["usage"].update(truncated=True, state_tokens_dropped=1, truncated_questions=["risk"])
        self.assertFalse(self.validate(result)["structurally_eligible"])
        result = prediction_fixture(); result["usage"]["options"] = {"risk": {"total": 4, "distinct": 2}}
        self.assertFalse(self.validate(result)["structurally_eligible"])

    def test_rejects_inconsistent_or_excessive_token_accounting(self):
        for change in ({"input_tokens": 5121}, {"output_tokens": 1}, {"state_tokens_dropped": 1},
                       {"truncated": True}, {"input_tokens": True}, {"truncated_questions": ["unknown"]}):
            result = prediction_fixture(); result["usage"].update(change)
            with self.subTest(change=change), self.assertRaises(LayaError): self.validate(result)


class OfflineRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.checkpoint = self.root / "checkpoint"
        for name in ARTIFACTS:
            path = self.checkpoint / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
        (self.checkpoint / "questions.json").write_text(json.dumps(questions_fixture()))
        self.settings = Settings(gemini_api_key="never-print-me", rom_path=self.root / "game.nds",
                                 laya_model_path=self.checkpoint, laya_source_path=self.root / "source")

    def test_requires_complete_local_checkpoint_never_remote_name(self):
        with self.assertRaises(LayaError): load_questions(Path("convai/model"))
        (self.checkpoint / "tokenizer/tokenizer.json").unlink()
        with patch("models.laya_local.subprocess.Popen") as spawn:
            record, path = run_laya_test(self.settings, self.root / "runs")
        spawn.assert_not_called()
        self.assertEqual(record["status"], "failed")
        self.assertIn("downloads are disabled", record["error"])
        self.assertEqual(json.loads(path.read_text())["execution_calls"], 0)

    def test_checkpoint_question_types_and_labels_are_verified(self):
        questions = questions_fixture(); questions["action"]["criteria"]["allow"] = questions["action"]["criteria"].pop("continue")
        (self.checkpoint / "questions.json").write_text(json.dumps(questions))
        with self.assertRaises(LayaError): load_questions(self.checkpoint)

    def test_offline_environment_excludes_credentials_and_python_path(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "secret", "HF_TOKEN": "token", "PYTHONPATH": "untrusted"}):
            env = offline_environment()
        for key in ("GEMINI_API_KEY", "HF_TOKEN", "PYTHONPATH"): self.assertNotIn(key, env)
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def fake_process(self, command, **kwargs):
        request_path, result_path = map(Path, command[-2:])
        self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(json.loads(request_path.read_text())["prompt"], "inspect [hidden]")
        atomic_json(result_path, {"status": "passed", "stage": "complete"})
        return Mock(returncode=0, poll=Mock(return_value=0))

    def test_isolated_success_records_zero_external_effects_and_redacts_text(self):
        with patch("models.laya_local.subprocess.Popen", side_effect=self.fake_process):
            record, path = run_laya_test(self.settings, self.root / "runs", prompt="inspect never-print-me")
        self.assertEqual(record["status"], "passed")
        self.assertEqual((record["api_calls"], record["screen_captures"], record["execution_calls"]), (0, 0, 0))
        self.assertFalse(record["guardrail_enabled"])
        self.assertNotIn("never-print-me", path.read_text())

    def test_timeout_terminates_worker_and_preserves_partial_progress(self):
        def spawn(command, **kwargs):
            atomic_json(Path(command[-1]), {"status": "in_progress", "stage": "load"})
            return process
        process = Mock(returncode=None, poll=Mock(return_value=None))
        process.wait.side_effect = [subprocess.TimeoutExpired("local-worker", 1), None]
        with patch("models.laya_local.subprocess.Popen", side_effect=spawn):
            record, path = run_laya_test(self.settings, self.root / "runs", timeout=1)
        process.terminate.assert_called_once()
        self.assertEqual(record["result"]["stage"], "load")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(json.loads(path.read_text())["status"], "failed")

    def test_cancellation_terminates_worker_and_records_no_input(self):
        process = Mock(returncode=None, poll=Mock(return_value=None))
        process.wait.side_effect = [KeyboardInterrupt, None]
        with patch("models.laya_local.subprocess.Popen", return_value=process):
            record, _ = run_laya_test(self.settings, self.root / "runs")
        process.terminate.assert_called_once()
        self.assertEqual(record["exit_code"], 130)
        self.assertEqual(record["execution_calls"], 0)

    def test_initial_persistence_failure_prevents_worker_start(self):
        with patch("models.laya_local.atomic_json", side_effect=OSError), patch("models.laya_local.subprocess.Popen") as spawn:
            with self.assertRaises(OSError): run_laya_test(self.settings, self.root / "runs")
        spawn.assert_not_called()

    def test_original_examples_match_checkpoint_questions_and_cover_actions(self):
        dataset = self.settings.laya_source_path / "data/typed_decisions/test.jsonl"
        dataset.parent.mkdir(parents=True)
        rows = [{"state": {"task": label}, "questions": questions_fixture(), "gold": {"action": {"label": label}}}
                for label in CHOICES["action"]]
        dataset.write_text("\n".join(json.dumps(row) for row in rows))
        examples, info = original_examples(self.settings.laya_source_path, questions_fixture())
        self.assertEqual(len(examples), 4)
        self.assertFalse(info["independent_holdout_verified"])
        rows[0]["questions"]["action"]["instructions"] = "Different task"
        dataset.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaises(LayaError): original_examples(self.settings.laya_source_path, questions_fixture())

    def test_cli_laya_operation_never_imports_vision_or_executor(self):
        original_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(("computer.", "models.planner", "models.perception", "google.genai")):
                raise AssertionError("Unexpected vision or desktop import")
            return original_import(name, *args, **kwargs)
        record = {"status": "failed", "elapsed_seconds": 0.1, "result": None, "error": "test failure", "exit_code": 4}
        with patch("harness.cli.load_settings", return_value=self.settings), patch("models.laya_local.run_laya_test", return_value=(record, self.root / "record.json")) as run, patch("builtins.__import__", side_effect=guarded), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = main(["--laya-test", "--prompt", "inspect text", "--laya-device", "cpu"])
        self.assertEqual(code, 4)
        self.assertEqual(run.call_args.kwargs["prompt"], "inspect text")

    def test_cli_rejects_laya_execution_and_mixed_operations(self):
        for args in (["--laya-test", "--execute"], ["--laya-test", "--dry-run"],
                     ["--laya-test", "--action", "{}"], ["--check", "--laya-device", "cpu"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(args)
            self.assertEqual(caught.exception.code, 2)

    def test_rejects_image_objects_as_prompt_before_spawning(self):
        with patch("models.laya_local.subprocess.Popen") as spawn:
            with self.assertRaises(LayaError): run_laya_test(self.settings, self.root / "runs", prompt={"image": "screenshot.png"})
        spawn.assert_not_called()

    def test_worker_blocks_socket_connections_even_when_local_loader_attempts_one(self):
        source = self.settings.laya_source_path
        (source / "laya").mkdir(parents=True)
        (source / "laya/__init__.py").touch()
        dataset = source / "data/typed_decisions/test.jsonl"
        dataset.parent.mkdir(parents=True)
        dataset.write_text("\n".join(json.dumps({"state": "text", "questions": questions_fixture(), "gold": {"action": {"label": label}}}) for label in CHOICES["action"]))
        (self.checkpoint / "rl_agent_config.json").write_text(json.dumps({"encoder": "answerdotai/ModernBERT-large", "max_len": 1024, "head_max_len": 256}))
        workspace = self.root / "worker"
        workspace.mkdir()
        request_path, result_path = workspace / "request.json", workspace / "result.json"
        atomic_json(request_path, {"checkpoint": str(self.checkpoint), "source": str(source), "device": "cpu", "prompt": None})
        torch = Mock(); torch.backends.mps.is_available.return_value = False
        laya = Mock(); laya.__file__ = str(source / "laya/__init__.py"); laya.__version__ = "test"
        laya.load.side_effect = lambda *args, **kwargs: socket.create_connection(("example.invalid", 443))
        with patch.dict(sys.modules, {"torch": torch, "laya": laya}), patch("models.laya_worker.importlib.metadata.version", return_value="test"), patch.object(sys, "path", list(sys.path)), patch.dict("os.environ"), patch("socket.socket.connect"), patch("socket.socket.connect_ex"), patch("socket.create_connection") as connect:
            status = worker_main(request_path, result_path)
            connect.assert_not_called()
        result = json.loads(result_path.read_text())
        self.assertEqual(status, 4)
        self.assertEqual(result["network_attempts_blocked"], 1)
        self.assertEqual(result["stage"], "load")


if __name__ == "__main__":
    unittest.main()
