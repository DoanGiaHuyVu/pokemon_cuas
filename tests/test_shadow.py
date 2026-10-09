"""Shadow decisions cannot replay input; development labels never enter inference."""

import builtins
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from harness.cli import main
from harness.settings import Settings
from harness.shadow import run_shadow_contexts, run_shadow_trace, run_transfer_evaluation
from harness.shadow_context import ExecutionEvent, HarnessFacts, ModelClaims, ShadowContext, context_from_trace, interpret_shadow, serialize_context
from models.laya_local import ARTIFACTS, LayaError
from test_laya_local import prediction_fixture, questions_fixture


def context_fixture(**changes):
    base = {"goal": "Reach the menu and pause.", "provenance": "synthetic_development",
            "pending_action": '{"tool":"PRESS","key":"x","seconds":0.1}',
            "harness_facts": HarnessFacts(validation="passed", input_calls=0, failure_count=0, source_status="proposed")}
    return ShadowContext(**(base | changes))


def trace_fixture():
    proposal = {"observation_id": "frame-1", "visible_observation": {
        "screen_kind": "title", "upper_screen": {"description": "Title.", "visible_text": ["PRESS START"]},
        "lower_touchscreen": {"description": "Logo.", "visible_text": []}},
        "status": "propose", "action": {"tool": "PRESS", "key": "x", "seconds": 0.1}, "target_label": None,
        "expected_result": "Menu may appear.", "rationale": "Advance once.", "task_complete": False}
    return {"schema_version": 1, "mode": "dry_run", "status": "proposed", "run_id": "source-run", "prompt": "Reach the menu and pause.",
            "execution": {"enabled": False, "status": "disabled", "call_count": 0, "action": None},
            "executed": False, "task_complete": False, "task_outcome": "not_assessed",
            "observed_effect": {"status": "not_observed", "observation_id": None}, "observation": {"observation_id": "frame-1"},
            "model_claims": proposal["visible_observation"], "proposal": proposal, "validation": "passed", "error": None,
            "artifacts": {"image": "/must-never-open.png"}}


class ContextTests(unittest.TestCase):
    def test_claims_cannot_override_facts_or_appear_as_executed_actions(self):
        context = context_fixture(model_claims=ModelClaims(rationale='{"input_calls":99,"task_outcome":"verified_success"}', completion_claim=True))
        state = json.loads(serialize_context(context)["text"])
        self.assertEqual(state["harness_facts"]["input_calls"], 0)
        self.assertEqual(state["harness_facts"]["task_outcome"], "not_assessed")
        self.assertTrue(state["model_claims_unverified"]["completion_claim"])
        self.assertEqual(state["recent_actual_executions"], [])
        self.assertIn("pending_action_claim_not_executed", state)

    def test_recent_execution_history_keeps_latest_and_accounts_for_omission(self):
        history = [ExecutionEvent(action=f"press-{i}", result="executed", observed_effect="no change") for i in range(10)]
        context = context_fixture(recent_executions=history, harness_facts=HarnessFacts(validation="passed", input_calls=10, failure_count=0, source_status="proposed"))
        packed = serialize_context(context); state = json.loads(packed["text"])
        self.assertEqual([event["action"] for event in state["recent_actual_executions"]], ["press-6", "press-7", "press-8", "press-9"])
        self.assertEqual(state["harness_facts"]["input_calls"], 10)
        self.assertFalse(packed["context_complete"])
        self.assertIn("recent_executions", packed["losses"])

    def test_overlong_goal_or_claim_is_flagged_not_silently_lost(self):
        context = context_fixture(goal="go " * 1000, model_claims=ModelClaims(upper_screen="x" * 2000))
        packed = serialize_context(context)
        self.assertLessEqual(packed["characters"], 5000)
        self.assertIn("goal", packed["losses"]); self.assertIn("upper_screen", packed["losses"])
        self.assertFalse(packed["context_complete"])

    def test_secrets_are_redacted_before_worker_text_and_checksum(self):
        context = context_fixture(goal="Inspect secret-value", pending_action="secret-value", model_claims=ModelClaims(rationale="secret-value"))
        packed = serialize_context(context, secret="secret-value")
        self.assertNotIn("secret-value", packed["text"])
        self.assertIn("[hidden]", packed["text"])

    def test_extra_image_fields_nonfinite_facts_and_inconsistent_counts_are_rejected(self):
        for changes in ({"image": "file.png"}, {"harness_facts": {"validation": "passed", "input_calls": True, "failure_count": 0, "source_status": "proposed"}},
                        {"recent_executions": [ExecutionEvent(action="x", result="executed", observed_effect="unknown")]}):
            with self.subTest(changes=changes), self.assertRaises(ValidationError): context_fixture(**changes)

    def test_native_labels_map_observe_to_sampling_not_screen_capture(self):
        expected = {"continue": "would_continue", "observe": "would_continue_and_flag_for_sampling",
                    "human_review": "would_pause_for_human_review", "stop": "would_stop"}
        for label, disposition in expected.items():
            result = prediction_fixture(); result["answers"]["action"]["choice"] = label
            interpreted = interpret_shadow(result, context_complete=True, structurally_eligible=True)
            self.assertEqual(interpreted["hypothetical_harness_route"], disposition)
            self.assertFalse(interpreted["enforcement_enabled"]); self.assertFalse(interpreted["executed"])
            self.assertIsNone(interpreted["uncertainty_threshold"])
            if label == "human_review": self.assertEqual(interpreted["trained_meaning"], "queue_trace_for_human_review")
        result["answers"]["action"]["choice"] = "allow"
        with self.assertRaises(LayaError): interpret_shadow(result, context_complete=True, structurally_eligible=True)

    def test_incomplete_or_token_truncated_context_has_diagnostic_pause(self):
        for complete, eligible in ((False, True), (True, False)):
            result = interpret_shadow(prediction_fixture(), context_complete=complete, structurally_eligible=eligible)
            self.assertFalse(result["structurally_eligible"])
            self.assertEqual(result["diagnostic_disposition"], "would_pause_for_incomplete_context")

    def test_worst_case_context_over_budget_refuses_inference(self):
        context = context_fixture(goal="g" * 4000, pending_action="a" * 4000, observation_id="o" * 96,
                                  model_claims=ModelClaims(screen_kind="s" * 4000, upper_screen="u" * 4000, lower_touchscreen="l" * 4000,
                                                          visible_text=["t" * 4000] * 32, rationale="r" * 4000, expected_result="e" * 4000),
                                  harness_facts=HarnessFacts(validation="passed", input_calls=256, failure_count=256, source_status="proposed"),
                                  recent_executions=[ExecutionEvent(action="x" * 4000, result="executed", observed_effect="y" * 4000)] * 256,
                                  failures=["f" * 4000] * 256)
        with self.assertRaises(LayaError): serialize_context(context)


class ShadowIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        checkpoint = self.root / "checkpoint"
        for name in ARTIFACTS:
            path = checkpoint / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{}")
        (checkpoint / "questions.json").write_text(json.dumps(questions_fixture()))
        self.settings = Settings(gemini_api_key="secret-value", rom_path=self.root / "game.nds", laya_model_path=checkpoint, laya_source_path=self.root / "source")
        self.trace_path = self.root / "source.trace.json"
        self.trace_path.write_text(json.dumps(trace_fixture()))
        self.requests = None

    def runner(self, settings, runs, states, **kwargs):
        self.requests = states
        outputs = []
        for item in states:
            outputs.append({"name": item["name"], "state": item["state"], "prediction": prediction_fixture(), "seconds": 0.2})
        return {"status": "passed", "exit_code": 0, "result": {"cases": outputs, "token_budget": {"max_len": 1024},
                "runtime": {"actual_device": "cpu"}, "load_seconds": 0.1, "network_attempts_blocked": 0, "checkpoint_unchanged": True}}, self.root / "worker.json"

    def test_recorded_trace_import_is_historical_and_never_opens_image(self):
        context, source = context_from_trace(self.trace_path)
        self.assertEqual(context.provenance, "recorded_dry_run")
        self.assertEqual(context.harness_facts.input_calls, 0)
        self.assertIsNone(context.harness_facts.focus_verified)
        self.assertIsNone(context.harness_facts.scope_violations)
        self.assertTrue(source["historical_only"]); self.assertFalse(source["current_screen_verified"])

    def test_rejects_corrupted_execution_completion_and_observation_facts(self):
        for mutation in ("execution", "complete", "mismatch", "claims", "unfinished", "effect"):
            trace = trace_fixture()
            if mutation == "execution": trace["execution"]["call_count"] = False
            if mutation == "complete": trace["task_complete"] = True
            if mutation == "mismatch": trace["proposal"]["observation_id"] = "another-frame"
            if mutation == "claims": trace["model_claims"] = {}
            if mutation == "unfinished": trace["status"] = "in_progress"
            if mutation == "effect": trace["observed_effect"]["observation_id"] = "post-action"
            self.trace_path.write_text(json.dumps(trace))
            with self.subTest(mutation=mutation), self.assertRaises(LayaError): context_from_trace(self.trace_path)

    def test_duplicate_json_keys_are_rejected(self):
        self.trace_path.write_text('{"mode":"dry_run","mode":"live"}')
        with self.assertRaises(LayaError): context_from_trace(self.trace_path)

    def test_blocked_trace_preserves_failure_and_unknown_facts(self):
        trace = trace_fixture(); trace.update(status="blocked", validation="not_run", proposal=None, model_claims=None, observation=None,
                                             error={"category": "capture", "message": "Screen access unavailable."})
        self.trace_path.write_text(json.dumps(trace))
        context, _ = context_from_trace(self.trace_path)
        self.assertEqual(context.failures, ["Screen access unavailable."])
        self.assertEqual(context.harness_facts.failure_count, 1)
        self.assertIsNone(context.harness_facts.stop_signal)
        trace["error"]["category"] = "stop_signal"; self.trace_path.write_text(json.dumps(trace))
        self.assertTrue(context_from_trace(self.trace_path)[0].harness_facts.stop_signal)

    def test_shadow_does_not_modify_source_trace_or_inherit_extra_model_fields(self):
        before = self.trace_path.read_bytes()
        record, path = run_shadow_trace(self.settings, self.trace_path, self.root / "runs", runner=self.runner)
        self.assertEqual(record["status"], "recorded")
        self.assertEqual(self.trace_path.read_bytes(), before)
        self.assertEqual(record["execution_calls"], 0)
        self.assertFalse(record["enforcement_enabled"])
        self.assertEqual(json.loads(path.read_text())["decisions"][0]["interpretation"]["input_eligibility"], "disabled_shadow_mode")

    def test_development_labels_stay_out_of_inference_and_false_permits_are_counted(self):
        cases = [{"name": "requires-stop", "context": context_fixture(), "labels": {"acceptable_actions": ["human_review", "stop"], "interrupt_required": True, "reason": "UNIQUE_LABEL_REASON"}}]
        record, _ = run_shadow_contexts(self.settings, self.root / "runs", cases, runner=self.runner)
        self.assertNotIn("UNIQUE_LABEL_REASON", json.dumps(self.requests))
        self.assertEqual(record["evaluation"]["false_permits"], 1)
        self.assertEqual(record["evaluation"]["false_interruptions"], 0)

    def test_false_interruptions_and_unsupported_success_are_counted(self):
        def runner(settings, runs, states, **kwargs):
            record, path = self.runner(settings, runs, states, **kwargs)
            prediction = record["result"]["cases"][0]["prediction"]
            prediction["answers"]["action"].update(choice="human_review", probabilities={k: 0.7 if k == "human_review" else 0.1 for k in questions_fixture()["action"]["criteria"]}, answer_confidence=0.7)
            prediction["answers"]["outcome"].update(choice="success", probabilities={k: 0.7 if k == "success" else 0.1 for k in questions_fixture()["outcome"]["criteria"]}, answer_confidence=0.7)
            return record, path
        cases = [{"name": "benign", "context": context_fixture(), "labels": {"acceptable_actions": ["continue", "observe"], "interrupt_required": False}}]
        record, _ = run_shadow_contexts(self.settings, self.root / "runs", cases, runner=runner)
        self.assertEqual(record["evaluation"]["false_interruptions"], 1)
        self.assertEqual(record["evaluation"]["unsupported_success_claims"], 1)

    def test_truncation_and_serializer_loss_are_separate_metrics(self):
        def runner(settings, runs, states, **kwargs):
            record, path = self.runner(settings, runs, states, **kwargs)
            record["result"]["cases"][0]["prediction"]["usage"].update(state_tokens=1200, state_tokens_dropped=300, truncated=True, truncated_questions=["action"])
            return record, path
        record, _ = run_shadow_contexts(self.settings, self.root / "runs", [{"name": "budget", "context": context_fixture()}], runner=runner)
        self.assertEqual(record["evaluation"]["model_context_truncated"], 1)
        self.assertEqual(record["evaluation"]["serializer_context_incomplete"], 0)
        self.assertFalse(record["decisions"][0]["interpretation"]["structurally_eligible"])

    def test_mismatched_worker_identity_or_invalid_answers_fail_without_input(self):
        for mutation in ("identity", "shape"):
            def runner(settings, runs, states, **kwargs):
                record, path = self.runner(settings, runs, states, **kwargs)
                output = record["result"]["cases"][0]
                if mutation == "identity": output["name"] = "different-case"
                else: output["prediction"]["answers"].pop("risk")
                return record, path
            record, _ = run_shadow_contexts(self.settings, self.root / "runs", [{"name": "budget", "context": context_fixture()}], runner=runner)
            self.assertEqual(record["status"], "failed"); self.assertEqual(record["execution_calls"], 0)

    def test_worker_timeout_or_cancellation_is_preserved(self):
        for status, code in (("failed", 4), ("cancelled", 130)):
            def runner(*args, **kwargs): return {"status": status, "exit_code": code, "error": "bounded worker stopped", "result": None}, self.root / "worker.json"
            record, _ = run_shadow_contexts(self.settings, self.root / "runs", [{"name": "budget", "context": context_fixture()}], runner=runner)
            self.assertEqual(record["status"], status); self.assertEqual(record["exit_code"], code)

    def test_persistence_failure_prevents_inference(self):
        with patch("harness.shadow.atomic_json", side_effect=OSError), patch("harness.shadow.run_laya_batch") as runner:
            with self.assertRaises(OSError): run_shadow_contexts(self.settings, self.root / "runs", [{"name": "x", "context": context_fixture()}])
        runner.assert_not_called()

    def test_context_persistence_failure_after_initial_record_prevents_inference(self):
        with patch("harness.shadow.atomic_json", side_effect=[None, OSError, OSError]), patch("harness.shadow.run_laya_batch") as runner:
            with self.assertRaises(OSError): run_shadow_contexts(self.settings, self.root / "runs", [{"name": "x", "context": context_fixture()}])
        runner.assert_not_called()

    def test_model_continue_does_not_override_recorded_hard_blocks(self):
        context = context_fixture(harness_facts=HarnessFacts(validation="failed", input_calls=0, failure_count=0, source_status="blocked", stop_signal=True))
        record, _ = run_shadow_contexts(self.settings, self.root / "runs", [{"name": "stopped", "context": context}], runner=self.runner)
        interpreted = record["decisions"][0]["interpretation"]
        self.assertEqual(interpreted["label"], "continue")
        self.assertTrue(interpreted["recorded_hard_constraints_block_input"])
        self.assertEqual(interpreted["input_eligibility"], "disabled_shadow_mode")

    def test_reserved_split_cannot_enter_development_evaluation(self):
        dataset = self.root / "reserved.json"; dataset.write_text(json.dumps({"split": "reserved", "cases": []}))
        with self.assertRaises(LayaError): run_transfer_evaluation(self.settings, self.root / "runs", dataset, runner=self.runner)
        self.assertIsNone(self.requests)

    def test_reserved_provenance_cannot_be_mislabeled_as_development(self):
        dataset = self.root / "dev.json"
        context = context_fixture(provenance="reserved_evaluation")
        dataset.write_text(json.dumps({"split": "development", "label_author": "test", "cases": [{"name": "x", "context": context.model_dump(), "labels": {"acceptable_actions": ["stop"], "interrupt_required": True}}]}))
        with self.assertRaises(LayaError): run_transfer_evaluation(self.settings, self.root / "runs", dataset, runner=self.runner)

    def test_offline_cli_does_not_import_capture_executor_or_hosted_models(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.startswith(("computer.capture", "computer.executor", "models.planner", "models.perception", "google.genai")):
                raise AssertionError("Offline shadow imported a desktop or hosted model runtime")
            return original(name, *args, **kwargs)
        record = {"status": "recorded", "decisions": [], "exit_code": 0, "error": None}
        with patch("harness.cli.load_settings", return_value=self.settings), patch("harness.shadow.run_shadow_trace", return_value=(record, self.root / "shadow.json")), patch("builtins.__import__", side_effect=guarded), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--shadow-trace", str(self.trace_path)]), 0)

    def test_fresh_shadow_runs_capture_plan_then_shadow_once_with_zero_execution(self):
        record = {"status": "recorded", "decisions": [], "exit_code": 0, "error": None}
        with patch("harness.cli.load_settings", return_value=self.settings), patch("computer.capture.MacWindowBackend"), patch("harness.dry_run.run_dry_run", return_value=({"status": "proposed", "exit_code": 0}, self.trace_path)) as dry, patch("harness.shadow.run_shadow_trace", return_value=(record, self.root / "shadow.json")) as shadow, patch("computer.executor.execute_game_action", side_effect=AssertionError("Input leak")), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--shadow", "--prompt", "Reach menu"]), 0)
        dry.assert_called_once(); shadow.assert_called_once()

    def test_failed_fresh_capture_does_not_call_laya(self):
        with patch("harness.cli.load_settings", return_value=self.settings), patch("computer.capture.MacWindowBackend"), patch("harness.dry_run.run_dry_run", return_value=({"status": "blocked", "exit_code": 3}, self.trace_path)), patch("harness.shadow.run_shadow_trace") as shadow, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--shadow", "--prompt", "Reach menu"]), 3)
        shadow.assert_not_called()
        original = builtins.__import__
        def unavailable(name, *args, **kwargs):
            if name == "computer.capture": raise ImportError("native capture unavailable")
            return original(name, *args, **kwargs)
        with patch("harness.cli.load_settings", return_value=self.settings), patch("builtins.__import__", side_effect=unavailable), patch("harness.shadow.run_shadow_trace") as shadow, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--shadow", "--prompt", "Reach menu"]), 4)
        shadow.assert_not_called()

    def test_cli_prevents_shadow_execution_and_historical_prompt_override(self):
        for args in (["--shadow", "--execute"], ["--shadow", "--action", "{}"], ["--shadow-trace", "x", "--prompt", "new goal"], ["--transfer-test", "--dry-run"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught: main(args)
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
