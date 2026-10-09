"""Step 5 validation and API previews; fake screens/targets are not visual grounding."""

import builtins
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from computer.capture import calibrate, capture_observation, confirm_calibration
from computer.geometry import Calibration
from harness.cli import main
from harness.proposals import ProposalError, parse_proposal
from models.perception import PerceptionError
from models.planner import propose_observation
from test_capture import FakeBackend
from test_perception import APIError, FakeClient, response


def proposal(identifier, *, kind="title", action=None, pause=False):
    return {
        "observation_id": identifier,
        "visible_observation": {
            "screen_kind": kind,
            "upper_screen": {"description": "Synthetic title fixture", "visible_text": ["PRESS START"]},
            "lower_touchscreen": {"description": "Synthetic lower screen", "visible_text": []},
        },
        "status": "pause" if pause else "propose",
        "action": None if pause else (action or {"tool": "PRESS", "key": "x", "seconds": 0.1}),
        "target_label": None,
        "expected_result": "A subsequent screenshot may show a menu.",
        "rationale": "The user reports physical x advances the title screen.",
        "task_complete": False,
    }


class ProposalTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.backend = FakeBackend()
        profile = self.root / "runs/calibration.json"
        calibrate(self.backend, profile, self.root / "screenshots")
        confirm_calibration(self.backend, profile)
        self.observation = capture_observation(self.backend, profile, self.root / "screenshots",
                                               self.root / "runs", "Reach the menu test-secret")
        self.identifier = self.observation["observation_id"]
        self.calibration = Calibration.model_validate(self.observation["calibration"])
        self.value = proposal(self.identifier)
        self.settings = SimpleNamespace(gemini_api_key=SecretStr("test-secret"), gemma_model="gemma-4-26b-a4b-it")
        self.path = Path(self.observation["metadata_path"]).with_suffix(".proposal.json")

    def parse(self, value=None):
        return parse_proposal(json.dumps(value or self.value), self.identifier, self.calibration)

    def run_model(self, client):
        return propose_observation(self.settings, self.observation, client_factory=lambda **kwargs: client)

    def test_title_and_opening_allow_one_short_physical_x(self):
        for kind in ("title", "opening"):
            value = proposal(self.identifier, kind=kind)
            parsed = self.parse(value)
            self.assertEqual(parsed.action.key, "x")
            self.assertEqual(parsed.action.seconds, 0.1)
            self.assertFalse(parsed.task_complete)

    def test_title_rejects_wrong_key_long_hold_wait_and_click(self):
        for action in ({"tool": "PRESS", "key": "enter", "seconds": 0.1},
                       {"tool": "PRESS", "key": "z", "seconds": 0.1},
                       {"tool": "PRESS", "key": "x", "seconds": 0.3},
                       {"tool": "WAIT", "seconds": 0.5},
                       {"tool": "CLICK", "x": 500, "y": 750}):
            value = proposal(self.identifier, action=action)
            if action["tool"] == "CLICK":
                value["target_label"] = "Start"
            with self.subTest(action=action), self.assertRaises(ProposalError):
                self.parse(value)

    def test_menu_pauses_without_claiming_completion(self):
        for label in ("New Game", "Continue"):
            value = proposal(self.identifier, kind="main_menu", pause=True)
            value["visible_observation"]["upper_screen"]["visible_text"] = [label]
            self.assertIsNone(self.parse(value).action)
            value["status"], value["action"] = "propose", {"tool": "PRESS", "key": "x", "seconds": 0.1}
            with self.assertRaises(ProposalError):
                self.parse(value)

    def test_menu_label_conflicts_rejected_even_if_screen_claim_is_opening(self):
        self.value["visible_observation"]["upper_screen"]["visible_text"] = ["  NEW GAME "]
        with self.assertRaises(ProposalError):
            self.parse()

    def test_unknown_pauses_and_transition_wait_is_bounded(self):
        self.assertIsNone(self.parse(proposal(self.identifier, kind="unknown", pause=True)).action)
        with self.assertRaises(ProposalError):
            self.parse(proposal(self.identifier, kind="unknown"))
        self.assertEqual(self.parse(proposal(self.identifier, kind="transition", action={"tool": "WAIT", "seconds": 0.5})).action.seconds, 0.5)
        with self.assertRaises(ProposalError):
            self.parse(proposal(self.identifier, kind="transition", action={"tool": "WAIT", "seconds": 2.1}))

    def test_rejects_malformed_fenced_arrays_multiple_objects_duplicate_and_nonfinite_json(self):
        raw = json.dumps(self.value)
        for text in ("{", "```json\n" + raw + "\n```", "[" + raw + "]", raw + raw,
                     raw.replace('"task_complete": false', '"task_complete": false, "task_complete": false'),
                     raw.replace('"seconds": 0.1', '"seconds": NaN'),
                     raw.replace('"seconds": 0.1', '"seconds": Infinity'), "x" * 12001):
            with self.subTest(text=text[:30]), self.assertRaises(ProposalError):
                parse_proposal(text, self.identifier, self.calibration)

    def test_rejects_extra_missing_and_wrong_types(self):
        variants = []
        for key in self.value:
            changed = copy.deepcopy(self.value)
            del changed[key]
            variants.append(changed)
        for field, value in (("task_complete", 0), ("task_complete", "false"), ("done", False),
                             ("status", "completed"), ("task_complete", True)):
            changed = copy.deepcopy(self.value)
            changed[field] = value
            variants.append(changed)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ProposalError):
                self.parse(value)

    def test_actions_reject_repeats_chords_unsupported_tools_and_durations(self):
        for action in ({"tool": "PRESS", "key": "x", "repeat": 10},
                       {"tool": "PRESS", "key": "ctrl+x"}, {"tool": "SHELL", "command": "ls"},
                       {"tool": "PRESS", "key": "x", "seconds": "0.1"},
                       {"tool": "PRESS", "key": "x", "seconds": True},
                       {"tool": "PRESS", "key": "x", "seconds": 0},
                       [{"tool": "PRESS", "key": "x"}]):
            with self.subTest(action=action), self.assertRaises(ProposalError):
                self.parse(proposal(self.identifier, action=action))

    def test_stale_identifier_and_unreviewed_geometry_rejected(self):
        self.value["observation_id"] = "old-observation"
        with self.assertRaises(ProposalError):
            self.parse()
        with self.assertRaises(ProposalError):
            parse_proposal(json.dumps(proposal(self.identifier)), self.identifier,
                           self.calibration.model_copy(update={"verified": False}))

    def name_click(self):
        value = proposal(self.identifier, kind="name_entry", action={"tool": "CLICK", "x": 500, "y": 750})
        value["visible_observation"]["lower_touchscreen"]["visible_text"] = ["H", "U", "Y"]
        value["target_label"] = "H"
        return value

    def test_synthetic_name_target_coordinates_use_full_crop(self):
        parsed = self.parse(self.name_click())
        self.assertEqual(self.calibration.normalized_to_pixel(parsed.action.x, parsed.action.y), (536, 1205))
        self.assertTrue(self.calibration.is_touch_target(parsed.action.x, parsed.action.y))

    def test_click_rejects_wrong_screen_missing_or_unclaimed_label_and_bool_coordinates(self):
        for updates in ({"y": 250}, {"x": True}, {"x": 1001}):
            value = self.name_click()
            value["action"].update(updates)
            with self.subTest(updates=updates), self.assertRaises(ProposalError):
                self.parse(value)
        for target in (None, "Z"):
            value = self.name_click()
            value["target_label"] = target
            with self.assertRaises(ProposalError):
                self.parse(value)

    def test_type_and_pause_with_action_are_rejected(self):
        with self.assertRaises(ProposalError):
            self.parse(proposal(self.identifier, kind="name_entry", action={"tool": "TYPE", "text": "HUY"}))
        value = proposal(self.identifier, pause=True)
        value["action"] = {"tool": "PRESS", "key": "x"}
        with self.assertRaises(ProposalError):
            self.parse(value)

    def test_model_record_payload_redaction_and_no_provider_schema_assumption(self):
        value = copy.deepcopy(self.value)
        value["rationale"] = "test-secret"
        client = FakeClient([response(json.dumps(value))])
        record, path = self.run_model(client)
        self.assertEqual(record["status"], "proposed")
        self.assertFalse(record["executed"] or record["laya_called"])
        self.assertIn("model_claims", record["grounding"])
        self.assertNotIn("test-secret", path.read_text())
        contents = client.calls[0]["contents"]
        request = json.loads(contents[4].split("\n", 1)[1])
        self.assertEqual(request["observation_id"], self.identifier)
        self.assertEqual(request["full_game_image"], {"width": 1074, "height": 1608})
        self.assertEqual(request["checkpoint_rules"]["opening_or_title_action"],
                         {"tool": "PRESS", "key": "x", "seconds": 0.1})
        self.assertNotIn("test-secret", contents[4])
        for index in (1, 3):
            from PIL import Image
            with Image.open(io.BytesIO(contents[index].inline_data.data)) as image:
                self.assertEqual(image.size, (1074, 804))
        config = client.calls[0]["config"]
        self.assertIsNone(config.response_json_schema)
        self.assertIsNone(config.tools)
        self.assertEqual(config.max_output_tokens, 1200)
        self.assertIn("Required JSON schema", config.system_instruction)
        self.assertTrue(client.closed and client.async_closed)

    def test_invalid_model_json_is_saved_once_and_rejected_without_semantic_repair(self):
        client = FakeClient([response("```json test-secret ```")])
        with self.assertRaises(ProposalError):
            self.run_model(client)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["status"], "rejected")
        self.assertEqual(saved["raw_response"], "```json [hidden] ```")
        self.assertNotIn("proposal", saved)
        self.assertFalse(saved["executed"])
        self.assertEqual(len(client.calls), 1)

    def test_api_and_pre_api_failures_distinguish_requests_without_echoing_secrets(self):
        client = FakeClient([APIError(403)])
        with self.assertRaises(PerceptionError):
            self.run_model(client)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["status"], "api_failed")
        self.assertTrue(saved["model_called"])
        self.observation["calibration"]["verified"] = False
        with self.assertRaises(ProposalError):
            self.run_model(client)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["attempts"], 0)
        self.assertFalse(saved["model_called"])
        self.assertEqual(len(client.calls), 1)

    def test_cancelled_proposal_does_not_claim_zero_remote_requests(self):
        with patch("models.planner.propose_images", new=lambda *args, **kwargs: None), patch(
            "models.planner.asyncio.run", side_effect=KeyboardInterrupt
        ), self.assertRaises(KeyboardInterrupt):
            self.run_model(FakeClient([]))
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["status"], "cancelled")
        self.assertIsNone(saved["model_called"])
        self.assertFalse(saved["executed"])

    def invoke_cli(self, client):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"pyautogui", "laya", "torch", "transformers"} or name == "computer.executor":
                raise AssertionError("Proposal imported executor or local model")
            return original_import(name, *args, **kwargs)
        output, error = io.StringIO(), io.StringIO()
        with patch("harness.cli.PROJECT_ROOT", self.root), patch("harness.cli.load_settings", return_value=self.settings), patch(
            "computer.capture.MacWindowBackend", return_value=self.backend
        ), patch("computer.capture.capture_observation", return_value=self.observation), patch(
            "google.genai.Client", return_value=client
        ), patch("builtins.__import__", side_effect=guarded_import), contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = main(["--propose", "--prompt", "Reach the menu"])
        return status, output.getvalue(), error.getvalue()

    def test_cli_generates_preview_without_importing_executor_or_local_model(self):
        status, output, error = self.invoke_cli(FakeClient([response(json.dumps(self.value))]))
        self.assertEqual(status, 0)
        self.assertEqual(error, "")
        self.assertIn("Proposal only", output)
        self.assertIn('"task_complete": false', output)
        self.assertNotIn("test-secret", output)

    def test_cli_rejected_proposal_reports_record_without_traceback(self):
        status, output, error = self.invoke_cli(FakeClient([response("not json test-secret")]))
        self.assertEqual(status, 4)
        self.assertIn("Proposal record:", output)
        self.assertNotIn("Traceback", error)
        self.assertNotIn("test-secret", output + error)

    def test_cli_cancellation_returns_130_and_records_no_execution(self):
        with patch("models.planner.propose_images", new=lambda *args, **kwargs: None), patch(
            "models.planner.asyncio.run", side_effect=KeyboardInterrupt
        ):
            status, output, error = self.invoke_cli(FakeClient([]))
        self.assertEqual(status, 130)
        self.assertEqual(error, "")
        self.assertIn("no game input executed", output)
        self.assertEqual(json.loads(self.path.read_text())["status"], "cancelled")

    def test_cli_propose_cannot_combine_with_execute_manual_action_or_preflight(self):
        for extra in (["--execute"], ["--action", '{}'], ["--mode", "preflight"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--propose", "--prompt", "Reach the menu", *extra])


if __name__ == "__main__":
    unittest.main()
