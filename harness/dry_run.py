"""Single-turn capture → plan → validate → durable trace. No execution dependency."""

import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from computer.capture import CaptureError, capture_observation, utc_now
from computer.geometry import Calibration, GeometryError
from harness.journal import atomic_json
from harness.proposals import ProposalError, parse_proposal
from models.perception import MAX_ATTEMPTS, TIMEOUT_SECONDS, PerceptionError
from models.planner import MAX_OUTPUT_TOKENS, SYSTEM_INSTRUCTION, propose_observation


class TraceWriteError(Exception):
    def __init__(self, message, path):
        super().__init__(message)
        self.path = path


class StopRequested(Exception):
    pass


class Trace:
    """Only this harness sets execution and outcome facts; model data stays nested."""

    def __init__(self, settings, prompt, runs, profile):
        self.secret = settings.gemini_api_key.get_secret_value()
        self.started = time.monotonic()
        identifier = uuid4().hex
        self.path = runs / (identifier + '.trace.json')
        self.data = {
            "schema_version": 1, "run_id": identifier, "mode": "dry_run",
            "prompt": prompt.replace(self.secret, "[hidden]"),
            "started_at": utc_now(), "updated_at": None, "finished_at": None,
            "status": "in_progress", "active_stage": "initialization", "exit_code": None,
            "artifacts": {"trace": str(self.path), "calibration": str(profile), "image": None, "observation": None, "proposal": None},
            "observation": None, "model_claims": None, "proposal": None,
            "validation": "not_run", "expected_effect": None,
            "observed_effect": {"status": "not_observed", "observation_id": None, "description": None,
                                "reason": "Dry run sends no input and captures no post-action frame."},
            "execution": {"enabled": False, "status": "disabled", "call_count": 0, "action": None},
            "executed": False, "task_complete": False, "task_outcome": "not_assessed",
            "laya": {"called": False, "mode": "not_connected", "checkpoint": str(settings.laya_model_path)},
            "model": {
                "provider": "Google Gemini API", "name": settings.gemma_model,
                "endpoint": "https://generativelanguage.googleapis.com/v1beta",
                "temperature": 0, "thinking_level": "minimal", "max_output_tokens": MAX_OUTPUT_TOKENS,
                "timeout_seconds_per_attempt": TIMEOUT_SECONDS, "max_attempts": MAX_ATTEMPTS,
                "sdk_max_attempts_per_call": 1, "automatic_function_calling": False,
                "schema_enforcement": "local", "system_instruction_sha256": hashlib.sha256(SYSTEM_INSTRUCTION.encode()).hexdigest(),
                "image_input_strategy": "two_labeled_screen_crops_from_one_observation",
                "called": False, "attempts": 0, "reported_latency_seconds": None, "usage": None,
            },
            "latency_seconds": {"capture": None, "planning_and_validation": None, "total": 0},
            "error": None, "events": [],
        }
        self.checkpoint("run_started")

    def checkpoint(self, event):
        self.data["updated_at"] = utc_now()
        self.data["latency_seconds"]["total"] = round(time.monotonic() - self.started, 3)
        self.data["events"].append({"sequence": len(self.data["events"]) + 1, "at": self.data["updated_at"],
                                    "event": event, "stage": self.data["active_stage"]})
        try:
            atomic_json(self.path, self.data, secret=self.secret)
        except (OSError, ValueError) as exc:
            raise TraceWriteError("Could not persist the dry-run trace; work stopped. The previous complete snapshot may remain.", self.path) from exc

    def finish(self, status, exit_code, error=None):
        self.data.update(status=status, exit_code=exit_code, finished_at=utc_now(), error=error)
        self.checkpoint("run_finished")
        return self.data, self.path


def _stop(path):
    if path.exists():
        raise StopRequested("Stop signal is set; start a new request after explicitly clearing it.")


def _model_metrics(trace, record):
    # Never merge a model/adapter record into harness facts.
    attempts = record.get("attempts")
    if type(attempts) is int and 0 <= attempts <= MAX_ATTEMPTS:
        trace.data["model"].update(attempts=attempts, called=attempts > 0)
    else:
        trace.data["model"].update(attempts=None, called=None)
    trace.data["model"].update(reported_latency_seconds=record.get("latency_seconds"), usage=record.get("usage"))


def run_dry_run(settings, prompt, backend, profile: Path, images: Path, runs: Path, *, client_factory=None):
    """Always fresh capture; no executor parameter, automatic loop, or replay path."""
    trace = Trace(settings, prompt, runs, profile)
    stop_file = runs / "STOP"
    proposal_path = None
    stage_started = None
    try:
        _stop(stop_file)
        trace.data["active_stage"] = "capture"
        trace.checkpoint("capture_started")
        stage_started = time.monotonic()
        observation = capture_observation(backend, profile, images, runs, trace.data["prompt"])
        trace.data["latency_seconds"]["capture"] = round(time.monotonic() - stage_started, 3)
        trace.data["observation"] = {
            "observation_id": observation["observation_id"], "captured_at": observation["captured_at"],
            "source_kind": observation.get("source_kind", "live_window_capture"),
            "image_sha256": hashlib.sha256(Path(observation["image_path"]).read_bytes()).hexdigest(),
            "calibration": observation["calibration"],
        }
        proposal_path = Path(observation["metadata_path"]).with_suffix(".proposal.json")
        trace.data["artifacts"].update(image=observation["image_path"], observation=observation["metadata_path"])
        trace.checkpoint("capture_recorded")
        _stop(stop_file)
        trace.data["active_stage"] = "planning_and_validation"
        # If killed/cancelled before a response is saved, request count is unknown.
        trace.data["model"].update(called=None, attempts=None)
        trace.checkpoint("planning_started")
        stage_started = time.monotonic()
        record, saved_path = propose_observation(settings, observation, client_factory=client_factory)
        trace.data["latency_seconds"]["planning_and_validation"] = round(time.monotonic() - stage_started, 3)
        trace.data["artifacts"]["proposal"] = str(saved_path)
        _model_metrics(trace, record)
        # Validate the raw response at this boundary too. Adapter metadata cannot authorize input.
        parsed = parse_proposal(record["raw_response"], observation["observation_id"],
                                Calibration.model_validate(observation["calibration"]))
        trace.data.update(validation="passed", proposal=parsed.model_dump(mode="json"),
                          model_claims=parsed.visible_observation.model_dump(mode="json"),
                          expected_effect={"source": "model_prediction", "description": parsed.expected_result})
        trace.checkpoint("proposal_validated")
        _stop(stop_file)
        trace.data["active_stage"] = "review"
        return trace.finish("proposed" if parsed.status == "propose" else "paused", 0)
    except TraceWriteError:
        # Never continue capture/planning after a failed durable checkpoint.
        raise
    except (Exception, KeyboardInterrupt) as exc:
        stage = trace.data["active_stage"]
        if stage_started is not None and stage in {"capture", "planning_and_validation"}:
            trace.data["latency_seconds"][stage] = round(time.monotonic() - stage_started, 3)
        if proposal_path is not None and proposal_path.is_file():
            trace.data["artifacts"]["proposal"] = str(proposal_path)
            try:
                failed_record = json.loads(proposal_path.read_text())
                _model_metrics(trace, failed_record)
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        if isinstance(exc, KeyboardInterrupt):
            status, code, category, message = "cancelled", 130, "cancelled", "Request cancelled; no game input executed."
        elif isinstance(exc, StopRequested):
            status, code, category, message = "blocked", 3, "stop_signal", str(exc)
        elif isinstance(exc, (CaptureError, GeometryError)):
            status, code, category, message = "blocked", 3, "capture", str(exc)
        elif isinstance(exc, ProposalError):
            trace.data["validation"] = "failed"
            status, code, category, message = "blocked", 4, "proposal_validation", str(exc)
        elif isinstance(exc, PerceptionError):
            trace.data["model"].update(attempts=exc.attempts, called=exc.attempts > 0)
            status, code, category, message = "blocked", 4, "model_request", str(exc)
        else:
            status, code, category, message = "failed", 3, "internal_error", "Dry-run system or artifact error; no input executed."
        return trace.finish(status, code, {"stage": stage, "category": category, "exception_type": type(exc).__name__,
                                          "message": message.replace(trace.secret, "[hidden]")})
