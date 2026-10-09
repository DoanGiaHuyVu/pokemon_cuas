"""Offline text checkpoint checks. No vision, planner, or executor imports."""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from harness.journal import atomic_json


QUESTION_IDS = ("action", "needs_review", "outcome", "risk", "urgency")
CHOICES = {
    "action": {"continue", "human_review", "observe", "stop"},
    "outcome": {"failure", "harmful", "partial", "success"},
}
ARTIFACTS = (
    "model.safetensors", "rl_agent_config.json", "questions.json",
    "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
)


class LayaError(Exception):
    """A local checkpoint or typed response failed validation."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_questions(checkpoint: Path) -> dict:
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise LayaError("An existing absolute local checkpoint directory is required.")
    for name in ARTIFACTS:
        if not (checkpoint / name).is_file():
            raise LayaError(f"Local checkpoint is missing {name}; downloads are disabled.")
    questions = json.loads((checkpoint / "questions.json").read_text())
    if not isinstance(questions, dict) or not all(key in questions for key in QUESTION_IDS):
        raise LayaError("Checkpoint lacks the five agent-trace questions.")
    selected = {key: questions[key] for key in QUESTION_IDS}
    for key, question in selected.items():
        if not isinstance(question, dict) or not isinstance(question.get("instructions"), str):
            raise LayaError("Invalid checkpoint question definition.")
        expected = "choice" if key in CHOICES else "noul" if key == "needs_review" else "score"
        if question.get("type") != expected:
            raise LayaError("Unexpected trained question type.")
        criteria = question.get("criteria")
        if expected in {"choice", "noul"}:
            labels = CHOICES[key] if expected == "choice" else {"false", "true"}
            if not isinstance(criteria, dict) or set(criteria) != labels:
                raise LayaError("Unexpected trained question labels.")
            values = list(criteria.values())
        else:
            if not isinstance(criteria, list) or len(criteria) != 4:
                raise LayaError("Expected four trained score levels.")
            values = criteria
        if not all(isinstance(value, str) and value for value in values):
            raise LayaError("Question criteria must contain text.")
    return selected


def probability(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise LayaError("Expected a finite probability between zero and one.")
    return float(value)


def validate_prediction(result: dict, questions: dict, max_len: int) -> dict:
    """Preserve native answer semantics; eligibility here is structural only."""
    if not isinstance(result, dict) or set(result) != {"model", "answers", "usage"}:
        raise LayaError("Unexpected Laya response envelope.")
    answers, usage = result["answers"], result["usage"]
    if not isinstance(result["model"], str) or not isinstance(answers, dict) or set(answers) != set(QUESTION_IDS):
        raise LayaError("Missing or unexpected typed answers.")
    labels = {}
    for key, answer in answers.items():
        kind = questions[key]["type"]
        common = {"type", "confidence", "answer_confidence", "action"}
        extra = {"choice", "probabilities"} if kind == "choice" else {"noul"} if kind == "noul" else {"score", "legend", "probabilities"}
        if not isinstance(answer, dict) or set(answer) != common | extra or answer["type"] != kind:
            raise LayaError("Unexpected typed answer structure.")
        probability(answer["confidence"])
        probability(answer["answer_confidence"])
        if not isinstance(answer["action"], dict) or set(answer["action"]) != {"act_probability"}:
            raise LayaError("Unexpected auxiliary act head.")
        probability(answer["action"]["act_probability"])
        if kind == "noul":
            p_true = probability(answer["noul"])
            labels[key] = "true" if p_true >= 0.5 else "false"
            continue
        expected = CHOICES[key] if kind == "choice" else {str(i) for i in range(4)}
        probs = answer["probabilities"]
        if not isinstance(probs, dict) or set(probs) != expected:
            raise LayaError("Unexpected probability labels.")
        for value in probs.values():
            probability(value)
        if abs(sum(probs.values()) - 1) > 0.001:
            raise LayaError("Probabilities do not sum to one.")
        modal = max(probs, key=probs.get)
        labels[key] = modal
        if kind == "choice":
            if not isinstance(answer["choice"], str) or answer["choice"] not in expected or probs[answer["choice"]] < max(probs.values()) - 0.0001:
                raise LayaError("Choice disagrees with the probabilities.")
            labels[key] = answer["choice"]
        else:
            score = answer["score"]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 3:
                raise LayaError("Invalid ordinal expected score.")
            if abs(score - sum(int(i) * p for i, p in probs.items())) > 0.001:
                raise LayaError("Score disagrees with the probabilities.")
            if answer["legend"] != {str(i): value for i, value in enumerate(questions[key]["criteria"])}:
                raise LayaError("Score legend differs from the checkpoint.")
    required = {"input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped", "truncated", "truncated_questions"}
    if not isinstance(usage, dict) or not required <= set(usage):
        raise LayaError("Missing token-budget accounting.")
    for key in ("input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped"):
        if type(usage[key]) is not int or usage[key] < 0:
            raise LayaError("Invalid token counts.")
    if usage["input_tokens"] > max_len * len(QUESTION_IDS) or usage["output_tokens"] != 0:
        raise LayaError("Inference exceeded the configured token budget.")
    truncated = usage["truncated"]
    truncated_questions = usage["truncated_questions"]
    if type(truncated) is not bool or not isinstance(truncated_questions, list) or any(key not in QUESTION_IDS for key in truncated_questions):
        raise LayaError("Invalid truncation accounting.")
    if truncated != (usage["state_tokens_dropped"] > 0) or truncated != bool(truncated_questions):
        raise LayaError("Inconsistent truncation accounting.")
    if usage["state_tokens_dropped"] > usage["state_tokens"]:
        raise LayaError("Dropped token count exceeds the original state.")
    return {"modal_labels": labels, "structurally_eligible": not truncated and not usage.get("options"),
            "semantic_guardrail_validated": False, "noul_label_rule": "P(true) >= 0.5; reporting only"}


def offline_environment() -> dict[str, str]:
    # Allowlist the process essentials; never pass API/HF credentials to the worker.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT") if key in os.environ}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               HF_HUB_DISABLE_IMPLICIT_TOKEN="1", USE_TF="0", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    return env


def run_laya_test(settings, runs: Path, *, prompt: str | None = None, device="auto", timeout=240,
                  _states: list[dict] | None = None) -> tuple[dict, Path]:
    """Run a bounded isolated offline check; terminal status is durably recorded."""
    if device not in {"auto", "mps", "cpu"}:
        raise LayaError("Device must be auto, mps, or cpu.")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000):
        raise LayaError("Text input must contain 1–4000 characters.")
    if _states is not None and (not isinstance(_states, list) or not 1 <= len(_states) <= 32 or any(
        not isinstance(item, dict) or set(item) != {"name", "state"} or not isinstance(item["name"], str)
        or not 1 <= len(item["name"]) <= 96 or not isinstance(item["state"], str)
        or not 1 <= len(item["state"]) <= 5000 for item in _states
    )):
        raise LayaError("Shadow inference requires 1–32 bounded text cases.")
    run_id = uuid.uuid4().hex
    path = runs / f"{run_id}.{'laya' if _states is None else 'laya-shadow-worker'}.json"
    workspace = runs / "laya-runtime" / run_id
    workspace.mkdir(parents=True)
    request_path, result_path = workspace / "request.json", workspace / "result.json"
    secret = settings.gemini_api_key.get_secret_value()
    record = {"schema_version": 1, "step": 7 if _states is None else 8, "run_id": run_id, "status": "in_progress",
              "device_requested": device, "checkpoint": str(settings.laya_model_path),
              "source": str(settings.laya_source_path), "offline": True, "screen_captures": 0,
              "api_calls": 0, "execution_calls": 0, "guardrail_enabled": False,
              "result": None, "error": None, "exit_code": None}
    atomic_json(path, record, secret=secret)
    start, process = time.perf_counter(), None
    try:
        load_questions(settings.laya_model_path)
        request = {"checkpoint": str(settings.laya_model_path), "source": str(settings.laya_source_path),
                   "device": device, "prompt": prompt}
        if _states is not None:
            request.update(mode="shadow", states=_states)
        atomic_json(request_path, request, secret=secret)
        process = subprocess.Popen([sys.executable, "-m", "models.laya_worker", str(request_path), str(result_path)],
                                   cwd=Path(__file__).resolve().parent.parent, env=offline_environment(),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        process.wait(timeout=timeout)
        if not result_path.is_file():
            raise LayaError("Worker exited without a result; inspect local runtime compatibility.")
        result = json.loads(result_path.read_text())
        record["result"] = result
        if process.returncode != 0 or result.get("status") != "passed":
            raise LayaError("Offline checkpoint test failed; see the result stage and error class.")
        record.update(status="passed", exit_code=0)
    except KeyboardInterrupt:
        record.update(status="cancelled", exit_code=130, error="Local inference cancelled.")
    except subprocess.TimeoutExpired:
        record.update(status="failed", exit_code=4, error="Offline checkpoint test exceeded its time limit.")
    except Exception as exc:
        record.update(status="failed", exit_code=4,
                      error=str(exc) if isinstance(exc, LayaError) else f"Local checkpoint error ({type(exc).__name__}).")
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if record["result"] is None and result_path.is_file():
            try:
                record["result"] = json.loads(result_path.read_text())
            except (OSError, ValueError):
                pass
        record["elapsed_seconds"] = round(time.perf_counter() - start, 4)
        atomic_json(path, record, secret=secret)
    return record, path


def run_laya_batch(settings, runs: Path, states: list[dict], *, device="auto", timeout=240):
    """Same isolated offline runtime, without Step 7's unrelated baseline suite."""
    return run_laya_test(settings, runs, device=device, timeout=timeout, _states=states)
