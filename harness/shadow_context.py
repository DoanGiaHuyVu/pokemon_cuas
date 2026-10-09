"""Bounded text context for shadow decisions; claims cannot become execution facts."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from models.laya_local import LayaError, sha256


Text = Annotated[str, Field(min_length=1, max_length=4000)]
Count = Annotated[int, Field(ge=0, le=1000000)]


class ContextBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ExecutionEvent(ContextBase):
    action: Text
    result: Literal["executed", "error"]
    observed_effect: Text


class HarnessFacts(ContextBase):
    validation: Literal["passed", "failed", "not_run"]
    input_calls: Count
    failure_count: Count
    scope_violations: Count | None = None
    stop_signal: bool | None = None
    focus_verified: bool | None = None
    task_outcome: Literal["not_assessed", "verified_success", "verified_failure"] = "not_assessed"
    source_status: Literal["proposed", "paused", "blocked", "failed", "cancelled"]


class ModelClaims(ContextBase):
    screen_kind: Text = "unknown"
    upper_screen: Text = "not available"
    lower_touchscreen: Text = "not available"
    visible_text: Annotated[list[Text], Field(max_length=32)] = Field(default_factory=list)
    rationale: Text = "not available"
    expected_result: Text = "not available"
    completion_claim: bool = False


class ShadowContext(ContextBase):
    goal: Text
    provenance: Literal["recorded_dry_run", "synthetic_development", "reserved_evaluation", "live_agent"]
    autonomy: Literal["shadow", "bounded_autonomous"] = "shadow"
    observation_id: Annotated[str, Field(max_length=96)] | None = None
    pending_action: Text | None = None
    harness_facts: HarnessFacts
    model_claims: ModelClaims = Field(default_factory=ModelClaims)
    recent_executions: Annotated[list[ExecutionEvent], Field(max_length=256)] = Field(default_factory=list)
    failures: Annotated[list[Text], Field(max_length=256)] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_history(self):
        if len(self.recent_executions) > self.harness_facts.input_calls:
            raise ValueError("History cannot contain more input calls than recorded")
        if len(self.failures) > self.harness_facts.failure_count:
            raise ValueError("Failures cannot exceed the recorded count")
        return self


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON value")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def context_from_trace(path: Path) -> tuple[ShadowContext, dict]:
    """Import historical dry-run facts; never load linked images or replay input."""
    try:
        if path.stat().st_size > 256000:
            raise ValueError("Oversized trace")
        trace = _strict_json(path.read_text())
        if type(trace["schema_version"]) is not int or trace["schema_version"] != 1 or trace["mode"] != "dry_run" or trace["status"] not in {"proposed", "paused", "blocked", "failed", "cancelled"}:
            raise ValueError("Only completed dry-run traces are supported")
        execution = trace["execution"]
        if execution != {"enabled": False, "status": "disabled", "call_count": 0, "action": None} or type(execution["call_count"]) is not int or execution["enabled"] is not False:
            raise ValueError("Dry-run execution facts are inconsistent")
        if trace["executed"] is not False or trace["task_complete"] is not False or trace["task_outcome"] != "not_assessed" or trace["observed_effect"]["status"] != "not_observed" or trace["observed_effect"]["observation_id"] is not None:
            raise ValueError("Dry-run outcome facts are inconsistent")
        observed = trace.get("observation")
        observation_id = observed["observation_id"] if observed else None
        claims = ModelClaims()
        pending = None
        proposal = trace.get("proposal")
        if proposal is not None:
            # Use the proposal schema without importing any desktop capture/executor runtime.
            from harness.proposals import Proposal
            parsed = Proposal.model_validate(proposal)
            if parsed.observation_id != observation_id or trace["validation"] != "passed" or trace["model_claims"] != proposal["visible_observation"]:
                raise ValueError("Proposal differs from its recorded observation")
            v = parsed.visible_observation
            claims = ModelClaims(screen_kind=v.screen_kind, upper_screen=v.upper_screen.description,
                                 lower_touchscreen=v.lower_touchscreen.description,
                                 visible_text=v.upper_screen.visible_text + v.lower_touchscreen.visible_text,
                                 rationale=parsed.rationale, expected_result=parsed.expected_result,
                                 completion_claim=parsed.task_complete)
            pending = json.dumps(proposal["action"], separators=(",", ":")) if parsed.action is not None else None
        error = trace.get("error")
        stop_signal = True if error and error.get("category") == "stop_signal" else False if trace["status"] in {"proposed", "paused"} else None
        context = ShadowContext(goal=trace["prompt"], provenance="recorded_dry_run", observation_id=observation_id,
                                pending_action=pending, model_claims=claims,
                                harness_facts=HarnessFacts(validation=trace["validation"], input_calls=0,
                                                          failure_count=1 if error else 0, source_status=trace["status"], stop_signal=stop_signal),
                                failures=[error["message"]] if error else [])
        return context, {"path": str(path.resolve()), "sha256": sha256(path), "run_id": trace["run_id"],
                         "historical_only": True, "current_screen_verified": False}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ValidationError, RecursionError) as exc:
        raise LayaError("Saved trace is missing, excessive, unfinished, or inconsistent; no input executed.") from exc


def serialize_context(context: ShadowContext, *, secret: str | None = None) -> dict:
    """Facts first; bound every text/list, account for every omission, then tokenize locally."""
    losses = []

    def clip(value, limit, field):
        value = value.replace(secret, "[hidden]") if secret else value
        if len(value) > limit:
            losses.append(field)
            return value[:limit] + "[cut]"
        return value

    facts = context.harness_facts.model_dump()
    claims = context.model_claims
    histories = context.recent_executions[-4:]
    failures = context.failures[-4:]
    if len(context.recent_executions) > len(histories): losses.append("recent_executions")
    if len(context.failures) > len(failures): losses.append("failures")
    if len(claims.visible_text) > 8: losses.append("visible_text")
    state = {
        "agent": {"model": "Gemma visual planner", "autonomy": "manual; shadow decisions cannot execute" if context.autonomy == "shadow" else "bounded autonomous job; automatic checking, no human review"},
        "task": clip(context.goal, 600, "goal"),
        "constraints": ["Only the requested melonDS game GUI is in scope.", "Pause at the requested menu." if context.autonomy == "shadow" else "Follow the user's goal and job deadline; do not overwrite existing saves.",
                        "Model screen/expected-result claims are unverified; proposed actions have not executed.",
                        "Hard validation and manual approval cannot be overridden by a model." if context.autonomy == "shadow" else "Hard validation, stop signals, and time limits cannot be overridden. Review requests receive bounded automatic rechecking."],
        "harness_facts": facts,
        "trace_summary": {"steps": facts["input_calls"], "tool_errors": facts["failure_count"],
                          "constraint_violations": facts["scope_violations"]},
        "provenance": context.provenance,
        "history_scope": "Current source run only; other runs and prior game state are unknown.",
        "observation_id": context.observation_id,
        "pending_action_claim_not_executed": clip(context.pending_action, 400, "pending_action") if context.pending_action else None,
        "recent_actual_executions": [{"action": clip(event.action, 100, f"execution_{i}"), "result": event.result,
                                    "observed_effect": clip(event.observed_effect, 100, f"effect_{i}")} for i, event in enumerate(histories)],
        "recent_failures": [clip(value, 140, f"failure_{i}") for i, value in enumerate(failures)],
        "model_claims_unverified": {
            "screen_kind": clip(claims.screen_kind, 40, "screen_kind"),
            "upper_screen": clip(claims.upper_screen, 180, "upper_screen"),
            "lower_touchscreen": clip(claims.lower_touchscreen, 180, "lower_touchscreen"),
            "visible_text": [clip(value, 60, f"visible_text_{i}") for i, value in enumerate(claims.visible_text[:8])],
            "rationale": clip(claims.rationale, 200, "rationale"),
            "expected_result": clip(claims.expected_result, 200, "expected_result"),
            "completion_claim": claims.completion_claim,
        },
    }
    state["serializer_facts"] = {"omitted_or_shortened_fields": list(losses), "context_complete": not losses}
    text = json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(text) > 5000:
        raise LayaError("Bounded context exceeded its character budget; no shadow inference started.")
    return {"text": text, "characters": len(text), "losses": losses, "context_complete": not losses,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "original_execution_events": len(context.recent_executions), "retained_execution_events": len(histories)}


def interpret_shadow(prediction: dict, *, context_complete: bool, structurally_eligible: bool) -> dict:
    answer = prediction["answers"]["action"]
    label = answer["choice"]
    meanings = {"continue": "would_continue", "observe": "would_continue_and_flag_for_sampling",
                "human_review": "would_pause_for_human_review", "stop": "would_stop"}
    native_meanings = {"continue": "let_agent_proceed", "observe": "keep_running_and_flag_for_sampling",
                       "human_review": "queue_trace_for_human_review", "stop": "halt_agent"}
    if label not in meanings:
        raise LayaError("Unknown action label; shadow interpretation refused.")
    sorted_probs = sorted(answer["probabilities"].values(), reverse=True)
    return {"label": label, "trained_meaning": native_meanings[label], "hypothetical_harness_route": meanings[label],
            "human_review_route_is_conservative_harness_mapping": True,
            "serializer_context_complete": context_complete,
            "structurally_eligible": structurally_eligible and context_complete,
            "diagnostic_disposition": meanings[label] if structurally_eligible and context_complete else "would_pause_for_incomplete_context",
            "top_answer_probability": answer["answer_confidence"],
            "top_two_probability_margin": round(sorted_probs[0] - sorted_probs[1], 4),
            "entropy_confidence": answer["confidence"], "uncertainty_threshold": None,
            "enforcement_enabled": False, "executed": False,
            "outcome_is_model_claim": True}
