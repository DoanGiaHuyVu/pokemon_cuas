"""Bounded autonomous state machine. Ports supply observation, models, and guarded input."""

import json
import time
from pathlib import Path
from typing import Annotated, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from computer.geometry import Calibration
from harness.agent_job import AgentJob
from harness.agent_proposals import parse_agent_proposal
from harness.actions import Click, Press, Wait
from harness.journal import atomic_json
from harness.shadow_context import ExecutionEvent, HarnessFacts, ModelClaims, ShadowContext, serialize_context
from models.laya_local import load_questions, validate_prediction


class AgentRecordError(Exception):
    """Persistence failure: no further dispatch is allowed."""
    def __init__(self, message, path):
        super().__init__(message)
        self.path = path


class HardStop(Exception):
    pass


class Recoverable(Exception):
    pass


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    observation_id: Annotated[str, Field(min_length=1, max_length=96)]
    captured_monotonic: Annotated[float, Field(ge=0)]
    image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    calibration: Calibration
    source: Literal["live", "simulation"]
    metadata: dict = Field(default_factory=dict)


class GuardResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    status: Literal["clear", "changed", "blocked"]
    current_observation_id: str
    reason: str = ""


class GoalAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    observation_id: str
    verdict: Literal["confirmed", "not_confirmed", "uncertain"]
    evidence: Annotated[str, Field(min_length=1, max_length=1000)]
    source: Literal["vision_model", "simulation"]


class AgentPorts(Protocol):
    """Every call gets the same absolute deadline. No saved-trace replay interface."""
    def capture(self, job, deadline) -> Observation: ...
    def plan(self, observation, job, history, deadline) -> str: ...
    def evaluate(self, text, deadline) -> dict: ...
    def guard(self, observation, deadline) -> GuardResult: ...
    def execute(self, action, observation, action_id, deadline) -> dict: ...
    def assess_goal(self, observation, job, deadline) -> GoalAssessment: ...
    def close(self): ...


def state_key(proposal):
    visible = proposal.visible_observation
    # Descriptions and animation pixels cannot by themselves prove useful progress.
    return json.dumps([visible.screen_kind, visible.upper_screen.visible_text, visible.lower_touchscreen.visible_text], ensure_ascii=False)


def action_cost(action):
    if isinstance(action, (Press, Wait)): return action.seconds
    if isinstance(action, Click): return 0.08
    return 0.02


def run_agent(job: AgentJob, ports: AgentPorts, runs: Path, checkpoint: Path, *, secret=None, clock=time.monotonic,
              simulation=False, stop_file: Path | None = None):
    """Follow a job automatically; review means fresh automatic checking, never a person."""
    start = clock()
    deadline = start + job.seconds
    identifier = uuid4().hex
    path = runs / f"{identifier}.agent.json"
    stop_file = stop_file or runs / "STOP"
    record = {"schema_version": 1, "step": getattr(ports, "step", 9), "run_id": identifier,
              "mode": "simulation" if simulation else "autonomous", "job": job.model_dump(),
              "status": "in_progress", "state": "capture", "reason": None, "events": [],
              "observations": [], "proposals": [], "decisions": [], "executions": [], "goal_assessments": [],
              "model_calls": 0, "execution_attempts": 0, "completed_inputs": 0, "recoveries": 0,
              "task_complete": False, "task_outcome": "not_assessed", "human_review_requests": 0,
              "real_game_input_attempts": 0, "real_model_inference": False if simulation else None,
              "laya_semantic_reliability_validated": False, "sampling_flags": 0, "exit_code": None}
    pending = None
    seen_ids = set()
    geometry = None
    failures = []
    history = []
    no_progress = 0
    consecutive_recoveries = 0
    previous_key = None
    review_recheck = None

    def save(event):
        record["elapsed_seconds"] = round(clock() - start, 4)
        record["events"].append({"sequence": len(record["events"]) + 1, "state": record["state"],
                                 "event": event, "elapsed_seconds": record["elapsed_seconds"]})
        try:
            atomic_json(path, record, secret=secret)
        except (OSError, ValueError) as exc:
            raise AgentRecordError("Agent record could not be persisted; further input stopped.", path) from exc

    def state(value, event=None):
        record["state"] = value
        save(event or value + "_started")

    def check():
        if stop_file.exists(): raise HardStop("persistent_stop")
        if clock() >= deadline: raise HardStop("time_limit")

    def model_call(function, *args):
        check()
        if record["model_calls"] >= job.max_model_calls: raise HardStop("model_call_limit")
        record["model_calls"] += 1
        save("model_call_started")
        result = function(*args, deadline)
        check()  # A late response can never authorize input.
        if not simulation:
            record["real_model_inference"] = True
        return result

    def capture():
        nonlocal geometry
        check()
        observed = ports.capture(job, deadline)
        check()
        if not isinstance(observed, Observation) or observed.source != ("simulation" if simulation else "live"):
            raise HardStop("invalid_observation_source")
        if observed.observation_id in seen_ids or not observed.calibration.verified:
            raise HardStop("reused_observation_or_unverified_geometry")
        if not 0 <= clock() - observed.captured_monotonic <= job.max_observation_age:
            raise Recoverable("stale_observation")
        current_geometry = observed.calibration.model_dump(mode="json")
        if geometry is not None and current_geometry != geometry: raise HardStop("geometry_changed")
        geometry = current_geometry
        seen_ids.add(observed.observation_id)
        record["observations"].append({"id": observed.observation_id, "image_sha256": observed.image_sha256,
                                       "captured_elapsed": observed.captured_monotonic - start,
                                       "source": observed.source, "metadata": observed.metadata})
        save("observation_recorded")
        return observed

    def recover(reason):
        nonlocal pending, consecutive_recoveries
        pending = None
        failures.append(reason)
        state("recover", reason)
        if consecutive_recoveries >= job.max_recoveries: raise HardStop("recovery_limit")
        record["recoveries"] += 1
        consecutive_recoveries += 1
        save("automatic_recapture_scheduled")

    try:
        save("job_started")
        questions = load_questions(checkpoint)
        observed = None
        while True:
            try:
                check()
                if observed is None:
                    state("capture")
                    observed = capture()
                state("plan")
                raw = model_call(ports.plan, observed, job, history[-4:])
                state("validate")
                pending = parse_agent_proposal(raw, observed.observation_id, observed.calibration, job)
                record["proposals"].append(pending.model_dump(mode="json"))
                current_key = state_key(pending)
                if previous_key is not None:
                    no_progress = no_progress + 1 if current_key == previous_key else 0
                    previous_key = None
                if no_progress >= job.max_no_progress: raise HardStop("no_progress_limit")
                save("proposal_validated")
                # A reported goal receives a separate fresh frame and assessment, never the outcome head.
                if job.mode == "objective" and pending.goal_status == "candidate_complete":
                    state("verify_goal")
                    goal_observation = capture()
                    assessment = model_call(ports.assess_goal, goal_observation, job)
                    if not isinstance(assessment, GoalAssessment) or assessment.observation_id != goal_observation.observation_id:
                        raise Recoverable("goal_evidence_identity_mismatch")
                    if assessment.source != ("simulation" if simulation else "vision_model"):
                        raise HardStop("invalid_goal_evidence_source")
                    record["goal_assessments"].append(assessment.model_dump())
                    save("fresh_goal_assessment_recorded")
                    if assessment.verdict == "confirmed":
                        record.update(status="simulation_complete" if simulation else "goal_reported",
                                      reason="fresh_goal_evidence", task_outcome="simulated_goal" if simulation else "visual_model_report")
                        break
                    raise Recoverable("goal_not_confirmed")
                if pending.status == "pause": raise Recoverable("planner_requested_automatic_recovery")
                if record["completed_inputs"] >= job.max_actions: raise HardStop("action_limit")
                state("evaluate")
                visible = pending.visible_observation
                context = ShadowContext(goal=job.prompt, provenance="synthetic_development" if simulation else "live_agent", autonomy="bounded_autonomous",
                    pending_action=pending.action.model_dump_json(), observation_id=observed.observation_id,
                    harness_facts=HarnessFacts(validation="passed", input_calls=record["execution_attempts"],
                                              failure_count=len(failures), scope_violations=0, stop_signal=False,
                                              source_status="proposed"),
                    recent_executions=history[-4:], failures=failures[-4:],
                    model_claims=ModelClaims(screen_kind=visible.screen_kind, upper_screen=visible.upper_screen.description,
                                            lower_touchscreen=visible.lower_touchscreen.description,
                                            visible_text=visible.upper_screen.visible_text + visible.lower_touchscreen.visible_text,
                                            rationale=pending.rationale, expected_result=pending.expected_result))
                packed = serialize_context(context, secret=secret)
                if not packed["context_complete"]: raise Recoverable("context_shortened")
                prediction = model_call(ports.evaluate, packed["text"])
                validated = validate_prediction(prediction, questions, 1024)
                label = prediction["answers"]["action"]["choice"]
                probabilities = sorted(prediction["answers"]["action"]["probabilities"].values(), reverse=True)
                tied = probabilities[0] - probabilities[1] <= 0.0001
                record["decisions"].append({"observation_id": observed.observation_id, "text_sha256": packed["text_sha256"],
                                            "prediction": prediction, "label": label, "route": None})
                decision = record["decisions"][-1]
                save("decision_recorded")
                if not validated["structurally_eligible"] or tied:
                    decision["route"] = "automatic_recovery_uncertain_or_incomplete"
                    raise Recoverable("uncertain_or_incomplete_decision")
                if label == "stop":
                    decision["route"] = "stop"
                    raise HardStop("laya_stop")
                review_identity = getattr(ports, "review_identity", state_key)(pending)
                attempt_key = (review_identity, pending.action.model_dump_json())
                needs_replan = getattr(ports, "review_requires_replan", lambda proposal: True)(pending)
                if label == "human_review" and needs_replan and review_recheck != attempt_key:
                    decision["route"] = "automatic_fresh_replan"
                    review_recheck = attempt_key
                    raise Recoverable("review_label_automatic_check")
                state("pre_execute")
                check()
                if clock() - observed.captured_monotonic > job.max_observation_age: raise Recoverable("stale_pending_action")
                guarded = model_call(ports.guard, observed) if getattr(ports, "guard_uses_model", False) else ports.guard(observed, deadline)
                check()
                if not isinstance(guarded, GuardResult): raise HardStop("invalid_guard_result")
                decision["hard_guard"] = guarded.model_dump()
                save("hard_guard_recorded")
                if guarded.status == "blocked": raise HardStop("hard_guard_blocked")
                if guarded.status == "changed" or guarded.current_observation_id != observed.observation_id:
                    raise Recoverable("new_observation_invalidated_action")
                if label == "human_review":
                    decision["route"] = ("automatic_check_cleared; no_human_required" if needs_replan else
                                         "automatic_fresh_entry_check_cleared; no_human_required")
                elif label == "observe":
                    decision["route"] = "continue_and_flag_for_sampling"
                    record["sampling_flags"] += 1
                else:
                    decision["route"] = "continue"
                if clock() + action_cost(pending.action) + 0.05 >= deadline: raise HardStop("time_limit")
                # Atomically log intent before entry. Delivery may be unknown if the process dies here.
                action_id = f"{identifier}:{len(record['executions']) + 1}"
                entry = {"action_id": action_id, "observation_id": observed.observation_id,
                         "action": pending.action.model_dump(), "status": "dispatch_intent",
                         "may_have_executed": True, "result": None, "post_observation_id": None}
                record["executions"].append(entry)
                state("execute", "dispatch_intent_persisted")
                check()
                record["execution_attempts"] += 1
                entry["dispatch_started_elapsed"] = round(clock() - start, 4)
                if not simulation: record["real_game_input_attempts"] += 1
                try:
                    result = ports.execute(pending.action, observed, action_id, deadline)
                    if not isinstance(result, dict) or result.get("executed") is not True or result.get("action") != entry["action"] or result.get("simulated", False) is not simulation:
                        raise ValueError("Unknown executor result")
                except Exception:
                    entry["status"] = "delivery_unknown_no_retry"
                    save("execution_error_terminal")
                    raise HardStop("execution_uncertain_no_retry")
                entry.update(status="completed", result=result, may_have_executed=False)
                record["completed_inputs"] += 1
                save("input_completed")
                # Clear before observing effects; no retry can redispatch this proposal.
                previous_key = current_key
                pending = None
                review_recheck = None
                consecutive_recoveries = 0
                history.append(ExecutionEvent(action=json.dumps(entry["action"], separators=(",", ":")), result="executed",
                                              observed_effect="Not observed yet; a post-action capture is required."))
                state("verify")
                after = capture()
                entry["post_observation_id"] = after.observation_id
                changed = after.image_sha256 != observed.image_sha256
                entry["observed_effect"] = "pixels_changed_only" if changed else "pixels_unchanged"
                history[-1] = ExecutionEvent(action=json.dumps(entry["action"], separators=(",", ":")), result="executed",
                                              observed_effect=entry["observed_effect"] + "; semantic effect not yet established")
                save("post_action_observation_recorded")
                if hasattr(ports, 'assess_effect'):
                    effect = model_call(ports.assess_effect, observed, after, entry['action'], job)
                    if not isinstance(effect, dict) or effect.get('before_id') != observed.observation_id or effect.get('after_id') != after.observation_id or effect.get('progress') not in {'confirmed', 'unchanged', 'uncertain'}:
                        raise HardStop('invalid_effect_assessment')
                    entry['semantic_effect'] = effect
                    history[-1] = ExecutionEvent(action=json.dumps(entry['action'], separators=(',', ':')), result='executed',
                        observed_effect=effect['progress'] + ': ' + effect['evidence'])
                    if effect['progress'] == 'confirmed':
                        previous_key = None
                        no_progress = 0
                    save('post_action_effect_assessed')
                observed = after
            except Recoverable as exc:
                recover(str(exc))
                observed = None
            except (HardStop, AgentRecordError, KeyboardInterrupt):
                raise
            except Exception as exc:
                # Model/capture failures can replan; input errors are caught above and are terminal.
                review_recheck = None
                recover("stage_failure_" + type(exc).__name__)
                observed = None
    except HardStop as exc:
        reason = str(exc)
        if record["executions"] and record["executions"][-1]["status"] == "dispatch_intent":
            record["executions"][-1].update(status="not_dispatched", may_have_executed=False)
        record.update(status="time_limit" if reason == "time_limit" else "stopped", reason=reason)
    except KeyboardInterrupt:
        record.update(status="cancelled", reason="user_cancelled")
    except AgentRecordError:
        raise
    except Exception as exc:
        record.update(status="failed", reason="initialization_" + type(exc).__name__)
    finally:
        pending = None
        try:
            ports.close()
        except Exception:
            stop_file.parent.mkdir(parents=True, exist_ok=True)
            stop_file.touch()
            record.update(status="stopped", reason="cleanup_failed_persistent_stop")
    record["state"] = "stop"
    record["exit_code"] = 0 if record["status"] in {"simulation_complete", "goal_reported", "time_limit"} else 130 if record["status"] == "cancelled" else 3
    save("job_finished")
    return record, path
