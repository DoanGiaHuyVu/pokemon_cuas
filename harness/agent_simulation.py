"""Deterministic game/model ports for Step 9. No native input or model inference."""

import hashlib
import json

from computer.geometry import Calibration, PixelRect, Rect, WindowInfo
from harness.agent import GoalAssessment, GuardResult, Observation, run_agent
from models.laya_local import load_questions


class SimClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def advance(self, seconds): self.now += seconds


def simulation_calibration():
    return Calibration(window=WindowInfo(window_id=1, pid=1, owner="simulation", title="simulated DS",
                       bounds=Rect(left=0, top=0, width=256, height=384)), frame_width=256, frame_height=384,
                       game_crop=PixelRect(left=0, top=0, width=256, height=384),
                       top_screen=PixelRect(left=0, top=0, width=256, height=192),
                       touchscreen=PixelRect(left=0, top=192, width=256, height=192),
                       created_at="simulation", verified=True)


def fixture_prediction(questions, label="continue"):
    answers = {}
    for name, q in questions.items():
        common = {"type": q["type"], "confidence": 0.3, "answer_confidence": 0.7, "action": {"act_probability": 0.9}}
        if q["type"] == "choice":
            choice = label if name == "action" else "partial"
            common.update(choice=choice, probabilities={key: 0.7 if key == choice else 0.1 for key in q["criteria"]})
        elif q["type"] == "noul":
            common.update(noul=0.2, confidence=0.8, answer_confidence=0.8)
        else:
            common.update(score=0.6, legend={str(i): value for i, value in enumerate(q["criteria"])},
                          probabilities={str(i): 0.7 if i == 0 else 0.1 for i in range(4)})
        answers[name] = common
    return {"model": "simulation_fixture_not_real_laya", "answers": answers,
            "usage": {"input_tokens": 500, "output_tokens": 0, "state_tokens": 40,
                      "state_tokens_dropped": 0, "truncated": False, "truncated_questions": []}}


class SimulatedPorts:
    stages = [
        ("title", ["PRESS START"], "PRESS", "x"),
        ("main_menu", ["New Game"], "PRESS", "x"),
        ("dialogue", ["Welcome to the world of Pokemon!"], "PRESS", "x"),
        ("character_select", ["Boy", "Girl"], "CLICK", "Boy"),
        ("name_entry", ["H", "U", "Y", "OK"], "CLICK", "H"),
        ("name_entry", ["H", "U", "Y", "OK", "H entered"], "CLICK", "U"),
        ("name_entry", ["H", "U", "Y", "OK", "HU entered"], "CLICK", "Y"),
        ("name_entry", ["H", "U", "Y", "OK", "HUY entered"], "CLICK", "OK"),
        ("starter_select", ["Snivy", "Tepig", "Oshawott"], "CLICK", "Snivy"),
        ("dialogue", ["You received Snivy!"], "PRESS", "x"),
        ("party", ["Snivy", "Trainer HUY"], "PRESS", "right"),
    ]

    def __init__(self, questions, clock, *, label="human_review"):
        self.questions, self.clock, self.label = questions, clock, label
        self.stage = 0
        self.captures = 0
        self.latest_id = None
        self.inputs = []
        self.contexts = []
        self.closed = False
        self.calibration = simulation_calibration()

    def capture(self, job, deadline):
        self.clock.advance(0.2)
        self.captures += 1
        self.latest_id = f"sim-frame-{self.captures}"
        return Observation(observation_id=self.latest_id, captured_monotonic=float(self.clock()),
                           image_sha256=hashlib.sha256(f"stage-{self.stage}".encode()).hexdigest(),
                           calibration=self.calibration, source="simulation", metadata={"stage": self.stage, "real_screen_captured": False})

    def plan(self, observation, job, history, deadline):
        self.clock.advance(0.4)
        stage = min(self.stage, len(self.stages) - 1)
        kind, labels, tool, key = self.stages[stage]
        labels = list(labels)
        if self.stage >= len(self.stages):
            kind, labels = "overworld", [f"Position {self.stage}"]
        goal_stage = 1 if job.goal_kind == 'menu' else 4 if job.goal_kind == 'introduction' else 10
        complete = job.mode == 'objective' and self.stage >= goal_stage
        action = None if complete else {"tool": "PRESS", "key": key, "seconds": 0.1} if tool == "PRESS" else {"tool": "CLICK", "x": 500, "y": 750}
        return json.dumps({"observation_id": observation.observation_id,
                          "visible_observation": {"screen_kind": kind, "upper_screen": {"description": "Simulated upper DS screen.", "visible_text": []},
                                                  "lower_touchscreen": {"description": "Simulated touch controls.", "visible_text": labels}},
                          "status": "pause" if complete else "propose", "action": action,
                          "target_label": key if tool == "CLICK" and not complete else None,
                          "expected_result": "A later simulated screen may advance.", "rationale": "Follow the bounded simulated job.",
                          "task_complete": False, "goal_status": "candidate_complete" if complete else "in_progress",
                          "goal_evidence": "Simulated goal screen is visible." if complete else "", "selected_menu_label": None})

    def evaluate(self, text, deadline):
        self.clock.advance(0.1)
        self.contexts.append(json.loads(text))
        return fixture_prediction(self.questions, self.label)

    def guard(self, observation, deadline):
        return GuardResult(status="clear", current_observation_id=self.latest_id)

    def execute(self, action, observation, action_id, deadline):
        if self.clock() >= deadline or action_id in {item["id"] for item in self.inputs}:
            raise RuntimeError("Late or duplicate simulation dispatch")
        self.inputs.append({"id": action_id, "observation_id": observation.observation_id, "action": action.model_dump(),
                            "started_monotonic": self.clock()})
        self.stage += 1
        self.clock.advance(min(3.0, max(0, deadline - self.clock())))
        return {"executed": True, "action": action.model_dump(), "simulated": True}

    def assess_goal(self, observation, job, deadline):
        self.clock.advance(0.3)
        return GoalAssessment(observation_id=observation.observation_id, verdict="confirmed",
                              evidence="Only the simulated screen reached the requested goal.", source="simulation")

    def close(self): self.closed = True


def run_agent_simulation(settings, job, runs):
    clock = SimClock()
    # Initialize the fixture only after the controller has persisted its run and
    # verified the checkpoint schema. Invalid questions produce a durable failure.
    ports = SimulatedPorts({}, clock)
    original_evaluate = ports.evaluate
    def evaluate(text, deadline):
        if not ports.questions:
            ports.questions = load_questions(settings.laya_model_path)
        return original_evaluate(text, deadline)
    ports.evaluate = evaluate
    return run_agent(job, ports, runs, settings.laya_model_path, clock=clock, simulation=True,
                     secret=settings.gemini_api_key.get_secret_value())
