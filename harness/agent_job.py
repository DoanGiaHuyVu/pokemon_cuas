"""Bounded autonomous requests. Natural-language conditions remain part of the goal."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentJob(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    mode: Literal["objective", "timed"]
    goal_kind: Literal["menu", "introduction", "first_pokemon", "party", "exploration", "time", "other"] = "other"
    seconds: Annotated[float, Field(gt=0, le=1800)]
    max_actions: Annotated[int, Field(ge=1, le=500)] = 100
    max_model_calls: Annotated[int, Field(ge=1, le=1500)] = 200
    max_recoveries: Annotated[int, Field(ge=0, le=10)] = 3
    max_no_progress: Annotated[int, Field(ge=1, le=10)] = 3
    max_observation_age: Annotated[float, Field(gt=0, le=60)] = 40.0
    human_review_required: Literal[False] = False


def job_from_prompt(prompt: str, *, seconds: float | None = None) -> AgentJob:
    prompt = prompt.strip()
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    durations = re.findall(r"\b(?:for|within|at most|no more than)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(seconds?|secs?|minutes?|mins?)\b", prompt, re.I)
    if len(durations) > 1:
        raise ValueError("Use one time limit in the prompt.")
    inferred = None
    if durations:
        number, unit = durations[0]
        inferred = (float(words[number.lower()]) if number.lower() in words else float(number)) * (60 if unit.lower().startswith("m") else 1)
    if seconds is not None and inferred is not None and seconds != inferred:
        raise ValueError("The prompt time limit and --seconds must agree.")
    duration = seconds if seconds is not None else inferred if inferred is not None else 600.0
    # A duration-only play request runs until the clock; a timed objective can finish sooner.
    timed = bool((inferred is not None or seconds is not None) and re.search(r"\bplay\b", prompt, re.I))
    if re.search(r"\b(?:until|get|obtain|reach|enter|entering)\b", prompt, re.I): timed = False
    kind = "time" if timed else "first_pokemon" if re.search(r"\b(?:first\s+pok[eé]mon|starter)\b", prompt, re.I) else "introduction" if re.search(r"\bname[ -]entry\b", prompt, re.I) else "menu" if re.search(r"\b(?:menu|new\s+game\s+(?:button\s+)?screen)\b", prompt, re.I) else "other"
    if kind in {'other', 'time'} or kind == 'menu' and not re.search(r'\b(?:new\s+game|continue)\b', prompt, re.I):
        if re.search(r'\b(?:wander|wandering|walk|walking|explore|exploring|navigate|navigating)\b', prompt, re.I):
            kind = 'exploration'
            timed = True  # Walking has no invented completion condition.
        elif re.search(r'\b(?:party|pok[eé]mon)\b', prompt, re.I) and re.search(r'\b(?:check|inspect|show|view|look)\b', prompt, re.I):
            kind = 'party'
            timed = False
    return AgentJob(prompt=prompt, mode="timed" if timed else "objective", goal_kind=kind, seconds=float(duration))


def live_menu_job(job: AgentJob) -> AgentJob:
    if job.mode != "objective" or job.goal_kind != "menu":
        raise ValueError("Step 10 live scope is reaching the menu without selection.")
    return job.model_copy(update={"max_actions":min(8, job.max_actions), "max_model_calls":min(48, job.max_model_calls),
                                 "max_recoveries":10,
                                 "max_observation_age":60.0})


def live_intro_job(job: AgentJob) -> AgentJob:
    if job.mode != 'objective' or job.goal_kind != 'introduction':
        raise ValueError('Step 11 live scope is introduction progression to name entry.')
    return job.model_copy(update={'max_actions':min(40, job.max_actions),
                                 'max_model_calls':min(150, job.max_model_calls),
                                 'max_recoveries':10, 'max_observation_age':60.0})


def live_gameplay_job(job: AgentJob) -> AgentJob:
    if job.goal_kind not in {'party', 'exploration', 'time'}:
        raise ValueError('Existing-save gameplay supports party inspection and bounded exploration.')
    return job.model_copy(update={'max_actions':min(40, job.max_actions),
                                 'max_model_calls':min(160, job.max_model_calls),
                                 'max_recoveries':5, 'max_observation_age':60.0})
