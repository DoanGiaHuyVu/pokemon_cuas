"""Strict proposals tied to one observation; model claims are never proof of success."""

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from computer.geometry import Calibration
from harness.actions import Action, ActionError, Click, Move, Press, validate_game_action


class ProposalError(Exception):
    pass


class ProposalBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ScreenSummary(ProposalBase):
    description: Annotated[str, Field(min_length=1, max_length=1600)]
    visible_text: Annotated[list[Annotated[str, Field(min_length=1, max_length=160)]], Field(max_length=16)]


class VisibleObservation(ProposalBase):
    screen_kind: Literal["opening", "title", "main_menu", "name_entry", "gameplay", "transition", "unknown"]
    upper_screen: ScreenSummary
    lower_touchscreen: ScreenSummary


class Proposal(ProposalBase):
    observation_id: Annotated[str, Field(min_length=1, max_length=96, pattern=r"^[a-zA-Z0-9_-]+$")]
    visible_observation: VisibleObservation
    status: Literal["propose", "pause"]
    action: Action | None
    target_label: Annotated[str, Field(min_length=1, max_length=160)] | None
    expected_result: Annotated[str, Field(min_length=1, max_length=600)]
    rationale: Annotated[str, Field(min_length=1, max_length=600)]
    task_complete: bool

    @field_validator("task_complete")
    @classmethod
    def forbid_completion_claim(cls, value):
        if value:
            raise ValueError("Completion claims require later independent verification")
        return value

    @model_validator(mode="after")
    def consistent_action(self):
        if self.status == "pause":
            if self.action is not None or self.target_label is not None:
                raise ValueError("A paused proposal cannot contain an action or target")
        elif self.action is None:
            raise ValueError("A proposed action is required")
        if isinstance(self.action, (Move, Click)):
            if self.target_label is None:
                raise ValueError("A mouse target must be named explicitly")
        elif self.target_label is not None:
            raise ValueError("Only mouse actions have a target label")
        return self


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _constant(_):
    raise ValueError("Non-finite JSON constant")


def parse_proposal(raw: str, observation_id: str, calibration: Calibration) -> Proposal:
    if not isinstance(raw, str) or not raw or len(raw) > 12000:
        raise ProposalError("Proposal text is empty or excessive; no input executed.")
    try:
        decoded = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
        proposal = Proposal.model_validate(decoded)
    except (ValueError, ValidationError, TypeError, RecursionError) as exc:
        # Never echo raw JSON, field values, or Pydantic's input-bearing errors.
        raise ProposalError("Proposal must be one JSON object matching the required schema; no input executed.") from exc
    if proposal.observation_id != observation_id:
        raise ProposalError("Proposal references a different observation; no input executed.")
    if not calibration.verified:
        raise ProposalError("Proposal requires reviewed calibration; no input executed.")
    observed = proposal.visible_observation
    menu_labels = {label.strip().casefold() for screen in (observed.upper_screen, observed.lower_touchscreen)
                   for label in screen.visible_text}
    menu_visible = bool(menu_labels & {"new game", "continue"})
    if menu_visible and observed.screen_kind != "main_menu":
        raise ProposalError("Visible menu labels conflict with the claimed screen kind; no input executed.")
    # Current milestone stops at the menu. This is a check of model claims, not
    # independent OCR or proof that the goal actually succeeded in the game.
    if observed.screen_kind in {"main_menu", "unknown"} and proposal.status != "pause":
        raise ProposalError("Pause at the main menu or an unknown screen; do not keep pressing X.")
    if proposal.action is not None:
        try:
            validate_game_action(proposal.action, calibration)
        except ActionError as exc:
            raise ProposalError(str(exc)) from exc
        if observed.screen_kind in {"opening", "title"}:
            if not isinstance(proposal.action, Press) or proposal.action.key != "x" or proposal.action.seconds > 0.2:
                raise ProposalError("Opening/title navigation allows one short PRESS x, then a fresh observation.")
        if isinstance(proposal.action, Click):
            targets = {label.strip().casefold() for label in observed.lower_touchscreen.visible_text}
            if proposal.target_label.strip().casefold() not in targets:
                raise ProposalError("Click target is absent from the claimed lower-screen labels; no input executed.")
    return proposal
