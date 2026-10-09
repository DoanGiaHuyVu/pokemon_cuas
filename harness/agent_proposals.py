"""One-action planning for autonomous jobs; earlier menu-only previews stay unchanged."""

import json
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from harness.actions import ActionError, Click, Press, validate_game_action
from harness.proposals import Proposal, ProposalError, VisibleObservation, _constant, _object


class AgentVisibleObservation(VisibleObservation):
    screen_kind: Literal["opening", "title", "main_menu", "dialogue", "character_select", "name_entry",
                         "starter_select", "battle", "overworld", "game_menu", "party", "party_options",
                         "summary", "bag", "transition", "unknown"]


class AgentProposal(Proposal):
    visible_observation: AgentVisibleObservation
    goal_status: Literal["in_progress", "candidate_complete"]
    goal_evidence: Annotated[str, Field(max_length=600)]
    selected_menu_label: Annotated[str, Field(max_length=160)] | None = None


def parse_agent_proposal(raw: str, observation_id: str, calibration, job) -> AgentProposal:
    if not isinstance(raw, str) or not 1 <= len(raw) <= 14000:
        raise ProposalError("Autonomous proposal is empty or excessive.")
    try:
        proposal = AgentProposal.model_validate(json.loads(raw, object_pairs_hook=_object, parse_constant=_constant))
    except (ValueError, TypeError, ValidationError, RecursionError) as exc:
        raise ProposalError("Autonomous proposal must match the one-action schema.") from exc
    if proposal.observation_id != observation_id or not calibration.verified:
        raise ProposalError("Autonomous proposal has stale identity or unverified geometry.")
    visible = proposal.visible_observation
    if visible.screen_kind == "unknown" and proposal.status != "pause":
        raise ProposalError("Unknown screens require automatic recovery, without input.")
    if proposal.goal_status == "candidate_complete" and (proposal.status != "pause" or not proposal.goal_evidence.strip()):
        raise ProposalError("A candidate goal must have visible evidence and no pending action.")
    if proposal.action is not None:
        try:
            validate_game_action(proposal.action, calibration)
        except ActionError as exc:
            raise ProposalError(str(exc)) from exc
        if visible.screen_kind in {"opening", "title"} and (
            not isinstance(proposal.action, Press) or proposal.action.key != "x" or proposal.action.seconds > 0.2
        ):
            raise ProposalError("Opening/title navigation requires one short physical x press.")
        if isinstance(proposal.action, Click):
            labels = {text.casefold().strip() for text in visible.lower_touchscreen.visible_text}
            if proposal.target_label.strip().casefold() not in labels:
                raise ProposalError("Click target is absent from the claimed lower screen.")
        labels = {text.strip().casefold() for screen in (visible.upper_screen, visible.lower_touchscreen) for text in screen.visible_text}
        # Do not let a general play request create a new save over an existing account.
        if "continue" in labels and "new game" in labels:
            if isinstance(proposal.action, Click) and proposal.target_label.strip().casefold() == "new game":
                raise ProposalError("An existing save is visible; New Game is outside this job's safe scope.")
            if isinstance(proposal.action, Press) and proposal.action.key == "x" and proposal.selected_menu_label != "Continue":
                raise ProposalError("Existing-save menu confirmation must identify Continue.")
        if visible.screen_kind == "main_menu" and job.goal_kind == "menu":
            raise ProposalError("The user's requested menu stop forbids further input.")
    return proposal
