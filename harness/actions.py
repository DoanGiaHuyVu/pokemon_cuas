"""Generic input proposals. No shell commands, chords, or arbitrary key names."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator


class ActionError(Exception):
    pass


Coordinate = Annotated[int, Field(strict=True, ge=0, le=1000)]
Duration = Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
GameKey = Literal["x", "z", "s", "a", "q", "w", "enter", "shiftleft", "up", "down", "left", "right"]


class ActionBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Move(ActionBase):
    tool: Literal["MOVE"]
    x: Coordinate
    y: Coordinate


class Click(ActionBase):
    tool: Literal["CLICK"]
    x: Coordinate
    y: Coordinate


class Press(ActionBase):
    tool: Literal["PRESS"]
    key: GameKey
    seconds: Duration = 0.1


class TypeText(ActionBase):
    tool: Literal["TYPE"]
    text: Annotated[str, Field(min_length=1, max_length=32)]

    @field_validator("text")
    @classmethod
    def printable_ascii(cls, value: str) -> str:
        if any(not 32 <= ord(character) <= 126 for character in value):
            raise ValueError("Only printable ASCII text is currently supported")
        return value


class Wait(ActionBase):
    tool: Literal["WAIT"]
    seconds: Annotated[float, Field(strict=True, gt=0, le=2, allow_inf_nan=False)]


Action = Annotated[Move | Click | Press | TypeText | Wait, Field(discriminator="tool")]
ACTION_ADAPTER = TypeAdapter(Action)


def parse_action(raw: str) -> Move | Click | Press | TypeText | Wait:
    try:
        return ACTION_ADAPTER.validate_json(raw)
    except ValidationError as exc:
        # Validation errors can contain user text or credentials: report names only.
        fields = sorted({str(error["loc"][-1]) if error["loc"] else "tool/JSON" for error in exc.errors(include_input=False)})
        raise ActionError("Invalid action. Check fields: " + ", ".join(fields)) from exc


def validate_game_action(action, calibration) -> None:
    if isinstance(action, TypeText):
        raise ActionError("TYPE is disabled for the game. Use touchscreen clicks; --input-test verifies typing separately.")
    if not calibration.verified:
        raise ActionError("Calibration must be reviewed and confirmed before input.")
    if isinstance(action, Click) and not calibration.is_touch_target(action.x, action.y):
        raise ActionError("Game clicks must target the lower DS touchscreen.")
