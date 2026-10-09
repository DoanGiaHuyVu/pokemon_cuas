"""Validated mapping between image pixels and macOS desktop points."""

import math
from numbers import Real
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


Positive = Annotated[float, Field(gt=0)]
NonnegativeInt = Annotated[int, Field(ge=0, strict=True)]
PositiveInt = Annotated[int, Field(gt=0, strict=True)]


class GeometryError(Exception):
    pass


class Rect(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    left: float
    top: float
    width: Positive
    height: Positive


class PixelRect(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    left: NonnegativeInt
    top: NonnegativeInt
    width: PositiveInt
    height: PositiveInt

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.left + self.width, self.top + self.height


class WindowInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    window_id: PositiveInt
    pid: PositiveInt
    owner: str
    title: str
    bounds: Rect


class Calibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    version: int = 1
    window: WindowInfo
    frame_width: PositiveInt
    frame_height: PositiveInt
    game_crop: PixelRect
    top_screen: PixelRect
    touchscreen: PixelRect
    created_at: str
    verified: bool = False

    @model_validator(mode="after")
    def validate_rectangles(self) -> "Calibration":
        crop = self.game_crop
        if self.version != 1:
            raise ValueError("Unsupported calibration version")
        if crop.box[2] > self.frame_width or crop.box[3] > self.frame_height:
            raise ValueError("Crop extends outside the captured window")
        for screen in (self.top_screen, self.touchscreen):
            if screen.box[2] > crop.width or screen.box[3] > crop.height:
                raise ValueError("Screen extends outside the game crop")
        if self.top_screen.box[3] > self.touchscreen.top:
            raise ValueError("DS screen rectangles overlap")
        return self

    @property
    def scale_x(self) -> float:
        return self.frame_width / self.window.bounds.width

    @property
    def scale_y(self) -> float:
        return self.frame_height / self.window.bounds.height

    @property
    def game_desktop(self) -> Rect:
        crop, bounds = self.game_crop, self.window.bounds
        return Rect(
            left=bounds.left + crop.left / self.scale_x,
            top=bounds.top + crop.top / self.scale_y,
            width=crop.width / self.scale_x,
            height=crop.height / self.scale_y,
        )

    @property
    def touchscreen_desktop(self) -> Rect:
        game, screen = self.game_desktop, self.touchscreen
        return Rect(
            left=game.left + screen.left / self.scale_x,
            top=game.top + screen.top / self.scale_y,
            width=screen.width / self.scale_x,
            height=screen.height / self.scale_y,
        )

    def normalized_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        if any(not isinstance(v, Real) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1000 for v in (x, y)):
            raise GeometryError("Normalized coordinates must be finite numbers from 0 to 1000.")
        return (
            round(x / 1000 * (self.game_crop.width - 1)),
            round(y / 1000 * (self.game_crop.height - 1)),
        )

    def normalized_to_desktop(self, x: float, y: float) -> tuple[float, float]:
        px, py = self.normalized_to_pixel(x, y)
        game = self.game_desktop
        return game.left + px / self.scale_x, game.top + py / self.scale_y

    def is_touch_target(self, x: float, y: float) -> bool:
        px, py = self.normalized_to_pixel(x, y)
        screen = self.touchscreen
        return screen.left <= px < screen.box[2] and screen.top <= py < screen.box[3]

    def check_window(self, current: WindowInfo) -> None:
        if (current.window_id, current.pid, current.owner, current.bounds) != (
            self.window.window_id, self.window.pid, self.window.owner, self.window.bounds
        ):
            raise GeometryError("The emulator moved, resized, or restarted. Run --calibrate again.")

    def check_frame(self, size: tuple[int, int]) -> None:
        if size != (self.frame_width, self.frame_height):
            raise GeometryError("Capture scale changed. Run --calibrate again.")
