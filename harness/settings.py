"""Local configuration only: no model loading, API requests, or desktop access."""

import os
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROM = Path(
    "/Users/adamvu/Pokemon_retro/nintendo_ds/pokemon_white/roms/"
    "Pokemon - White Version (USA, Europe) (NDSi Enhanced).nds"
)
DEFAULT_LAYA_MODEL = Path(
    "/Users/adamvu/laya-project/laya/laya_finetuned_typed_decisions"
)
DEFAULT_LAYA_SOURCE = Path("/Users/adamvu/laya-project/laya")


class ConfigurationError(Exception):
    """A safe, value-free configuration error suitable for terminal output."""


class RuntimeMode(StrEnum):
    PREFLIGHT = "preflight"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    mode: RuntimeMode = RuntimeMode.PREFLIGHT
    gemini_api_key: SecretStr
    gemma_model: Annotated[str, Field(min_length=1)] = "gemma-4-26b-a4b-it"
    laya_model_path: Path = DEFAULT_LAYA_MODEL
    laya_source_path: Path = DEFAULT_LAYA_SOURCE
    rom_path: Path = DEFAULT_ROM
    emu_left: int | None = None
    emu_top: int | None = None
    emu_width: Annotated[int, Field(gt=0)] | None = None
    emu_height: Annotated[int, Field(gt=0)] | None = None

    @field_validator("gemini_api_key")
    @classmethod
    def require_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("API key must be configured")
        if value.get_secret_value() == "your_google_ai_studio_key_here":
            raise ValueError("Replace the example API key")
        return value

    @field_validator("gemma_model", mode="before")
    @classmethod
    def trim_model(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("laya_model_path", "laya_source_path", "rom_path", mode="before")
    @classmethod
    def normalize_path(cls, value: str | Path, info: ValidationInfo) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError("Path cannot be empty")
        path = Path(value).expanduser()
        root = (info.context or {}).get("project_root", PROJECT_ROOT)
        return (path if path.is_absolute() else root / path).resolve()

    @model_validator(mode="after")
    def complete_geometry(self) -> "Settings":
        coordinates = (self.emu_left, self.emu_top, self.emu_width, self.emu_height)
        if any(value is not None for value in coordinates) and any(
            value is None for value in coordinates
        ):
            raise ValueError("Supply all four EMU geometry fields or none")
        return self

    @property
    def geometry_configured(self) -> bool:
        return self.emu_width is not None


ENV_FIELDS = {
    "GEMINI_API_KEY": "gemini_api_key",
    "GEMMA_MODEL": "gemma_model",
    "LAYA_MODEL_PATH": "laya_model_path",
    "LAYA_SOURCE_PATH": "laya_source_path",
    "ROM_PATH": "rom_path",
    "EMU_LEFT": "emu_left",
    "EMU_TOP": "emu_top",
    "EMU_WIDTH": "emu_width",
    "EMU_HEIGHT": "emu_height",
}


def load_settings(
    env_file: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> Settings:
    """Read only known settings, with process environment taking precedence.

    Do not modify os.environ or interpolate values. Relative paths are anchored
    to the project, independently of the terminal's working directory.
    """
    path = project_root / ".env" if env_file is None else env_file
    try:
        local = dotenv_values(path, interpolate=False) if path.is_file() else {}
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError("Cannot read the configuration file.") from exc
    process = os.environ if environ is None else environ
    values = {
        field: process[name] if name in process else local[name]
        for name, field in ENV_FIELDS.items()
        if name in process or name in local
    }
    try:
        settings = Settings.model_validate(values, context={"project_root": project_root})
    except ValidationError as exc:
        reverse = {field: name for name, field in ENV_FIELDS.items()}
        fields = sorted(
            {
                reverse.get(str(error["loc"][0]), "configuration")
                if error["loc"]
                else "EMU_LEFT/EMU_TOP/EMU_WIDTH/EMU_HEIGHT (all four required)"
                for error in exc.errors(include_input=False)
            }
        )
        # Never print Pydantic's raw exception: it can include secret input values.
        raise ConfigurationError("Missing or invalid settings: " + ", ".join(fields)) from exc
    verify_paths(settings)
    return settings


def verify_paths(settings: Settings) -> None:
    """Check artifacts without loading weights or opening the game."""
    issues = []
    if not settings.rom_path.is_file() or settings.rom_path.suffix.lower() != ".nds":
        issues.append("ROM_PATH must identify an existing .nds file")
    source = settings.laya_source_path
    if not (source / "pyproject.toml").is_file() or not (source / "laya" / "__init__.py").is_file():
        issues.append("LAYA_SOURCE_PATH must identify the local Laya package source")
    model = settings.laya_model_path
    for relative in (
        "model.safetensors",
        "rl_agent_config.json",
        "questions.json",
        "encoder/config.json",
    ):
        if not (model / relative).is_file():
            issues.append(f"LAYA_MODEL_PATH is missing {relative}")
    if not (model / "tokenizer").is_dir():
        issues.append("LAYA_MODEL_PATH is missing the tokenizer directory")
    if issues:
        raise ConfigurationError("; ".join(issues))
