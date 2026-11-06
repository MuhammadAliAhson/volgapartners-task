"""Environment-driven settings.

Every tunable lives here so behavior can be changed without editing code and
so tests can construct a small, explicit configuration.
"""

import os
from typing import Annotated, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _parse_extensions(value: object) -> object:
    """Accept a comma-separated env string as well as a sequence of extensions."""
    if isinstance(value, str):
        return tuple(part.strip().lower().lstrip(".") for part in value.split(",") if part.strip())
    return value


class Settings(BaseSettings):
    """Process configuration loaded from the environment and an optional ``.env``.

    The prefix is ``TRANSCRIBER_``. The ElevenLabs key is the exception: it is
    read from ``ELEVENLABS_API_KEY`` so it can be shared with the vendor's own
    tools. A blank value is treated as unset.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        populate_by_name=True,
    )

    default_engine: str = "whisper"  # whisper or elevenlabs
    chunk_seconds: float = Field(default=600, gt=0)  # target chunk length
    silence_window_seconds: float = Field(default=60, ge=0)  # search ±N s around a cut
    silence_noise_db: float = -35  # silence threshold passed to silencedetect
    silence_min_duration: float = Field(default=0.4, gt=0)  # minimum silence length
    language_samples: int = Field(default=3, ge=1)  # clips used for language detection
    language_clip_seconds: float = Field(default=30, gt=0)  # length of each detection clip
    max_retries: int = Field(default=3, ge=1)  # attempts per engine call
    retry_base_delay: float = Field(default=1.0, ge=0)  # delay = base * 2^attempt + jitter
    whisper_model: str = "base"  # tiny / base / small / medium / large-v3
    whisper_compute_type: str = "int8"  # CPU-friendly compute type
    elevenlabs_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ELEVENLABS_API_KEY", "TRANSCRIBER_ELEVENLABS_API_KEY"),
    )
    elevenlabs_model: str = "scribe_v2"
    request_timeout_seconds: float = Field(default=600, gt=0)  # HTTP and subprocess timeout
    supported_extensions: Annotated[tuple[str, ...], NoDecode] = (
        "wav",
        "mp3",
        "m4a",
        "flac",
        "ogg",
        "opus",
        "aac",
        "webm",
        "mp4",
    )

    @field_validator("elevenlabs_api_key", mode="before")
    @classmethod
    def _empty_key_is_missing(cls, value: object) -> object:
        if value is None or value == "":
            return None
        return value

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def _normalize_extensions(cls, value: object) -> object:
        return _parse_extensions(value)

    @field_validator("default_engine")
    @classmethod
    def _known_engine(cls, value: str) -> str:
        if value not in {"whisper", "elevenlabs"}:
            raise ValueError("default_engine must be 'whisper' or 'elevenlabs'")
        return value

    @model_validator(mode="after")
    def _key_from_canonical_env(self) -> Self:
        # Alias handling has varied across pydantic-settings releases. Reading
        # the canonical variable here keeps ``ELEVENLABS_API_KEY`` working even
        # when the prefix is applied to aliases.
        if self.elevenlabs_api_key is not None:
            return self
        raw = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not raw:
            return self
        return self.model_copy(update={"elevenlabs_api_key": SecretStr(raw)})
