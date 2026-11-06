"""Immutable records shared by the pipeline, engines, and exporters.

Frozen models keep a finished transcript from being edited after it is built.
"""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Segment(BaseModel):
    """One time-aligned piece of transcript text.

    ``start`` and ``end`` are seconds. Engines emit times relative to the chunk
    they received; the pipeline shifts them onto the original file's timeline
    before anything is saved.
    """

    model_config = ConfigDict(frozen=True)

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str

    @model_validator(mode="after")
    def _start_not_after_end(self) -> Self:
        if self.start > self.end:
            raise ValueError("start must be less than or equal to end")
        return self


class ChunkSpec(BaseModel):
    """A slice of the normalized audio that will be transcribed on its own."""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _start_not_after_end(self) -> Self:
        if self.start > self.end:
            raise ValueError("start must be less than or equal to end")
        return self


class TranscriptionResult(BaseModel):
    """Structured transcript ready for search, subtitles, or summarization."""

    model_config = ConfigDict(frozen=True)

    engine: str
    language: str | None
    duration: float = Field(ge=0)
    word_count: int = Field(ge=0)
    text: str
    segments: tuple[Segment, ...] = Field(default_factory=tuple)
