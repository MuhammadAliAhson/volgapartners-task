"""Speech-to-text engine contract.

Timestamps returned by ``transcribe`` are relative to the start of the chunk.
The pipeline adds the chunk offset so every exported time is on the original
file's timeline. Keeping that shift in one place avoids each engine reimplementing it.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from transcriber.models import Segment


class SpeechToTextEngine(Protocol):
    """One interchangeable transcription backend."""

    name: str

    def detect_language(self, clips: Sequence[Path]) -> str | None:
        """Return a language code, or ``None`` when the clips are empty or silent."""

    def transcribe(self, chunk: Path, language: str | None) -> list[Segment]:
        """Transcribe ``chunk``. Times are relative to the start of that file."""
