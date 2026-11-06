"""Engine factory.

The pipeline never constructs an engine itself. Callers (the CLI, the API, and
tests) choose one here and pass it in.
"""

from transcriber.config import Settings
from transcriber.engines.base import SpeechToTextEngine
from transcriber.engines.elevenlabs import ElevenLabsEngine
from transcriber.engines.whisper import WhisperEngine
from transcriber.exceptions import ConfigurationError

__all__ = [
    "ElevenLabsEngine",
    "SpeechToTextEngine",
    "WhisperEngine",
    "get_engine",
]


def get_engine(name: str, settings: Settings) -> SpeechToTextEngine:
    """Build the named engine. Misconfiguration fails before any audio is read."""
    if name == "whisper":
        return WhisperEngine(settings)
    if name == "elevenlabs":
        return ElevenLabsEngine(settings)
    raise ConfigurationError(f"Unknown engine: {name}")
