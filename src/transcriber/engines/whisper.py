"""faster-whisper backend.

The model is loaded on first use and then reused. Reviewers can run this
engine with no API key. Language detection is a probability-weighted vote
across several clips so one noisy clip cannot pick the language alone.
"""

import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from transcriber.config import Settings
from transcriber.exceptions import PermanentEngineError, TransientEngineError
from transcriber.models import Segment

logger = logging.getLogger(__name__)

BEAM_SIZE = 5
SAMPLE_RATE = 16_000


class WhisperEngine:
    """Local faster-whisper engine. One model instance per engine object."""

    name: str = "whisper"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._lock = threading.Lock()

    def detect_language(self, clips: Sequence[Path]) -> str | None:
        """Vote across clips using every language probability, not just the top label."""
        if not clips:
            return None
        try:
            model = self._get_model()
            votes = [self._probabilities(model, clip) for clip in clips]
        except (TransientEngineError, PermanentEngineError):
            raise
        except Exception as exc:
            raise _classify(exc) from exc
        return weighted_language_vote(votes)

    def transcribe(self, chunk: Path, language: str | None) -> list[Segment]:
        """Transcribe with VAD on and the language forced when one was resolved."""
        try:
            model = self._get_model()
            segments, _info = model.transcribe(
                str(chunk),
                language=language,
                vad_filter=True,
                beam_size=BEAM_SIZE,
            )
            return [_to_segment(raw) for raw in segments]
        except (TransientEngineError, PermanentEngineError):
            raise
        except Exception as exc:
            raise _classify(exc) from exc

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                logger.info("Loading faster-whisper model %s", self._settings.whisper_model)
                self._model = self._load_model()
            return self._model

    def _load_model(self) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._settings.whisper_model,
            device="auto",
            compute_type=self._settings.whisper_compute_type,
        )

    def _probabilities(self, model: Any, clip: Path) -> list[tuple[str, float]]:
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(clip), sampling_rate=SAMPLE_RATE)
        _language, _probability, all_probs = model.detect_language(audio)
        return [(str(code), float(prob)) for code, prob in all_probs]


def weighted_language_vote(votes: Sequence[Sequence[tuple[str, float]]]) -> str | None:
    """Pick the language with the highest summed probability across clips."""
    totals: dict[str, float] = {}
    for vote in votes:
        for code, probability in vote:
            totals[code] = totals.get(code, 0.0) + probability
    if not totals:
        return None
    return max(totals, key=lambda code: totals[code])


def _to_segment(raw: Any) -> Segment:
    start = max(0.0, float(raw.start))
    end = max(start, float(raw.end))
    text = str(raw.text).strip()
    return Segment(start=start, end=end, text=text)


def _classify(exc: Exception) -> TransientEngineError | PermanentEngineError:
    text = str(exc).lower()
    if isinstance(exc, MemoryError) or "out of memory" in text:
        return TransientEngineError("Whisper ran out of memory")
    if isinstance(exc, (ValueError, FileNotFoundError)) or "invalid" in text:
        return PermanentEngineError("Whisper could not read the audio")
    return TransientEngineError("Whisper failed temporarily")
