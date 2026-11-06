"""ElevenLabs Scribe backend.

Scribe has no separate language-detection route, so detection transcribes the
middle sample clip and reads ``language_code``. Word timestamps are grouped
into subtitle-sized segments by a pure function so the HTTP client stays thin.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from transcriber.config import Settings
from transcriber.exceptions import (
    ConfigurationError,
    PermanentEngineError,
    TransientEngineError,
)
from transcriber.models import Segment

logger = logging.getLogger(__name__)

SPEECH_TO_TEXT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_SENTENCE_END = (".", "!", "?")


class ElevenLabsEngine:
    """HTTP client for ``POST /v1/speech-to-text``. The API key never appears in errors."""

    name: str = "elevenlabs"

    def __init__(self, settings: Settings) -> None:
        key = _require_key(settings)
        self._settings = settings
        self._client = httpx.Client(
            timeout=httpx.Timeout(settings.request_timeout_seconds),
            headers={"xi-api-key": key},
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __repr__(self) -> str:
        return f"ElevenLabsEngine(model={self._settings.elevenlabs_model!r})"

    def detect_language(self, clips: Sequence[Path]) -> str | None:
        """Transcribe the middle clip. One clip is enough; Scribe returns the code."""
        if not clips:
            return None
        payload = self._request(clips[len(clips) // 2], language=None)
        code = payload.get("language_code")
        if isinstance(code, str) and code.strip():
            return code.strip()
        return None

    def transcribe(self, chunk: Path, language: str | None) -> list[Segment]:
        """Send the chunk with the language forced, then group word timestamps."""
        payload = self._request(chunk, language)
        words = payload.get("words", [])
        if not isinstance(words, list):
            raise PermanentEngineError("ElevenLabs response is missing word timestamps")
        return words_to_segments(words)

    def _request(self, path: Path, language: str | None) -> dict[str, Any]:
        data = {
            "model_id": self._settings.elevenlabs_model,
            "timestamps_granularity": "word",
        }
        if language:
            data["language_code"] = language
        try:
            with path.open("rb") as handle:
                response = self._client.post(
                    SPEECH_TO_TEXT_URL,
                    data=data,
                    files={"file": (path.name, handle, "application/octet-stream")},
                )
        except httpx.TimeoutException as exc:
            raise TransientEngineError("ElevenLabs request timed out") from exc
        except httpx.TransportError as exc:
            raise TransientEngineError("ElevenLabs connection failed") from exc
        return _payload(response)


def words_to_segments(
    words: Sequence[Mapping[str, Any]],
    *,
    max_len: float = 15.0,
    max_gap: float = 1.0,
) -> list[Segment]:
    """Group word timestamps into segments.

    Non-word items (spacing, audio events) are ignored. A new segment starts
    after sentence punctuation, after a pause longer than ``max_gap``, or when
    adding the next word would push the segment past ``max_len``.
    """
    tokens = [token for token in (_as_token(word) for word in words) if token is not None]
    return _group_tokens(tokens, max_len=max_len, max_gap=max_gap)


def _require_key(settings: Settings) -> str:
    secret = settings.elevenlabs_api_key
    if secret is None or not secret.get_secret_value().strip():
        raise ConfigurationError("ELEVENLABS_API_KEY is not set")
    return secret.get_secret_value()


def _payload(response: httpx.Response) -> dict[str, Any]:
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientEngineError(f"ElevenLabs request failed with status {response.status_code}")
    if response.status_code >= 400:
        raise PermanentEngineError(f"ElevenLabs request failed with status {response.status_code}")
    try:
        body = response.json()
    except ValueError as exc:
        raise PermanentEngineError("ElevenLabs returned invalid JSON") from exc
    if not isinstance(body, dict):
        raise PermanentEngineError("ElevenLabs returned an unexpected payload")
    return body


def _as_token(word: Mapping[str, Any]) -> tuple[float, float, str] | None:
    if not isinstance(word, Mapping) or word.get("type") != "word":
        return None
    text = str(word.get("text", "")).strip()
    start = word.get("start")
    end = word.get("end")
    if not text or isinstance(start, bool) or isinstance(end, bool):
        return None
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    if float(start) < 0:
        return None
    return (float(start), max(float(end), float(start)), text)


def _group_tokens(
    tokens: Sequence[tuple[float, float, str]],
    *,
    max_len: float,
    max_gap: float,
) -> list[Segment]:
    segments: list[Segment] = []
    buffer: list[tuple[float, float, str]] = []

    def flush() -> None:
        if not buffer:
            return
        text = " ".join(item[2] for item in buffer)
        segments.append(Segment(start=buffer[0][0], end=buffer[-1][1], text=text))
        buffer.clear()

    for start, end, text in tokens:
        if buffer and _should_split(buffer, start, end, max_gap=max_gap, max_len=max_len):
            flush()
        buffer.append((start, end, text))
        if text.endswith(_SENTENCE_END):
            flush()
    flush()
    return segments


def _should_split(
    buffer: Sequence[tuple[float, float, str]],
    start: float,
    end: float,
    *,
    max_gap: float,
    max_len: float,
) -> bool:
    gap = start - buffer[-1][1]
    duration = end - buffer[0][0]
    return gap > max_gap or duration > max_len
