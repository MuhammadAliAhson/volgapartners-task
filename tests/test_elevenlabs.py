"""ElevenLabs request mapping, retries, and word grouping."""

import logging
from pathlib import Path

import httpx
import pytest
import respx
from tests.conftest import make_settings

from transcriber.engines.elevenlabs import SPEECH_TO_TEXT_URL, ElevenLabsEngine, words_to_segments
from transcriber.exceptions import (
    ConfigurationError,
    PermanentEngineError,
    TransientEngineError,
)
from transcriber.models import Segment
from transcriber.retry import retry

KEY = "super-secret-key"


def test_words_to_segments_splits_on_punctuation_gap_and_length() -> None:
    """Spacing is ignored. Punctuation, a long pause, and max length each flush."""
    words = [
        {"type": "word", "text": "Hello", "start": 0.0, "end": 0.4},
        {"type": "spacing", "text": " ", "start": 0.4, "end": 0.42},
        {"type": "word", "text": "world.", "start": 0.5, "end": 1.0},
        {"type": "word", "text": "Next", "start": 1.1, "end": 1.4},
        {"type": "audio_event", "text": "(laughter)", "start": 1.4, "end": 1.6},
        {"type": "word", "text": "A", "start": 4.0, "end": 4.2},
        {"type": "word", "text": "B", "start": 4.3, "end": 4.5},
        {"type": "word", "text": "C", "start": 4.6, "end": 5.2},
    ]
    segments = words_to_segments(words, max_len=1.0, max_gap=1.0)
    assert [segment.text for segment in segments] == ["Hello world.", "Next", "A B", "C"]
    assert segments[0] == Segment(start=0.0, end=1.0, text="Hello world.")
    assert segments[1].start == 1.1


def test_missing_api_key_fails_before_any_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construction is the configuration boundary, so no audio is read."""
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="ELEVENLABS_API_KEY"):
        ElevenLabsEngine(make_settings(elevenlabs_api_key=None))


def test_api_key_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical variable is accepted without the TRANSCRIBER_ prefix."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "abc123")
    monkeypatch.setenv("TRANSCRIBER_WHISPER_MODEL", "tiny")
    monkeypatch.setenv("TRANSCRIBER_SUPPORTED_EXTENSIONS", "wav, MP3")
    settings = make_settings()
    assert settings.elevenlabs_api_key is not None
    assert settings.elevenlabs_api_key.get_secret_value() == "abc123"
    assert settings.whisper_model == "tiny"
    assert settings.supported_extensions == ("wav", "mp3")


@respx.mock
def test_transcribe_posts_model_and_word_timestamps(tmp_path: Path) -> None:
    """The documented form fields are sent and words become segments."""
    route = respx.post(SPEECH_TO_TEXT_URL).mock(return_value=httpx.Response(200, json=_ok("en")))
    engine = _engine()
    try:
        segments = engine.transcribe(_audio(tmp_path), "en")
    finally:
        engine.close()
    assert segments[0].text == "Hello"
    body = route.calls.last.request.content
    assert b"scribe_v2" in body
    assert b"timestamps_granularity" in body
    assert b'name="language_code"' in body
    assert b"word" in body


@respx.mock
def test_429_then_200_succeeds_with_one_retry(tmp_path: Path) -> None:
    """A rate limit is transient, so the retry helper gets a second chance."""
    respx.post(SPEECH_TO_TEXT_URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=_ok("en"))]
    )
    engine = _engine()
    delays: list[float] = []
    try:
        result = retry(
            lambda: engine.transcribe(_audio(tmp_path), "en"),
            attempts=3,
            base_delay=1.0,
            sleep=delays.append,
            jitter=lambda _base: 0,
        )
    finally:
        engine.close()
    assert result[0].text == "Hello"
    assert delays == [1.0]


@respx.mock
def test_401_fails_without_retry(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Authentication failures are permanent, and the key is not echoed."""
    caplog.set_level(logging.DEBUG)
    route = respx.post(SPEECH_TO_TEXT_URL).mock(return_value=httpx.Response(401))
    engine = _engine()
    delays: list[float] = []
    try:
        with pytest.raises(PermanentEngineError, match="401") as exc_info:
            retry(
                lambda: engine.transcribe(_audio(tmp_path), None),
                attempts=3,
                base_delay=1.0,
                sleep=delays.append,
                jitter=lambda _base: 0,
            )
    finally:
        engine.close()
    assert delays == []
    assert route.call_count == 1
    assert KEY not in str(exc_info.value)
    assert KEY not in caplog.text
    assert KEY not in repr(engine)


@respx.mock
def test_repeated_500_raises_after_the_budget(tmp_path: Path) -> None:
    """Server errors are retried and then surfaced as transient."""
    route = respx.post(SPEECH_TO_TEXT_URL).mock(return_value=httpx.Response(500))
    engine = _engine()
    try:
        with pytest.raises(TransientEngineError, match="500"):
            retry(
                lambda: engine.transcribe(_audio(tmp_path), "en"),
                attempts=3,
                base_delay=0.0,
                sleep=lambda _delay: None,
                jitter=lambda _base: 0,
            )
    finally:
        engine.close()
    assert route.call_count == 3


@respx.mock
def test_language_comes_from_the_middle_clip(tmp_path: Path) -> None:
    """Scribe has no detect route, so the middle sample is transcribed."""
    route = respx.post(SPEECH_TO_TEXT_URL).mock(return_value=httpx.Response(200, json=_ok("es")))
    clips = [_audio(tmp_path, name) for name in ("a.wav", "b.wav", "c.wav")]
    engine = _engine()
    try:
        assert engine.detect_language(clips) == "es"
    finally:
        engine.close()
    assert b'filename="b.wav"' in route.calls.last.request.content


@respx.mock
def test_connection_error_and_invalid_json_are_classified(tmp_path: Path) -> None:
    """Transport problems are transient. A 200 with garbage is permanent."""
    respx.post(SPEECH_TO_TEXT_URL).mock(side_effect=httpx.ConnectError("down"))
    engine = _engine()
    try:
        with pytest.raises(TransientEngineError, match="connection"):
            engine.transcribe(_audio(tmp_path), None)
    finally:
        engine.close()
    respx.post(SPEECH_TO_TEXT_URL).mock(return_value=httpx.Response(200, text="nope"))
    engine = _engine()
    try:
        with pytest.raises(PermanentEngineError, match="JSON"):
            engine.transcribe(_audio(tmp_path), None)
    finally:
        engine.close()


def _engine() -> ElevenLabsEngine:
    from pydantic import SecretStr

    return ElevenLabsEngine(make_settings(elevenlabs_api_key=SecretStr(KEY)))


def _audio(directory: Path, name: str = "chunk.wav") -> Path:
    path = directory / name
    path.write_bytes(b"RIFF")
    return path


def _ok(language: str) -> dict[str, object]:
    return {
        "language_code": language,
        "text": "Hello",
        "words": [{"text": "Hello", "start": 0.0, "end": 0.4, "type": "word"}],
    }
