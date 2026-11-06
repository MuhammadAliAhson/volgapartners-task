"""Whisper language vote, segment mapping, and error classification."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import av
import pytest
from tests.conftest import make_settings, write_wav

from transcriber.engines import get_engine
from transcriber.engines.whisper import WhisperEngine, _relax_pyav, weighted_language_vote
from transcriber.exceptions import (
    ConfigurationError,
    PermanentEngineError,
    TransientEngineError,
)
from transcriber.models import Segment


def test_weighted_vote_sums_probabilities() -> None:
    """The winner is the language with the largest total, not the most top-1 votes."""
    votes = [
        [("en", 0.9), ("es", 0.1)],
        [("es", 0.6), ("en", 0.4)],
        [("en", 0.8), ("es", 0.2)],
    ]
    assert weighted_language_vote(votes) == "en"
    assert weighted_language_vote([]) is None


def test_detect_language_uses_the_vote(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Each clip is decoded and its full probability list is fed to the vote."""
    model = MagicMock()
    model.detect_language.side_effect = [
        ("en", 0.9, [("en", 0.9), ("es", 0.1)]),
        ("es", 0.6, [("es", 0.6), ("en", 0.4)]),
    ]
    _install_audio(monkeypatch)
    engine = WhisperEngine(make_settings())
    monkeypatch.setattr(engine, "_get_model", lambda: model)
    clips = [_touch(tmp_path, "a.wav"), _touch(tmp_path, "b.wav")]
    assert engine.detect_language(clips) == "en"
    assert model.detect_language.call_count == 2


def test_transcribe_maps_segments_and_forces_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VAD and the forced language are passed through, and text is trimmed."""
    raw = SimpleNamespace(start=1.25, end=2.5, text=" Hello ")
    model = MagicMock()
    model.transcribe.return_value = (iter([raw]), SimpleNamespace(language="en"))
    engine = WhisperEngine(make_settings())
    monkeypatch.setattr(engine, "_get_model", lambda: model)
    assert engine.transcribe(Path("chunk.wav"), "fr") == [
        Segment(start=1.25, end=2.5, text="Hello")
    ]
    kwargs = model.transcribe.call_args.kwargs
    assert kwargs["vad_filter"] is True
    assert kwargs["beam_size"] == 5
    assert kwargs["language"] == "fr"


def test_memory_error_is_transient_and_invalid_audio_is_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-memory can be retried; a rejected file cannot."""
    model = MagicMock()
    engine = WhisperEngine(make_settings())
    monkeypatch.setattr(engine, "_get_model", lambda: model)
    model.transcribe.side_effect = MemoryError("out of memory")
    with pytest.raises(TransientEngineError):
        engine.transcribe(Path("chunk.wav"), "en")
    model.transcribe.side_effect = ValueError("invalid audio")
    with pytest.raises(PermanentEngineError):
        engine.transcribe(Path("chunk.wav"), "en")


def test_model_is_loaded_once_with_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=auto and the configured compute type are passed, then cached."""
    created: dict[str, Any] = {}

    class FakeModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
            created["args"] = (model_name, device, compute_type)

    package = ModuleType("faster_whisper")
    package.WhisperModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    engine = WhisperEngine(make_settings(whisper_model="tiny", whisper_compute_type="int8"))
    first = engine._get_model()
    second = engine._get_model()
    assert first is second
    assert created["args"] == ("tiny", "auto", "int8")


def test_unknown_engine_name() -> None:
    """The factory fails before any audio work when the name is not supported."""
    with pytest.raises(ConfigurationError, match="Unknown engine"):
        get_engine("nope", make_settings())


def test_pyav19_open_accepts_removed_metadata_argument(tmp_path: Path) -> None:
    """faster-whisper still passes metadata_errors, which PyAV 19 removed."""
    audio = tmp_path / "clip.wav"
    write_wav(audio, 0.2)
    _relax_pyav()
    with av.open(str(audio), metadata_errors="ignore") as container:
        assert len(container.streams.audio) == 1


def test_empty_clips_skip_the_model() -> None:
    """There is nothing to vote on when the sampler produced no clips."""
    engine = WhisperEngine(make_settings())
    assert engine.detect_language([]) is None


def _install_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = ModuleType("faster_whisper.audio")
    audio.decode_audio = lambda *_args, **_kwargs: [0.0]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", audio)


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"x")
    return path
