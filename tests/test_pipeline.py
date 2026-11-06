"""Pipeline offsets, resume, progress, and temporary-directory cleanup."""

from pathlib import Path

import pytest
from tests.conftest import FakeEngine, make_settings, patch_media, render_pattern, requires_ffmpeg

from transcriber.audio.chunking import build_plan, find_cuts
from transcriber.exceptions import PermanentEngineError, UnsupportedFormatError
from transcriber.pipeline import TranscriptionPipeline
from transcriber.workspace import JobWorkspace


def test_offsets_language_once_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chunk starts are added to relative timestamps, and detection runs a single time."""
    patch_media(monkeypatch, duration=100, silences=[30.0, 60.0])
    engine = FakeEngine()
    pipeline = TranscriptionPipeline(
        engine,
        make_settings(chunk_seconds=30, silence_window_seconds=5, language_clip_seconds=10),
    )
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    progress: list[float] = []
    result = pipeline.run(audio, on_progress=progress.append)
    expected = build_plan(find_cuts([30.0, 60.0], 100, 30, 5), 100)
    assert [segment.start for segment in result.segments] == [chunk.start for chunk in expected]
    assert [segment.text for segment in result.segments] == [
        "hello 1",
        "hello 2",
        "hello 3",
        "hello 4",
    ]
    assert progress == [0.25, 0.5, 0.75, 1.0]
    assert len(engine.detect_calls) == 1
    assert {language for _path, language in engine.transcribe_calls} == {"en"}
    assert result.language == "en"
    assert result.word_count == 8


def test_resume_skips_finished_chunks_and_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second run with the same workdir does not call the engine again."""
    counts = patch_media(monkeypatch, duration=100, silences=[30.0, 60.0])
    engine = FakeEngine()
    settings = make_settings(chunk_seconds=30, silence_window_seconds=5, language_clip_seconds=10)
    pipeline = TranscriptionPipeline(engine, settings)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    workdir = tmp_path / "job"
    pipeline.run(audio, workdir=workdir)
    first_calls = len(engine.transcribe_calls)
    engine.transcribe_calls.clear()
    pipeline.run(audio, workdir=workdir)
    assert first_calls == 4
    assert engine.transcribe_calls == []
    assert len(engine.detect_calls) == 1
    assert counts["normalize"] == 1


def test_language_change_retranscribes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A different override drops chunk checkpoints so languages are not mixed."""
    patch_media(monkeypatch, duration=40, silences=[])
    engine = FakeEngine()
    settings = make_settings(chunk_seconds=600, silence_window_seconds=60)
    pipeline = TranscriptionPipeline(engine, settings)
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    workdir = tmp_path / "job"
    pipeline.run(audio, language="en", workdir=workdir)
    engine.transcribe_calls.clear()
    result = pipeline.run(audio, language="fr", workdir=workdir)
    assert result.language == "fr"
    assert len(engine.transcribe_calls) == 1
    assert engine.transcribe_calls[0][1] == "fr"


def test_temp_dir_removed_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Owned workspaces disappear after both a clean run and a forced failure."""
    roots: list[Path] = []

    class Recording(JobWorkspace):
        def __init__(self, root: Path | None, engine: str) -> None:
            super().__init__(root, engine)
            roots.append(self.root)

    monkeypatch.setattr("transcriber.pipeline.JobWorkspace", Recording)
    patch_media(monkeypatch, duration=40, silences=[])
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"wav")
    settings = make_settings(chunk_seconds=600, silence_window_seconds=60)
    TranscriptionPipeline(FakeEngine(), settings).run(audio)
    assert roots
    assert not roots[0].exists()
    roots.clear()
    engine = FakeEngine(fail_at=1, error=PermanentEngineError("boom"))
    with pytest.raises(PermanentEngineError):
        TranscriptionPipeline(engine, settings).run(audio)
    assert roots
    assert not roots[0].exists()


def test_missing_file_and_unsupported_extension(tmp_path: Path) -> None:
    """Validation fails before any engine call."""
    pipeline = TranscriptionPipeline(FakeEngine(), make_settings())
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "missing.wav")
    notes = tmp_path / "notes.txt"
    notes.write_text("nope", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        pipeline.run(notes)


def test_no_audio_stream(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """probe, not the extension, decides that a file is not audio."""
    patch_media(monkeypatch, has_audio=False)
    audio = tmp_path / "empty.wav"
    audio.write_bytes(b"nope")
    with pytest.raises(UnsupportedFormatError):
        TranscriptionPipeline(FakeEngine(), make_settings()).run(audio)


def test_zero_duration_is_an_empty_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Silence or an empty decode is a valid result, not an error."""
    patch_media(monkeypatch, duration=0, has_audio=True)
    engine = FakeEngine()
    audio = tmp_path / "quiet.wav"
    audio.write_bytes(b"wav")
    result = TranscriptionPipeline(engine, make_settings()).run(audio)
    assert result.word_count == 0
    assert result.segments == ()
    assert engine.transcribe_calls == []


@requires_ffmpeg
@pytest.mark.integration
def test_half_hour_file_splits_near_silences(tmp_path: Path) -> None:
    """A 31-minute file is cut on the synthetic silences and times stay absolute."""
    audio = tmp_path / "long.wav"
    render_pattern(audio, 1860)
    engine = FakeEngine()
    settings = make_settings(chunk_seconds=600, silence_window_seconds=60, language_clip_seconds=5)
    result = TranscriptionPipeline(engine, settings).run(audio)
    starts = [segment.start for segment in result.segments]
    assert starts[0] == 0
    assert starts[1] == pytest.approx(600.5, abs=1.0)
    assert starts[2] == pytest.approx(1200.5, abs=1.0)
    assert result.duration == pytest.approx(1860, abs=1.5)
    assert len(engine.detect_calls) == 1
    assert {language for _path, language in engine.transcribe_calls} == {"en"}
