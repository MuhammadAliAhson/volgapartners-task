"""Language override, checkpoint reuse, and clip placement."""

from pathlib import Path

import pytest
from tests.conftest import FakeEngine, make_settings, requires_ffmpeg

from transcriber.language import clip_starts, resolve_language, sample_clips
from transcriber.workspace import JobWorkspace


class MemoryLanguage:
    """Minimal checkpoint used to prove detection runs once."""

    def __init__(self, saved: str | None = None) -> None:
        self.value = saved
        self.saves = 0

    def load_language(self) -> str | None:
        return self.value

    def save_language(self, language: str) -> None:
        self.value = language
        self.saves += 1


def test_clip_starts_for_three_samples_and_for_short_audio() -> None:
    """Three clips sit at 25, 50, and 75 percent. Short audio uses the start."""
    assert clip_starts(100, count=3, clip_seconds=30) == [10.0, 35.0, 60.0]
    assert clip_starts(20, count=3, clip_seconds=30) == [0.0]


def test_override_wins_without_detection() -> None:
    """A caller-supplied language is returned and stored, and the engine is not called."""
    engine = FakeEngine()
    store = MemoryLanguage()
    assert resolve_language(engine, [], store, "fr", attempts=2, base_delay=0) == "fr"
    assert engine.detect_calls == []
    assert store.value == "fr"


def test_saved_language_is_reused() -> None:
    """A checkpointed code skips detection on the next run."""
    engine = FakeEngine()
    store = MemoryLanguage("es")
    assert resolve_language(engine, [], store, None, attempts=2, base_delay=0) == "es"
    assert engine.detect_calls == []


def test_detection_is_called_once_and_saved() -> None:
    """The second resolution reads the checkpoint instead of calling the engine."""
    engine = FakeEngine(language="en")
    store = MemoryLanguage()
    clips = [Path("a.wav")]
    first = resolve_language(engine, clips, store, None, attempts=2, base_delay=0)
    second = resolve_language(engine, clips, store, None, attempts=2, base_delay=0)
    assert first == second == "en"
    assert len(engine.detect_calls) == 1


def test_workspace_namespaces_checkpoints_by_engine(tmp_path: Path) -> None:
    """A whisper checkpoint must not satisfy an elevenlabs resume."""
    whisper = JobWorkspace(tmp_path, "whisper")
    whisper.save_language("en")
    other = JobWorkspace(tmp_path, "elevenlabs")
    assert whisper.load_language() == "en"
    assert other.load_language() is None
    assert not list(whisper.engine_dir.rglob("*.tmp"))


def test_temporary_workspace_is_removed() -> None:
    """Owning the directory means close deletes it, including after a successful write."""
    with JobWorkspace(None, "whisper") as workspace:
        root = workspace.root
        workspace.save_language("en")
        assert root.exists()
    assert not root.exists()


@requires_ffmpeg
@pytest.mark.integration
def test_sample_clips_on_long_and_short_audio(tone_with_silences: Path, tmp_path: Path) -> None:
    """Long audio yields one file per sample; short audio yields a single clip."""
    long_dir = tmp_path / "long"
    long_clips = sample_clips(
        tone_with_silences,
        40,
        long_dir,
        count=3,
        clip_seconds=2,
        timeout=30,
    )
    assert len(long_clips) == 3
    short_dir = tmp_path / "short"
    short_clips = sample_clips(
        tone_with_silences,
        2,
        short_dir,
        count=3,
        clip_seconds=30,
        timeout=30,
    )
    assert len(short_clips) == 1


def test_make_settings_helper_is_used_by_language_retries() -> None:
    """The shared helper builds a Settings object tests can pass around."""
    assert make_settings().max_retries == 3
