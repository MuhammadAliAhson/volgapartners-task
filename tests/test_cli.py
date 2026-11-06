"""CLI exit codes and the files a successful run writes."""

from pathlib import Path

import pytest
from tests.conftest import FakeEngine, requires_ffmpeg, write_wav

from transcriber.cli import main
from transcriber.config import Settings


def test_unsupported_file_exits_2(tmp_path: Path) -> None:
    """A disallowed extension is a usage error, not a crash."""
    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    assert main([str(notes), "--out-dir", str(tmp_path)]) == 2


def test_missing_file_exits_1(tmp_path: Path) -> None:
    """A path that does not exist is a runtime failure with a clean message."""
    assert main([str(tmp_path / "missing.mp3"), "--out-dir", str(tmp_path)]) == 1


def test_bad_format_exits_2(tmp_path: Path) -> None:
    """An unknown export format is rejected before transcription."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    assert main([str(audio), "--format", "pdf", "--out-dir", str(tmp_path)]) == 2


@requires_ffmpeg
@pytest.mark.integration
def test_smoke_writes_requested_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The default formats are written and the process exits 0."""
    audio = tmp_path / "clip.wav"
    write_wav(audio, seconds=0.4)
    out = tmp_path / "out"
    fake = FakeEngine()

    def factory(name: str, settings: Settings) -> FakeEngine:
        assert name == "whisper"
        assert settings.whisper_model == "tiny"
        return fake

    monkeypatch.setattr("transcriber.cli.get_engine", factory)
    code = main([str(audio), "--engine", "whisper", "--model", "tiny", "--out-dir", str(out), "-v"])
    assert code == 0
    assert (out / "transcript.json").is_file()
    assert (out / "transcript.srt").is_file()
    assert (out / "transcript.txt").is_file()
    assert not (out / "transcript.vtt").exists()
    assert "hello 1" in (out / "transcript.txt").read_text(encoding="utf-8")
    assert "words=" in capsys.readouterr().out
