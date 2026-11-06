"""Shared fakes and synthetic audio for the transcription tests."""

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from transcriber.audio.ffmpeg import AudioInfo
from transcriber.config import Settings
from transcriber.exceptions import PermanentEngineError
from transcriber.models import Segment

FFMPEG_MISSING = shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None
requires_ffmpeg = pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg and ffprobe are required")


class FakeEngine:
    """Records calls and returns one segment per chunk, numbered so none collapse."""

    name = "fake"

    def __init__(
        self,
        *,
        language: str | None = "en",
        fail_at: int | None = None,
        error: Exception | None = None,
    ) -> None:
        self.language = language
        self.fail_at = fail_at
        self.error = error or PermanentEngineError("forced failure")
        self.detect_calls: list[list[Path]] = []
        self.transcribe_calls: list[tuple[Path, str | None]] = []

    def detect_language(self, clips: Sequence[Path]) -> str | None:
        """Remember the clips and return the configured language."""
        self.detect_calls.append(list(clips))
        return self.language

    def transcribe(self, chunk: Path, language: str | None) -> list[Segment]:
        """Return a chunk-relative segment, or raise once ``fail_at`` is reached."""
        self.transcribe_calls.append((chunk, language))
        if self.fail_at is not None and len(self.transcribe_calls) >= self.fail_at:
            raise self.error
        number = len(self.transcribe_calls)
        return [Segment(start=0.0, end=1.0, text=f"hello {number}")]


def make_settings(**overrides: object) -> Settings:
    """Settings with instant retries unless a test asks otherwise."""
    values: dict[str, object] = {
        "retry_base_delay": 0.0,
        "request_timeout_seconds": 30,
        "max_retries": 3,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def patch_media(
    monkeypatch: pytest.MonkeyPatch,
    *,
    duration: float = 100.0,
    has_audio: bool = True,
    silences: Sequence[float] = (),
) -> dict[str, int]:
    """Replace ffmpeg with in-memory stand-ins and return call counts."""
    from transcriber.audio import ffmpeg

    counts = {"normalize": 0, "probe": 0, "cut": 0, "silences": 0}

    def fake_probe(_path: Path, _timeout: float) -> AudioInfo:
        counts["probe"] += 1
        return AudioInfo(duration=duration, has_audio=has_audio)

    def fake_normalize(_src: Path, dst: Path, _timeout: float) -> None:
        counts["normalize"] += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"RIFF")

    def fake_cut(_src: Path, _start: float, _length: float, dst: Path, _timeout: float) -> None:
        counts["cut"] += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"RIFF")

    def fake_silences(_wav: Path, _noise: float, _minimum: float, _timeout: float) -> list[float]:
        counts["silences"] += 1
        return list(silences)

    monkeypatch.setattr(ffmpeg, "probe", fake_probe)
    monkeypatch.setattr(ffmpeg, "normalize", fake_normalize)
    monkeypatch.setattr(ffmpeg, "cut", fake_cut)
    monkeypatch.setattr(ffmpeg, "detect_silences", fake_silences)
    return counts


def write_wav(path: Path, seconds: float = 0.5) -> None:
    """Write a real mono PCM WAV so ffprobe accepts it without a model."""
    import math
    import struct
    import wave

    rate = 16_000
    count = max(1, int(seconds * rate))
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = bytearray()
        for index in range(count):
            sample = int(8000 * math.sin(2 * math.pi * 440 * index / rate))
            frames += struct.pack("<h", sample)
        handle.writeframes(frames)


def render_pattern(path: Path, seconds: int) -> None:
    """Synthesize tone with one second of silence every ten seconds."""
    expression = f"aevalsrc=0.2*sin(2*PI*440*t)*gte(mod(t\\,10)\\,1):s=16000:d={seconds}"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            expression,
            "-ac",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def tone_with_silences(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Forty seconds of tone with silence at 0-1, 10-11, 20-21, and 30-31."""
    if FFMPEG_MISSING:
        pytest.skip("ffmpeg is not installed")
    path = tmp_path_factory.mktemp("audio") / "tone.wav"
    render_pattern(path, 40)
    return path
