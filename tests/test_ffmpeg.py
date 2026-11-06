"""ffmpeg normalization, silence detection, and probe behavior."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from tests.conftest import requires_ffmpeg

from transcriber.audio.ffmpeg import _run, detect_silences, normalize, probe
from transcriber.exceptions import AudioProcessingError


def test_missing_executable_is_a_clear_error() -> None:
    """A missing binary is reported as audio processing, not a raw FileNotFoundError."""
    with pytest.raises(AudioProcessingError, match="not installed"):
        _run(["transcriber-missing-bin"], timeout=5)


def test_subprocess_timeout_is_wrapped() -> None:
    """The helper enforces the timeout and does not leak TimeoutExpired."""
    with pytest.raises(AudioProcessingError, match="timed out"):
        _run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.2)


@requires_ffmpeg
@pytest.mark.integration
def test_normalize_yields_16k_mono(tmp_path: Path) -> None:
    """Any input layout becomes 16 kHz mono PCM, checked with ffprobe."""
    source = tmp_path / "stereo.wav"
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
            "sine=frequency=440:sample_rate=44100:duration=0.4",
            "-ac",
            "2",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    destination = tmp_path / "out.wav"
    normalize(source, destination, timeout=30)
    stream = _first_stream(destination)
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1
    assert stream["codec_name"] == "pcm_s16le"


@requires_ffmpeg
@pytest.mark.integration
def test_detect_silences_finds_known_gaps(tone_with_silences: Path) -> None:
    """The synthetic file has a one-second gap every ten seconds."""
    midpoints = detect_silences(tone_with_silences, noise_db=-35, min_duration=0.4, timeout=30)
    expected = [0.5, 10.5, 20.5, 30.5]
    assert len(midpoints) >= len(expected)
    for target in expected:
        assert any(abs(point - target) < 0.25 for point in midpoints)


@requires_ffmpeg
@pytest.mark.integration
def test_probe_rejects_non_audio(tmp_path: Path) -> None:
    """A renamed text file has no audio stream, even with a .wav suffix."""
    fake = tmp_path / "notes.wav"
    fake.write_text("this is not audio", encoding="utf-8")
    info = probe(fake, timeout=30)
    assert info.has_audio is False


@requires_ffmpeg
@pytest.mark.integration
def test_normalize_of_corrupt_input_raises(tmp_path: Path) -> None:
    """Decoder failures stay inside AudioProcessingError and keep stderr short."""
    source = tmp_path / "bad.wav"
    source.write_text("not a wav", encoding="utf-8")
    with pytest.raises(AudioProcessingError, match="ffmpeg failed") as exc_info:
        normalize(source, tmp_path / "out.wav", timeout=30)
    assert len(str(exc_info.value)) < 800
    assert not (tmp_path / "out.wav").exists()


def _first_stream(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    stream = payload["streams"][0]
    if not isinstance(stream, dict):
        raise AssertionError("ffprobe returned no stream object")
    return stream
