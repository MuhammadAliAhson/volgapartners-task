"""ffmpeg and ffprobe operations.

Every media file is decoded to 16 kHz mono PCM so the rest of the pipeline has
one layout to cut, sample, and transcribe. Commands are argument lists with a
timeout; the shell is never involved.
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from transcriber.exceptions import AudioProcessingError

logger = logging.getLogger(__name__)

_STDERR_TAIL = 500
_START_RE = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_END_RE = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class AudioInfo:
    """Duration and whether ffprobe found an audio stream.

    Extensions are not trusted. Callers turn ``has_audio=False`` into
    ``UnsupportedFormatError``.
    """

    duration: float
    has_audio: bool


def probe(path: Path, timeout: float) -> AudioInfo:
    """Read duration and stream types. A non-media file reports no audio."""
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout,
    )
    payload = _payload_or_none(completed.stdout)
    if payload is None:
        _raise_failed(completed, "ffprobe")
    info = _info_from_payload(payload)
    if completed.returncode != 0 and info.has_audio:
        _raise_failed(completed, "ffprobe")
    return info


def normalize(src: Path, dst: Path, timeout: float) -> None:
    """Decode any container to 16 kHz mono PCM WAV, dropping video streams.

    The output is written to a sibling temp file and then renamed, so a crash
    cannot leave a half-written file at ``dst`` for a later resume to trust.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Keep a media extension. ffmpeg refuses to mux to "*.wav.partial".
    partial = dst.with_name(f".{dst.stem}.partial{dst.suffix}")
    try:
        completed = _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(partial),
            ],
            timeout,
        )
        _raise_if_failed(completed, "ffmpeg")
        os.replace(partial, dst)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def cut(src: Path, start: float, length: float, dst: Path, timeout: float) -> None:
    """Copy a time range out of a WAV. ``-ss`` is before ``-i`` for a fast seek."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src),
            "-t",
            f"{length:.3f}",
            "-c",
            "copy",
            str(dst),
        ],
        timeout,
    )
    _raise_if_failed(completed, "ffmpeg")


def detect_silences(
    wav: Path,
    noise_db: float,
    min_duration: float,
    timeout: float,
) -> list[float]:
    """Return the midpoint of each silence, in seconds.

    Midpoints are where a cut is least likely to land inside a word. The
    filter logs at info level, so the log level is pinned rather than left to
    the caller's environment.
    """
    silence_filter = f"silencedetect=noise={noise_db}dB:d={min_duration}"
    completed = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(wav),
            "-af",
            silence_filter,
            "-f",
            "null",
            "-",
        ],
        timeout,
    )
    _raise_if_failed(completed, "ffmpeg")
    return _midpoints(completed.stderr or "")


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    logger.debug("Running %s", cmd[0])
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AudioProcessingError(f"{cmd[0]} is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(f"{cmd[0]} timed out after {timeout:.0f}s") from exc


def _payload_or_none(stdout: str) -> dict[str, object] | None:
    text = (stdout or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _info_from_payload(payload: dict[str, object]) -> AudioInfo:
    streams = payload.get("streams")
    has_audio = False
    if isinstance(streams, list):
        has_audio = any(
            isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
        )
    return AudioInfo(duration=_duration(payload), has_audio=has_audio)


def _duration(payload: dict[str, object]) -> float:
    fmt = payload.get("format")
    if isinstance(fmt, dict):
        parsed = _as_float(fmt.get("duration"))
        if parsed is not None:
            return parsed
    streams = payload.get("streams")
    if isinstance(streams, list):
        for stream in streams:
            if isinstance(stream, dict) and stream.get("codec_type") == "audio":
                parsed = _as_float(stream.get("duration"))
                if parsed is not None:
                    return parsed
    return 0.0


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _midpoints(stderr: str) -> list[float]:
    starts = [float(match) for match in _START_RE.findall(stderr)]
    ends = [float(match) for match in _END_RE.findall(stderr)]
    count = min(len(starts), len(ends))
    return [(starts[index] + ends[index]) / 2 for index in range(count)]


def _raise_if_failed(completed: subprocess.CompletedProcess[str], tool: str) -> None:
    if completed.returncode != 0:
        _raise_failed(completed, tool)


def _raise_failed(completed: subprocess.CompletedProcess[str], tool: str) -> NoReturn:
    tail = (completed.stderr or "").strip()[-_STDERR_TAIL:]
    raise AudioProcessingError(f"{tool} failed: {tail}")
