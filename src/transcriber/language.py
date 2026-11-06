"""Language sampling and one-shot resolution.

Detecting the language once, then forcing it on every chunk, avoids a quiet or
noisy chunk flipping the language and producing garbage. It is also faster.
Code-switched audio is the known case this cannot represent.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from transcriber.audio import ffmpeg
from transcriber.engines.base import SpeechToTextEngine
from transcriber.retry import retry


class LanguageStore(Protocol):
    """The slice of the workspace the language step needs."""

    def load_language(self) -> str | None:
        """Return a previously stored language code."""

    def save_language(self, language: str) -> None:
        """Persist the language that later chunks must use."""


def clip_starts(total: float, *, count: int, clip_seconds: float) -> list[float]:
    """Return clip start times. Short audio yields a single clip at the start.

    For three clips the fractions are 25%, 50%, and 75%. One clip can easily be
    music or silence, so several evenly spaced clips are combined by the engine.
    """
    if total <= 0 or count <= 0 or clip_seconds <= 0:
        return []
    if total <= clip_seconds or count <= 1:
        return [0.0]
    fractions = [(index + 1) / (count + 1) for index in range(count)]
    starts = [_aligned_start(total, fraction, clip_seconds) for fraction in fractions]
    unique: list[float] = []
    for start in starts:
        if unique and abs(start - unique[-1]) < 0.01:
            continue
        unique.append(start)
    return unique


def sample_clips(
    wav: Path,
    total: float,
    directory: Path,
    *,
    count: int,
    clip_seconds: float,
    timeout: float,
) -> list[Path]:
    """Cut the clips described by :func:`clip_starts` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, start in enumerate(clip_starts(total, count=count, clip_seconds=clip_seconds)):
        length = min(clip_seconds, max(0.0, total - start))
        if length <= 0:
            continue
        destination = directory / f"clip-{index:02d}.wav"
        ffmpeg.cut(wav, start, length, destination, timeout)
        paths.append(destination)
    return paths


def resolve_language(
    engine: SpeechToTextEngine,
    clips: Sequence[Path],
    store: LanguageStore,
    override: str | None,
    *,
    attempts: int,
    base_delay: float,
) -> str | None:
    """Honor an override, else a checkpoint, else detect once and store it.

    Detection goes through the shared retry helper because it is an engine call.
    An override is stored too, so a resumed run does not detect a different code.
    """
    if override:
        if store.load_language() != override:
            store.save_language(override)
        return override
    saved = store.load_language()
    if saved:
        return saved
    detected = retry(
        lambda: engine.detect_language(clips),
        attempts=attempts,
        base_delay=base_delay,
    )
    if detected:
        store.save_language(detected)
    return detected


def _aligned_start(total: float, fraction: float, clip_seconds: float) -> float:
    start = total * fraction - clip_seconds / 2
    latest = max(0.0, total - clip_seconds)
    if start < 0:
        return 0.0
    if start > latest:
        return latest
    return start
