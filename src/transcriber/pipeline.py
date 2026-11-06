"""Orchestrate validation, normalization, language, chunking, and export prep.

The pipeline receives its engine and settings. It does not construct engines,
so tests can pass a fake and the API can choose the engine per request.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

from transcriber.audio import ffmpeg
from transcriber.audio.chunking import build_plan, find_cuts
from transcriber.audio.ffmpeg import AudioInfo
from transcriber.config import Settings
from transcriber.engines.base import SpeechToTextEngine
from transcriber.exceptions import UnsupportedFormatError
from transcriber.language import resolve_language, sample_clips
from transcriber.models import ChunkSpec, Segment, TranscriptionResult
from transcriber.postprocess import postprocess
from transcriber.retry import retry
from transcriber.workspace import JobWorkspace

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    """Run one audio file through the shared normalize → chunk → merge flow."""

    def __init__(self, engine: SpeechToTextEngine, settings: Settings) -> None:
        self._engine = engine
        self._settings = settings

    def run(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        workdir: Path | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe ``audio_path``. A supplied ``workdir`` is kept for resume."""
        logger.info("Starting transcription engine=%s", self._engine.name)
        workspace = JobWorkspace(workdir, self._engine.name)
        try:
            return self._run(audio_path, language, workspace, on_progress)
        finally:
            if workdir is None:
                workspace.close()

    def _run(
        self,
        audio_path: Path,
        language: str | None,
        workspace: JobWorkspace,
        on_progress: Callable[[float], None] | None,
    ) -> TranscriptionResult:
        self._validate(audio_path)
        self._ensure_normalized(audio_path, workspace)
        duration = ffmpeg.probe(
            workspace.normalized_path(),
            self._settings.request_timeout_seconds,
        ).duration
        if duration <= 0:
            logger.info("Audio duration is zero; returning an empty transcript")
            return postprocess([], engine=self._engine.name, language=language, duration=0)
        resolved = self._resolve_language(workspace, duration, language)
        plan = self._plan(workspace, duration)
        segments = self._transcribe_plan(workspace, plan, resolved, on_progress)
        result = postprocess(
            segments,
            engine=self._engine.name,
            language=resolved,
            duration=duration,
        )
        logger.info(
            "Finished language=%s duration=%.2f words=%s segments=%s",
            result.language,
            result.duration,
            result.word_count,
            len(result.segments),
        )
        return result

    def _validate(self, audio_path: Path) -> AudioInfo:
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        extension = audio_path.suffix.lower().lstrip(".")
        if extension not in self._settings.supported_extensions:
            raise UnsupportedFormatError(f"Unsupported audio extension: {extension or 'none'}")
        info = ffmpeg.probe(audio_path, self._settings.request_timeout_seconds)
        if not info.has_audio:
            raise UnsupportedFormatError(f"No audio stream in {audio_path.name}")
        logger.info("Validated %s (%.2fs)", audio_path.name, info.duration)
        return info

    def _ensure_normalized(self, source: Path, workspace: JobWorkspace) -> None:
        destination = workspace.normalized_path()
        if destination.is_file():
            logger.info("Reusing normalized audio")
            return
        logger.info("Normalizing audio to 16 kHz mono WAV")
        ffmpeg.normalize(source, destination, self._settings.request_timeout_seconds)

    def _resolve_language(
        self,
        workspace: JobWorkspace,
        duration: float,
        override: str | None,
    ) -> str | None:
        previous = workspace.load_language()
        clips = self._sample_if_needed(workspace, duration, override, previous)
        try:
            resolved = resolve_language(
                self._engine,
                clips,
                workspace,
                override,
                attempts=self._settings.max_retries,
                base_delay=self._settings.retry_base_delay,
            )
        finally:
            for clip in clips:
                clip.unlink(missing_ok=True)
        if previous and resolved != previous:
            logger.info("Language changed; dropping chunk checkpoints")
            workspace.clear_chunks()
        logger.info("Language for this run: %s", resolved or "unknown")
        return resolved

    def _sample_if_needed(
        self,
        workspace: JobWorkspace,
        duration: float,
        override: str | None,
        previous: str | None,
    ) -> list[Path]:
        if override is not None or previous is not None:
            return []
        return sample_clips(
            workspace.normalized_path(),
            duration,
            workspace.clip_dir(),
            count=self._settings.language_samples,
            clip_seconds=self._settings.language_clip_seconds,
            timeout=self._settings.request_timeout_seconds,
        )

    def _plan(self, workspace: JobWorkspace, duration: float) -> list[ChunkSpec]:
        existing = workspace.load_plan()
        if existing is not None:
            logger.info("Reusing chunk plan with %s chunks", len(existing))
            return existing
        target = self._settings.chunk_seconds
        window = self._settings.silence_window_seconds
        cuts = self._cuts(workspace, duration, target, window)
        plan = build_plan(cuts, duration)
        workspace.save_plan(plan)
        logger.info("Planned %s chunk(s)", len(plan))
        return plan

    def _cuts(
        self,
        workspace: JobWorkspace,
        duration: float,
        target: float,
        window: float,
    ) -> list[float]:
        if duration <= target + window:
            logger.info("Audio fits in a single chunk")
            return []
        logger.info("Detecting silences for chunk planning")
        silences = ffmpeg.detect_silences(
            workspace.normalized_path(),
            self._settings.silence_noise_db,
            self._settings.silence_min_duration,
            self._settings.request_timeout_seconds,
        )
        return find_cuts(silences, duration, target, window)

    def _transcribe_plan(
        self,
        workspace: JobWorkspace,
        plan: Sequence[ChunkSpec],
        language: str | None,
        on_progress: Callable[[float], None] | None,
    ) -> list[Segment]:
        merged: list[Segment] = []
        total = len(plan)
        for position, chunk in enumerate(plan):
            merged.extend(self._one_chunk(workspace, chunk, language))
            if on_progress is not None and total:
                on_progress((position + 1) / total)
        return merged

    def _one_chunk(
        self,
        workspace: JobWorkspace,
        chunk: ChunkSpec,
        language: str | None,
    ) -> list[Segment]:
        cached = workspace.load_chunk(chunk.index)
        if cached is not None:
            logger.info("Reusing checkpoint for chunk %s", chunk.index)
            return cached
        logger.info("Transcribing chunk %s", chunk.index)
        audio = workspace.chunk_audio_path(chunk.index)
        try:
            ffmpeg.cut(
                workspace.normalized_path(),
                chunk.start,
                max(0.0, chunk.end - chunk.start),
                audio,
                self._settings.request_timeout_seconds,
            )
            relative = retry(
                lambda: self._engine.transcribe(audio, language),
                attempts=self._settings.max_retries,
                base_delay=self._settings.retry_base_delay,
            )
            shifted = _shift(relative, chunk.start)
            workspace.save_chunk(chunk.index, shifted)
            return shifted
        finally:
            audio.unlink(missing_ok=True)


def _shift(segments: Sequence[Segment], offset: float) -> list[Segment]:
    if offset == 0:
        return list(segments)
    return [
        segment.model_copy(update={"start": segment.start + offset, "end": segment.end + offset})
        for segment in segments
    ]
