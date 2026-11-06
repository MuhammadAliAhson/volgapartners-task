"""In-memory job tracking and the worker that runs the pipeline.

The store is a protocol so tests can substitute it. The in-memory implementation
is guarded by a lock because the API thread and the worker thread both touch it.
"""

import logging
import shutil
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from fastapi import UploadFile

from transcriber.api.errors import ApiError
from transcriber.audio.ffmpeg import probe
from transcriber.config import Settings
from transcriber.engines.base import SpeechToTextEngine
from transcriber.exceptions import AudioProcessingError, TranscriberError
from transcriber.models import TranscriptionResult
from transcriber.pipeline import TranscriptionPipeline

logger = logging.getLogger(__name__)

STREAM_CHUNK_BYTES = 1024 * 1024
EngineFactory = Callable[[str, Settings], SpeechToTextEngine]


@dataclass(frozen=True)
class Job:
    """One accepted transcription. Replaced wholesale on every state change."""

    id: str
    status: str
    progress: float
    engine: str
    created_at: datetime
    updated_at: datetime
    language: str | None
    error: str | None
    result: TranscriptionResult | None
    audio_path: Path
    workdir: Path


class JobStore(Protocol):
    """Persistence for job records. Implementations must be safe for the worker."""

    def add(self, job: Job) -> None:
        """Insert a new job."""

    def get(self, job_id: str) -> Job | None:
        """Return the job, or ``None`` when it was never created or was deleted."""

    def save(self, job: Job) -> None:
        """Replace an existing job. A deleted id is left deleted."""

    def delete(self, job_id: str) -> Job | None:
        """Remove the job and return it, or ``None`` if it was already gone."""


class InMemoryJobStore:
    """Process-local job map. Lost on restart; files under ``data_dir`` may remain."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def save(self, job: Job) -> None:
        with self._lock:
            if job.id in self._jobs:
                self._jobs[job.id] = job

    def delete(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.pop(job_id, None)


def new_job_id() -> str:
    """Identifier that is never derived from the client's filename."""
    return uuid.uuid4().hex


def utc_now() -> datetime:
    """Timezone-aware timestamp so API responses do not depend on the host zone."""
    return datetime.now(UTC)


async def write_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    """Stream an upload to disk and abort once it passes ``max_bytes``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with dest.open("wb") as handle:
            while True:
                piece = await upload.read(STREAM_CHUNK_BYTES)
                if not piece:
                    break
                written += len(piece)
                if written > max_bytes:
                    raise ApiError(413, "payload_too_large", "Upload exceeds the configured limit")
                handle.write(piece)
    except Exception:
        dest.unlink(missing_ok=True)
        raise


def assert_audio(path: Path, settings: Settings) -> None:
    """Reject uploads ffprobe does not consider audio. The partial file is removed."""
    try:
        info = probe(path, settings.request_timeout_seconds)
    except AudioProcessingError as exc:
        path.unlink(missing_ok=True)
        raise ApiError(422, "audio_processing_error", str(exc)) from exc
    if not info.has_audio:
        path.unlink(missing_ok=True)
        raise ApiError(415, "unsupported_media_type", "File does not contain an audio stream")


def extension_of(filename: str | None, allowed: tuple[str, ...]) -> str:
    """Return a safe extension, or raise 415. The original name is not kept."""
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    if suffix not in allowed:
        message = f"Unsupported audio extension: {suffix or 'none'}"
        raise ApiError(415, "unsupported_media_type", message)
    return suffix


def submit_job(
    executor: Executor,
    store: JobStore,
    settings: Settings,
    engine_factory: EngineFactory,
    job_id: str,
) -> None:
    """Queue the pipeline on the bounded worker pool."""
    executor.submit(execute_job, store, settings, engine_factory, job_id)


def execute_job(
    store: JobStore,
    settings: Settings,
    engine_factory: EngineFactory,
    job_id: str,
) -> None:
    """Run one job to completion. A deleted job is left deleted."""
    job = store.get(job_id)
    if job is None:
        return
    store.save(replace(job, status="processing", updated_at=utc_now()))
    engine: SpeechToTextEngine | None = None
    try:
        engine = engine_factory(job.engine, settings)
        pipeline = TranscriptionPipeline(engine, settings)
        result = pipeline.run(
            job.audio_path,
            language=job.language,
            workdir=job.workdir,
            on_progress=lambda fraction: _record_progress(store, job_id, fraction),
        )
        _finish(store, job_id, result)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        _fail(store, job_id, _failure_message(exc))
    finally:
        closer = getattr(engine, "close", None)
        if callable(closer):
            closer()


def delete_job_files(job: Job) -> None:
    """Remove the upload directory, including checkpoints."""
    shutil.rmtree(job.audio_path.parent, ignore_errors=True)


def _record_progress(store: JobStore, job_id: str, fraction: float) -> None:
    current = store.get(job_id)
    if current is None:
        return
    store.save(replace(current, progress=fraction, status="processing", updated_at=utc_now()))


def _finish(store: JobStore, job_id: str, result: TranscriptionResult) -> None:
    current = store.get(job_id)
    if current is None:
        return
    store.save(
        replace(
            current,
            status="completed",
            progress=1.0,
            result=result,
            error=None,
            updated_at=utc_now(),
        )
    )


def _fail(store: JobStore, job_id: str, message: str) -> None:
    current = store.get(job_id)
    if current is None:
        return
    store.save(replace(current, status="failed", error=message, updated_at=utc_now()))


def _failure_message(exc: Exception) -> str:
    if isinstance(exc, TranscriberError):
        return str(exc)
    if isinstance(exc, FileNotFoundError):
        return "Audio file is missing"
    return "Transcription failed"
