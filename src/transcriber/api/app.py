"""FastAPI application.

``create_app`` accepts the settings and an engine factory so tests can inject a
fake engine. The module-level ``app`` is the object uvicorn serves.
"""

import logging
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, Query, Request, UploadFile
from fastapi.responses import Response

from transcriber.api.errors import ApiError, register_error_handlers
from transcriber.api.jobs import (
    EngineFactory,
    InMemoryJobStore,
    Job,
    JobStore,
    assert_audio,
    delete_job_files,
    extension_of,
    new_job_id,
    submit_job,
    utc_now,
    write_upload,
)
from transcriber.api.schemas import JobView, TranscriptionAccepted
from transcriber.config import Settings
from transcriber.engines import get_engine
from transcriber.exporters import to_json, to_srt, to_txt, to_vtt

logger = logging.getLogger(__name__)

_EXPORTERS = {
    "json": (to_json, "application/json"),
    "srt": (to_srt, "application/x-subrip"),
    "vtt": (to_vtt, "text/vtt"),
    "txt": (to_txt, "text/plain; charset=utf-8"),
}
_ENGINES = {"whisper", "elevenlabs"}


def create_app(
    settings: Settings | None = None,
    engine_factory: EngineFactory | None = None,
    store: JobStore | None = None,
) -> FastAPI:
    """Build the API. The executor is created in the lifespan and shut down on exit."""
    resolved_settings = settings or Settings()
    resolved_factory = engine_factory or get_engine
    resolved_store = store or InMemoryJobStore()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        executor = ThreadPoolExecutor(
            max_workers=resolved_settings.max_concurrent_jobs,
            thread_name_prefix="transcriber",
        )
        app.state.executor = executor
        try:
            yield
        finally:
            executor.shutdown(wait=True, cancel_futures=False)

    app = FastAPI(title="Transcription pipeline", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.engine_factory = resolved_factory
    app.state.store = resolved_store
    register_error_handlers(app)
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/v1/transcriptions", create_transcription, methods=["POST"], status_code=202)
    app.add_api_route("/v1/transcriptions/{job_id}", read_transcription, methods=["GET"])
    app.add_api_route("/v1/transcriptions/{job_id}/export", export_transcription, methods=["GET"])
    app.add_api_route(
        "/v1/transcriptions/{job_id}",
        delete_transcription,
        methods=["DELETE"],
        status_code=204,
    )
    return app


def health() -> dict[str, str]:
    """Liveness probe. It does not touch the model or the network."""
    return {"status": "ok"}


async def create_transcription(
    request: Request,
    file: UploadFile,
    engine: str | None = Form(default=None),
    language: str | None = Form(default=None),
) -> TranscriptionAccepted:
    """Accept an upload, probe it, and queue transcription."""
    settings: Settings = request.app.state.settings
    chosen = _engine_name(engine, settings)
    language_code = language.strip() if language and language.strip() else None
    job_id = new_job_id()
    ext = extension_of(file.filename, settings.supported_extensions)
    directory = settings.data_dir / job_id
    audio_path = directory / f"audio.{ext}"
    try:
        await write_upload(file, audio_path, settings.max_upload_mb * 1024 * 1024)
        assert_audio(audio_path, settings)
    except Exception:
        delete_job_files(
            _placeholder(job_id, chosen, audio_path, directory / "work", language_code)
        )
        raise
    job = _placeholder(job_id, chosen, audio_path, directory / "work", language_code)
    request.app.state.store.add(job)
    submit_job(
        request.app.state.executor,
        request.app.state.store,
        settings,
        request.app.state.engine_factory,
        job_id,
    )
    logger.info("Queued job %s engine=%s", job_id, chosen)
    return TranscriptionAccepted(id=job_id, status="queued")


def read_transcription(job_id: str, request: Request) -> JobView:
    """Return status, progress, and the result once the job has finished."""
    job = _require(request.app.state.store, job_id)
    return _view(job)


def export_transcription(
    job_id: str,
    request: Request,
    export_format: str = Query(default="json", alias="format"),
) -> Response:
    """Download a finished transcript. Unfinished jobs return 409."""
    chosen = export_format.lower()
    if chosen not in _EXPORTERS:
        raise ApiError(422, "validation_error", "Unsupported export format")
    job = _require(request.app.state.store, job_id)
    if job.status != "completed" or job.result is None:
        raise ApiError(409, "conflict", "Transcription is not complete")
    render, media_type = _EXPORTERS[chosen]
    return Response(
        content=render(job.result),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="transcript.{chosen}"'},
    )


def delete_transcription(job_id: str, request: Request) -> Response:
    """Delete the job record and its files."""
    job = request.app.state.store.delete(job_id)
    if job is None:
        raise ApiError(404, "not_found", "Transcription not found")
    delete_job_files(job)
    return Response(status_code=204)


def _engine_name(requested: str | None, settings: Settings) -> str:
    chosen = requested or settings.default_engine
    if chosen not in _ENGINES:
        raise ApiError(422, "validation_error", "Unsupported engine")
    return chosen


def _require(store: JobStore, job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise ApiError(404, "not_found", "Transcription not found")
    return job


def _view(job: Job) -> JobView:
    status = job.status
    if status not in {"queued", "processing", "completed", "failed"}:
        status = "failed"
    return JobView(
        id=job.id,
        status=status,  # type: ignore[arg-type]
        progress=job.progress,
        engine=job.engine,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        result=job.result,
    )


def _placeholder(
    job_id: str,
    engine: str,
    audio_path: Path,
    workdir: Path,
    language: str | None,
) -> Job:
    now = utc_now()
    return Job(
        id=job_id,
        status="queued",
        progress=0.0,
        engine=engine,
        created_at=now,
        updated_at=now,
        language=language,
        error=None,
        result=None,
        audio_path=audio_path,
        workdir=workdir,
    )


app = create_app()
