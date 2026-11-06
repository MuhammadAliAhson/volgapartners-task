"""Upload, poll, export, and the documented error codes."""

import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.conftest import FakeEngine, make_settings, requires_ffmpeg, write_wav

from transcriber.api.app import create_app
from transcriber.api.jobs import Job, utc_now
from transcriber.config import Settings


@pytest.fixture
def api(tmp_path: Path) -> Iterator[tuple[TestClient, FakeEngine, Settings]]:
    """An API whose engine is the fake and whose files stay in tmp."""
    settings = make_settings(
        data_dir=tmp_path / "data",
        max_upload_mb=1,
        max_concurrent_jobs=1,
        chunk_seconds=600,
        silence_window_seconds=60,
    )
    engine = FakeEngine()
    application = create_app(settings, lambda _name, _settings: engine)
    with TestClient(application) as client:
        yield client, engine, settings


def test_health(api: tuple[TestClient, FakeEngine, Settings]) -> None:
    """The probe does not depend on a model."""
    client, _engine, _settings = api
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_job_is_404(api: tuple[TestClient, FakeEngine, Settings]) -> None:
    """Unknown ids are a not-found error in the shared JSON shape."""
    client, _engine, _settings = api
    response = client.get("/v1/transcriptions/missing")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_validation_error_shape(api: tuple[TestClient, FakeEngine, Settings]) -> None:
    """A missing file is 422 and does not use FastAPI's default body."""
    client, _engine, _settings = api
    response = client.post("/v1/transcriptions", data={"engine": "whisper"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_unknown_engine_is_422(
    api: tuple[TestClient, FakeEngine, Settings], tmp_path: Path
) -> None:
    """Engine names outside the two supported values are rejected."""
    client, _engine, _settings = api
    audio = tmp_path / "clip.wav"
    write_wav(audio, 0.2)
    with audio.open("rb") as handle:
        response = client.post(
            "/v1/transcriptions",
            files={"file": ("clip.wav", handle, "audio/wav")},
            data={"engine": "other"},
        )
    assert response.status_code == 422


def test_export_before_completion_is_409(api: tuple[TestClient, FakeEngine, Settings]) -> None:
    """Exporting a queued job conflicts instead of returning a partial file."""
    client, _engine, _settings = api
    job = _queued_job()
    client.app.state.store.add(job)  # type: ignore[attr-defined]
    response = client.get(f"/v1/transcriptions/{job.id}/export", params={"format": "txt"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "conflict"


def test_upload_over_the_limit_is_413(api: tuple[TestClient, FakeEngine, Settings]) -> None:
    """The limit is enforced while streaming, before the file is probed."""
    client, _engine, _settings = api
    payload = b"0" * (1024 * 1024 + 8)
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("big.wav", payload, "audio/wav")},
        data={"engine": "whisper"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


@requires_ffmpeg
@pytest.mark.integration
def test_rejects_non_audio_with_415(
    api: tuple[TestClient, FakeEngine, Settings],
) -> None:
    """A .wav suffix does not make a text file audio."""
    client, _engine, _settings = api
    response = client.post(
        "/v1/transcriptions",
        files={"file": ("notes.wav", b"this is text", "audio/wav")},
        data={"engine": "whisper"},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_delete_missing_is_404(api: tuple[TestClient, FakeEngine, Settings]) -> None:
    """Deleting an unknown job is 404."""
    client, _engine, _settings = api
    response = client.delete("/v1/transcriptions/missing")
    assert response.status_code == 404


@requires_ffmpeg
@pytest.mark.integration
def test_upload_poll_export_and_delete(
    api: tuple[TestClient, FakeEngine, Settings], tmp_path: Path
) -> None:
    """The happy path is 202, then completed, then an SRT download, then 204."""
    client, _engine, _settings = api
    audio = tmp_path / "clip.wav"
    write_wav(audio, 0.4)
    with audio.open("rb") as handle:
        created = client.post(
            "/v1/transcriptions",
            files={"file": ("clip.wav", handle, "audio/wav")},
            data={"engine": "whisper", "language": "en"},
        )
    assert created.status_code == 202
    job_id = created.json()["id"]
    assert created.json()["status"] == "queued"
    body = _wait(client, job_id)
    assert body["status"] == "completed"
    assert body["result"]["text"] == "hello 1"
    assert body["result"]["segments"][0]["start"] == 0
    exported = client.get(f"/v1/transcriptions/{job_id}/export", params={"format": "srt"})
    assert exported.status_code == 200
    assert "hello 1" in exported.text
    bad = client.get(f"/v1/transcriptions/{job_id}/export", params={"format": "pdf"})
    assert bad.status_code == 422
    deleted = client.delete(f"/v1/transcriptions/{job_id}")
    assert deleted.status_code == 204
    assert client.get(f"/v1/transcriptions/{job_id}").status_code == 404


def _wait(client: TestClient, job_id: str) -> dict[str, object]:
    deadline = time.time() + 20
    body: dict[str, object] = {}
    while time.time() < deadline:
        response = client.get(f"/v1/transcriptions/{job_id}")
        body = response.json()
        if body.get("status") in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    return body


def _queued_job() -> Job:
    now = utc_now()
    return Job(
        id="job-queued",
        status="queued",
        progress=0.0,
        engine="whisper",
        created_at=now,
        updated_at=now,
        language=None,
        error=None,
        result=None,
        audio_path=Path("data/job-queued/audio.wav"),
        workdir=Path("data/job-queued/work"),
    )
