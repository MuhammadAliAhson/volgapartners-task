"""Request and response models for the HTTP API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from transcriber.models import TranscriptionResult

JobStatus = Literal["queued", "processing", "completed", "failed"]


class TranscriptionAccepted(BaseModel):
    """Body returned as soon as an upload has been queued."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: Literal["queued"]


class JobView(BaseModel):
    """Public view of a job. Paths and secrets are intentionally absent."""

    model_config = ConfigDict(frozen=True)

    id: str
    status: JobStatus
    progress: float
    engine: str
    created_at: datetime
    updated_at: datetime
    error: str | None
    result: TranscriptionResult | None
