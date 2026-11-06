"""JSON error responses.

Clients always receive ``{"error": {"code", "message"}}``. Stack traces stay in
the server log.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from transcriber.exceptions import TranscriberError

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An HTTP failure with a stable machine-readable code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build the single error document used by every handler."""
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def register_error_handlers(app: FastAPI) -> None:
    """Attach handlers so unexpected failures cannot leak a traceback."""

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(HTTPException)
    async def handle_http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return error_response(exc.status_code, _http_code(exc.status_code), message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_request: Request, _exc: RequestValidationError) -> JSONResponse:
        return error_response(422, "validation_error", "Request validation failed")

    @app.exception_handler(TranscriberError)
    async def handle_transcriber(_request: Request, exc: TranscriberError) -> JSONResponse:
        return error_response(500, "transcriber_error", str(exc))

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return error_response(500, "internal_error", "Internal server error")


def _http_code(status_code: int) -> str:
    return {
        404: "not_found",
        409: "conflict",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
    }.get(status_code, "http_error")
