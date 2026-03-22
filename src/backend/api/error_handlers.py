"""
The following are exceptions handlers for our custom exceptions which may arise 
during the OCR, summarization, search, or document retrieval phases.
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.exceptions import AppError

logger = logging.getLogger(__name__)


def _error_payload(code: str, message: str, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.code, exc.message, exc.details),
        )

    # @app.exception_handler(RequestValidationError)
    # async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    #     return JSONResponse(
    #         status_code=422,
    #         content=_error_payload("validation_error", "Request validation failed", exc.errors()),
    #     )

    # @app.exception_handler(HTTPException)
    # async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    #     details = exc.detail if isinstance(exc.detail, dict) else None
    #     message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    #     return JSONResponse(
    #         status_code=exc.status_code,
    #         content=_error_payload("http_error", message, details),
    #     )

    # @app.exception_handler(Exception)
    # async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    #     logger.exception("Unhandled application error")
    #     return JSONResponse(
    #         status_code=500,
    #         content=_error_payload("internal_server_error", "Internal server error"),
    #     )
