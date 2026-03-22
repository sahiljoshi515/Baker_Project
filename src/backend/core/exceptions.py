"""
The following are custom exceptions which may arise handled globally 
with FastAPI. The exception handlers corresponding to these exceptions can be 
found in the api subpackage.
"""
from typing import Any


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "app_error",
        status_code: int = 500,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class BadRequestError(AppError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            message,
            code="bad_request",
            status_code=400,
            details=details,
        )


class ConfigurationError(AppError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            message,
            code="configuration_error",
            status_code=503,
            details=details,
        )


class OCRProcessingError(AppError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            message,
            code="ocr_processing_error",
            status_code=502,
            details=details,
        )


class MetadataExtractionError(AppError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            message,
            code="metadata_extraction_error",
            status_code=502,
            details=details,
        )


class PersistenceError(AppError):
    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(
            message,
            code="persistence_error",
            status_code=500,
            details=details,
        )

