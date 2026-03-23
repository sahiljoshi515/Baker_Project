import logging
from typing import Protocol

from core.exceptions import AppError, BadRequestError
from models.schemas import MetadataResponse

logger = logging.getLogger(__name__)

"""
Interface which user can inject a concrete implementation of at runtime
"""
class MetadataProvider(Protocol):
    def extract(self, ocr_output: str) -> MetadataResponse:
        ...


class MetadataService:
    def __init__(self, provider: MetadataProvider) -> None:
        self.provider = provider

    def extract_metadata(self, ocr_output: str, doc_name: str) -> MetadataResponse:
        if not ocr_output.strip():
            raise BadRequestError("OCR output cannot be empty")

        try:
            metadata = self.provider.extract(ocr_output)
        except AppError:
            logger.exception("Metadata extraction failed for document '%s'", doc_name)
            raise
        return metadata
        # return MetadataResponse(metadata=metadata)
