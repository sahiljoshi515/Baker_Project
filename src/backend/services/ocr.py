import logging
from typing import Protocol

from core.exceptions import AppError, BadRequestError
from models.schemas import OCRResult

logger = logging.getLogger(__name__)

"""
Interface which user can inject a concrete implementation of at runtime
"""
class OCRProvider(Protocol):
    def extract(self, file_name: str, content: bytes) -> tuple[list[str], str]:
        ...


class OCRService:
    def __init__(self, provider: OCRProvider) -> None:
        self.provider = provider

    def extract_pdf(self, file_name: str, content: bytes) -> OCRResult:
        if not file_name or not file_name.lower().endswith(".pdf"):
            raise BadRequestError("Only PDF files are supported")
        if not content:
            raise BadRequestError("Uploaded PDF is empty")

        try:
            pages, markdown = self.provider.extract(file_name, content)
        except AppError:
            logger.exception("OCR failed for document '%s'", file_name)
            raise

        return OCRResult(pages=pages, markdown=markdown)
