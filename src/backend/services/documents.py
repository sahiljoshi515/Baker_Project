import logging
from typing import Protocol

from core.exceptions import AppError
from models.schemas import Doc, DocCreate

logger = logging.getLogger(__name__)

"""
Interface which user can inject a concrete implementation of at runtime
"""
class DocumentRepositoryProtocol(Protocol):
    def create(self, document: DocCreate) -> Doc:
        ...


class DocumentService:
    def __init__(self, repository: DocumentRepositoryProtocol) -> None:
        self.repository = repository

    def create_document(self, document: DocCreate) -> Doc:
        try:
            return self.repository.create(document)
        except AppError:
            logger.exception("Document persistence failed for '%s'", document.doc_name)
            raise
