"""
To utilize Dependency Injection in our application, we create
FastAPI dependencies so that Document creation, OCR, and Metadata Summarization
functionality is interface-like. In other words, our Metadata service, OCR service,
and Document service rely on interfaces (e.g. MistralOCRProvider for OCR Service) which can be injected at runtime.
Here, we display that by performing dependency injection with our concrete implementations (see the services sub-package
subpackage for more detail).
"""

from functools import lru_cache
from typing import Annotated, Generator

from fastapi import Depends
from sqlmodel import Session, create_engine

from core.config import Settings, get_settings
from repositories.documents import DocumentRepository
from services.documents import DocumentService
from services.extract import OpenAIMetadataProvider
from services.metadata import MetadataService
from services.mistral import MistralOCRProvider
from services.ocr import OCRService


SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache
def get_engine():
    settings = get_settings()
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    return create_engine(
        settings.database_url,
        echo=settings.debug,
        connect_args=connect_args,
    )


def get_session() -> Generator[Session, None, None]:
    engine = get_engine()
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


"""
Below are concrete implementations of the interfaces
which OCRService, MetadataService, and DocumentService expect.
"""

def get_ocr_provider(settings: SettingsDep) -> MistralOCRProvider:
    return MistralOCRProvider(
        api_key=settings.mistral_api_key,
        model_name=settings.mistral_ocr_model,
    )


def get_metadata_provider(settings: SettingsDep) -> OpenAIMetadataProvider:
    return OpenAIMetadataProvider(
        api_key=settings.openai_api_key,
        model_name=settings.openai_chat_model,
    )

"""
Dependency Injection can be seen below
"""

def get_document_repository(session: SessionDep) -> DocumentRepository:
    return DocumentRepository(session)


def get_ocr_service(provider: Annotated[MistralOCRProvider, Depends(get_ocr_provider)]) -> OCRService:
    return OCRService(provider)


def get_metadata_service(
    provider: Annotated[OpenAIMetadataProvider, Depends(get_metadata_provider)],
) -> MetadataService:
    return MetadataService(provider)


def get_document_service(
    repository: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> DocumentService:
    return DocumentService(repository)


OCRServiceDep = Annotated[OCRService, Depends(get_ocr_service)]
MetadataServiceDep = Annotated[MetadataService, Depends(get_metadata_service)]
DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
