from fastapi import APIRouter

from api.dependencies import DocumentServiceDep
from models.schemas import DocCreate, DocPublic

router = APIRouter(prefix="/api/db", tags=["documents"])


@router.post("/process", response_model=DocPublic)
def create_db_obj(doc: DocCreate, document_service: DocumentServiceDep) -> DocPublic:
    return document_service.create_document(doc)
