from fastapi import APIRouter

from api.dependencies import MetadataServiceDep
from models.schemas import ExtractRequest, MetadataResponse

router = APIRouter(prefix="/api/pdf", tags=["extract"])


@router.post("/extract", response_model=MetadataResponse)
async def metadata_extraction(
    request: ExtractRequest,
    metadata_service: MetadataServiceDep,
) -> MetadataResponse:
    return metadata_service.extract_metadata(request.ocr_output, request.doc_name)
