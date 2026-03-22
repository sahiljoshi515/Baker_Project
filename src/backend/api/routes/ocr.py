from fastapi import APIRouter, UploadFile

from api.dependencies import OCRServiceDep
from models.schemas import OCRResult

router = APIRouter(prefix="/api/pdf", tags=["ocr"])


@router.post("/ocr", response_model=OCRResult)
async def handle_upload(e: UploadFile, ocr_service: OCRServiceDep) -> OCRResult:
    file_name = e.filename or "uploaded.pdf"
    ocr_input = await e.read()
    return ocr_service.extract_pdf(file_name, ocr_input)
