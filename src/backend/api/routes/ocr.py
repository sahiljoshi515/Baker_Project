from fastapi import APIRouter
import logging

from services.mistral import mistral_ocr

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pdf", tags=["ocr"])
from fastapi import File, UploadFile


@router.post("/ocr")
async def handle_upload(e: UploadFile):
    # collection, entities, location, description, date, subject_lst, accessibility
    # print("receieved")
    # return {"file":e.filename}
    # ocr pdf
    ocr_input = await e.read() # Read the content
    pages, markdown_to_display = mistral_ocr(e.filename, ocr_input)
    if pages == None:
        return pages, "Failed to OCR"
    # add to Elastic Search
    # ...
    # Send Post Request from backend to frontend
    return {"pages":pages, "markdown": markdown_to_display}