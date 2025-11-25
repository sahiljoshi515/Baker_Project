from fastapi import APIRouter
import logging

from services.extract import gpt_extract
from models.schemas import ExtractRequest
import json

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
router = APIRouter(prefix="/api/pdf", tags=["extract"])


@router.post("/extract")
async def metadata_extraction(request: ExtractRequest) -> dict[str, str]:
    text = request.ocr_output
    # collection, entities, location, description, date, subject_lst, accessibility
    metadata = await gpt_extract(text)
    if metadata == "":
        return {"metadata":"Failed to Extract"}
    return {"metadata":metadata}

