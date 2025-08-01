from fastapi import APIRouter
import logging

from services.extract import gpt_extract
from models.schemas import ExtractRequest

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
router = APIRouter(prefix="/api/pdf", tags=["extract"])

from fastapi import File, UploadFile, Form
import json


@router.post("/extract")
async def metadata_extraction(input: ExtractRequest):
    # logger.info(f'ocr sent to backend')
    # collection, entities, location, description, date, subject_lst, accessibility
    metadata = await gpt_extract(input["ocr_output"])
    logger.info(metadata)
    if metadata == "":
        return "Failed to OCR"
    # add to Elastic Search
    # ...
    # serialize
    data = json.loads(metadata)
    # Send Post Request from backend to frontend
    return {"metadata":data}