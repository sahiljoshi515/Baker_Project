from fastapi import APIRouter
import logging

from services.extract import gpt_extract
from models.schemas import ExtractRequest

logger = logging.getLogger(__name__)
logging.basicConfig(filename='myapp.log', level=logging.DEBUG)
router = APIRouter(prefix="/api/pdf", tags=["extract"])


# submit data (pdf to ocr, itemize, and extract metadata) to update DB
# @router.post("/process", response_model=DocPublic)
# def create_db_obj(doc: DocCreate, session: SessionDep):
#     db_doc = Doc.model_validate(doc)
#     session.add(db_doc)
#     session.commit()
#     session.refresh(db_doc)
#     # why ???????
#     return db_doc

@router.post("/extract")
async def metadata_extraction(request: ExtractRequest) -> dict[str, str]:
    text = request.ocr_output
    # logger.info(f'ocr sent to backend:\n{text}')
    # collection, entities, location, description, date, subject_lst, accessibility
    metadata = await gpt_extract(text)
    # logger.info(f"metadata: {metadata}")
    if metadata == "":
        return "Failed to Extract"
    # add to Elastic Search
    # ...
    # deserialize
    # data = json.loads(metadata)
    # Send Post Request from backend to frontend
    # metadata += f", doc_name: {request.doc_name}\n"
    # json_metadata = json.loads(metadata)
    # logger.info(f"json metadata: {json_metadata}")
    # newDoc = DocCreate(**json_metadata)
    # collection
    # entities
    # description
    # date
    # subject_lst
    # doc_name: str
    # async with httpx.AsyncClient() as client:
    #     resp = await client.post('http://localhost:8000/api/pdf/process', json = {
    #     'doc':newDoc, 'session':SessionDep},
    #     timeout=None)
    #     body = resp.json()
    #     if resp.status_code == 422:
    #         return "Failed to add to DB"
    #     logger.log(f"body: {body}")
    return {"metadata":metadata}

