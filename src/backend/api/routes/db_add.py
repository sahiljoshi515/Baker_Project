from typing import Annotated, List

from fastapi import Depends, FastAPI, HTTPException, Query, File, UploadFile, APIRouter
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware # import here

# from backend.services.mistral import mistral_ocr
import httpx
import typing
from pydantic import BaseModel

from main import SessionDep
from models.schemas import Doc, DocBase, DocCreate, DocPublic

router = APIRouter(prefix="/api/db", tags=["add"])


# submit data (pdf to ocr, itemize, and extract metadata) to update DB
@router.post("/process", response_model=DocPublic)
def create_db_obj(doc: DocCreate, session: SessionDep):
    db_doc = Doc.model_validate(doc)
    session.add(db_doc)
    session.commit()
    session.refresh(db_doc)
    # why ???????
    return db_doc