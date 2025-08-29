from typing import Annotated, List
import uvicorn

from fastapi import Depends, FastAPI, HTTPException, Query, File, UploadFile, APIRouter
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware # import here

# from backend.services.mistral import mistral_ocr
import httpx
import typing
from pydantic import BaseModel

"""
TEAM TABLE
id, name, collection, title, people+orgs, location, description, date, subjects, accessibility
"""
class DocBase(SQLModel):
    people_and_organizations: typing.Union[str, None] = Field(default=None) # separated by , or |
    # index not enabled on following 2
    # location: typing.Union[str, None] = Field(default = None)
    description: typing.Union[str, None] = Field(default=None)
    date: typing.Union[str, None] = Field(default = None, index = True)  # expect 'YYYYMM'
    subject: typing.Union[str, None] = Field(default=None) # separated by , or |
    # index not enabled
    # accessibility: typing.Union[str, None] = Field(default=None)

# Actual data model
class Doc(DocBase, table = True):
    id: typing.Union[int, None] = Field(default=None, primary_key=True)
    doc_name: str 

# to be returned to API user
class DocPublic(DocBase):
    id: int

class DocCreate(DocBase):
    title: str

"""Extraction API request body"""
class ExtractRequest(BaseModel):
    ocr_output: str
    doc_name:str

"""
CRUD
"""

# # retrieve data (in our case, we are retrieving metadata + PDF to serve to users)
# @app.get("/docs/", response_model= list[DocPublic])
# def retrieve_db_objs(session: SessionDep, offset: int, limit: Annotated[int, Query(le = 100)]):
#     docs = session.exec(select(Doc).offset(offset).limit(limit)).all()
#     return docs

# # retrieve one piece of data
# @app.get("/doc/{doc_id}", response_model=DocPublic)
# def retrieve_db_obj(doc_id: int, session: SessionDep):
#     doc = session.exec(select(Doc).where(Doc.id == doc_id))
#     if not doc:
#         raise HTTPException(404, "id not found")
#     return doc


# @app.delete("/doc/{doc_id}")
# def remove_db_objs(doc_id: int, session: SessionDep):
#     doc = session.exec(select(Doc).where(Doc.id == doc_id))
#     if not doc:
#         raise HTTPException(404, "id not found")
#     session.delete(doc)
#     # FFLUSH / Msync type operation
#     session.commit()
#     return {"ok": True}




