"""
This file defines the types we use internally and return to the user
"""

from pydantic import BaseModel
from sqlmodel import Field, SQLModel

# used to represent a document in our SQL database
class MetadataFields(SQLModel):
    title: str = Field(default=None)
    people_and_organizations: str = Field(default=None)
    description: str = Field(default=None)
    date: str = Field(default=None, index=True)
    subject: str  = Field(default=None)

# extends MetadataFields to add a primary key (id) generated internally and returned to the user
# through DocPublic so that the ID field can be used to query our database in GET requests
class Doc(MetadataFields, table=True):
    __tablename__ = "documents"

    id: int = Field(default = None, primary_key=True)
    doc_name: str


class DocCreate(MetadataFields):
    doc_name: str


class DocPublic(MetadataFields):
    id: int
    # doc_name: str


class ExtractRequest(BaseModel):
    ocr_output: str
    doc_name: str


class OCRResult(BaseModel):
    pages: list[str]
    markdown: str


class MetadataResponse(BaseModel):
    title: str 
    people_and_organizations: str 
    description: str 
    date: str 
    subject: str  