from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from sqlmodel import Field, Session, SQLModel, create_engine, select, Relationship
from typing import Annotated, List

from api.routes import ocr
from api.routes import extraction

from core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

# to ensure a single request can use multiple threads: MAKE SURE TO SERIALIZE WRITES
connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, echo= True, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # SQLModel.metadata.sess

def drop_db_and_tables():
    SQLModel.metadata.drop_all(engine)
    
def get_session():
    with Session(engine) as session:
        yield session

""" 
Since Depends(get_session) yields a session for each request,
annotated is used to declare the yielded type as a class named Session of type SessionDep
"""
SessionDep = Annotated[Session, Depends(get_session)]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # executed before app starts taking requests
    create_db_and_tables()
    # add code for when you are stopping the application
    yield
    # drop_db_and_tables()

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# cors_origins = [
#     "http://0.0.0.0:8000",
#     "http://127.0.0.1:5173",
#     "http://localhost:5173",
#     "https://digital-collections-explorer.com",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=cors_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# declare origin/s
origins = [
    "http://localhost:8080",
    "localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ocr.router)
app.include_router(extraction.router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# frontend_dir = Path(f"src/frontend/{settings.collection_type}/dist")
frontend_dir = Path("frontend.py")

# if frontend_dir.exists():
#     app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
#     logger.info(f"Serving frontend from {frontend_dir}")
# else:
#     logger.warning(f"Frontend directory not found at {frontend_dir}")
#     logger.warning("The API will run without serving the frontend.")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
