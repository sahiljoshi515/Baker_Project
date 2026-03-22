import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel

from api.error_handlers import register_exception_handlers
from api.routes import db_add, extraction, ocr
from api.dependencies import get_engine
from core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    

    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        SQLModel.metadata.create_all(get_engine())
        logger.info("Application startup complete")
        yield
        logger.info("Application shutdown complete")

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(ocr.router)
    app.include_router(extraction.router)
    app.include_router(db_add.router)

    @app.get("/api/health")
    async def health_check() -> dict[str, str]:
        return {"status": "healthy"}

    if settings.frontend_dist_dir:
        frontend_dir = Path(settings.frontend_dist_dir).expanduser()
        if frontend_dir.exists():
            app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
            logger.info("Serving frontend assets from %s", frontend_dir)
        else:
            logger.warning("Configured frontend directory does not exist: %s", frontend_dir)

    return app
