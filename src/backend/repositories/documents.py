from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from core.exceptions import PersistenceError
from models.schemas import Doc, DocCreate


class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, document: DocCreate) -> Doc:
        db_doc = Doc.model_validate(document)
        # we generate a unique ID primary key automatically to store our document
        try:
            self.session.add(db_doc)
            self.session.commit()
            self.session.refresh(db_doc)
        except SQLAlchemyError as exc:
            self.session.rollback()
            raise PersistenceError("Failed to save extracted metadata") from exc
        return db_doc
