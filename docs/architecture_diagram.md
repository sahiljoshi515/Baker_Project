# Baker Project Architecture Diagram

This diagram reflects the current codebase and the terminology used in the design document.

- Solid arrows show runtime calls, composition, or library usage.
- Dashed arrows show dependency inversion: high-level services depend on abstractions, while concrete adapters implement those abstractions.
- `frontend.py` currently combines the user-facing view and UI controller responsibilities.

```mermaid
flowchart LR
  classDef ui fill:#dbeafe,stroke:#1d4ed8,color:#0f172a,stroke-width:1.5px;
  classDef controller fill:#e0f2fe,stroke:#0284c7,color:#0f172a,stroke-width:1.5px;
  classDef domain fill:#dcfce7,stroke:#16a34a,color:#0f172a,stroke-width:1.5px;
  classDef abstract fill:#fef3c7,stroke:#d97706,color:#0f172a,stroke-width:1.5px,stroke-dasharray: 5 3;
  classDef infra fill:#ede9fe,stroke:#7c3aed,color:#0f172a,stroke-width:1.5px;
  classDef external fill:#f3f4f6,stroke:#6b7280,color:#111827,stroke-width:1.2px;

  subgraph UI["View + UI Controller"]
    FE["src/frontend/frontend.py<br/>NiceGUI pages, UI state,<br/>upload/extract button handlers"]
  end

  subgraph DELIVERY["HTTP Controller Layer"]
    MAIN["src/backend/main.py<br/>uvicorn entrypoint"]
    FACTORY["src/backend/app_factory.py<br/>create_app()"]
    ROUTES["src/backend/api/routes/*<br/>ocr.py, extraction.py, db_add.py"]
    DEPS["src/backend/api/dependencies.py<br/>FastAPI composition root"]
    ERR["src/backend/api/error_handlers.py<br/>maps AppError -> JSON"]
  end

  subgraph DOMAIN["Model / Application Core"]
    CFG["src/backend/core/config.py<br/>Settings from .env"]
    EXC["src/backend/core/exceptions.py<br/>AppError hierarchy"]
    SCHEMAS["src/backend/models/schemas.py<br/>Pydantic + SQLModel types"]
    OCRSVC["src/backend/services/ocr.py<br/>OCRService"]
    OCRPORT["OCRProvider<br/>Protocol"]
    MDSVC["src/backend/services/metadata.py<br/>MetadataService"]
    MDPORT["MetadataProvider<br/>Protocol"]
    DOCSVC["src/backend/services/documents.py<br/>DocumentService"]
    DOCPORT["DocumentRepositoryProtocol<br/>Protocol"]
  end

  subgraph ADAPTERS["Infrastructure Adapters"]
    MISTRAL["src/backend/services/mistral.py<br/>MistralOCRProvider"]
    OPENAI["src/backend/services/extract.py<br/>OpenAIMetadataProvider"]
    REPO["src/backend/repositories/documents.py<br/>DocumentRepository"]
    ENGINE["get_engine() / get_session()<br/>SQLModel engine + Session"]
  end

  subgraph EXTERNAL["External Libraries / Systems"]
    NICEGUI["NiceGUI"]
    HTTPX["httpx.AsyncClient"]
    FASTAPI["FastAPI / Depends / APIRouter<br/>CORSMiddleware / StaticFiles"]
    UVICORN["uvicorn"]
    PYD["Pydantic / pydantic-settings"]
    SQLMODEL["SQLModel / SQLAlchemy"]
    SQLITE["SQLite database.db"]
    MISTRALAPI["Mistral OCR API"]
    OPENAIAPI["OpenAI Responses API"]
    TIKTOKEN["tiktoken"]
    ENV[".env"]
  end

  FE -->|"renders pages"| NICEGUI
  FE -->|"POST /api/pdf/ocr<br/>POST /api/pdf/extract<br/>POST /api/db/process"| HTTPX
  HTTPX --> ROUTES

  MAIN --> UVICORN
  MAIN --> FACTORY
  FACTORY --> FASTAPI
  FACTORY --> ROUTES
  FACTORY --> ERR
  FACTORY --> CFG
  FACTORY -->|"lifespan startup<br/>SQLModel.metadata.create_all(...)"| ENGINE

  ROUTES -->|"request / response models"| SCHEMAS
  ROUTES -.->|"Depends(...)"| DEPS

  DEPS --> CFG
  DEPS --> ENGINE
  DEPS --> OCRSVC
  DEPS --> MDSVC
  DEPS --> DOCSVC
  DEPS -->|"inject concrete OCR adapter"| MISTRAL
  DEPS -->|"inject concrete metadata adapter"| OPENAI
  DEPS -->|"inject concrete repository"| REPO

  OCRSVC -.->|"depends on abstraction"| OCRPORT
  MDSVC -.->|"depends on abstraction"| MDPORT
  DOCSVC -.->|"depends on abstraction"| DOCPORT

  MISTRAL -.->|"implements"| OCRPORT
  OPENAI -.->|"implements"| MDPORT
  REPO -.->|"implements"| DOCPORT

  OCRSVC --> SCHEMAS
  MDSVC --> SCHEMAS
  DOCSVC --> SCHEMAS
  REPO --> SCHEMAS

  MISTRAL --> EXC
  OPENAI --> EXC
  REPO --> EXC
  ERR --> EXC

  MISTRAL --> MISTRALAPI
  MISTRAL --> TIKTOKEN
  OPENAI --> OPENAIAPI
  ENGINE --> SQLMODEL
  REPO --> SQLMODEL
  SQLMODEL --> SQLITE
  CFG --> PYD
  CFG --> ENV

  class FE ui;
  class MAIN,FACTORY,ROUTES,DEPS,ERR controller;
  class CFG,EXC,SCHEMAS,OCRSVC,MDSVC,DOCSVC domain;
  class OCRPORT,MDPORT,DOCPORT abstract;
  class MISTRAL,OPENAI,REPO,ENGINE infra;
  class NICEGUI,HTTPX,FASTAPI,UVICORN,PYD,SQLMODEL,SQLITE,MISTRALAPI,OPENAIAPI,TIKTOKEN,ENV external;
```

## Dependency Inversion Focus

The key inversion points in the current implementation are:

1. `OCRService` depends on the `OCRProvider` protocol, while `MistralOCRProvider` is injected by `get_ocr_service()`.
2. `MetadataService` depends on the `MetadataProvider` protocol, while `OpenAIMetadataProvider` is injected by `get_metadata_service()`.
3. `DocumentService` depends on `DocumentRepositoryProtocol`, while `DocumentRepository` is injected by `get_document_service()`.

That makes `src/backend/api/dependencies.py` the composition root for the application: the high-level services stay stable, and the low-level providers or persistence adapters can be replaced without rewriting the service layer or route handlers.
