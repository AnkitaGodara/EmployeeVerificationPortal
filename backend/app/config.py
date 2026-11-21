"""All settings in one place. Values come from environment variables or a .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


class Settings:
    # ---- folders -------------------------------------------------------
    DATA_DIR = Path(_env("DATA_DIR", str(BACKEND_DIR / "data")))
    FACES_DIR = DATA_DIR / "faces"          # cropped 100x100 faces, one folder per employee
    PHOTOS_DIR = DATA_DIR / "photos"        # one profile photo per employee
    MODEL_DIR = DATA_DIR / "model"          # trained CNN + labels + metrics
    INDEX_DIR = DATA_DIR / "rag_index"      # FAISS vector index
    UPLOADED_DOCS_DIR = DATA_DIR / "uploaded_docs"
    POLICY_DIR = BACKEND_DIR / "knowledge_base" / "policies"
    DB_URL = _env("DATABASE_URL", f"sqlite:///{DATA_DIR / 'portal.db'}")

    # ---- auth ------------------------------------------------------------
    JWT_SECRET = _env("JWT_SECRET", "change-this-secret-in-production")
    JWT_EXPIRE_MINUTES = int(_env("JWT_EXPIRE_MINUTES", "480"))
    ADMIN_USER_ID = _env("ADMIN_USER_ID", "admin")
    ADMIN_PASSWORD = _env("ADMIN_PASSWORD", "Admin@123")

    # ---- face model --------------------------------------------------------
    IMG_SIZE = 100
    CONFIDENCE_THRESHOLD = float(_env("CONFIDENCE_THRESHOLD", "0.70"))
    EPOCHS = int(_env("EPOCHS", "40"))
    BATCH_SIZE = int(_env("BATCH_SIZE", "8"))
    MIN_IMAGES_PER_PERSON = int(_env("MIN_IMAGES_PER_PERSON", "5"))

    # ---- RAG ------------------------------------------------------------------
    LLM_PROVIDER = _env("LLM_PROVIDER", "groq")              # groq | gemini | openai | none
    GROQ_API_KEY = _env("GROQ_API_KEY", "")
    GROQ_MODEL = _env("GROQ_MODEL", "llama-3.1-8b-instant")
    GOOGLE_API_KEY = _env("GOOGLE_API_KEY", "")
    GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.0-flash")
    OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
    OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
    EMBEDDINGS_PROVIDER = _env("EMBEDDINGS_PROVIDER", "tfidf")  # tfidf | fastembed | huggingface
    CHUNK_SIZE = int(_env("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "100"))
    TOP_K = int(_env("TOP_K", "4"))
    RETRIEVAL_MODE = _env("RETRIEVAL_MODE", "hybrid")            # hybrid | vector | bm25

    CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "http://localhost:5173").split(",")]


settings = Settings()

for folder in (settings.DATA_DIR, settings.FACES_DIR, settings.PHOTOS_DIR, settings.MODEL_DIR,
               settings.INDEX_DIR, settings.UPLOADED_DOCS_DIR):
    folder.mkdir(parents=True, exist_ok=True)
