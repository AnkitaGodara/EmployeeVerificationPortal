"""FastAPI entry point. Run:  uvicorn app.main:app --reload  (from the backend folder)."""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import BACKEND_DIR, settings
from .db import SessionLocal, init_db
from .rag.index import rag_index
from .routers import admin, auth, employees, knowledge, recognition
from .security import ensure_default_admin
from .seed import run_startup_seed

FRONTEND_DIST = Path(BACKEND_DIR).parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        ensure_default_admin(db)
    finally:
        db.close()
    try:
        rag_index.ensure()
    except Exception as exc:
        print("RAG index not ready:", exc)
    run_startup_seed()          # only does something when DEMO_* / SEED_* env vars are set
    yield


app = FastAPI(title="Employee Verification Portal",
              description="Face recognition (Keras CNN) + RAG assistant for company policies.",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

for r in (auth.router, employees.router, recognition.router, knowledge.router, admin.router):
    app.include_router(r)


@app.get("/api/health")
def health():
    return {"status": "ok"}


# In production the built React app is served by the same server (one container, one URL).
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        file = FRONTEND_DIST / full_path
        if full_path and file.is_file() and FRONTEND_DIST in file.resolve().parents:
            return FileResponse(file)
        return FileResponse(FRONTEND_DIST / "index.html")
