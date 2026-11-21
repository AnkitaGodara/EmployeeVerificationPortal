"""RAG assistant + knowledge base (policy documents) management."""
import re

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..db import Employee, User, get_db
from ..rag.assistant import answer
from ..rag.documents import DOC_EXTS, policy_files, read_file, title_of
from ..rag.index import rag_index
from ..rag.llm import provider_name
from ..schemas import AskRequest
from ..security import get_current_user, require_admin

router = APIRouter(prefix="/api", tags=["knowledge"])


@router.post("/assistant/ask")
def ask(body: AskRequest, _: User = Depends(get_current_user), db: Session = Depends(get_db)):
    employee = None
    if body.employee_id is not None:
        employee = db.get(Employee, body.employee_id)
        if employee is None:
            raise HTTPException(404, "Employee not found.")
    return answer(body.question.strip(), employee)


@router.get("/assistant/status")
def assistant_status(_: User = Depends(get_current_user)):
    rag_index.ensure()
    return {"llm_provider": provider_name(), "embeddings": settings.EMBEDDINGS_PROVIDER,
            "top_k": settings.TOP_K, "index": rag_index.meta}


@router.get("/documents")
def list_documents(_: User = Depends(get_current_user)):
    docs = []
    for path in policy_files():
        text = read_file(path)
        docs.append({"name": path.name, "title": title_of(path, text), "characters": len(text),
                     "origin": "uploaded" if path.parent == settings.UPLOADED_DOCS_DIR else "built-in"})
    return docs


@router.get("/documents/{name}")
def read_document(name: str, _: User = Depends(get_current_user)):
    for path in policy_files():
        if path.name == name:
            return {"name": name, "title": title_of(path, read_file(path)), "content": read_file(path)}
    raise HTTPException(404, "Document not found.")


@router.post("/documents", status_code=201)
async def upload_document(file: UploadFile = File(...), _: User = Depends(require_admin)):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", file.filename or "document")
    if not any(name.lower().endswith(ext) for ext in DOC_EXTS):
        raise HTTPException(400, "Only .md, .txt and .pdf files are supported.")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Document is larger than 10 MB.")
    path = settings.UPLOADED_DOCS_DIR / name
    path.write_bytes(data)
    if not read_file(path).strip():
        path.unlink()
        raise HTTPException(400, "No text could be read from this file (scanned PDFs are not supported).")
    return {"name": name, "index": rag_index.build()}


@router.delete("/documents/{name}", status_code=204)
def delete_document(name: str, _: User = Depends(require_admin)):
    path = settings.UPLOADED_DOCS_DIR / name
    if not path.exists() or path.parent != settings.UPLOADED_DOCS_DIR:
        raise HTTPException(404, "Only uploaded documents can be deleted.")
    path.unlink()
    rag_index.build()


@router.post("/documents/reindex")
def reindex(_: User = Depends(require_admin)):
    return rag_index.build()
