"""Loading the knowledge base: company policy files + one profile document per employee."""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..config import settings

DOC_EXTS = {".md", ".txt", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def title_of(path: Path, text: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("_", " ").title()


def policy_files() -> list[Path]:
    files = []
    for folder, origin in ((settings.POLICY_DIR, "built-in"), (settings.UPLOADED_DOCS_DIR, "uploaded")):
        if folder.exists():
            files += [p for p in sorted(folder.iterdir()) if p.suffix.lower() in DOC_EXTS]
    return files


def load_policy_documents() -> list[Document]:
    docs = []
    for path in policy_files():
        text = read_file(path)
        if text.strip():
            docs.append(Document(page_content=text, metadata={
                "type": "policy", "source": path.name, "title": title_of(path, text),
                "origin": "uploaded" if path.parent == settings.UPLOADED_DOCS_DIR else "built-in"}))
    return docs


def employee_profile_text(emp) -> str:
    """Plain-language profile, so the same text works for retrieval and for the LLM."""
    return (
        f"Employee profile: {emp.name}\n"
        f"Employee code: {emp.employee_code}\n"
        f"Department: {emp.department or 'not recorded'}\n"
        f"Designation: {emp.designation or 'not recorded'}\n"
        f"Access level: {emp.access_level}\n"
        f"Employment status: {emp.status}\n"
        f"Shift: {emp.shift or 'not recorded'}\n"
        f"Office location: {emp.office_location or 'not recorded'}\n"
        f"Reporting manager: {emp.manager or 'not recorded'}\n"
        f"Joining date: {emp.joining_date or 'not recorded'}\n"
        f"Email: {emp.email or 'not recorded'}\n"
        f"Notes: {emp.notes or 'none'}"
    )


def load_employee_documents(employees) -> list[Document]:
    return [Document(page_content=employee_profile_text(e), metadata={
        "type": "employee", "employee_id": e.id, "source": f"employee:{e.employee_code}",
        "title": f"Employee profile - {e.name}", "section": "Profile"}) for e in employees]


def split_documents(docs: list[Document]) -> list[Document]:
    """Markdown files are split by headings first (so each chunk keeps its section name),
    then long sections are cut into ~CHUNK_SIZE character pieces with overlap."""
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    chunks = []
    for doc in docs:
        if doc.metadata.get("type") == "employee":
            chunks.append(doc)
            continue
        sections = header_splitter.split_text(doc.page_content) if doc.metadata["source"].endswith(".md") \
            else [Document(page_content=doc.page_content)]
        for sec in sections:
            section = sec.metadata.get("h2") or sec.metadata.get("h1") or ""
            for piece in char_splitter.split_text(sec.page_content):
                # Prefix document title + section so every chunk makes sense on its own.
                prefix = f"[{doc.metadata['title']}" + (f" > {section}]" if section else "]")
                content = f"{prefix} {piece}"
                chunks.append(Document(page_content=content,
                                       metadata={**doc.metadata, "section": section}))
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
