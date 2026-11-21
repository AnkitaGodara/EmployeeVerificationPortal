"""Optional start-up seeding for hosted demos (e.g. Hugging Face Spaces, where disk is wiped on restart).

Environment variables:
  DEMO_USER_ID / DEMO_PASSWORD  create a 'security' login for visitors (verify + assistant only)
  SEED_ZIP_URL                  zip with one folder of photos per person (e.g. a PRIVATE HF dataset file)
  SEED_ZIP_TOKEN                optional bearer token for that URL (e.g. a read-only HF token)
  SEED_DETAILS                  JSON file with employee details by label
                                (default: scripts/employees.sample.json)

If SEED_ZIP_URL is set and no model has been trained yet, the app downloads the zip, enrols everyone,
applies the details and trains in the background. Nothing happens when the variables are not set.
"""
import json
import os
import threading
import urllib.request
from pathlib import Path

from .config import BACKEND_DIR
from .db import Employee, SessionLocal, User
from .face import model as cnn
from .face.dataset import group_zip_by_person, ingest_images, save_profile_photo
from .security import hash_password

MAX_SEED_BYTES = 300 * 1024 * 1024


def ensure_demo_user():
    user_id, password = os.getenv("DEMO_USER_ID"), os.getenv("DEMO_PASSWORD")
    if not (user_id and password):
        return
    db = SessionLocal()
    try:
        if not db.query(User).filter(User.user_id == user_id).first():
            db.add(User(user_id=user_id, full_name="Demo Security Officer",
                        password_hash=hash_password(password), role="security"))
            db.commit()
    finally:
        db.close()


def _download(url: str, token: str | None) -> bytes:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=120) as resp:      # noqa: S310 (URL comes from the admin's env)
        data = resp.read(MAX_SEED_BYTES + 1)
    if len(data) > MAX_SEED_BYTES:
        raise ValueError("Seed zip is larger than 300 MB")
    return data


def _next_code(db) -> str:
    n = db.query(Employee).count() + 1
    while db.query(Employee).filter(Employee.employee_code == f"ONX{n:04d}").first():
        n += 1
    return f"ONX{n:04d}"


def seed_from_zip(url: str, token: str | None = None, details_path: Path | None = None) -> dict:
    groups = group_zip_by_person(_download(url, token))
    details = {}
    if details_path and details_path.exists():
        details = {row["label"]: row for row in json.loads(details_path.read_text())}
    db = SessionLocal()
    summary = {}
    try:
        for label, images in sorted(groups.items()):
            emp = db.query(Employee).filter(Employee.label == label).first()
            if emp is None:
                emp = Employee(label=label, employee_code=_next_code(db), name=label.replace("_", " ").title())
                db.add(emp)
            for key, value in details.get(label, {}).items():
                if key != "label" and hasattr(emp, key):
                    setattr(emp, key, value)
            db.commit()
            report = ingest_images(label, images)
            first = report.pop("first_good_image")
            if first and not emp.photo_path:
                emp.photo_path = save_profile_photo(label, first)
                db.commit()
            summary[label] = report["saved"]
    finally:
        db.close()
    return summary


def _seed_job(url, token, details_path):
    from .face.service import face_service
    from .rag.index import rag_index
    try:
        print("Seeding demo data from SEED_ZIP_URL ...")
        print("Seeded faces:", seed_from_zip(url, token, details_path))
        rag_index.build()
        face_service.start_training(started_by="seed")
    except Exception as exc:          # a failed seed must never stop the app
        print("Demo seeding failed:", exc)


def run_startup_seed():
    ensure_demo_user()
    url = os.getenv("SEED_ZIP_URL")
    if not url or cnn.MODEL_PATH.exists():
        return
    details = Path(os.getenv("SEED_DETAILS", str(BACKEND_DIR / "scripts" / "employees.sample.json")))
    threading.Thread(target=_seed_job, args=(url, os.getenv("SEED_ZIP_TOKEN"), details), daemon=True).start()
