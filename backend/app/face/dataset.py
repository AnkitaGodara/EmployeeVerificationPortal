"""Turning uploaded photos (a folder for one person, or a zip with one folder per person)
into a clean training set: data/faces/<label>/<id>.jpg, all 100x100 face crops."""
import io
import re
import uuid
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from PIL import Image

from ..config import settings
from .detector import extract_face, profile_photo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_ZIP_BYTES = 300 * 1024 * 1024
MAX_ZIP_FILES = 5000


def make_label(text: str) -> str:
    label = re.sub(r"[^a-z0-9_]+", "_", text.strip().lower()).strip("_")
    return label or "person"


def is_image_name(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in IMAGE_EXTS


def face_dir(label: str) -> Path:
    path = settings.FACES_DIR / label
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_faces(label: str) -> int:
    path = settings.FACES_DIR / label
    return sum(1 for p in path.glob("*.jpg")) if path.exists() else 0


def ingest_images(label: str, files: list[tuple[str, bytes]]) -> dict:
    """Detect, crop and save every face. Returns a per-file report."""
    saved, skipped, report = 0, 0, []
    out_dir = face_dir(label)
    first_good = None
    for name, data in files:
        if not is_image_name(name):
            skipped += 1
            report.append({"file": name, "status": "skipped", "reason": "not an image"})
            continue
        try:
            result = extract_face(data, allow_precropped=True)
        except ValueError as exc:
            skipped += 1
            report.append({"file": name, "status": "skipped", "reason": str(exc)})
            continue
        if result.face is None:
            skipped += 1
            report.append({"file": name, "status": "skipped", "reason": result.reason})
            continue
        Image.fromarray(result.face).save(out_dir / f"{uuid.uuid4().hex[:12]}.jpg", quality=95)
        saved += 1
        first_good = first_good or data
        report.append({"file": name, "status": "saved",
                       "reason": "face detected" if result.found else result.reason})
    return {"label": label, "saved": saved, "skipped": skipped, "total_faces": count_faces(label),
            "first_good_image": first_good, "files": report}


def save_profile_photo(label: str, image_bytes: bytes) -> str:
    path = settings.PHOTOS_DIR / f"{label}.jpg"
    photo = profile_photo(image_bytes)
    if photo:
        path.write_bytes(photo)
    return str(path)


def group_zip_by_person(zip_bytes: bytes) -> dict[str, list[tuple[str, bytes]]]:
    """Read a zip like  project 1/harshal/a.jpg, project 1/kishan/b.jpg ...
    The folder that directly contains each image is the person's label.
    Files are read in memory; nothing is extracted to disk by its zip path (safe from zip-slip).
    """
    if len(zip_bytes) > MAX_ZIP_BYTES:
        raise ValueError("Zip is larger than 300 MB")
    groups: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        if len(members) > MAX_ZIP_FILES:
            raise ValueError(f"Zip has more than {MAX_ZIP_FILES} files")
        for m in members:
            path = PurePosixPath(m.filename.replace("\\", "/"))
            if "__MACOSX" in path.parts or path.name.startswith(".") or not is_image_name(path.name):
                continue
            if len(path.parts) < 2:
                continue      # image at the root of the zip has no person folder
            person = make_label(path.parts[-2])
            groups[person].append((path.name, zf.read(m)))
    if not groups:
        raise ValueError("No person folders with images found. Expected: <person_name>/<photo>.jpg")
    return dict(groups)


def load_training_set():
    """Return (images uint8 N x 100 x 100 x 3, labels list[str]) from data/faces/."""
    import numpy as np
    images, labels = [], []
    for person_dir in sorted(p for p in settings.FACES_DIR.iterdir() if p.is_dir()):
        for img_path in sorted(person_dir.glob("*.jpg")):
            img = Image.open(img_path).convert("RGB").resize((settings.IMG_SIZE, settings.IMG_SIZE))
            images.append(np.array(img))
            labels.append(person_dir.name)
    if not images:
        return np.zeros((0, settings.IMG_SIZE, settings.IMG_SIZE, 3), dtype="uint8"), []
    return np.stack(images).astype("uint8"), labels
