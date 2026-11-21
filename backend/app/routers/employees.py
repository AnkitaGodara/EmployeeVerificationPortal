"""Employees and enrolment: add people, upload a folder of photos per person, or one zip with a
folder per person. Every upload is face-detected and cropped before it is saved."""
import shutil

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..db import Employee, User, get_db
from ..face.dataset import (count_faces, group_zip_by_person, ingest_images, make_label,
                            save_profile_photo)
from ..rag.index import rag_index
from ..schemas import EmployeeCreate, EmployeeOut, EmployeeUpdate
from ..security import get_current_user, require_admin

router = APIRouter(prefix="/api", tags=["employees"])
MAX_FILE_BYTES = 15 * 1024 * 1024


def to_out(emp: Employee) -> EmployeeOut:
    out = EmployeeOut.model_validate(emp)
    out.has_photo = bool(emp.photo_path)
    out.face_images = count_faces(emp.label)
    return out


def next_code(db: Session) -> str:
    n = db.query(Employee).count() + 1
    while db.query(Employee).filter(Employee.employee_code == f"ONX{n:04d}").first():
        n += 1
    return f"ONX{n:04d}"


def refresh_index():
    try:
        rag_index.build()
    except Exception as exc:          # never block an employee update on the index
        print("RAG index rebuild failed:", exc)


@router.get("/employees", response_model=list[EmployeeOut])
def list_employees(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [to_out(e) for e in db.query(Employee).order_by(Employee.name).all()]


@router.post("/employees", response_model=EmployeeOut, status_code=201)
def create_employee(body: EmployeeCreate, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    label = body.label or make_label(body.name)
    if db.query(Employee).filter(Employee.label == label).first():
        raise HTTPException(409, f"An employee with label '{label}' already exists.")
    code = body.employee_code or next_code(db)
    if db.query(Employee).filter(Employee.employee_code == code).first():
        raise HTTPException(409, f"Employee code {code} already exists.")
    data = body.model_dump(exclude={"label", "employee_code"})
    emp = Employee(label=label, employee_code=code, **data)
    db.add(emp)
    db.commit()
    db.refresh(emp)
    refresh_index()
    return to_out(emp)


@router.get("/employees/{emp_id}", response_model=EmployeeOut)
def get_employee(emp_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)):
    emp = db.get(Employee, emp_id)
    if not emp:
        raise HTTPException(404, "Employee not found.")
    return to_out(emp)


@router.patch("/employees/{emp_id}", response_model=EmployeeOut)
def update_employee(emp_id: int, body: EmployeeUpdate, _: User = Depends(require_admin),
                    db: Session = Depends(get_db)):
    emp = db.get(Employee, emp_id)
    if not emp:
        raise HTTPException(404, "Employee not found.")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(emp, key, value)
    db.commit()
    db.refresh(emp)
    refresh_index()
    return to_out(emp)


@router.delete("/employees/{emp_id}", status_code=204)
def delete_employee(emp_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Deletes the employee, their face images and photo (privacy policy). Retrain afterwards."""
    emp = db.get(Employee, emp_id)
    if not emp:
        raise HTTPException(404, "Employee not found.")
    shutil.rmtree(settings.FACES_DIR / emp.label, ignore_errors=True)
    (settings.PHOTOS_DIR / f"{emp.label}.jpg").unlink(missing_ok=True)
    db.delete(emp)
    db.commit()
    refresh_index()


@router.get("/employees/{emp_id}/photo")
def employee_photo(emp_id: int, _: User = Depends(get_current_user), db: Session = Depends(get_db)):
    emp = db.get(Employee, emp_id)
    if not emp or not emp.photo_path:
        raise HTTPException(404, "No photo.")
    return FileResponse(emp.photo_path, media_type="image/jpeg")


async def _read_uploads(files: list[UploadFile]) -> list[tuple[str, bytes]]:
    out = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than 15 MB.")
        # Folder uploads send names like "harshal/img1.jpg"; keep only the file name.
        out.append(((f.filename or "image").replace("\\", "/").split("/")[-1], data))
    return out


@router.post("/employees/{emp_id}/images")
async def upload_images(emp_id: int, files: list[UploadFile] = File(...),
                        _: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Upload a folder (or several photos) of ONE person."""
    emp = db.get(Employee, emp_id)
    if not emp:
        raise HTTPException(404, "Employee not found.")
    report = ingest_images(emp.label, await _read_uploads(files))
    first = report.pop("first_good_image")
    if first and not emp.photo_path:
        emp.photo_path = save_profile_photo(emp.label, first)
        db.commit()
    report["employee_id"] = emp.id
    report["ready_for_training"] = report["total_faces"] >= settings.MIN_IMAGES_PER_PERSON
    return report


@router.post("/enroll/zip")
async def enroll_zip(file: UploadFile = File(...), _: User = Depends(require_admin),
                     db: Session = Depends(get_db)):
    """Upload ONE zip with a folder per person, e.g. project 1/harshal/*.jpg, project 1/kishan/*.jpg.
    New folder names become new employees (details can be edited afterwards)."""
    try:
        groups = group_zip_by_person(await file.read())
    except Exception as exc:
        raise HTTPException(400, f"Could not read zip: {exc}")
    people = []
    for label, images in sorted(groups.items()):
        emp = db.query(Employee).filter(Employee.label == label).first()
        created = emp is None
        if created:
            emp = Employee(label=label, employee_code=next_code(db),
                           name=label.replace("_", " ").title(),
                           notes="Created automatically from a zip upload. Please complete the details.")
            db.add(emp)
            db.commit()
            db.refresh(emp)
        report = ingest_images(label, images)
        first = report.pop("first_good_image")
        if first and not emp.photo_path:
            emp.photo_path = save_profile_photo(label, first)
            db.commit()
        people.append({"employee_id": emp.id, "label": label, "name": emp.name, "created": created,
                       "saved": report["saved"], "skipped": report["skipped"],
                       "total_faces": report["total_faces"],
                       "skipped_files": [f for f in report["files"] if f["status"] == "skipped"][:20]})
    refresh_index()
    return {"people": people, "total_saved": sum(p["saved"] for p in people),
            "total_skipped": sum(p["skipped"] for p in people)}
