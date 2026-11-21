"""Training the CNN and verifying a photo."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from ..config import settings
from ..db import Employee, TrainingRun, User, VerificationLog, get_db
from ..face import model as cnn
from ..face.detector import extract_face
from ..face.service import face_service
from ..security import get_current_user, require_admin
from .employees import to_out

router = APIRouter(prefix="/api", tags=["recognition"])

# Same table as section 3 of the Access Control Policy (knowledge_base/policies/01_...).
ZONES = {
    "L1": ["Lobby", "Cafeteria", "Ground-floor meeting rooms"],
    "L2": ["Office work floors", "Library", "Training rooms"],
    "L3": ["R&D lab", "Network operations room", "Records room"],
    "L4": ["Server room (data centre)", "Finance vault"],
}
STATUS_WARNINGS = {
    "On Leave": "Employee is ON LEAVE: entry only with HR/manager approval; no L3/L4 zones.",
    "Suspended": "Employee is SUSPENDED: deny entry and inform HR and the Security Supervisor.",
    "Exited": "Employee has EXITED the company: treat as a visitor; all access revoked.",
}


def allowed_zones(level: str, status: str) -> list[str]:
    if status in ("Suspended", "Exited"):
        return []
    n = int(level[1]) if level and level[1:].isdigit() else 1
    if status == "On Leave":
        n = min(n, 2)
    return [z for i in range(1, n + 1) for z in ZONES[f"L{i}"]]


# ---------------- training -----------------------------------------------------
@router.post("/train", status_code=202)
def start_training(epochs: int | None = None, admin: User = Depends(require_admin)):
    if epochs is not None and not 1 <= epochs <= 200:
        raise HTTPException(400, "epochs must be between 1 and 200")
    try:
        run_id = face_service.start_training(admin.user_id, epochs)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    return {"run_id": run_id, "status": face_service.status}


@router.get("/train/status")
def training_status(_: User = Depends(require_admin)):
    return {**face_service.status, "model_ready": face_service.ready,
            "classes": face_service.classes}


@router.get("/train/metrics")
def training_metrics(_: User = Depends(require_admin)):
    """Metrics of the current model, or null if no model has been trained yet."""
    return cnn.load_metrics()


@router.get("/train/runs")
def training_runs(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    runs = db.query(TrainingRun).order_by(TrainingRun.id.desc()).limit(20).all()
    return [{"id": r.id, "status": r.status, "started_by": r.started_by, "num_images": r.num_images,
             "num_classes": r.num_classes, "accuracy": r.accuracy, "macro_f1": r.macro_f1,
             "message": r.message, "started_at": r.started_at, "finished_at": r.finished_at}
            for r in runs]


# ---------------- verification ---------------------------------------------------
@router.post("/verify")
async def verify(file: UploadFile = File(...), user: User = Depends(get_current_user),
                 db: Session = Depends(get_db)):
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "Image is larger than 15 MB.")
    if not face_service.ready:
        raise HTTPException(409, "No trained model yet. An admin must upload photos and train the model.")
    try:
        face = extract_face(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    log = VerificationLog(checked_by=user.user_id)
    if face.face is None:
        log.result = "NO_FACE"
        db.add(log)
        db.commit()
        return {"result": "NO_FACE", "message": "No face detected. Retake the photo facing the camera "
                "with good light, and without a mask, cap or sunglasses.", "top": [], "employee": None}

    top = face_service.predict(face.face)
    result = face_service.decide(top)
    by_label = {e.label: e for e in db.query(Employee).filter(
        Employee.label.in_([t["label"] for t in top])).all()}
    for t in top:
        emp = by_label.get(t["label"])
        t["name"] = emp.name if emp else t["label"]
        t["employee_id"] = emp.id if emp else None

    best = by_label.get(top[0]["label"]) if result == "MATCH" else None
    log.predicted_label, log.confidence, log.result = top[0]["label"], top[0]["confidence"], result
    log.employee_id = best.id if best else None
    db.add(log)
    db.commit()

    response = {"result": result, "confidence": top[0]["confidence"], "top": top, "face_box": face.box,
                "threshold": settings.CONFIDENCE_THRESHOLD, "employee": None, "warning": None,
                "allowed_zones": []}
    if result == "UNKNOWN":
        response["message"] = (f"Low confidence ({top[0]['confidence']:.0%} < "
                               f"{settings.CONFIDENCE_THRESHOLD:.0%}). Treat as UNKNOWN: check the "
                               "physical ID card and call the reporting manager.")
    elif best is not None:
        response["employee"] = to_out(best).model_dump()
        response["warning"] = STATUS_WARNINGS.get(best.status)
        response["allowed_zones"] = allowed_zones(best.access_level, best.status)
        response["message"] = f"Verified as {best.name}."
    else:
        response["message"] = "The model matched a label that has no employee record. Retrain the model."
    return response
