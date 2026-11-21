"""Verification logs and dashboard numbers."""
from datetime import datetime, time, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db import Employee, User, VerificationLog, get_db
from ..face.service import face_service
from ..security import get_current_user, require_admin

router = APIRouter(prefix="/api", tags=["admin"])


@router.get("/logs")
def logs(limit: int = 100, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    rows = (db.query(VerificationLog).order_by(VerificationLog.id.desc())
            .limit(max(1, min(limit, 500))).all())
    return [{"id": r.id, "checked_by": r.checked_by, "result": r.result, "confidence": r.confidence,
             "predicted_label": r.predicted_label, "employee_name": r.employee.name if r.employee else None,
             "created_at": r.created_at} for r in rows]


@router.get("/stats")
def stats(_: User = Depends(get_current_user), db: Session = Depends(get_db)):
    start_of_day = datetime.combine(datetime.now(timezone.utc).date(), time.min, tzinfo=timezone.utc)
    today = db.query(VerificationLog).filter(VerificationLog.created_at >= start_of_day)
    by_result = dict(db.query(VerificationLog.result, func.count()).group_by(VerificationLog.result).all())
    total = sum(by_result.values())
    return {"employees": db.query(Employee).count(),
            "active_employees": db.query(Employee).filter(Employee.status == "Active").count(),
            "model_ready": face_service.ready, "model_classes": len(face_service.classes),
            "verifications_today": today.count(), "verifications_total": total,
            "match_rate": round(by_result.get("MATCH", 0) / total, 4) if total else None,
            "by_result": by_result}
