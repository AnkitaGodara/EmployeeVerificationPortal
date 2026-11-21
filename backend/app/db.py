"""Database tables (SQLAlchemy). SQLite by default; set DATABASE_URL for PostgreSQL."""
from datetime import datetime, timezone

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
                        create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import settings

connect_args = {"check_same_thread": False} if settings.DB_URL.startswith("sqlite") else {}
engine = create_engine(settings.DB_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def now_utc():
    return datetime.now(timezone.utc)


class User(Base):
    """People who can log in to the portal (not the employees being recognised)."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    full_name = Column(String(100), nullable=False)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), nullable=False, default="security")   # admin | security
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)


class Employee(Base):
    """A person the model can recognise. `label` is the class name the CNN predicts."""
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    label = Column(String(50), unique=True, nullable=False, index=True)
    employee_code = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    department = Column(String(100), default="")
    designation = Column(String(100), default="")
    access_level = Column(String(10), default="L1")        # L1..L4, see access policy
    email = Column(String(120), default="")
    phone = Column(String(20), default="")
    office_location = Column(String(100), default="")
    shift = Column(String(50), default="General (9:30-18:30)")
    manager = Column(String(100), default="")
    joining_date = Column(String(20), default="")
    status = Column(String(20), default="Active")           # Active | On Leave | Suspended | Exited
    notes = Column(Text, default="")
    photo_path = Column(String(300), default="")
    created_at = Column(DateTime(timezone=True), default=now_utc)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)


class VerificationLog(Base):
    __tablename__ = "verification_logs"
    id = Column(Integer, primary_key=True)
    checked_by = Column(String(50), ForeignKey("users.user_id"))
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)
    predicted_label = Column(String(50), default="")
    confidence = Column(Float, default=0.0)
    result = Column(String(20), default="")    # MATCH | UNKNOWN | NO_FACE
    created_at = Column(DateTime(timezone=True), default=now_utc)
    employee = relationship("Employee")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    id = Column(Integer, primary_key=True)
    started_by = Column(String(50))
    status = Column(String(20), default="running")    # running | done | failed
    num_classes = Column(Integer, default=0)
    num_images = Column(Integer, default=0)
    accuracy = Column(Float, nullable=True)
    macro_f1 = Column(Float, nullable=True)
    message = Column(Text, default="")
    started_at = Column(DateTime(timezone=True), default=now_utc)
    finished_at = Column(DateTime(timezone=True), nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
