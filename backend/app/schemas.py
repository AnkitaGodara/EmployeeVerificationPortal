"""Request and response shapes for the API."""
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    full_name: str
    role: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    user_id: str
    full_name: str
    role: str
    is_active: bool


class UserCreate(BaseModel):
    user_id: str = Field(min_length=3, max_length=50)
    full_name: str = Field(min_length=2, max_length=100)
    password: str = Field(min_length=8, max_length=72)
    role: str = Field(default="security", pattern="^(admin|security)$")


class EmployeeBase(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    employee_code: Optional[str] = Field(default=None, max_length=20)
    department: str = ""
    designation: str = ""
    access_level: str = Field(default="L1", pattern="^L[1-4]$")
    email: str = ""
    phone: str = ""
    office_location: str = ""
    shift: str = "General (9:30-18:30)"
    manager: str = ""
    joining_date: str = ""
    status: str = Field(default="Active", pattern="^(Active|On Leave|Suspended|Exited)$")
    notes: str = ""


class EmployeeCreate(EmployeeBase):
    label: Optional[str] = None   # folder / class name; made from the name if empty

    @field_validator("label")
    @classmethod
    def clean_label(cls, v):
        if v is None or v.strip() == "":
            return None
        v = re.sub(r"[^a-z0-9_]+", "_", v.strip().lower()).strip("_")
        if not v:
            raise ValueError("label must contain letters or digits")
        return v


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    department: Optional[str] = None
    designation: Optional[str] = None
    access_level: Optional[str] = Field(default=None, pattern="^L[1-4]$")
    email: Optional[str] = None
    phone: Optional[str] = None
    office_location: Optional[str] = None
    shift: Optional[str] = None
    manager: Optional[str] = None
    joining_date: Optional[str] = None
    status: Optional[str] = Field(default=None, pattern="^(Active|On Leave|Suspended|Exited)$")
    notes: Optional[str] = None


class EmployeeOut(EmployeeBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    label: str
    employee_code: str
    has_photo: bool = False
    face_images: int = 0


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    employee_id: Optional[int] = None
