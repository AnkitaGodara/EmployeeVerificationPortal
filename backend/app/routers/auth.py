"""Login, current user, and portal user management (admin only)."""
import time
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..db import User, get_db
from ..schemas import LoginResponse, UserCreate, UserOut
from ..security import create_token, get_current_user, hash_password, require_admin, verify_password

router = APIRouter(prefix="/api", tags=["auth"])

# Simple brute-force protection: 5 wrong passwords -> account locked for 5 minutes.
MAX_FAILS, LOCK_SECONDS = 5, 300
_fails: dict[str, list[float]] = defaultdict(list)


def _locked(user_id: str) -> bool:
    now = time.time()
    _fails[user_id] = [t for t in _fails[user_id] if now - t < LOCK_SECONDS]
    return len(_fails[user_id]) >= MAX_FAILS


@router.post("/auth/login", response_model=LoginResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user_id = form.username.strip()
    if _locked(user_id):
        raise HTTPException(429, "Too many failed attempts. Try again in 5 minutes.")
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user or not verify_password(form.password, user.password_hash):
        _fails[user_id].append(time.time())
        raise HTTPException(401, "Incorrect user ID or password.")
    if not user.is_active:
        raise HTTPException(403, "This account is disabled. Contact an admin.")
    _fails.pop(user_id, None)
    return LoginResponse(access_token=create_token(user), user_id=user.user_id,
                         full_name=user.full_name, role=user.role)


@router.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.get("/users", response_model=list[UserOut])
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(User).order_by(User.created_at).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    if db.query(User).filter(User.user_id == body.user_id).first():
        raise HTTPException(409, "That user ID already exists.")
    user = User(user_id=body.user_id, full_name=body.full_name,
                password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.patch("/users/{user_id}/active", response_model=UserOut)
def set_active(user_id: str, active: bool, admin: User = Depends(require_admin),
               db: Session = Depends(get_db)):
    if user_id == admin.user_id and not active:
        raise HTTPException(400, "You cannot disable your own account.")
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found.")
    user.is_active = active
    db.commit()
    db.refresh(user)
    return user
