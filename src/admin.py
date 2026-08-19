import os
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db import get_session, User, UsageLog, AuditLog, AppSetting
from auth import get_current_user
import errors

router = APIRouter(prefix="/admin", tags=["admin"])


def require_admin(user: User = Depends(get_current_user)) -> User:
    if not user.is_admin:
        raise errors.admin_required()
    return user


class UserUpdate(BaseModel):
    credits: Optional[int] = None
    is_admin: Optional[bool] = None
    is_banned: Optional[bool] = None
    is_unlimited: Optional[bool] = None
    can_use_cpu: Optional[bool] = None


def _user_to_dict(u: User):
    return {
        "id": str(u.id),
        "email": u.email,
        "credits": u.credits,
        "is_admin": u.is_admin,
        "is_banned": u.is_banned,
        "is_unlimited": u.is_unlimited,
        "can_use_cpu": u.can_use_cpu,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    }


@router.get("/users")
async def list_users(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    users = db.query(User).order_by(desc(User.created_at)).limit(1000).all()
    return [_user_to_dict(u) for u in users]


@router.patch("/users/{email}")
async def update_user(
    email: str,
    body: UserUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if user is None:
        raise errors.admin_user_not_found()

    before = {
        "credits": user.credits,
        "is_admin": user.is_admin,
        "is_banned": user.is_banned,
        "is_unlimited": user.is_unlimited,
        "can_use_cpu": user.can_use_cpu,
    }
    changes = body.model_dump(exclude_unset=True)
    for k, v in changes.items():
        setattr(user, k, v)
    after = {
        "credits": user.credits,
        "is_admin": user.is_admin,
        "is_banned": user.is_banned,
        "is_unlimited": user.is_unlimited,
        "can_use_cpu": user.can_use_cpu,
    }

    db.add(AuditLog(
        actor_email=admin.email,
        action="user_update",
        target_email=user.email,
        before=before,
        after=after,
    ))
    db.commit()
    return _user_to_dict(user)


@router.get("/users/{email}/usage")
async def user_usage(
    email: str,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if user is None:
        raise errors.admin_user_not_found()
    logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user.id)
        .order_by(desc(UsageLog.created_at))
        .limit(200)
        .all()
    )
    return [
        {
            "id": str(l.id),
            "endpoint": l.endpoint,
            "input_tokens": l.input_tokens,
            "output_tokens": l.output_tokens,
            "credits_deducted": l.credits_deducted,
            "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]


@router.get("/audit")
async def get_audit_log(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    logs = db.query(AuditLog).order_by(desc(AuditLog.created_at)).limit(200).all()
    return [
        {
            "id": str(l.id),
            "actor_email": l.actor_email,
            "action": l.action,
            "target_email": l.target_email,
            "before": l.before,
            "after": l.after,
            "created_at": l.created_at.isoformat(),
        }
        for l in logs
    ]


# ─── Operational settings ────────────────────────────────────────────────────
# The OpenAI key normally comes from the k8s secret `tritondft-secrets`, which
# needs kubectl plus a rollout to change. When the key runs out of credit mid
# workshop that is not a workable rotation path, so an admin can override it
# here and the worker picks it up on the next job.
#
# SECURITY: the override is stored in Postgres in plaintext, and so appears in
# database backups. That is a real widening of exposure and it is deliberate:
# encrypting it with a key that also lives in this cluster would not raise the
# bar for anyone who can already read the database. Mitigations that DO matter
# are in place — it is admin-only, the value is never returned by any endpoint
# (only a last-4 fingerprint), and every change is audit-logged without the
# value. Clearing the override falls back to the k8s secret.

OPENAI_KEY_SETTING = "openai_api_key"


class ApiKeyUpdate(BaseModel):
    key: str


def _fingerprint(secret: str) -> str:
    """Enough to tell two keys apart, not enough to use one."""
    s = (secret or "").strip()
    return f"…{s[-4:]}" if len(s) >= 8 else "(too short)"


def get_openai_override(db: Session) -> Optional[str]:
    """The admin-set key, or None to fall back to the environment.

    Imported by the worker, so keep it free of FastAPI dependencies.
    """
    try:
        row = db.query(AppSetting).filter(AppSetting.key == OPENAI_KEY_SETTING).first()
        return (row.value or "").strip() or None if row else None
    except Exception:
        return None


@router.get("/settings/openai-key")
async def read_openai_key(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    row = db.query(AppSetting).filter(AppSetting.key == OPENAI_KEY_SETTING).first()
    if row is None:
        return {"source": "environment", "fingerprint": _fingerprint(os.getenv("OPENAI_API_KEY", "")),
                "updated_at": None, "updated_by": None}
    return {
        "source": "override",
        "fingerprint": _fingerprint(row.value),
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "updated_by": row.updated_by,
    }


@router.put("/settings/openai-key")
async def set_openai_key(
    body: ApiKeyUpdate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    key = (body.key or "").strip()
    # Cheap shape check only. Whether the key WORKS is not knowable without
    # spending a request against it, and a typo shows up on the next job.
    if len(key) < 20 or " " in key or "\n" in key:
        raise errors.bad_api_key()

    row = db.query(AppSetting).filter(AppSetting.key == OPENAI_KEY_SETTING).first()
    before_fp = _fingerprint(row.value) if row else _fingerprint(os.getenv("OPENAI_API_KEY", ""))
    if row is None:
        row = AppSetting(key=OPENAI_KEY_SETTING, value=key)
        db.add(row)
    else:
        row.value = key
    row.updated_at = datetime.utcnow()
    row.updated_by = admin.email

    # Fingerprints only — an audit log that records secrets is a second place to
    # leak them from.
    db.add(AuditLog(
        actor_email=admin.email,
        action="openai_key_set",
        target_email=admin.email,
        before={"fingerprint": before_fp},
        after={"fingerprint": _fingerprint(key)},
    ))
    db.commit()
    return {"source": "override", "fingerprint": _fingerprint(key),
            "updated_at": row.updated_at.isoformat(), "updated_by": row.updated_by}


@router.delete("/settings/openai-key")
async def clear_openai_key(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_session),
):
    """Drop the override and go back to whatever the k8s secret provides."""
    row = db.query(AppSetting).filter(AppSetting.key == OPENAI_KEY_SETTING).first()
    if row is not None:
        db.add(AuditLog(
            actor_email=admin.email, action="openai_key_cleared",
            target_email=admin.email,
            before={"fingerprint": _fingerprint(row.value)}, after={"fingerprint": None},
        ))
        db.delete(row)
        db.commit()
    return {"source": "environment", "fingerprint": _fingerprint(os.getenv("OPENAI_API_KEY", "")),
            "updated_at": None, "updated_by": None}
