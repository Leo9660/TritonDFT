from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import desc

from db import get_session, User, UsageLog, AuditLog
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


def _user_to_dict(u: User):
    return {
        "id": str(u.id),
        "email": u.email,
        "credits": u.credits,
        "is_admin": u.is_admin,
        "is_banned": u.is_banned,
        "is_unlimited": u.is_unlimited,
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
    }
    changes = body.model_dump(exclude_unset=True)
    for k, v in changes.items():
        setattr(user, k, v)
    after = {
        "credits": user.credits,
        "is_admin": user.is_admin,
        "is_banned": user.is_banned,
        "is_unlimited": user.is_unlimited,
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
