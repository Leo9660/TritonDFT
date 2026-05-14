import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Request, Response, Cookie
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from db import get_session, User, MagicLink
from email_sender import send_magic_link_email
import errors

router = APIRouter(prefix="/auth", tags=["auth"])

JWT_SECRET = os.environ.get("JWT_SECRET", "")
JWT_ALG = "HS256"
JWT_TTL_DAYS = 365  # 1 year as requested
MAGIC_LINK_TTL_MINUTES = 15
FRONTEND_BASE_URL = os.environ.get(
    "FRONTEND_BASE_URL", "https://yil384.github.io/TritonDFT-frontend"
)
COOKIE_NAME = "tritondft_token"


class RequestLinkBody(BaseModel):
    email: EmailStr


class VerifyResponse(BaseModel):
    ok: bool
    token: str
    email: str
    credits: int
    is_admin: bool


class MeResponse(BaseModel):
    email: str
    credits: int
    is_admin: bool
    is_unlimited: bool


def issue_jwt(email: str):
    if not JWT_SECRET:
        return None, "JWT_SECRET not set"
    payload = {
        "sub": email,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(days=JWT_TTL_DAYS),
    }
    try:
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG), None
    except Exception as e:
        return None, str(e)


def verify_jwt(token: str):
    if not JWT_SECRET:
        return None, "JWT_SECRET not set"
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return payload.get("sub"), None
    except JWTError as e:
        return None, str(e)


def _extract_token(request: Request, cookie: Optional[str]) -> Optional[str]:
    if cookie:
        return cookie
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None


def get_current_user(
    request: Request,
    db: Session = Depends(get_session),
    cookie: Optional[str] = Cookie(default=None, alias=COOKIE_NAME),
) -> User:
    token = _extract_token(request, cookie)
    if not token:
        raise errors.not_authenticated()
    email, err = verify_jwt(token)
    if err or not email:
        raise errors.invalid_token()
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise errors.user_not_found()
    if user.is_banned:
        raise errors.banned()
    return user


@router.post("/request-link")
async def request_link(body: RequestLinkBody, db: Session = Depends(get_session)):
    email = body.email.lower().strip()
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        admin_emails = {
            e.strip().lower()
            for e in os.environ.get("ADMIN_EMAILS", "").split(",")
            if e.strip()
        }
        is_admin = email in admin_emails
        user = User(
            email=email,
            credits=1000,
            is_admin=is_admin,
            is_unlimited=is_admin,
        )
        db.add(user)
        db.flush()
    if user.is_banned:
        raise errors.banned()

    token = secrets.token_urlsafe(32)
    db.add(MagicLink(
        token=token,
        email=email,
        expires_at=datetime.utcnow() + timedelta(minutes=MAGIC_LINK_TTL_MINUTES),
        used=False,
    ))
    db.commit()

    link = f"{FRONTEND_BASE_URL}/auth/callback?token={token}"
    _, err = send_magic_link_email(email, link)
    if err:
        # Still 200 to avoid leaking which emails are valid; log server-side
        print(f"[auth] email send failed for {email}: {err}")
    return {"ok": True, "message": "If that email is valid, a sign-in link is on the way."}


@router.post("/verify")
async def verify_magic_link(
    token: str,
    response: Response,
    db: Session = Depends(get_session),
):
    ml = db.query(MagicLink).filter(MagicLink.token == token).first()
    if ml is None:
        raise errors.magic_link_invalid()
    if ml.used:
        raise errors.magic_link_used()
    if ml.expires_at < datetime.utcnow():
        raise errors.magic_link_expired()

    ml.used = True
    user = db.query(User).filter(User.email == ml.email).first()
    if user is None:
        raise errors.user_not_found()
    if user.is_banned:
        raise errors.banned()
    user.last_login_at = datetime.utcnow()
    db.commit()

    jwt_token, err = issue_jwt(user.email)
    if err:
        raise errors.jwt_signing_failed(err)

    # Best-effort cookie (works only if same-site). Frontend should also store
    # the returned token and send Authorization: Bearer for cross-origin.
    response.set_cookie(
        key=COOKIE_NAME,
        value=jwt_token,
        max_age=JWT_TTL_DAYS * 86400,
        httponly=True,
        secure=True,
        samesite="none",
    )
    return VerifyResponse(
        ok=True,
        token=jwt_token,
        email=user.email,
        credits=user.credits,
        is_admin=user.is_admin,
    )


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
async def me(user: User = Depends(get_current_user)):
    return MeResponse(
        email=user.email,
        credits=user.credits,
        is_admin=user.is_admin,
        is_unlimited=user.is_unlimited,
    )
