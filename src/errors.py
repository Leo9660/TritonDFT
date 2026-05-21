"""Unified API error shape.

All user-facing API errors return:
    {
      "detail": {
        "code":    "insufficient_credits",       # stable identifier (lookup key)
        "message": "You're out of credits.",      # human-readable fallback
        "details": {"credits_needed": 13, ...}    # optional structured extras
      }
    }
Frontend uses `code` for i18n lookup and shows the localized string; if no
translation is registered it falls back to `message`.
"""
from typing import Any, Dict, Optional
from fastapi import HTTPException


class APIError(HTTPException):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        detail: Dict[str, Any] = {"code": code, "message": message}
        if details:
            detail["details"] = details
        super().__init__(status_code=status_code, detail=detail)


# ───── Auth ─────
def not_authenticated():
    return APIError(401, "not_authenticated", "You need to sign in to continue.")


def invalid_token():
    return APIError(401, "invalid_token", "Your session is invalid. Please sign in again.")


def user_not_found():
    return APIError(401, "user_not_found", "We couldn't find your account.")


def banned():
    return APIError(403, "banned", "Your account has been suspended. Contact an admin.")


def magic_link_invalid():
    return APIError(400, "magic_link_invalid", "This sign-in link is invalid.")


def magic_link_used():
    return APIError(400, "magic_link_used", "This sign-in link has already been used. Request a new one.")


def magic_link_expired():
    return APIError(400, "magic_link_expired", "This sign-in link has expired. Request a new one.")


def jwt_signing_failed(err: str):
    return APIError(500, "jwt_signing_failed", "Sign-in failed due to an internal error. Try again.", {"err": err})


# ───── Admin ─────
def admin_required():
    return APIError(403, "admin_required", "Admin access required.")


def admin_user_not_found():
    return APIError(404, "admin_user_not_found", "No user with that email exists.")


# ───── Chat / credits ─────
def conversation_too_long(actual: int, maximum: int):
    return APIError(
        413, "conversation_too_long",
        f"Your conversation is too long ({actual} chars). Please start a new chat.",
        {"actual_chars": actual, "max_chars": maximum},
    )


def message_too_long(actual: int, maximum: int):
    return APIError(
        413, "message_too_long",
        f"Your message is too long. Please shorten it (max ~{maximum} characters).",
        {"actual_chars": actual, "max_chars": maximum},
    )


def insufficient_credits(needed: int, remaining: int):
    return APIError(
        402, "insufficient_credits",
        "You're out of credits. Contact an admin to top up.",
        {"credits_needed": needed, "credits_remaining": remaining},
    )


def agent_busy():
    return APIError(
        429, "agent_busy",
        "Another DFT job is currently running. Please try again in a minute.",
    )


# ───── Jobs ─────
def job_not_found():
    return APIError(404, "job_not_found", "That job doesn't exist or isn't yours.")


def empty_message():
    return APIError(400, "empty_message", "Your message is empty.")
