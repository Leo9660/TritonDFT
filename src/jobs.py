"""Job queue API: enqueue a DFT request, poll its status/output, cancel it.

Execution happens in a separate worker process (worker.py) — the HTTP request
lifecycle is fully decoupled from the (possibly 30-minute) agent run.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db import get_session, Job, User
from auth import get_current_user
from credits import count_tokens, pre_charge, reconcile
from ratelimit import limiter, PER_IP_RATE
import errors

router = APIRouter(prefix="/jobs", tags=["jobs"])

MAX_MESSAGE_CHARS = 8000
MAX_CONVERSATION_CHARS = 2_000_000
MAX_OUTPUT_TOKENS = 4096   # worst-case for pre-charge


class CreateJobBody(BaseModel):
    messages: list


def _extract_user_message(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content", ""))
    if messages and isinstance(messages[-1], dict):
        return str(messages[-1].get("content", ""))
    return ""


def _queue_position(db: Session, job: Job) -> int:
    """0 = next to run."""
    return (
        db.query(Job)
        .filter(Job.status == "queued", Job.created_at < job.created_at)
        .count()
    )


@router.post("")
@limiter.limit(PER_IP_RATE)
async def create_job(
    request: Request,
    body: CreateJobBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    messages = body.messages or []
    total_chars = sum(len(str(m.get("content", ""))) for m in messages if isinstance(m, dict))
    if total_chars > MAX_CONVERSATION_CHARS:
        raise errors.conversation_too_long(total_chars, MAX_CONVERSATION_CHARS)

    user_msg = _extract_user_message(messages)
    if not user_msg.strip():
        raise errors.empty_message()
    if len(user_msg) > MAX_MESSAGE_CHARS:
        raise errors.message_too_long(len(user_msg), MAX_MESSAGE_CHARS)

    # Pre-charge worst-case credits up front; refunded by the worker on finish.
    full_input = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    input_tokens, _ = count_tokens(full_input)
    log, err = pre_charge(db, user, input_tokens, MAX_OUTPUT_TOKENS, endpoint="/jobs")
    if err:
        raise errors.insufficient_credits(err["needed"], err["remaining"])

    job = Job(
        user_id=user.id,
        status="queued",
        query=user_msg,
        output="",
        usage_log_id=log.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "job_id": str(job.id),
        "status": job.status,
        "queue_position": _queue_position(db, job),
        "credits_remaining": user.credits,
    }


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    # Treat "not yours" as not-found so job existence isn't leaked.
    if job is None or (job.user_id != user.id and not user.is_admin):
        raise errors.job_not_found()

    return {
        "job_id": str(job.id),
        "status": job.status,
        "output": job.output or "",
        "error": job.error,
        "queue_position": _queue_position(db, job) if job.status == "queued" else None,
        "credits_remaining": user.credits,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None or (job.user_id != user.id and not user.is_admin):
        raise errors.job_not_found()

    if job.status in ("queued", "running"):
        job.status = "cancelled"
        job.finished_at = datetime.utcnow()
        db.commit()
        # Refund the pre-charge. If a worker is mid-run it will also reconcile
        # on finish — reconcile is idempotent so no double refund.
        try:
            reconcile(db, job.usage_log_id, job.user_id, count_tokens(job.output or "")[0])
        except Exception as e:
            print(f"[jobs] cancel reconcile failed for {job_id}: {e}")

    return {"ok": True, "status": job.status}
