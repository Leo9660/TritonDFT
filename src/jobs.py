"""Job queue API: enqueue a DFT request, poll its status/output, cancel it.

Execution happens in a separate worker process (worker.py) — the HTTP request
lifecycle is fully decoupled from the (possibly 30-minute) agent run.
"""
import uuid as _uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from typing import Optional

from db import get_session, Job, User
from auth import get_current_user
from credits import count_tokens, pre_charge, reconcile, resolve_model
from ratelimit import limiter, PER_IP_RATE
import artifacts
import errors

router = APIRouter(prefix="/jobs", tags=["jobs"])

MAX_MESSAGE_CHARS = 8000
MAX_CONVERSATION_CHARS = 2_000_000
MAX_OUTPUT_TOKENS = 4096   # worst-case for pre-charge
# Cap on a user's simultaneously queued+running jobs (bounds queue + $ spend).
MAX_ACTIVE_JOBS_REGULAR = 3
MAX_ACTIVE_JOBS_PRIVILEGED = 10


class CreateJobBody(BaseModel):
    messages: list
    model: Optional[str] = None
    script_only: Optional[bool] = None
    mode: Optional[str] = None   # "auto" (default) | "assistant" (human-in-the-loop)


class StepActionBody(BaseModel):
    action: str                          # approve | suggest | cancel
    scripts: Optional[list] = None       # [{filename, content}] — user-edited inputs
    suggestion: Optional[str] = None     # natural-language revision request


def _valid_uuid(s: str) -> bool:
    """Guard before querying — a non-UUID would make Postgres raise on the
    uuid column comparison (500) instead of a clean 404."""
    try:
        _uuid.UUID(str(s))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


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

    model = resolve_model(body.model)

    # Per-user cap on in-flight jobs — bounds queue depth and (with the credit
    # system) real model spend. Privileged accounts get a higher cap.
    privileged_cap = user.is_admin or user.is_unlimited
    max_active = MAX_ACTIVE_JOBS_PRIVILEGED if privileged_cap else MAX_ACTIVE_JOBS_REGULAR
    active = (
        db.query(Job)
        .filter(Job.user_id == user.id,
                Job.status.in_(("queued", "running", "awaiting_approval")))
        .count()
    )
    if active >= max_active:
        raise errors.too_many_active_jobs(active, max_active)

    # CPU policy: only admins and unlimited accounts may run real DFT (CPU).
    # Everyone else is forced to script-only (generate inputs, no execution),
    # regardless of what the client requested.
    privileged = user.is_admin or user.is_unlimited
    if privileged:
        # Default for privileged users is CPU on (script_only=False).
        script_only = bool(body.script_only) if body.script_only is not None else False
    else:
        script_only = True

    # Pre-charge worst-case credits up front; reconciled to real usage on finish.
    full_input = "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    input_tokens, _ = count_tokens(full_input)
    log, err = pre_charge(db, user, model, input_tokens, MAX_OUTPUT_TOKENS, endpoint="/jobs")
    if err:
        raise errors.insufficient_credits(err["needed"], err["remaining"])

    mode = "assistant" if (body.mode or "").lower() == "assistant" else "auto"

    job = Job(
        user_id=user.id,
        status="queued",
        query=user_msg,
        output="",
        usage_log_id=log.id,
        model=model,
        script_only=script_only,
        mode=mode,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    return {
        "job_id": str(job.id),
        "status": job.status,
        "queue_position": _queue_position(db, job),
        "credits_remaining": user.credits,
        "model": model,
        "script_only": script_only,
        "mode": mode,
    }


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    if not _valid_uuid(job_id):
        raise errors.job_not_found()
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
        "result": job.result,
        "has_artifacts": bool(job.run_dir),
        "model": job.model,
        "script_only": bool(job.script_only),
        "mode": job.mode or "auto",
        # The step + generated scripts awaiting the user's review (assistant mode).
        "pending_step": job.pending_step if job.status == "awaiting_approval" else None,
    }


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    if not _valid_uuid(job_id):
        raise errors.job_not_found()
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
            reconcile(db, job.usage_log_id, job.user_id, job.model,
                      None, count_tokens(job.output or "")[0])
        except Exception as e:
            print(f"[jobs] cancel reconcile failed for {job_id}: {e}")

    return {"ok": True, "status": job.status}


@router.post("/{job_id}/step-action")
async def step_action(
    job_id: str,
    body: StepActionBody,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    """Assistant mode: submit the user's decision for a step awaiting review.

    action=approve  → run the generated script (optionally replaced by `scripts`)
    action=suggest  → ask the LLM to revise the script per `suggestion`
    action=cancel   → cancel the whole job
    """
    if not _valid_uuid(job_id):
        raise errors.job_not_found()
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None or (job.user_id != user.id and not user.is_admin):
        raise errors.job_not_found()

    action = (body.action or "").lower()
    if action not in ("approve", "suggest", "cancel"):
        return Response(status_code=400)

    if action == "cancel":
        if job.status in ("queued", "running", "awaiting_approval"):
            job.status = "cancelled"
            job.finished_at = datetime.utcnow()
            db.commit()
            try:
                reconcile(db, job.usage_log_id, job.user_id, job.model,
                          None, count_tokens(job.output or "")[0])
            except Exception as e:
                print(f"[jobs] step-action cancel reconcile failed for {job_id}: {e}")
        return {"ok": True, "status": job.status}

    # approve / suggest are only meaningful while a step is actually awaiting review.
    if job.status != "awaiting_approval":
        return Response(status_code=409)

    payload = {"action": action}
    if action == "approve" and body.scripts:
        payload["scripts"] = body.scripts
    if action == "suggest":
        payload["suggestion"] = body.suggestion or ""

    # The worker's gate polls step_action; it flips the job back to 'running'.
    job.step_action = payload
    db.commit()
    return {"ok": True}


# ───── Artifacts ─────

def _owned_job(job_id: str, user: User, db: Session) -> Job:
    if not _valid_uuid(job_id):
        raise errors.job_not_found()
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None or (job.user_id != user.id and not user.is_admin):
        raise errors.job_not_found()
    return job


@router.get("/{job_id}/files")
async def list_job_files(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    job = _owned_job(job_id, user, db)
    run_dir = artifacts.safe_run_dir(job.run_dir)
    if run_dir is None:
        return {"files": []}
    return {"files": artifacts.list_files(run_dir)}


@router.get("/{job_id}/bands")
async def get_job_bands(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    job = _owned_job(job_id, user, db)
    run_dir = artifacts.safe_run_dir(job.run_dir)
    if run_dir is None:
        return {"bands": None}
    return {"bands": artifacts.parse_bands(run_dir)}


@router.get("/{job_id}/download")
async def download_job_zip(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    job = _owned_job(job_id, user, db)
    run_dir = artifacts.safe_run_dir(job.run_dir)
    if run_dir is None:
        raise errors.job_not_found()
    data = artifacts.build_zip(run_dir)
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="tritondft-{job_id}.zip"'},
    )


@router.get("/{job_id}/files/{name}")
async def get_job_file(
    job_id: str,
    name: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    job = _owned_job(job_id, user, db)
    run_dir = artifacts.safe_run_dir(job.run_dir)
    if run_dir is None or not artifacts.is_safe_filename(name):
        raise errors.job_not_found()
    # Only serve whitelisted files that actually exist in the listing.
    allowed = {f["name"] for f in artifacts.list_files(run_dir)}
    if name not in allowed:
        raise errors.job_not_found()
    fp = run_dir / name
    # Defense in depth: the resolved path must still sit inside run_dir (guards
    # against a symlink that slipped past the listing filter).
    try:
        resolved = fp.resolve()
        if resolved != run_dir and run_dir not in resolved.parents:
            raise errors.job_not_found()
        data = fp.read_bytes()
    except OSError:
        raise errors.job_not_found()
    return Response(content=data, media_type="text/plain; charset=utf-8")
