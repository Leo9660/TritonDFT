"""Background worker: claims queued jobs from Postgres and runs the DFT agent.

Each worker pod is a separate process — its own sys.stdout — and processes one
job at a time. Concurrency = number of worker replicas. Scale via the worker
Deployment's `replicas`.
"""
import os
import re
import sys
import time
import socket
import threading
import traceback
from pathlib import Path
from datetime import datetime, timedelta

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sqlalchemy import text
from db import SessionLocal, Job, init_db
from credits import count_tokens, reconcile
from artifacts import extract_result
from DFTAgent import DFTAgent, JobCancelled

WORKER_ID = os.environ.get("HOSTNAME", socket.gethostname())
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "1800"))   # 30 min
# Assistant mode: how long a step's script waits for human review before the
# worker auto-continues with the generated script as-is. The execution-timeout
# clock is paused while waiting, so this never eats into JOB_TIMEOUT_S.
APPROVAL_TIMEOUT_S = int(os.environ.get("APPROVAL_TIMEOUT_S", "600"))   # 10 min
APPROVAL_POLL_S = 2.0
OUTPUT_CAP = 80_000
FLUSH_INTERVAL_S = 1.5
POLL_INTERVAL_S = 2.0
# Unique work dir per worker pod so concurrent workers don't collide on the
# shared RWX PVC.
WORK_DIR = f"/workspace/tmp/{WORKER_ID}"

_REAL_STDOUT = sys.stdout
_REAL_STDERR = sys.stderr

_TQDM_RE = re.compile(r"\d+%\s*\|.*?\|\s*\d+/\d+\s*\[")
_BLANK_RE = re.compile(r"^[\s\r\n]*$")
_ERR_RE = re.compile(r"\[error\]|\[exception\]|\[fatal\]|Traceback", re.IGNORECASE)


def log(msg: str):
    """Worker-level logging — always to the real stdout, never the hijacked one."""
    _REAL_STDOUT.write(f"[worker {WORKER_ID}] {msg}\n")
    _REAL_STDOUT.flush()


# ───── DB helpers (each opens a short-lived session) ─────

def claim_job():
    """Atomically claim one queued job.
    Returns (id, user_id, usage_log_id, query, model, script_only, mode) or None."""
    db = SessionLocal()
    try:
        row = db.execute(text("""
            UPDATE jobs SET status='running', worker_id=:wid, started_at=now()
            WHERE id = (
                SELECT id FROM jobs WHERE status='queued'
                ORDER BY created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id
        """), {"wid": WORKER_ID}).fetchone()
        db.commit()
        if row is None:
            return None
        job = db.query(Job).filter(Job.id == row[0]).first()
        return (job.id, job.user_id, job.usage_log_id, job.query,
                job.model, bool(job.script_only), job.mode or "auto")
    finally:
        db.close()


# ───── Approval-gate DB helpers (assistant mode) ─────

def set_awaiting(job_id, pending: dict):
    """Publish the pending step for review and flip the job to awaiting_approval."""
    db = SessionLocal()
    try:
        db.query(Job).filter(Job.id == job_id).update(
            {"status": "awaiting_approval", "pending_step": pending, "step_action": None})
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def read_gate(job_id):
    """Return (status, step_action) for the gate poll loop."""
    db = SessionLocal()
    try:
        j = db.query(Job).filter(Job.id == job_id).first()
        if j is None:
            return (None, None)
        return (j.status, j.step_action)
    finally:
        db.close()


def resume_after_gate(job_id, cancelled: bool):
    """Clear the pending-step fields once the gate returns. Keep a cancelled job
    cancelled; otherwise flip back to running so the agent can proceed."""
    db = SessionLocal()
    try:
        fields = {"pending_step": None, "step_action": None}
        if not cancelled:
            fields["status"] = "running"
        db.query(Job).filter(Job.id == job_id).update(fields)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def reap_stale():
    """Mark jobs stuck 'running' past the timeout (their worker pod died) as timed-out."""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=JOB_TIMEOUT_S + 120)
        # Assistant-mode jobs legitimately pause for human review, so their
        # wall-clock (started_at → now) can far exceed JOB_TIMEOUT_S without the
        # worker being dead. Excluding them avoids falsely reaping a resumed job.
        stale = db.query(Job).filter(
            Job.status == "running",
            Job.started_at < cutoff,
            (Job.mode == None) | (Job.mode != "assistant"),  # noqa: E711
        ).all()
        for job in stale:
            job.status = "timeout"
            job.error = "Worker did not finish in time (it likely crashed)."
            job.finished_at = datetime.utcnow()
            try:
                reconcile(db, job.usage_log_id, job.user_id, job.model,
                          None, count_tokens(job.output or "")[0])
            except Exception:
                pass
        if stale:
            db.commit()
            log(f"reaped {len(stale)} stale job(s)")
    except Exception as e:
        db.rollback()
        log(f"reap_stale error: {e}")
    finally:
        db.close()


def flush_output(job_id, output: str):
    db = SessionLocal()
    try:
        db.query(Job).filter(Job.id == job_id).update({"output": output[:OUTPUT_CAP]})
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def is_cancelled(job_id) -> bool:
    db = SessionLocal()
    try:
        j = db.query(Job).filter(Job.id == job_id).first()
        return j is not None and j.status == "cancelled"
    finally:
        db.close()


def finalize(job_id, user_id, usage_log_id, status, output, error,
             run_dir=None, result=None, model=None,
             prompt_tokens=None, output_tokens=None, cost_usd=None):
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job is None:
            return
        # Don't overwrite a user-initiated cancellation.
        if job.status != "cancelled":
            job.status = status
        # Caller already capped to OUTPUT_CAP (+ truncation notice); don't
        # re-slice here or the notice gets clipped off.
        job.output = output
        job.error = error
        if run_dir:
            job.run_dir = run_dir
        if result:
            job.result = result
        job.finished_at = datetime.utcnow()
        db.commit()
        # Bill the REAL OpenAI usage the generator tallied for this job. Fall
        # back to a stdout-length estimate if the generator counts are missing
        # (e.g. the agent crashed before any LLM call).
        if output_tokens is None:
            output_tokens, _ = count_tokens(output)
        reconcile(db, usage_log_id, user_id, model or job.model,
                  prompt_tokens, output_tokens, cost_usd=cost_usd)
    except Exception as e:
        db.rollback()
        log(f"finalize error for {job_id}: {e}")
    finally:
        db.close()


# ───── Job execution ─────

def run_job(agent, job_id, user_id, usage_log_id, query, model=None, script_only=False, mode="auto"):
    # Reconfigure the (reused) agent for THIS job: model, script-only mode, and
    # a fresh token tally so billing reflects only this job's usage.
    if model:
        agent.model = model
        try:
            agent.generator.model = model
        except Exception:
            pass
    agent.script_only = bool(script_only)
    try:
        agent.generator.reset_token_counters()
    except Exception:
        pass
    log(f"job {job_id}: model={agent.model} script_only={agent.script_only} mode={mode}")

    buf = []
    buf_lock = threading.Lock()

    # Execution-timeout clock, shared with the approval gate so human-wait time
    # never counts against JOB_TIMEOUT_S.
    state = {"deadline": time.time() + JOB_TIMEOUT_S, "paused": False}

    def emit(msg: str):
        with buf_lock:
            buf.append(msg)

    def approval_gate(step_meta, scripts):
        """Blocks the agent thread: publish the step, wait for the user's action
        (or auto-continue after APPROVAL_TIMEOUT_S). Returns the decision dict."""
        idx = step_meta.get("step_index")
        total = step_meta.get("total_steps")
        label = f"step {idx}" + (f"/{total}" if total else "")
        emit(f"\n\n⏸️ Assistant mode — review the script for {label} "
             f"({step_meta.get('tool', '')}). Approve, edit, or suggest a change. "
             f"Auto-continues in {APPROVAL_TIMEOUT_S // 60} min.\n")
        set_awaiting(job_id, {**step_meta, "scripts": scripts})
        state["paused"] = True
        wait_start = time.time()
        decision = {"action": "approve"}
        try:
            while True:
                time.sleep(APPROVAL_POLL_S)
                status, action = read_gate(job_id)
                if status is None or status == "cancelled":
                    decision = {"action": "cancel"}
                    break
                if action:
                    decision = action
                    break
                if time.time() - wait_start > APPROVAL_TIMEOUT_S:
                    emit("\n▶️ No response — continuing with the generated script.\n")
                    decision = {"action": "approve"}
                    break
        finally:
            resume_after_gate(job_id, cancelled=(decision.get("action") == "cancel"))
            state["deadline"] += (time.time() - wait_start)
            state["paused"] = False
        act = decision.get("action")
        if act == "suggest":
            emit("\n▶️ Revising the script per your suggestion…\n")
        elif act == "approve" and decision.get("scripts"):
            emit("\n▶️ Running your edited script.\n")
        return decision

    gate = approval_gate if mode == "assistant" else None

    class Catcher:
        def write(self, t):
            if not t or _TQDM_RE.search(t) or _BLANK_RE.match(t):
                return
            cleaned = t.replace("\r", "")
            if not cleaned:
                return
            with buf_lock:
                buf.append(cleaned)
            # Tee to the real stdout so `kubectl logs` on the worker is useful.
            prefix = "AGENT-ERR " if _ERR_RE.search(cleaned) else "AGENT "
            line = cleaned if cleaned.endswith("\n") else cleaned + "\n"
            _REAL_STDOUT.write(prefix + line)
            _REAL_STDOUT.flush()

        def flush(self):
            pass

    crashed = {"err": None}
    done = threading.Event()

    def agent_thread():
        try:
            agent.run(query, approval_gate=gate)
        except JobCancelled:
            # Clean stop — the user cancelled at an approval gate. The DB status
            # is already 'cancelled'; nothing to report as an error.
            with buf_lock:
                buf.append("\n\n> ⏹️ Cancelled at your request.\n")
        except Exception as e:
            crashed["err"] = str(e)
            with buf_lock:
                buf.append(f"\n\n> ⚠️ The agent hit an error and stopped.\n> {e}\n")
            _REAL_STDERR.write(f"AGENT-CRASH {traceback.format_exc()}\n")
            _REAL_STDERR.flush()
        finally:
            done.set()

    sys.stdout = Catcher()
    sys.stderr = Catcher()

    t = threading.Thread(target=agent_thread, daemon=True)
    t.start()

    status = "done"
    # try/finally guarantees stdout is restored even if the poll loop raises —
    # otherwise a hijacked stdout would leak into the next job on this worker.
    try:
        while not done.wait(timeout=FLUSH_INTERVAL_S):
            # Don't enforce the execution timeout while paused for human review;
            # the gate extends state["deadline"] by the waited time on resume.
            if not state["paused"] and time.time() > state["deadline"]:
                status = "timeout"
                with buf_lock:
                    buf.append(f"\n\n> ⏱️ Request timed out after {JOB_TIMEOUT_S // 60} minutes.\n")
                break
            with buf_lock:
                snapshot = "".join(buf)
            flush_output(job_id, snapshot)
            if is_cancelled(job_id):
                status = "cancelled"
                break
    finally:
        sys.stdout = _REAL_STDOUT
        sys.stderr = _REAL_STDERR

    if crashed["err"]:
        status = "failed"
    with buf_lock:
        final_output = "".join(buf)
    if len(final_output) > OUTPUT_CAP:
        final_output = final_output[:OUTPUT_CAP] + "\n\n[... output truncated at 80KB ...]"

    # Capture artifacts — agent.work_dir points at this run's directory.
    run_dir = None
    result = None
    try:
        wd = Path(str(agent.work_dir))
        if wd != Path(WORK_DIR) and (wd / "run_meta.json").exists():
            run_dir = str(wd)
            result = extract_result(wd)
            log(f"job {job_id} artifacts: run_dir={run_dir} result={result}")
    except Exception as e:
        log(f"artifact capture failed for {job_id}: {e}")

    # Real token usage the generator tallied across all LLM calls in this job.
    prompt_tokens = getattr(agent.generator, "total_prompt_tokens", None)
    output_tokens = getattr(agent.generator, "total_output_tokens", None)
    # Claude Code CLI reports exact USD cost; 0 for the OpenAI (token-billed) path.
    cost_usd = getattr(agent.generator, "total_cost_usd", 0.0) or None

    finalize(job_id, user_id, usage_log_id, status,
             final_output, crashed["err"], run_dir=run_dir, result=result,
             model=agent.model, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
             cost_usd=cost_usd)
    log(f"job {job_id} finished: status={status} tokens(in/out)={prompt_tokens}/{output_tokens} cost_usd={cost_usd}")

    # If the agent thread is still alive (timeout/cancel — agent.run() can't be
    # killed), exit the process so k8s restarts a clean pod. Otherwise the
    # orphan thread keeps print()-ing into the NEXT job's hijacked stdout.
    if t.is_alive():
        log(f"agent thread still alive after {job_id}; exiting for a clean restart")
        os._exit(0)


def main():
    os.makedirs(WORK_DIR, exist_ok=True)
    log(f"starting, work_dir={WORK_DIR}, job_timeout={JOB_TIMEOUT_S}s")

    init_db()  # idempotent — ensures tables exist even if worker starts first

    agent = DFTAgent(
        model=os.environ.get("DEFAULT_MODEL", "gpt-4o"),
        dft_tool="quantum espresso",
        verbose=True,
        backend="openai",
        work_dir=WORK_DIR,
        max_new_tokens=4096,
        temperature=0.0,
        top_p=0.9,
        need_query_info=True,
        parallel_exec=True,
        # MPI ranks per QE run — keep ≤ the container CPU limit (8) so we don't
        # oversubscribe cores.
        parallel_np=int(os.environ.get("JOB_NP", "8")),
        # Per-pw.x cap, kept just under the JOB_TIMEOUT_S wall (300s) so a single
        # runaway step is killed cleanly by the executor (agent finalizes, pod
        # survives) instead of tripping the job-level wall that hard-restarts the pod.
        qe_timeout_seconds=270,
        # MP gives un-relaxed initial structures, so the planner always prepends
        # a vc-relax step. Set FORCE_VC_RELAX=0 to let the planner decide instead.
        force_vc_relax=os.environ.get("FORCE_VC_RELAX", "1").lower() not in ("0", "false", "no"),
    )
    log("agent loaded, polling for jobs")

    while True:
        try:
            reap_stale()
            claimed = claim_job()
        except Exception as e:
            log(f"claim loop error: {e}")
            claimed = None

        if claimed is None:
            time.sleep(POLL_INTERVAL_S)
            continue

        job_id, user_id, usage_log_id, query, model, script_only, mode = claimed
        log(f"claimed job {job_id}")
        try:
            run_job(agent, job_id, user_id, usage_log_id, query, model, script_only, mode)
        except Exception as e:
            log(f"run_job crashed for {job_id}: {e}")
            try:
                finalize(job_id, user_id, usage_log_id, "failed", "", str(e))
            except Exception:
                pass


if __name__ == "__main__":
    main()
