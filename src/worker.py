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
from DFTAgent import DFTAgent

WORKER_ID = os.environ.get("HOSTNAME", socket.gethostname())
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "1800"))   # 30 min
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
    """Atomically claim one queued job. Returns (id, user_id, usage_log_id, query) or None."""
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
        return (job.id, job.user_id, job.usage_log_id, job.query)
    finally:
        db.close()


def reap_stale():
    """Mark jobs stuck 'running' past the timeout (their worker pod died) as timed-out."""
    db = SessionLocal()
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=JOB_TIMEOUT_S + 120)
        stale = db.query(Job).filter(Job.status == "running", Job.started_at < cutoff).all()
        for job in stale:
            job.status = "timeout"
            job.error = "Worker did not finish in time (it likely crashed)."
            job.finished_at = datetime.utcnow()
            try:
                reconcile(db, job.usage_log_id, job.user_id, count_tokens(job.output or "")[0])
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
             run_dir=None, result=None):
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
        actual_tokens, _ = count_tokens(output)
        reconcile(db, usage_log_id, user_id, actual_tokens)
    except Exception as e:
        db.rollback()
        log(f"finalize error for {job_id}: {e}")
    finally:
        db.close()


# ───── Job execution ─────

def run_job(agent, job_id, user_id, usage_log_id, query):
    buf = []
    buf_lock = threading.Lock()

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
            agent.run(query)
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

    deadline = time.time() + JOB_TIMEOUT_S
    status = "done"
    # try/finally guarantees stdout is restored even if the poll loop raises —
    # otherwise a hijacked stdout would leak into the next job on this worker.
    try:
        while not done.wait(timeout=FLUSH_INTERVAL_S):
            if time.time() > deadline:
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

    finalize(job_id, user_id, usage_log_id, status,
             final_output, crashed["err"], run_dir=run_dir, result=result)
    log(f"job {job_id} finished: status={status}")

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
        model="gpt-4o",
        dft_tool="quantum espresso",
        verbose=True,
        backend="openai",
        work_dir=WORK_DIR,
        max_new_tokens=4096,
        temperature=0.0,
        top_p=0.9,
        need_query_info=True,
        parallel_exec=True,
        parallel_np=12,
        qe_timeout_seconds=540,
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

        job_id, user_id, usage_log_id, query = claimed
        log(f"claimed job {job_id}")
        try:
            run_job(agent, job_id, user_id, usage_log_id, query)
        except Exception as e:
            log(f"run_job crashed for {job_id}: {e}")
            try:
                finalize(job_id, user_id, usage_log_id, "failed", "", str(e))
            except Exception:
                pass


if __name__ == "__main__":
    main()
