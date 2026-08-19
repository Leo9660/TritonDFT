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
from artifacts import extract_result, cleanup_scratch, sweep_stale_scratch
from DFTAgent import DFTAgent, JobCancelled

WORKER_ID = os.environ.get("HOSTNAME", socket.gethostname())
JOB_TIMEOUT_S = int(os.environ.get("JOB_TIMEOUT_S", "1800"))   # 30 min
# Per-pw.x wall. A band-structure workflow is 4 chained pw.x runs, so this has to
# leave room for several of them inside JOB_TIMEOUT_S.
QE_TIMEOUT_S = int(os.environ.get("QE_TIMEOUT_S", "600"))      # 10 min
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
ARTIFACT_ROOT = "/workspace/tmp"
# Age past which a run directory cannot belong to a live job, so its QE
# intermediates are safe to sweep. Generous: a job is capped at JOB_TIMEOUT_S and
# assistant-mode gates auto-continue after APPROVAL_TIMEOUT_S each.
SWEEP_MIN_AGE_S = int(os.environ.get("SWEEP_MIN_AGE_S", str(6 * 3600)))
SWEEP_INTERVAL_S = int(os.environ.get("SWEEP_INTERVAL_S", str(3600)))
# Delete QE binary intermediates when a job finishes. On by default for the
# hosted service: jobs are atomic, nothing is resumed, and the PVC is shared.
# Set CLEANUP_SCRATCH=0 to keep everything (useful when debugging a run).
CLEANUP_SCRATCH = os.environ.get("CLEANUP_SCRATCH", "1").lower() not in ("0", "false", "no")

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
    Returns (id, user_id, usage_log_id, query, model, script_only, mode, pseudo_choice)
    or None."""
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
                job.model, bool(job.script_only), job.mode or "auto", job.pseudo_choice)
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


# ───── Plan-gate DB helpers ─────

def save_plan(job_id, steps):
    """Record the agent's plan. Runs in both modes — the UI renders from this
    instead of parsing <subproblem> blocks out of the log."""
    db = SessionLocal()
    try:
        db.query(Job).filter(Job.id == job_id).update({"plan": steps})
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def set_awaiting_plan(job_id, pending: dict):
    """Publish the plan for review and flip the job to awaiting_plan."""
    db = SessionLocal()
    try:
        db.query(Job).filter(Job.id == job_id).update(
            {"status": "awaiting_plan", "pending_plan": pending, "plan_action": None})
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def read_plan_gate(job_id):
    """Return (status, plan_action) for the plan gate poll loop."""
    db = SessionLocal()
    try:
        j = db.query(Job).filter(Job.id == job_id).first()
        if j is None:
            return (None, None)
        return (j.status, j.plan_action)
    finally:
        db.close()


def resume_after_plan_gate(job_id, cancelled: bool):
    db = SessionLocal()
    try:
        fields = {"pending_plan": None, "plan_action": None}
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


def publish_run_dir(job_id, agent) -> bool:
    """Record the agent's run directory mid-flight so /files can serve the inputs
    while the job is still running. Returns True once it has been written."""
    try:
        wd = Path(str(agent.work_dir))
        if wd == Path(WORK_DIR) or not (wd / "run_meta.json").exists():
            return False
        db = SessionLocal()
        try:
            db.query(Job).filter(Job.id == job_id).update({"run_dir": str(wd)})
            db.commit()
        finally:
            db.close()
        return True
    except Exception:
        return False


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

def _apply_openai_override(agent):
    """Swap in the admin-set OpenAI key if one is configured.

    The agent (and the OpenAI client inside its generator) is built once and
    reused across jobs, so a key rotated through the admin page would otherwise
    not take effect until the pod restarted — which defeats the point of having
    a rotation path that does not need kubectl.

    Only rebuilds the client when the key actually changed, so the normal case
    costs one indexed SELECT.
    """
    db = SessionLocal()
    try:
        from admin import get_openai_override
        key = get_openai_override(db) or os.getenv("OPENAI_API_KEY")
    except Exception:
        return
    finally:
        db.close()
    if not key or getattr(agent, "_applied_openai_key", None) == key:
        return
    try:
        from openai import OpenAI
        agent.generator._oa_client = OpenAI(
            api_key=key, base_url=os.getenv("OPENAI_BASE_URL") or None
        )
        agent._applied_openai_key = key
        log(f"openai key applied: \u2026{key[-4:]}")
    except Exception as e:
        log(f"openai key override failed: {e}")


def run_job(agent, job_id, user_id, usage_log_id, query, model=None, script_only=False, mode="auto", pseudo_choice=None):
    # Reconfigure the (reused) agent for THIS job: model, script-only mode, and
    # a fresh token tally so billing reflects only this job's usage.
    if model:
        agent.model = model
        try:
            agent.generator.model = model
        except Exception:
            pass
    agent.script_only = bool(script_only)
    _apply_openai_override(agent)
    # Re-resolve the pseudopotential library for THIS job (the agent is reused
    # across jobs, so a previous job's choice must not leak into this one).
    agent.pseudo_choice = dict(pseudo_choice) if pseudo_choice else None
    agent.pseudo_dir = agent.config.pseudo.PBE
    agent._apply_pseudo_choice()
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

    def plan_gate(payload):
        """Record the plan; in assistant mode also block for the user's review.

        Runs in BOTH modes so the plan is always persisted structurally — auto
        mode records and approves in one shot without touching job status.
        """
        steps = payload.get("steps") or []
        save_plan(job_id, steps)
        if mode != "assistant":
            return {"action": "approve"}

        emit(f"\n\n⏸️ Assistant mode — review the {len(steps)}-step plan. "
             f"Approve, edit the steps, or ask for a revision. "
             f"Auto-continues in {APPROVAL_TIMEOUT_S // 60} min.\n")
        set_awaiting_plan(job_id, payload)
        state["paused"] = True
        wait_start = time.time()
        decision = {"action": "approve"}
        try:
            while True:
                time.sleep(APPROVAL_POLL_S)
                status, action = read_plan_gate(job_id)
                if status is None or status == "cancelled":
                    decision = {"action": "cancel"}
                    break
                if action:
                    decision = action
                    break
                if time.time() - wait_start > APPROVAL_TIMEOUT_S:
                    emit("\n▶️ No response — continuing with the generated plan.\n")
                    break
        finally:
            resume_after_plan_gate(job_id, cancelled=(decision.get("action") == "cancel"))
            state["deadline"] += (time.time() - wait_start)
            state["paused"] = False
        act = decision.get("action")
        if act == "suggest":
            emit("\n▶️ Revising the plan per your suggestion…\n")
        elif act == "approve" and decision.get("steps"):
            emit("\n▶️ Running your edited plan.\n")
        return decision

    gate = approval_gate if mode == "assistant" else None

    class Catcher:
        def write(self, t):
            if not t or _TQDM_RE.search(t):
                return
            # print() emits the text and its trailing "\n" as SEPARATE write()
            # calls. Dropping every whitespace-only chunk therefore swallowed
            # EVERY newline, concatenating the entire log into one line — which
            # broke any consumer that anchors on line starts. Keep chunks that
            # carry a newline (normalised to a single one); still drop pure
            # spaces/tabs and blank padding.
            if _BLANK_RE.match(t):
                if "\n" not in t:
                    return
                # Collapse runs of blank lines — a filtered chunk (tqdm, pure
                # indentation) still emits its own trailing newline, which would
                # otherwise pile up and eat into OUTPUT_CAP.
                with buf_lock:
                    if buf and buf[-1].endswith("\n"):
                        return
                t = "\n"
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
            agent.run(query, approval_gate=gate, plan_gate=plan_gate)
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
    published_run_dir = False
    # try/finally guarantees stdout is restored even if the poll loop raises —
    # otherwise a hijacked stdout would leak into the next job on this worker.
    try:
        while not done.wait(timeout=FLUSH_INTERVAL_S):
            # Publish the run directory as soon as the agent creates it, not at
            # finalize. Until this lands the /files endpoint has nothing to serve,
            # so a user watching a live run — or one who just hit stop — sees no
            # scripts at all, even though they are already on disk.
            if not published_run_dir:
                published_run_dir = publish_run_dir(job_id, agent)
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
            # Drop the QE binary intermediates now that everything the user can
            # see has been extracted. They are regenerable, never served, and on
            # a shared PVC they only accumulate — a phonon run's _ph0 alone can
            # be tens of GB.
            freed = cleanup_scratch(wd, enabled=CLEANUP_SCRATCH)
            if freed:
                log(f"job {job_id} scratch cleaned: {freed / 1048576:.0f} MB freed")
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

    # A worker starting up is precisely when a previous pod's leftovers exist.
    if CLEANUP_SCRATCH:
        n, freed = sweep_stale_scratch(ARTIFACT_ROOT, SWEEP_MIN_AGE_S)
        if n:
            log(f"startup sweep: cleaned {n} stale run dir(s), {freed / 1048576:.0f} MB freed")

    agent = DFTAgent(
        model=os.environ.get("DEFAULT_MODEL", "gpt-5.2"),
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
        # Per-pw.x cap. Must stay below the JOB_TIMEOUT_S wall so a single runaway
        # step is killed cleanly by the executor (agent finalizes, pod survives)
        # instead of tripping the job-level wall that hard-restarts the pod — but
        # comfortably under it, since a band workflow chains 4 pw.x runs.
        qe_timeout_seconds=QE_TIMEOUT_S,
        # MP gives un-relaxed initial structures, so the planner always prepends
        # a vc-relax step. Set FORCE_VC_RELAX=0 to let the planner decide instead.
        force_vc_relax=os.environ.get("FORCE_VC_RELAX", "1").lower() not in ("0", "false", "no"),
    )
    log("agent loaded, polling for jobs")
    last_sweep = time.time()

    while True:
        try:
            reap_stale()
            claimed = claim_job()
        except Exception as e:
            log(f"claim loop error: {e}")
            claimed = None

        if claimed is None:
            # Idle is the safe moment to sweep: nothing of ours is being written,
            # and the age threshold keeps us off other workers' live runs.
            if CLEANUP_SCRATCH and time.time() - last_sweep > SWEEP_INTERVAL_S:
                last_sweep = time.time()
                n, freed = sweep_stale_scratch(ARTIFACT_ROOT, SWEEP_MIN_AGE_S)
                if n:
                    log(f"sweep: cleaned {n} stale run dir(s), {freed / 1048576:.0f} MB freed")
            time.sleep(POLL_INTERVAL_S)
            continue

        job_id, user_id, usage_log_id, query, model, script_only, mode, pseudo_choice = claimed
        log(f"claimed job {job_id}")
        try:
            run_job(agent, job_id, user_id, usage_log_id, query, model, script_only, mode, pseudo_choice)
        except Exception as e:
            log(f"run_job crashed for {job_id}: {e}")
            try:
                finalize(job_id, user_id, usage_log_id, "failed", "", str(e))
            except Exception:
                pass


if __name__ == "__main__":
    main()
