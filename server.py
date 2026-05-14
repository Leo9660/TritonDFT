import sys
from pathlib import Path
import os
import re
import threading
from queue import Queue, Empty
import time
import random

# Load .env in local dev. In NRP the values come from k8s Secrets (env block).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))

from DFTAgent import DFTAgent
from db import init_db, get_session, SessionLocal, User
from auth import router as auth_router, get_current_user
from admin import router as admin_router
from credits import count_tokens, pre_charge, reconcile
import errors

# ============================================================
# Rate-limit configuration
# ============================================================

MAX_MESSAGE_CHARS = 8000          # per-message cap (the only one the agent actually sees)
MAX_CONVERSATION_CHARS = 2_000_000  # raw payload guard only — agent uses just the latest user msg,
                                    # so huge histories from streamed agent logs are harmless
REQUEST_TIMEOUT_S = 300
MAX_OUTPUT_TOKENS = 4096   # worst-case for pre-charge

PER_IP_RATE = "5/minute;30/day"
agent_lock = threading.Lock()


def get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=get_real_ip, default_limits=[])

# ============================================================
# App
# ============================================================

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup():
    _, err = init_db()
    if err:
        # Don't crash the app; admin can inspect logs and fix DB.
        # Auth endpoints will surface DB errors directly.
        print(f"[startup] init_db failed: {err}")
    else:
        print("✅ DB initialized (tables + admin seed)")


app.include_router(auth_router)
app.include_router(admin_router)

# ========== Agent init ==========
agent = DFTAgent(
    model="gpt-4o",
    dft_tool="quantum espresso",
    verbose=True,
    backend="openai",
    work_dir="",
    max_new_tokens=4096,
    temperature=0.0,
    top_p=0.9,
    need_query_info=True,
    parallel_exec=True,
    parallel_np=12,
)

print("✅ DFT Agent Loaded")


class ChatRequest(BaseModel):
    messages: list


def extract_user_message(messages):
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return messages[-1]["content"]


def stream_generator(query: str, deadline: float, user_id, log_id):
    """Stream agent stdout/stderr; on completion, reconcile credit refund.

    Also tees agent output to the real pod stderr so kubectl logs is useful
    for debugging — lines matching error patterns are flagged with a prefix
    so they're easy to grep.
    """
    q = Queue()
    old_stdout, old_stderr = sys.stdout, sys.stderr

    tqdm_re = re.compile(r"\d+%\s*\|.*?\|\s*\d+/\d+\s*\[")
    blank_re = re.compile(r"^[\s\r\n]*$")
    err_re = re.compile(r"\[error\]|\[exception\]|\[fatal\]|Traceback", re.IGNORECASE)
    yielded_chunks = []

    class StreamCatcher:
        def write(self, text):
            if not text or tqdm_re.search(text) or blank_re.match(text):
                return
            cleaned = text.replace("\r", "")
            if cleaned:
                q.put(cleaned)
                # Tee to real pod stderr so debugging via `kubectl logs` works.
                # Tag error-ish lines with a marker so they're easy to grep.
                prefix = "AGENT-ERR " if err_re.search(cleaned) else "AGENT "
                try:
                    old_stderr.write(prefix + cleaned if cleaned.endswith("\n") else prefix + cleaned + "\n")
                    old_stderr.flush()
                except Exception:
                    pass

        def flush(self):
            pass

    sys.stdout = StreamCatcher()
    sys.stderr = StreamCatcher()

    def run_agent():
        try:
            agent.run(query)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                old_stderr.write(f"AGENT-CRASH agent.run() raised:\n{tb}\n")
                old_stderr.flush()
            except Exception:
                pass
            q.put(f"\n\n> ⚠️ **The agent hit an error and stopped.**\n> {str(e)}\n")
        finally:
            q.put(None)

    t = threading.Thread(target=run_agent, daemon=True)
    t.start()

    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                yield "\n\n> ⏱️ **Request timed out** after 5 minutes. The agent has been stopped.\n"
                yielded_chunks.append("[TIMEOUT]")
                break
            try:
                msg = q.get(timeout=min(remaining, 1.0))
            except Empty:
                continue

            if msg is None:
                break

            text = str(msg)
            yielded_chunks.append(text)
            for ch in text:
                yield ch
                time.sleep(random.uniform(0.001, 0.005))
            yield "\n"

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        # Reconcile: refund unused output budget
        try:
            actual_output_text = "".join(yielded_chunks)
            actual_tokens, _ = count_tokens(actual_output_text)
            db = SessionLocal()
            try:
                _, err = reconcile(db, log_id, user_id, actual_tokens)
                if err:
                    print(f"[credits] reconcile failed: {err}")
            finally:
                db.close()
        except Exception as e:
            print(f"[credits] reconcile exception: {e}")

        if agent_lock.locked():
            try:
                agent_lock.release()
            except RuntimeError:
                pass


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
@limiter.limit(PER_IP_RATE)
async def chat_completions(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_session),
):
    body = await request.json()
    messages = body.get("messages", [])

    # ---- Input size cap ----
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    if total_chars > MAX_CONVERSATION_CHARS:
        raise errors.conversation_too_long(total_chars, MAX_CONVERSATION_CHARS)

    user_msg = extract_user_message(messages)
    if len(user_msg) > MAX_MESSAGE_CHARS:
        raise errors.message_too_long(len(user_msg), MAX_MESSAGE_CHARS)

    # ---- Credits: pre-charge worst case ----
    full_input_text = "\n".join(str(m.get("content", "")) for m in messages)
    input_tokens, _ = count_tokens(full_input_text)
    log, err = pre_charge(
        db, user, input_tokens, MAX_OUTPUT_TOKENS, endpoint="/v1/chat/completions"
    )
    if err:
        raise errors.insufficient_credits(err["needed"], err["remaining"])

    user_id = user.id
    log_id = log.id

    # ---- Global concurrency: one agent run at a time ----
    if not agent_lock.acquire(blocking=False):
        try:
            reconcile(db, log_id, user_id, 0)
        except Exception:
            pass
        raise errors.agent_busy()

    print(f"\n📩 Incoming query from {user.email} (IP={get_real_ip(request)}):\n{user_msg}\n")

    deadline = time.time() + REQUEST_TIMEOUT_S
    return StreamingResponse(
        stream_generator(user_msg, deadline, user_id, log_id),
        media_type="text/plain",
    )
