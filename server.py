import sys
from pathlib import Path

# Load .env in local dev. In NRP the values come from k8s Secrets (env block).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))

from db import init_db
from auth import router as auth_router
from admin import router as admin_router
from jobs import router as jobs_router
from ratelimit import limiter

# ============================================================
# App — lightweight API: auth, admin, job enqueue/status.
# Actual DFT execution lives in worker.py (separate Deployment).
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


# Route handlers are declared `def`, so FastAPI runs them in anyio's threadpool
# rather than on the event loop — see the note in jobs.py. That threadpool
# defaults to 40 threads, but each in-flight request holds a SQLAlchemy session
# for its whole duration and the engine gives out pool_size + max_overflow = 10
# connections per process. Forty threads competing for ten connections turns a
# burst into connection-pool timeouts, so the limits are matched here instead.
_DB_CONCURRENCY = 10


@app.on_event("startup")
def _match_threadpool_to_db_pool():
    try:
        import anyio.to_thread
        anyio.to_thread.current_default_thread_limiter().total_tokens = _DB_CONCURRENCY
    except Exception as e:
        print(f"[startup] could not size the threadpool: {e}")


@app.on_event("startup")
def _on_startup():
    _, err = init_db()
    if err:
        print(f"[startup] init_db failed: {err}")
    else:
        print("✅ DB initialized (tables + admin seed)")


app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(jobs_router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
