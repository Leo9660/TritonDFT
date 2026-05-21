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
