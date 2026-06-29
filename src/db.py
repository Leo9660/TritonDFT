import os
import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey, JSON, Text, Index, text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://tritondft:tritondft@localhost:5432/tritondft",
)

Base = declarative_base()
# Bound the pool: with ~10 pods (2 API + 8 workers) the default 5+10 per process
# could exceed Postgres' default max_connections=100. 5+5 per process → ≤100.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=5,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    credits = Column(Integer, default=100, nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)
    is_banned = Column(Boolean, default=False, nullable=False)
    is_unlimited = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime)


class MagicLink(Base):
    __tablename__ = "magic_links"
    token = Column(String, primary_key=True)
    email = Column(String, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class UsageLog(Base):
    __tablename__ = "usage_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    endpoint = Column(String, nullable=False)
    model = Column(String)                      # which model this usage was billed at
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    credits_deducted = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    actor_email = Column(String, nullable=False)
    action = Column(String, nullable=False)
    target_email = Column(String)
    before = Column(JSON)
    after = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class Job(Base):
    """A queued DFT request. Workers claim queued rows, run the agent, and
    append output. The HTTP request lifecycle is decoupled from execution —
    the user can close the browser and the job keeps running."""
    __tablename__ = "jobs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    # queued | running | awaiting_approval | done | failed | timeout | cancelled
    status = Column(String, default="queued", nullable=False)
    # auto = run end-to-end; assistant = pause for human review before each step's
    # script is executed (human-in-the-loop).
    mode = Column(String, default="auto", nullable=False)
    # When status=awaiting_approval: the step + generated scripts awaiting review.
    pending_step = Column(JSON)
    # The user's decision for the pending step; the worker's gate consumes it.
    step_action = Column(JSON)
    query = Column(Text, nullable=False)
    output = Column(Text, default="", nullable=False)
    error = Column(Text)
    usage_log_id = Column(UUID(as_uuid=True))   # links to UsageLog for credit reconcile
    worker_id = Column(String)                  # which worker pod claimed it
    model = Column(String)                      # OpenAI model to run this job with
    script_only = Column(Boolean, default=False, nullable=False)  # generate inputs, skip CPU execution
    run_dir = Column(Text)                      # absolute path to the agent's run directory (on the PVC)
    result = Column(JSON)                       # extracted key values (material, energy, band gap, ...)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)


# Worker claim query filters status='queued' ordered by created_at; this index
# makes both the claim and the queue-position count fast.
Index("ix_jobs_status_created", Job.status, Job.created_at)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_migrations():
    """Idempotent column additions.

    Base.metadata.create_all() only CREATES missing tables — it never ALTERs an
    existing one. New columns added to a model after its table already exists
    must be applied explicitly here.
    """
    migrations = [
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS run_dir TEXT",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS result JSON",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS model TEXT",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS script_only BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'auto'",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS pending_step JSON",
        "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS step_action JSON",
        "ALTER TABLE usage_log ADD COLUMN IF NOT EXISTS model TEXT",
    ]
    with engine.begin() as conn:
        for stmt in migrations:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                print(f"[migrate] failed: {stmt} -> {e}")


def init_db():
    """Create tables, apply migrations, and seed admin users from ADMIN_EMAILS."""
    try:
        Base.metadata.create_all(engine)
        _run_migrations()
        admin_emails = [
            e.strip().lower()
            for e in os.environ.get("ADMIN_EMAILS", "").split(",")
            if e.strip()
        ]
        if not admin_emails:
            return None, None
        db = SessionLocal()
        try:
            for email in admin_emails:
                user = db.query(User).filter(User.email == email).first()
                if user is None:
                    db.add(User(email=email, credits=1000, is_admin=True, is_unlimited=True))
                else:
                    user.is_admin = True
                    user.is_unlimited = True
            db.commit()
        finally:
            db.close()
        return None, None
    except Exception as e:
        return None, str(e)
