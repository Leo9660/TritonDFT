import os
import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Integer, Boolean, DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://tritondft:tritondft@localhost:5432/tritondft",
)

Base = declarative_base()
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    credits = Column(Integer, default=1000, nullable=False)
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


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create tables and seed admin users from ADMIN_EMAILS env var."""
    try:
        Base.metadata.create_all(engine)
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
