import tiktoken
from sqlalchemy.orm import Session

from db import User, UsageLog

# 1 credit = 1K input tokens, 1 credit = 0.33K output tokens (output is 3x input cost)
CREDITS_PER_1K_INPUT = 1
CREDITS_PER_1K_OUTPUT = 3

try:
    _ENC = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    _ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str):
    if not text:
        return 0, None
    try:
        return len(_ENC.encode(text)), None
    except Exception as e:
        return max(1, len(text) // 4), str(e)


def estimate_cost(input_tokens: int, output_tokens: int):
    in_credits = (input_tokens * CREDITS_PER_1K_INPUT + 999) // 1000
    out_credits = (output_tokens * CREDITS_PER_1K_OUTPUT + 999) // 1000
    return max(1, in_credits + out_credits), None


def pre_charge(db: Session, user: User, input_tokens: int, max_output_tokens: int, endpoint: str):
    """Pre-deduct worst-case credits. Returns (UsageLog, None) or (None, err)."""
    cost, _ = estimate_cost(input_tokens, max_output_tokens)
    if not user.is_unlimited and user.credits < cost:
        return None, f"insufficient_credits: need {cost}, have {user.credits}"
    if not user.is_unlimited:
        user.credits -= cost
    log = UsageLog(
        user_id=user.id,
        endpoint=endpoint,
        input_tokens=input_tokens,
        output_tokens=max_output_tokens,
        credits_deducted=0 if user.is_unlimited else cost,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log, None


def reconcile(db: Session, log_id, user_id, actual_output_tokens: int):
    """Refund unused budget after streaming completes. Uses fresh session-safe ids."""
    try:
        log = db.query(UsageLog).filter(UsageLog.id == log_id).first()
        user = db.query(User).filter(User.id == user_id).first()
        if log is None or user is None:
            return 0, "log_or_user_not_found"
        if user.is_unlimited:
            log.output_tokens = actual_output_tokens
            db.commit()
            return 0, None
        actual_cost, _ = estimate_cost(log.input_tokens, actual_output_tokens)
        refund = max(0, log.credits_deducted - actual_cost)
        if refund > 0:
            user.credits += refund
            log.credits_deducted = actual_cost
        log.output_tokens = actual_output_tokens
        db.commit()
        return refund, None
    except Exception as e:
        return 0, str(e)
