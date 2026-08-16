"""Per-model credit accounting.

A credit is a unit of prepaid usage. 1 credit ≈ ``USD_PER_CREDIT`` of real
OpenAI spend, so the cost of a job scales with BOTH the model's price and the
number of tokens it actually consumed:

    credits = ceil( (in_tok·price_in + out_tok·price_out) / 1e6 / USD_PER_CREDIT )

Prices below are USD per 1,000,000 tokens (OpenAI list prices). Update them
from https://openai.com/api/pricing/ when they change — they're the single
source of truth for billing.
"""
import math
import tiktoken
from sqlalchemy.orm import Session

from db import User, UsageLog

# USD of model spend that one credit buys. 100 starting credits ≈ $3.
USD_PER_CREDIT = 0.03

# model id -> (input_usd_per_1M, output_usd_per_1M)
# Claude models run through the Claude Code CLI, which reports exact USD cost per
# call — for those, billing uses that real cost (see reconcile). The prices here
# are only a pre-charge reservation estimate.
MODEL_PRICES = {
    "gpt-5.2":      (1.25, 10.00),
    "gpt-5.1":      (1.25, 10.00),
    "gpt-5":        (1.25, 10.00),
    "gpt-5-mini":   (0.25, 2.00),
    "gpt-5-nano":   (0.05, 0.40),
    "gpt-4o":       (2.50, 10.00),
    "gpt-4o-mini":  (0.15, 0.60),
    "claude-opus-4-8":   (15.00, 75.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00, 5.00),
}

# What the frontend dropdown may request. Anything else falls back to DEFAULT.
ALLOWED_MODELS = list(MODEL_PRICES.keys())
DEFAULT_MODEL = "gpt-4o"
# Unknown models are priced conservatively (the most expensive flagship input
# + output) so we never undercharge for something we don't recognize.
_FALLBACK_PRICE = (2.50, 10.00)

# Encoding init must NEVER crash module import: tiktoken downloads the BPE file
# from openaipublic.blob.core.windows.net on first use, and workers may have no
# egress / flaky DNS to that host. The image pre-caches both encodings (see
# TIKTOKEN_CACHE_DIR in the Dockerfile); if even that is missing we degrade to
# the len//4 heuristic in count_tokens() rather than killing the worker.
try:
    _ENC = tiktoken.encoding_for_model("gpt-4o")
except Exception:
    try:
        _ENC = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _ENC = None


def resolve_model(model):
    """Return a validated model id (falls back to DEFAULT_MODEL)."""
    if model and isinstance(model, str) and model in MODEL_PRICES:
        return model
    return DEFAULT_MODEL


def count_tokens(text: str):
    if not text:
        return 0, None
    try:
        return len(_ENC.encode(text)), None
    except Exception as e:
        return max(1, len(text) // 4), str(e)


def _price_for(model: str):
    return MODEL_PRICES.get(model, _FALLBACK_PRICE)


def estimate_cost(model: str, input_tokens: int, output_tokens: int):
    """Credits for a given model + token counts. Always at least 1."""
    p_in, p_out = _price_for(model)
    usd = (input_tokens * p_in + output_tokens * p_out) / 1_000_000.0
    return credits_from_usd(usd), None


def credits_from_usd(usd: float) -> int:
    """Convert real USD spend to credits (always at least 1)."""
    return max(1, math.ceil((usd or 0.0) / USD_PER_CREDIT))


def pre_charge(db: Session, user: User, model: str, input_tokens: int,
               max_output_tokens: int, endpoint: str):
    """Reserve worst-case credits up front; reconcile() settles to actual usage.

    Returns (UsageLog, None) on success, or (None, {"needed": int, "remaining": int})
    when the user lacks credits — caller maps this to errors.insufficient_credits.

    The deduction is only FLUSHED (not committed) so the caller can commit it in
    the SAME transaction as the Job it creates — if job creation fails, the
    charge rolls back (no orphan deduction). The user row is locked FOR UPDATE so
    concurrent jobs from the same user can't race to a lost update.
    """
    cost, _ = estimate_cost(model, input_tokens, max_output_tokens)
    if not user.is_unlimited:
        # Lock the row and re-read the authoritative balance.
        db.refresh(user, with_for_update=True)
        if user.credits < cost:
            return None, {"needed": cost, "remaining": user.credits}
        user.credits -= cost
    log = UsageLog(
        user_id=user.id,
        endpoint=endpoint,
        model=model,
        input_tokens=input_tokens,
        output_tokens=max_output_tokens,
        credits_deducted=0 if user.is_unlimited else cost,
    )
    db.add(log)
    db.flush()   # assign/persist within the transaction; caller commits
    return log, None


def reconcile(db: Session, log_id, user_id, model: str,
              actual_input_tokens=None, actual_output_tokens: int = 0,
              cost_usd=None):
    """Settle the pre-charge to the REAL usage.

    If ``cost_usd`` is given (Claude Code CLI reports exact cost), bill on that;
    otherwise bill on model price × tokens.

    Unlike a refund-only model, this charges the difference if the job used more
    than the reservation (clamped so a balance never goes negative) and refunds
    if it used less. Concurrency-safe: the user row is locked FOR UPDATE first,
    so concurrent reconciles (and a cancel racing a worker finalize) serialize
    and settle to the same credits_deducted rather than losing an update.
    """
    try:
        # Lock the user row FIRST (populate_existing forces a fresh read even if
        # the object is already in this session), THEN read the now-stable log.
        user = (
            db.query(User)
            .filter(User.id == user_id)
            .with_for_update()
            .populate_existing()
            .first()
        )
        log = db.query(UsageLog).filter(UsageLog.id == log_id).first()
        if log is None or user is None:
            return 0, "log_or_user_not_found"

        in_tok = actual_input_tokens if actual_input_tokens is not None else (log.input_tokens or 0)
        model = model or log.model or DEFAULT_MODEL

        if user.is_unlimited:
            log.model = model
            log.input_tokens = in_tok
            log.output_tokens = actual_output_tokens
            db.commit()
            return 0, None

        if cost_usd is not None and cost_usd > 0:
            actual_cost = credits_from_usd(cost_usd)
        else:
            actual_cost, _ = estimate_cost(model, in_tok, actual_output_tokens)
        delta = actual_cost - (log.credits_deducted or 0)
        refund = 0
        if delta > 0:
            # Charge the overrun, but never push the balance below zero.
            user.credits -= min(delta, user.credits)
        elif delta < 0:
            refund = -delta
            user.credits += refund
        log.credits_deducted = actual_cost
        log.model = model
        log.input_tokens = in_tok
        log.output_tokens = actual_output_tokens
        db.commit()
        return refund, None
    except Exception as e:
        return 0, str(e)
