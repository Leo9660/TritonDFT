"""Shared slowapi limiter so both server.py and routers can apply rate limits
without a circular import."""
from fastapi import Request
from slowapi import Limiter


def get_real_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


limiter = Limiter(key_func=get_real_ip, default_limits=[])

# Enqueue is cheap; credits are the real spend limiter. Keep this lenient —
# just enough to stop someone hammering the endpoint.
PER_IP_RATE = "20/minute;200/day"
