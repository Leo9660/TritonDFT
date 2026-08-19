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
# Keyed on the client IP, which at a workshop is ONE NAT address for the whole
# room — 20/minute was a budget for every attendee combined, not per person.
# Credits remain the real spend limiter; this only exists to blunt a script.
PER_IP_RATE = "180/minute;5000/day"
