from __future__ import annotations
from typing import Dict, Iterable, Any
import math

def mae(x: float, y: float) -> float:
    return abs(float(x) - float(y))

def relative_err(x: float, y: float, eps: float = 1e-12) -> float:
    denom = max(abs(float(y)), eps)
    return abs(float(x) - float(y)) / denom

def aggregate_metrics(per_item: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple average aggregation for common scalar keys if they appear."""
    keys = {}
    count = 0
    for r in per_item:
        m = r.get("metrics", {})
        for k, v in m.items():
            if isinstance(v, (int, float)) and math.isfinite(v):
                keys.setdefault(k, 0.0)
                keys[k] += float(v)
        count += 1
    if count == 0:
        return {}
    return {k: v / count for k, v in keys.items()}