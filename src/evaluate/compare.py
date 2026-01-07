from __future__ import annotations

from typing import Dict, Optional, Tuple

# Static comparison rules: key -> "relative" (numeric relative error)
#                        or "exact" (string / integer exact match).
COMPARISON_RULES: Dict[str, str] = {
    # Lattice lengths (Å): allow numerical tolerance
    "a": "relative",
    "b": "relative",
    "c": "relative",

    # Lattice angles (degree): allow numerical tolerance
    "alpha": "relative",
    "beta": "relative",
    "gamma": "relative",

    # Symmetry / group theory descriptors: must match exactly
    "space_group": "exact",          # e.g. "Fd-3m"
    "space_group_number": "exact",   # e.g. 227
    "point_group": "exact",          # e.g. "m-3m"
    "crystal_system": "exact",       # e.g. "cubic"
}


def compare_evaluation(
    ground_truth: Dict[str, object],
    evaluation: Dict[str, object],
    rules: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[float], bool]:
    """
    Compare evaluation dict against ground_truth.
    - Only iterates over keys in evaluation.
    - For "relative": compare numeric relative error.
    - For "exact": compare string equality.
    Returns: (max_relative_error, all_exact_match)
    """
    if not evaluation:
        return None, True
    rules = rules or COMPARISON_RULES
    max_rel_error: Optional[float] = None
    all_exact_match = True
    eps = 1e-12

    for key, eval_value in evaluation.items():
        compare_type = rules.get(key, "exact")
        gt_value = ground_truth.get(key)
        if compare_type == "relative":
            try:
                eval_num = float(eval_value)
                gt_num = float(gt_value)
            except (TypeError, ValueError):
                rel_error = float("inf")
            else:
                denom = max(abs(gt_num), eps)
                rel_error = abs(eval_num - gt_num) / denom
            if max_rel_error is None or rel_error > max_rel_error:
                max_rel_error = rel_error
        else:
            if str(eval_value) != str(gt_value):
                all_exact_match = False

    return max_rel_error, all_exact_match
