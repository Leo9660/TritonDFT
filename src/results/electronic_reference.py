from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List


def electronic_references(run_dir: str | Path) -> Dict[str, float]:
    """Extract branch-local electronic references, preferring the final SOC branch."""
    root = Path(run_dir)
    records: List[tuple[int, Dict[str, float]]] = []
    for path in root.rglob("*.out"):
        if "probe" in path.stem.lower():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        values: Dict[str, float] = {}
        fermi = re.findall(r"the Fermi energy is\s+([-+0-9.Ee]+)\s*ev", text, re.I)
        pairs = re.findall(
            r"highest occupied, lowest unoccupied level\s*\(ev\):\s*"
            r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", text, re.I,
        )
        occupied = re.findall(r"highest occupied(?: level)?\s*\(ev\):\s*([-+0-9.Ee]+)", text, re.I)
        if fermi:
            values["fermi"] = float(fermi[-1])
        if pairs:
            values["vbm"] = float(pairs[-1][0])
            values["cbm"] = float(pairs[-1][1])
            values["midgap"] = (values["vbm"] + values["cbm"]) / 2.0
        elif occupied:
            values["vbm"] = float(occupied[-1])
        if not values:
            continue
        name = path.name.lower()
        path_text = str(path).lower()
        priority = 10 if "soc_refinement" in path_text else 0
        priority += 4 if "nscf" in name else 3 if "scf" in name else 1
        records.append((priority, values))
    result: Dict[str, float] = {"absolute": 0.0}
    for _priority, values in sorted(records, key=lambda item: item[0]):
        result.update(values)
    if "vbm" not in result and "fermi" in result:
        result["vbm"] = result["fermi"]
    if "fermi" not in result and "midgap" in result:
        result["fermi"] = result["midgap"]
    return result


def electronic_reference(run_dir: str | Path, mode: str = "vbm") -> float:
    refs = electronic_references(run_dir)
    key = mode.strip().lower()
    return refs.get(key, refs.get("vbm", refs.get("fermi", 0.0)))
