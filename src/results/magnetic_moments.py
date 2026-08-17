from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


def extract_magnetic_moments(run_dir: str | Path) -> Dict[str, Any]:
    """Extract explicitly reported QE moments with their provenance."""
    directory = Path(run_dir)
    total = None
    absolute = None
    sites: Dict[int, Dict[str, Any]] = {}
    sources: List[str] = []
    for path in sorted(directory.rglob("*.out")):
        if "probe" in path.stem:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        totals = re.findall(r"total magnetization\s*=\s*([-+0-9.Ee]+)\s+Bohr mag/cell", text, re.I)
        absolutes = re.findall(r"absolute magnetization\s*=\s*([-+0-9.Ee]+)\s+Bohr mag/cell", text, re.I)
        if totals:
            total = float(totals[-1])
            sources.append(str(path.relative_to(directory)))
        if absolutes:
            absolute = float(absolutes[-1])
            sources.append(str(path.relative_to(directory)))
        for match in re.finditer(
            r"atom\s*[:#]?\s*(\d+)(?:\s*\(([^)]+)\))?.{0,100}?magn(?:etization)?\s*[:=]\s*([-+0-9.Ee]+)",
            text,
            re.I,
        ):
            sites[int(match.group(1))] = {
                "atom_index": int(match.group(1)),
                "species": (match.group(2) or "").strip(),
                "moment_muB": float(match.group(3)),
                "method": "QE reported site/Lowdin projection",
                "source": str(path.relative_to(directory)),
            }
    result: Dict[str, Any] = {
        "method_note": "Local moments are projection/partition dependent; values are reported using QE's own output convention.",
        "site_moments": [sites[key] for key in sorted(sites)],
        "sources": sorted(set(sources)),
    }
    if total is not None:
        result["total_magnetization_muB_cell"] = total
    if absolute is not None:
        result["absolute_magnetization_muB_cell"] = absolute
    if total is None and absolute is None and not sites:
        return {}
    output = directory / "magnetic_moments.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["path"] = str(output)
    return result
