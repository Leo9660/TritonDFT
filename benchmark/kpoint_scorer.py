"""Score an agent-generated QE k-point mesh against hidden expert references.

This module is intentionally outside the prompt path. It should be used after
the agent has already generated input files, so benchmark runs can preserve the
raw model guess while still measuring under/over-estimation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "benchmark" / "kpoint_reference.json"


def _normalise_material(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_reference(path: Path = DEFAULT_REFERENCE) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_material_entry(reference: dict[str, Any], material: str) -> tuple[str, dict[str, Any]]:
    wanted = _normalise_material(material)
    for name, entry in reference["materials"].items():
        candidates = [name, *entry.get("aliases", [])]
        if any(_normalise_material(candidate) == wanted for candidate in candidates):
            return name, entry
    raise KeyError(f"No k-point reference found for material {material!r}")


def infer_material(input_path: Path, run_meta: dict[str, Any] | None) -> str:
    if run_meta and run_meta.get("material_name"):
        return str(run_meta["material_name"])

    text = input_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("&"):
            continue
        species_match = re.match(r"^([A-Z][a-z]?)\s+[-+0-9.]+\s+\S+\.upf\b", stripped, re.I)
        if species_match:
            return species_match.group(1)

    raise ValueError("Could not infer material; pass --material explicitly")


def infer_accuracy_tier(path: Path, query: str | None) -> str:
    haystack = f"{path} {query or ''}".lower()
    if "1 mev" in haystack or "1mev" in haystack or "highacc" in haystack or "high_acc" in haystack:
        return "high"
    if "10 mev" in haystack or "10mev" in haystack or "mediumacc" in haystack or "medium_acc" in haystack:
        return "medium"
    if "20 mev" in haystack or "20mev" in haystack or "lowacc" in haystack or "low_acc" in haystack:
        return "low"
    raise ValueError("Could not infer accuracy tier; pass --tier high|medium|low explicitly")


def load_run_meta(input_path: Path) -> dict[str, Any] | None:
    meta_path = input_path.with_name("run_meta.json")
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_automatic_kpoints(input_path: Path) -> list[int]:
    lines = input_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if line.strip().lower() == "k_points automatic":
            if index + 1 >= len(lines):
                raise ValueError("K_POINTS automatic found without mesh line")
            values = [int(part) for part in lines[index + 1].split()[:6]]
            if len(values) != 6:
                raise ValueError("K_POINTS automatic mesh line must contain six integers")
            return values[:3]
    raise ValueError("Only K_POINTS automatic inputs are supported by this scorer")


def parse_ecutwfc(input_path: Path) -> float:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\becutwfc\s*=\s*([-+]?\d*\.?\d+)", text, re.I)
    if not match:
        raise ValueError("ecutwfc not found in QE input")
    return float(match.group(1))


def compare_mesh_and_ecut(
    agent_mesh: list[int],
    reference_mesh: list[int],
    agent_ecut: float | None,
    reference_ecut: float | None,
) -> dict[str, Any]:
    mesh_delta = [agent_i - ref_i for agent_i, ref_i in zip(agent_mesh, reference_mesh)]
    ecut_delta = None if agent_ecut is None or reference_ecut is None else agent_ecut - reference_ecut

    if all(value == 0 for value in mesh_delta) and (ecut_delta is None or ecut_delta == 0):
        status = "match"
    elif any(value < 0 for value in mesh_delta) or (ecut_delta is not None and ecut_delta < 0):
        status = "underestimate"
    else:
        status = "overestimate"

    return {
        "status": status,
        "delta": mesh_delta,
        "ecut_delta": ecut_delta,
        "agent_kmesh": agent_mesh,
        "reference_kmesh": reference_mesh,
        "agent_ecutwfc_ry": agent_ecut,
        "reference_ecutwfc_ry": reference_ecut,
    }


def score_input(
    input_path: Path,
    material: str | None = None,
    tier: str | None = None,
    reference_path: Path = DEFAULT_REFERENCE,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    run_meta = load_run_meta(input_path)
    reference = load_reference(reference_path)

    material_name = material or infer_material(input_path, run_meta)
    tier_name = tier or infer_accuracy_tier(input_path, run_meta.get("query") if run_meta else None)
    if tier_name not in {"high", "medium", "low"}:
        raise ValueError("--tier must be one of: high, medium, low")

    reference_key, material_entry = find_material_entry(reference, material_name)
    agent_kmesh = parse_automatic_kpoints(input_path)
    agent_ecut = parse_ecutwfc(input_path)
    reference_kmesh = material_entry[tier_name]
    reference_ecut = material_entry.get(f"{tier_name}_ecutwfc_ry")
    comparison = compare_mesh_and_ecut(agent_kmesh, reference_kmesh, agent_ecut, reference_ecut)

    return {
        "input_file": str(input_path),
        "material": material_name,
        "reference_material_key": reference_key,
        "accuracy_tier": tier_name,
        "accuracy_target": reference["accuracy_tiers"][tier_name],
        **comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score QE K_POINTS against hidden benchmark references.")
    parser.add_argument("input_file", type=Path, help="Path to a generated QE input file")
    parser.add_argument("--material", help="Material name; inferred from run_meta.json or ATOMIC_SPECIES if omitted")
    parser.add_argument("--tier", choices=["high", "medium", "low"], help="Accuracy tier; inferred from path/query if omitted")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Reference JSON path")
    args = parser.parse_args()

    result = score_input(args.input_file, args.material, args.tier, args.reference)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
