from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRUCTURE_TERMS = {
    "structure", "structural", "lattice", "cell", "volume", "density", "symmetry",
    "space group", "crystal system", "bond", "distance", "angle", "coordination",
    "neighbor", "neighbour", "fractional coordinate", "cartesian coordinate",
    "alpha", "beta", "gamma",
}


@dataclass(frozen=True)
class StructuralToolCall:
    task: str
    species1: str = ""
    species2: str = ""
    species3: str = ""


def is_structural_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in STRUCTURE_TERMS)


def _parseable(paths: list[Path]) -> list[Path]:
    result = []
    for path in paths:
        try:
            _load_structure(path)
            result.append(path)
        except Exception:
            continue
    return result


def _executed_stage_inputs(run_dir: Path, stage: str) -> list[Path]:
    candidates = [path for path in sorted(run_dir.rglob(f"{stage}.in"))
                  if "attempts" in {part.lower() for part in path.parts}]
    return _parseable(candidates)


def _structure_path(run_dir: Path, question: str = "") -> Path:
    lowered = question.lower()
    # A stage-specific question must cite the concrete input submitted for that
    # stage, never a placeholder in approved_inputs or a different-stage file.
    for stage, terms in (("scf", ("scf", "self-consistent")),
                         ("nscf", ("nscf", "non-self-consistent")),
                         ("bands", ("bands calculation", "band structure calculation"))):
        if any(term in lowered for term in terms):
            candidates = _executed_stage_inputs(run_dir, stage)
            if candidates:
                return candidates[-1]
    if any(term in lowered for term in ("before relaxation", "initial structure", "starting structure", "unrelaxed")):
        candidates = _executed_stage_inputs(run_dir, "vc-relax")
        if candidates:
            return candidates[0]
    relaxed = _parseable(sorted(run_dir.rglob("relaxed_structure.in")))
    if relaxed:
        return relaxed[-1]
    # The concrete input to the first successful post-relaxation SCF contains
    # the materialized final geometry and is a reliable fallback.
    scf = _executed_stage_inputs(run_dir, "scf")
    if scf:
        return scf[-1]
    inputs = _parseable([path for path in sorted(run_dir.rglob("*.in")) if "approved_inputs" in path.parts])
    if inputs:
        return inputs[0]
    raise FileNotFoundError("No parseable structure file was found in this workflow.")


def _load_structure(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    from structure_paths import _parse_relaxed_structure
    if not re.search(r"(?mi)^\s*CELL_PARAMETERS", text):
        # Initial QE inputs frequently use ibrav/celldm rather than explicit
        # vectors. Reuse TritonDFT's existing normalization before comparison.
        from evaluate.relax_eval import _normalize_qe_for_pymatgen
        text = _normalize_qe_for_pymatgen(text)
    try:
        # Isolate the geometry cards so trailing K_POINTS or other numeric QE
        # cards cannot be mistaken for atoms by the lightweight parser.
        cell = re.search(
            r"(?mis)^\s*CELL_PARAMETERS\s*\(\s*angstrom\s*\)\s*\n"
            r"\s*[^\n]+\n\s*[^\n]+\n\s*[^\n]+",
            text,
        )
        positions = re.search(
            r"(?mis)^\s*ATOMIC_POSITIONS\s*\(\s*crystal\s*\)\s*\n"
            r"((?:\s*[A-Za-z][A-Za-z0-9_+-]*\s+[-+0-9.EeDd]+\s+[-+0-9.EeDd]+\s+[-+0-9.EeDd]+[^\n]*\n?)+)",
            text,
        )
        if cell and positions:
            geometry = cell.group(0) + "\nATOMIC_POSITIONS (crystal)\n" + positions.group(1)
            return _parse_relaxed_structure(geometry)
        return _parse_relaxed_structure(text)
    except ValueError:
        from pymatgen.io.pwscf import PWInput
        return PWInput.from_str(text).structure


def _source_lines(path: Path, species: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected: list[int] = []
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "cell_parameters" in lowered:
            selected.extend(range(index, min(index + 4, len(lines))))
        if "atomic_positions" in lowered:
            selected.append(index)
        first = line.split(maxsplit=1)[0] if line.split() else ""
        if first in species:
            selected.append(index)
    return [{"path": str(path), "line_number": index + 1, "line": lines[index]}
            for index in sorted(set(selected))]


def _species_symbols(question: str, available: set[str], *, unique: bool = True) -> list[str]:
    # Hyphens are separators here, so both "Mo-S" and "Mo S" work.
    tokens = re.findall(r"[A-Z][a-z]?", question.replace("–", "-").replace("—", "-"))
    found = []
    for token in tokens:
        if token in available and (not unique or token not in found):
            found.append(token)
    return found


def plan_structural_tool_call(question: str, structure) -> StructuralToolCall | None:
    lowered = question.lower()
    available = {site.specie.symbol for site in structure}
    species_sequence = _species_symbols(question, available, unique=False)
    species = list(dict.fromkeys(species_sequence))
    if "angle" in lowered and len(species_sequence) >= 3:
        return StructuralToolCall("species_triplet_angles", species_sequence[0], species_sequence[1], species_sequence[2])
    if ("bond" in lowered or "distance" in lowered or "neighbor" in lowered or "neighbour" in lowered) and len(species) >= 2:
        return StructuralToolCall("species_pair_bonds", species[0], species[1])
    if any(term in lowered for term in ("alpha", "beta", "gamma", "lattice", "cell", "volume", "density")):
        return StructuralToolCall("lattice")
    if ("symmetry" in lowered or "space group" in lowered) and any(term in lowered for term in ("change", "changed", "before and after", "after relaxation")):
        return StructuralToolCall("symmetry_comparison")
    if any(term in lowered for term in ("symmetry", "space group", "crystal system")):
        return StructuralToolCall("symmetry")
    if any(term in lowered for term in ("coordination", "neighbor", "neighbour")):
        return StructuralToolCall("coordination", species[0] if species else "")
    if "structure" in lowered or "structural" in lowered:
        return StructuralToolCall("summary")
    return None


def _lattice(structure) -> dict[str, Any]:
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    return {"a_angstrom": a, "b_angstrom": b, "c_angstrom": c,
            "alpha_degree": alpha, "beta_degree": beta, "gamma_degree": gamma,
            "volume_angstrom3": structure.volume, "density_g_cm3": structure.density}


def _symmetry(structure) -> dict[str, Any]:
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    analyzer = SpacegroupAnalyzer(structure, symprec=1e-3, angle_tolerance=5)
    return {"symbol": analyzer.get_space_group_symbol(), "number": analyzer.get_space_group_number(),
            "point_group": analyzer.get_point_group_symbol(), "crystal_system": analyzer.get_crystal_system()}


def _species_pair_bonds(structure, species1: str, species2: str) -> dict[str, Any]:
    indices1 = [i for i, site in enumerate(structure) if site.specie.symbol == species1]
    indices2 = [i for i, site in enumerate(structure) if site.specie.symbol == species2]
    if not indices1 or not indices2:
        raise ValueError(f"The structure does not contain both {species1} and {species2}.")
    pair_distances = []
    for i in indices1:
        for j in indices2:
            if species1 == species2 and i == j:
                continue
            distance, image = structure.lattice.get_distance_and_image(
                structure[i].frac_coords, structure[j].frac_coords
            )
            pair_distances.append((float(distance), i, j, tuple(int(value) for value in image)))
    if not pair_distances:
        raise ValueError("No distinct atom pairs were available for this measurement.")
    minimum = min(item[0] for item in pair_distances)
    cutoff = minimum * 1.25 + 1e-6
    bonds = []
    seen = set()
    for i in indices1:
        for neighbor in structure.get_neighbors(structure[i], cutoff):
            if neighbor.specie.symbol != species2:
                continue
            image = tuple(int(value) for value in neighbor.image)
            key = (i, int(neighbor.index), image)
            if species1 == species2:
                reverse = (int(neighbor.index), i, tuple(-value for value in image))
                if reverse in seen:
                    continue
            seen.add(key)
            bonds.append({"atom1_index": i, "atom2_index": int(neighbor.index),
                          "periodic_image": list(image), "distance_angstrom": float(neighbor.nn_distance)})
    bonds.sort(key=lambda item: item["distance_angstrom"])
    groups = []
    for bond in bonds:
        distance = bond["distance_angstrom"]
        group = next((item for item in groups if abs(item["distance_angstrom"] - distance) <= 0.01), None)
        if group is None:
            groups.append({"distance_angstrom": distance, "multiplicity": 1})
        else:
            group["multiplicity"] += 1
    return {"species_pair": [species1, species2], "shortest_distance_angstrom": minimum,
            "first_shell_cutoff_angstrom": cutoff, "distinct_first_shell_distances": groups,
            "first_shell_bonds": bonds}


def _coordination(structure, species: str = "") -> dict[str, Any]:
    from pymatgen.analysis.local_env import CrystalNN
    cnn = CrystalNN()
    sites = []
    for index, site in enumerate(structure):
        if species and site.specie.symbol != species:
            continue
        try:
            neighbors = cnn.get_nn_info(structure, index)
            sites.append({"site_index": index, "species": site.specie.symbol,
                          "coordination_number": len(neighbors),
                          "neighbors": [{"site_index": int(item["site_index"]),
                                         "species": item["site"].specie.symbol,
                                         "weight": float(item["weight"])} for item in neighbors]})
        except Exception as exc:
            sites.append({"site_index": index, "species": site.specie.symbol, "warning": str(exc)})
    return {"sites": sites, "method": "pymatgen CrystalNN"}


def _nearest_species_distance(structure, center_indices: list[int], target: str) -> float:
    distances = []
    for center in center_indices:
        for index, site in enumerate(structure):
            if site.specie.symbol != target or index == center:
                continue
            distances.append(structure.get_distance(center, index))
    if not distances:
        raise ValueError(f"No {target} neighbors are available for the requested angle.")
    return min(distances)


def _species_triplet_angles(structure, outer1: str, center_species: str, outer2: str) -> dict[str, Any]:
    """Measure outer1-center-outer2 angles using periodic first-shell neighbors."""
    import math
    import numpy as np

    centers = [i for i, site in enumerate(structure) if site.specie.symbol == center_species]
    if not centers:
        raise ValueError(f"The structure does not contain the central species {center_species}.")
    nearest1 = _nearest_species_distance(structure, centers, outer1)
    nearest2 = _nearest_species_distance(structure, centers, outer2)
    cutoff = max(nearest1, nearest2) * 1.25 + 1e-6
    measurements = []
    for center_index in centers:
        center = structure[center_index]
        neighbors = structure.get_neighbors(center, cutoff)
        left = [neighbor for neighbor in neighbors if neighbor.specie.symbol == outer1]
        right = [neighbor for neighbor in neighbors if neighbor.specie.symbol == outer2]
        for left_index, first in enumerate(left):
            for right_index, second in enumerate(right):
                first_key = (int(first.index), tuple(int(value) for value in first.image))
                second_key = (int(second.index), tuple(int(value) for value in second.image))
                if first_key == second_key:
                    continue
                if outer1 == outer2 and right_index <= left_index:
                    continue
                vector1 = np.asarray(first.coords) - np.asarray(center.coords)
                vector2 = np.asarray(second.coords) - np.asarray(center.coords)
                cosine = float(np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
                angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
                measurements.append({
                    "outer1_index": int(first.index), "center_index": center_index,
                    "outer2_index": int(second.index),
                    "outer1_periodic_image": list(first_key[1]), "outer2_periodic_image": list(second_key[1]),
                    "angle_degree": angle,
                })
    if not measurements:
        raise ValueError("No periodic first-shell atom triplets were found for the requested species.")
    measurements.sort(key=lambda item: item["angle_degree"])
    groups = []
    for measurement in measurements:
        angle = measurement["angle_degree"]
        group = next((item for item in groups if abs(item["angle_degree"] - angle) <= 0.1), None)
        if group is None:
            groups.append({"angle_degree": angle, "multiplicity": 1})
        else:
            group["multiplicity"] += 1
    return {"species_triplet": [outer1, center_species, outer2],
            "first_shell_cutoff_angstrom": cutoff, "distinct_angles": groups,
            "angle_measurements": measurements}


def call_structural_analysis_tool(run_dir: Path, question: str) -> dict[str, Any] | None:
    """Plan and execute a deterministic structural tool call for a natural-language question."""
    if not is_structural_question(question):
        return None
    run_dir = run_dir.resolve()
    path = _structure_path(run_dir, question)
    structure = _load_structure(path)
    call = plan_structural_tool_call(question, structure)
    if call is None:
        return None
    extra_evidence = []
    if call.task == "lattice":
        result = _lattice(structure)
    elif call.task == "symmetry":
        result = _symmetry(structure)
    elif call.task == "species_pair_bonds":
        result = _species_pair_bonds(structure, call.species1, call.species2)
    elif call.task == "species_triplet_angles":
        result = _species_triplet_angles(structure, call.species1, call.species2, call.species3)
    elif call.task == "coordination":
        result = _coordination(structure, call.species1)
    elif call.task == "symmetry_comparison":
        initial_candidates = _executed_stage_inputs(run_dir, "vc-relax")
        final_candidates = _executed_stage_inputs(run_dir, "scf")
        initial_path = initial_candidates[0] if initial_candidates else _structure_path(run_dir, "initial structure before relaxation")
        final_path = final_candidates[-1] if final_candidates else _structure_path(run_dir, "relaxed structure")
        initial_symmetry = _symmetry(_load_structure(initial_path))
        final_symmetry = _symmetry(_load_structure(final_path))
        changed = (initial_symmetry["number"], initial_symmetry["symbol"]) != (final_symmetry["number"], final_symmetry["symbol"])
        result = {"changed": changed, "initial": initial_symmetry, "final": final_symmetry,
                  "initial_structure_file": str(initial_path), "final_structure_file": str(final_path)}
        path = final_path
        extra_evidence = _source_lines(initial_path)
    else:
        result = {"formula": structure.composition.formula, "reduced_formula": structure.composition.reduced_formula,
                  "num_sites": len(structure), "lattice": _lattice(structure), "symmetry": _symmetry(structure)}
    species = tuple(value for value in (call.species1, call.species2, call.species3) if value)
    return {"tool": "structural_analysis", "call": call.__dict__, "structure_file": str(path),
            "result": result, "evidence": extra_evidence + _source_lines(path, species)}


def format_structural_tool_result(payload: dict[str, Any]) -> str:
    call, result = payload["call"], payload["result"]
    lines = ["Structural analysis tool", f"Structure: {payload['structure_file']}", f"Operation: {call['task']}", ""]
    if call["task"] == "species_pair_bonds":
        pair = "–".join(result["species_pair"])
        lines.append(f"Shortest {pair} distance: {result['shortest_distance_angstrom']:.6f} Å")
        lines.append("Symmetry/numerically distinct first-shell distances:")
        for group in result["distinct_first_shell_distances"]:
            lines.append(f"  {group['distance_angstrom']:.6f} Å  multiplicity={group['multiplicity']}")
    elif call["task"] == "species_triplet_angles":
        triplet = "–".join(result["species_triplet"])
        lines.append(f"Distinct first-shell {triplet} angles:")
        for group in result["distinct_angles"]:
            lines.append(f"  {group['angle_degree']:.6f}°  multiplicity={group['multiplicity']}")
        lines.append(f"Periodic triplets measured: {len(result['angle_measurements'])}")
    elif call["task"] == "lattice":
        lines.extend([f"a={result['a_angstrom']:.6f} Å, b={result['b_angstrom']:.6f} Å, c={result['c_angstrom']:.6f} Å",
                      f"α={result['alpha_degree']:.6f}°, β={result['beta_degree']:.6f}°, γ={result['gamma_degree']:.6f}°",
                      f"Volume={result['volume_angstrom3']:.6f} Å³; density={result['density_g_cm3']:.6f} g/cm³"])
    elif call["task"] == "symmetry_comparison":
        state = "changed" if result["changed"] else "did not change"
        lines.extend([
            f"Symmetry {state} after relaxation.",
            f"Initial: {result['initial']['symbol']} (No. {result['initial']['number']}), file: {result['initial_structure_file']}",
            f"Final/SCF: {result['final']['symbol']} (No. {result['final']['number']}), file: {result['final_structure_file']}",
        ])
    else:
        import json
        lines.append(json.dumps(result, indent=2))
    lines.append("\nVerified structural evidence")
    for item in payload["evidence"]:
        lines.append(f"\n{item['path']}:{item['line_number']}\n{item['line']}")
    return "\n".join(lines) + "\n"
