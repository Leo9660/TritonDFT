from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple


def _parse_relaxed_structure(block: str):
    from pymatgen.core import Lattice, Structure

    cell_match = re.search(
        r"(?mis)^\s*CELL_PARAMETERS\s*\(\s*angstrom\s*\)\s*\n"
        r"\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*\n"
        r"\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*\n"
        r"\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
        block,
    )
    pos_match = re.search(
        r"(?mis)^\s*ATOMIC_POSITIONS\s*\(\s*crystal\s*\)\s*\n(.*)$",
        block,
    )
    if not cell_match or not pos_match:
        raise ValueError("Relaxed structure must contain angstrom cell vectors and crystal positions.")
    values = [float(value) for value in cell_match.groups()]
    lattice = Lattice([values[0:3], values[3:6], values[6:9]])
    species: List[str] = []
    coords: List[List[float]] = []
    for raw in pos_match.group(1).splitlines():
        tokens = raw.split()
        if len(tokens) < 4:
            continue
        try:
            xyz = [float(value) for value in tokens[1:4]]
        except ValueError:
            continue
        # Magnetic aliases such as Fe_up still need the chemical element for
        # symmetry analysis. The original aliases remain in the QE input.
        symbol_match = re.match(r"[A-Z][a-z]?", tokens[0])
        if not symbol_match:
            raise ValueError(f"Cannot infer element from species label {tokens[0]!r}.")
        species.append(symbol_match.group(0))
        coords.append(xyz)
    if not species:
        raise ValueError("No relaxed atomic positions were found.")
    return Structure(lattice, species, coords, coords_are_cartesian=False)


def _path_nodes(structure) -> List[Tuple[str, List[float]]]:
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    primitive = SpacegroupAnalyzer(structure, symprec=1e-3).find_primitive()
    if primitive is None or len(primitive) != len(structure) or abs(primitive.volume / structure.volume - 1.0) > 1e-3:
        raise ValueError(
            "The calculation cell is not the symmetry primitive cell; automatic path coordinates "
            "would require an explicit reciprocal-basis transformation."
        )
    kpath = HighSymmKpath(structure)
    points = kpath.kpath["kpoints"]
    segments = kpath.kpath["path"]
    nodes: List[Tuple[str, List[float]]] = []
    for segment in segments:
        for label in segment:
            if not nodes or nodes[-1][0] != label:
                nodes.append((label, list(points[label])))
    if len(nodes) < 2:
        raise ValueError("Symmetry analysis did not produce a usable high-symmetry path.")
    return nodes


def materialize_relaxed_band_path(
    input_path: str,
    relaxed_structure: str,
    *,
    default_points_per_segment: int = 40,
) -> List[str]:
    """Replace a model-proposed path with one derived from the relaxed cell."""
    path = Path(input_path)
    text = path.read_text(encoding="utf-8")
    header = re.search(r"(?mi)^\s*K_POINTS\s+crystal_b\s*$", text)
    if not header:
        return []
    existing = text[header.end():].strip().splitlines()
    points_per_segment = default_points_per_segment
    if len(existing) >= 2:
        tokens = existing[1].split()
        if len(tokens) == 4:
            try:
                points_per_segment = max(1, int(tokens[3]))
            except ValueError:
                pass
    structure = _parse_relaxed_structure(relaxed_structure)
    nodes = _path_nodes(structure)
    rows = [f"K_POINTS crystal_b", str(len(nodes))]
    labels: List[str] = []
    for index, (label, coords) in enumerate(nodes):
        count = 1 if index == len(nodes) - 1 else points_per_segment
        rows.append(f"{coords[0]:.10f} {coords[1]:.10f} {coords[2]:.10f} {count}")
        labels.append(label)
    path.write_text(text[:header.start()].rstrip() + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return labels
