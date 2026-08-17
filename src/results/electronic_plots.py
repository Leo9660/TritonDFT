from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .electronic_reference import electronic_reference


def _numeric_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) >= 2:
            rows.append(values)
    return rows


def _energy_reference(run_dir: Path) -> float:
    reference = None
    for path in sorted(run_dir.rglob("*.out")):
        if "probe" in path.stem:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"the Fermi energy is\s+([-+0-9.Ee]+)\s*ev", text, re.I)
        if matches:
            reference = float(matches[-1])
            continue
        matches = re.findall(r"highest occupied(?: level)?\s*\(ev\):\s*([-+0-9.Ee]+)", text, re.I)
        if matches:
            reference = float(matches[-1])
            continue
        matches = re.findall(
            r"highest occupied, lowest unoccupied level\s*\(ev\):\s*([-+0-9.Ee]+)",
            text,
            re.I,
        )
        if matches:
            reference = float(matches[-1])
    return reference if reference is not None else 0.0


def _plot_bands(run_dir: Path, plt) -> str:
    candidates = sorted(run_dir.rglob("*.band.gnu")) + sorted(run_dir.rglob("*.bands.dat.gnu"))
    if not candidates:
        return ""
    blocks: List[List[Tuple[float, float]]] = []
    block: List[Tuple[float, float]] = []
    for raw in candidates[0].read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw.strip():
            if block:
                blocks.append(block)
                block = []
            continue
        try:
            x, energy = (float(v) for v in raw.split()[:2])
        except (ValueError, IndexError):
            continue
        block.append((x, energy))
    if block:
        blocks.append(block)
    if not blocks:
        return ""
    reference = electronic_reference(run_dir, "vbm")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for band in blocks:
        ax.plot([p[0] for p in band], [p[1] - reference for p in band], color="black", linewidth=0.9)
    ax.axhline(0.0, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("k-path distance")
    ax.set_ylabel("Energy - VBM (eV)")
    ax.set_title("Electronic band structure")
    fig.tight_layout()
    output = run_dir / "band_structure.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def _plot_total_dos(run_dir: Path, plt) -> str:
    candidates = [
        path for path in sorted(run_dir.rglob("*.dos"))
        if "pdos_" not in path.name.lower()
    ]
    if not candidates:
        return ""
    rows = _numeric_rows(candidates[0])
    if not rows:
        return ""
    ncol = min(len(row) for row in rows)
    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    reference = electronic_reference(run_dir, "vbm")
    energy = [row[0] - reference for row in rows]
    if ncol >= 4:  # spin-polarized dos.x: E, DOSup, DOSdown, IntDOS
        ax.plot(energy, [row[1] for row in rows], label="spin up")
        ax.plot(energy, [-row[2] for row in rows], label="spin down")
        ax.legend()
    else:
        ax.plot(energy, [row[1] for row in rows], color="black", label="total DOS")
    ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Energy - VBM (eV)")
    ax.set_ylabel("DOS (states/eV)")
    ax.set_title("Total density of states")
    fig.tight_layout()
    output = run_dir / "total_dos.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def _pdos_group(path: Path) -> tuple[str, str]:
    # Standard QE name: prefix.pdos_atm#1(Mo)_wfc#3(d)
    match = re.search(r"atm#\d+\(([^)]+)\).*wfc#\d+\(([spdf])", path.name, re.I)
    if not match:
        return "", ""
    species = re.sub(r"_(?:up|down)$", "", match.group(1), flags=re.I)
    return species, match.group(2).lower()


def _plot_pdos(run_dir: Path, plt) -> str:
    grouped: Dict[tuple[str, str], List[List[List[float]]]] = defaultdict(list)
    for path in sorted(run_dir.rglob("*pdos_atm*")):
        species, orbital = _pdos_group(path)
        rows = _numeric_rows(path)
        if species and orbital and rows:
            grouped[(species, orbital)].append(rows)
    if not grouped:
        return ""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    plotted = 0
    for (species, orbital), datasets in sorted(grouped.items()):
        length = min(len(rows) for rows in datasets)
        if not length:
            continue
        reference = electronic_reference(run_dir, "vbm")
        energy = [datasets[0][i][0] - reference for i in range(length)]
        ncol = min(len(row) for rows in datasets for row in rows[:length])
        up = [sum(rows[i][1] for rows in datasets) for i in range(length)]
        ax.plot(energy, up, label=f"{species}-{orbital}")
        if ncol >= 3:
            down = [sum(rows[i][2] for rows in datasets) for i in range(length)]
            ax.plot(energy, [-value for value in down], linestyle="--", label=f"{species}-{orbital} down")
        plotted += 1
    if not plotted:
        plt.close(fig)
        return ""
    ax.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Projected DOS (states/eV)")
    ax.set_title("Orbital-projected density of states")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output = run_dir / "projected_dos.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return str(output)


def generate_electronic_plots(run_dir: str | Path, query: str = "") -> List[str]:
    """Render available electronic data; never use the LLM to alter data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    directory = Path(run_dir)
    outputs: List[str] = []
    for renderer in (_plot_bands, _plot_total_dos, _plot_pdos):
        try:
            output = renderer(directory, plt)
        except Exception:
            output = ""
        if output:
            outputs.append(output)
    return outputs
