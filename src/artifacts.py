"""Helpers for exposing a DFT run's artifacts: file listing, key-result
extraction, band-structure parsing, and zip packaging.

Run directories live on the shared RWX PVC under /workspace/tmp. The worker
records the absolute path in Job.run_dir; the API serves files from it.
"""
import io
import os
import re
import json
import zipfile
from pathlib import Path

# Q1.A: ship only the useful files — exclude wavefunction / charge-density
# intermediates (*.wfc*, *.save/, *.mix*, etc.).
USEFUL_EXTS = {
    ".in", ".out", ".xml", ".json", ".band", ".gnu",
    ".dat", ".txt", ".log", ".cif", ".csv",
}
# Internal bookkeeping files surfaced elsewhere (result cards / Analysis section)
# — hide them from the user-facing file list and zip.
HIDDEN_FILES = {"analysis.json", "run_meta.json"}
# Everything we whitelist is plain text → safe to preview inline.
TEXT_EXTS = USEFUL_EXTS

RY_TO_EV = 13.605693

# Run directories must stay under this root (guards against a bad/old run_dir).
ARTIFACT_ROOT = "/workspace/tmp"


def safe_run_dir(run_dir: str):
    """Return a validated Path for run_dir, or None if missing/outside the root."""
    if not run_dir:
        return None
    try:
        p = Path(run_dir).resolve()
    except Exception:
        return None
    # Must be the root itself or strictly inside it (trailing sep guards
    # against a sibling like /workspace/tmp-evil).
    if str(p) != ARTIFACT_ROOT and not str(p).startswith(ARTIFACT_ROOT + os.sep):
        return None
    if not p.is_dir():
        return None
    return p


def is_safe_filename(name: str) -> bool:
    """Reject path traversal — only a bare filename is allowed."""
    return (
        bool(name)
        and name not in (".", "..")
        and name == os.path.basename(name)
        and "/" not in name
        and "\\" not in name
    )


def list_files(run_dir: Path):
    """List useful (whitelisted) files directly inside run_dir."""
    out = []
    try:
        for f in sorted(run_dir.iterdir()):
            # Skip symlinks: a job could plant `x.out -> ../<other-job>/...` or
            # -> /etc/passwd and read it through its own /files endpoint.
            if f.is_symlink() or not f.is_file():
                continue
            if f.name in HIDDEN_FILES:
                continue
            if f.suffix.lower() not in USEFUL_EXTS:
                continue
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            out.append({
                "name": f.name,
                "size": size,
                "ext": f.suffix.lower(),
                "text": f.suffix.lower() in TEXT_EXTS,
            })
    except Exception:
        pass
    return out


def extract_result(run_dir: Path) -> dict:
    """Best-effort extraction of headline numbers. Never raises — returns
    whatever it could find."""
    result = {}
    try:
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(errors="ignore"))
            if meta.get("material_name"):
                result["material"] = meta["material_name"]
            if meta.get("task_type"):
                result["task_type"] = meta["task_type"]
    except Exception:
        pass

    # Natural-language conclusion the agent wrote — the answer to the user's
    # question (distinct from the raw streamed log).
    try:
        analysis_path = run_dir / "analysis.json"
        if analysis_path.exists():
            adata = json.loads(analysis_path.read_text(errors="ignore"))
            atext = (adata.get("analysis") or "").strip()
            if atext:
                result["analysis"] = atext
    except Exception:
        pass

    energies = []
    # gap candidates keyed by the calculation that produced them, so we can pick
    # the trustworthy one instead of "whichever file sorted last".
    gaps = {}
    try:
        for f in _sorted_outputs(run_dir):
            text = f.read_text(errors="ignore")
            for m in re.finditer(r"total energy\s*=\s*(-?\d+\.\d+)\s*Ry", text):
                energies.append(float(m.group(1)))
            gm = re.search(
                r"highest occupied, lowest unoccupied level \(ev\):\s*"
                r"(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                text,
            )
            if gm:
                occ, unocc = float(gm.group(1)), float(gm.group(2))
                g = round(unocc - occ, 4)
                if g >= 0:
                    gaps[_calculation_of(f)] = g
    except Exception:
        pass

    # A 'bands' run samples only the high-symmetry path, so its highest-occupied /
    # lowest-unoccupied pair is the extremum over THAT PATH — it overestimates the
    # gap whenever the true band edge lies off the path (for silicon the CBM sits
    # at ~0.85 along Gamma-X). The uniform-grid nscf is the authoritative source;
    # scf is the fallback. This used to be "last output file wins", which meant a
    # trailing bands step silently overwrote the nscf value.
    gap = next((gaps[k] for k in ("nscf", "scf", "relax", "vc-relax") if k in gaps), None)

    if energies:
        result["final_energy_ry"] = round(energies[-1], 6)
        result["final_energy_ev"] = round(energies[-1] * RY_TO_EV, 4)
    if gap is not None:
        result["band_gap_ev"] = gap
    return result


def _sorted_outputs(run_dir: Path):
    """Run outputs in step order.

    Names are output_<step>_<n>.out, so a plain lexicographic sort puts step 10
    before step 2. Sort on the numeric fields instead.
    """
    def key(p: Path):
        nums = [int(x) for x in re.findall(r"\d+", p.stem)]
        return (nums, p.name)
    return sorted(run_dir.glob("output_*.out"), key=key)


def _calculation_of(output_path: Path) -> str:
    """The pw.x `calculation` that produced an output, read from its sibling
    input file (output_2_1.out -> input_2_1.in). Empty string if unknown."""
    in_path = output_path.parent / (output_path.stem.replace("output_", "input_", 1) + ".in")
    try:
        text = in_path.read_text(errors="ignore")
    except OSError:
        return ""
    m = re.search(r"^\s*calculation\s*=\s*['\"]([\w-]+)['\"]", text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).lower() if m else ""


def _extract_fermi(run_dir: Path):
    """Fermi energy (eV) from QE output — used to zero-reference band plots.

    Metals print 'the Fermi energy is X ev'; insulators with fixed occupations
    print the highest-occupied level instead (use that as the reference).

    Scans all output_*.out steps and keeps the LAST value: vc-relax changes the
    cell so its early-step energies use a different reference than the final
    scf/nscf — and bands.x runs on that final structure. The last step's value
    is the one consistent with the band data.
    """
    fermi = None
    try:
        for f in _sorted_outputs(run_dir):
            # A 'bands' run does not determine the Fermi level (its k-point
            # weights are path points, not a BZ sampling). Skip it so it can't
            # overwrite the scf/nscf value the plot should be referenced to.
            if _calculation_of(f) == "bands":
                continue
            text = f.read_text(errors="ignore")
            m = re.findall(r"the Fermi energy is\s+(-?\d+\.\d+)\s*ev", text)
            if m:
                fermi = float(m[-1])
                continue
            m = re.findall(
                r"highest occupied, lowest unoccupied level \(ev\):\s*"
                r"(-?\d+\.\d+)\s+(-?\d+\.\d+)",
                text,
            )
            if m:
                fermi = float(m[-1][0])   # highest occupied = valence band max
                continue
            m = re.findall(r"highest occupied level \(ev\):\s*(-?\d+\.\d+)", text)
            if m:
                fermi = float(m[-1])
    except Exception:
        pass
    return fermi


def parse_bands(run_dir: Path):
    """Parse a QE bands.x *.band.gnu file into polylines.

    Format: blank-line-separated blocks, each block one band, each line
    `k_distance  energy`. Returns None if no band file.
    """
    gnu = next(iter(sorted(run_dir.glob("*.band.gnu"))), None)
    if gnu is None:
        return None
    bands = []
    cur = []
    try:
        for line in gnu.read_text(errors="ignore").splitlines():
            s = line.strip()
            if not s:
                if cur:
                    bands.append(cur)
                    cur = []
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    cur.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    pass
        if cur:
            bands.append(cur)
    except Exception:
        return None

    bands = [b for b in bands if len(b) >= 2]
    if not bands:
        return None
    all_e = [e for b in bands for (_, e) in b]
    all_k = [k for b in bands for (k, _) in b]
    return {
        "bands": bands,
        "n_bands": len(bands),
        "e_min": min(all_e),
        "e_max": max(all_e),
        "k_min": min(all_k),
        "k_max": max(all_k),
        "e_fermi": _extract_fermi(run_dir),
    }


def build_zip(run_dir: Path) -> bytes:
    """Zip all useful files in run_dir (flat — no nested paths)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fmeta in list_files(run_dir):
            fp = run_dir / fmeta["name"]
            try:
                z.write(fp, arcname=fmeta["name"])
            except Exception:
                pass
    buf.seek(0)
    return buf.getvalue()
