"""Helpers for exposing a DFT run's artifacts: file listing, key-result
extraction, band-structure parsing, and zip packaging.

Run directories live on the shared RWX PVC under /workspace/tmp. The worker
records the absolute path in Job.run_dir; the API serves files from it.
"""
import io
import os
import re
import json
import shutil
import zipfile
from pathlib import Path

# Q1.A: ship only the useful files — exclude wavefunction / charge-density
# intermediates (*.wfc*, *.save/, *.mix*, etc.).
USEFUL_EXTS = {
    ".in", ".out", ".xml", ".json", ".band", ".gnu",
    ".dat", ".txt", ".log", ".cif", ".csv", ".png",
    ".sh",   # run_all.sh — the runner emitted for script-only bundles
}
# Internal bookkeeping files surfaced elsewhere (result cards / Analysis section)
# — hide them from the user-facing file list and zip.
HIDDEN_FILES = {"analysis.json", "run_meta.json"}
# Binary plot artifacts are downloadable but must not be previewed as text.
TEXT_EXTS = USEFUL_EXTS - {".png"}

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

    Step files are named <NN>_<task>.out (01_vc-relax.out, 07_ph.out). Older runs
    used output_<step>_<n>.out, so both are matched; either way sort on the
    leading numbers, since lexicographically "10" precedes "2".
    """
    def key(p: Path):
        nums = [int(x) for x in re.findall(r"\d+", p.stem)]
        return (nums, p.name)
    seen = {}
    for pattern in ("[0-9][0-9]_*.out", "output_*.out"):
        for f in run_dir.glob(pattern):
            seen[f.name] = f
    return sorted(seen.values(), key=key)


def _calculation_of(output_path: Path) -> str:
    """The pw.x `calculation` that produced an output, read from its sibling
    input file. Current runs pair by name (07_ph.out -> 07_ph.in); older runs used
    the output_/input_ prefix pair. Empty string if unknown."""
    in_path = output_path.with_suffix(".in")
    if not in_path.exists():
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
    # bands.x writes <filband>.gnu, and filband is whatever the model chose. The
    # prompt suggests '<prefix>.band' but also allows a fixed 'bands.dat', which
    # produces bands.dat.gnu — so matching only *.band.gnu silently lost the plot
    # for the second, equally valid, choice.
    gnu = next(iter(sorted(run_dir.glob("*.band.gnu"))), None)
    if gnu is None:
        gnu = next(iter(sorted(run_dir.glob("*.gnu"))), None)
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


# ─────────────────────────────────────────────────────────────────────────────
# Generic plot data (experimental)
#
# Deliberately shape-driven, not property-driven: QE's text outputs collapse to
# two forms, so DOS, projected DOS, phonon dispersion and anything similar added
# later are covered without a per-property parser.
#
#   A) commented XY table   "# E (eV)  dos(E)  Int dos(E)"  + numeric rows
#   B) multi-branch curves  band.gnu blocks, or matdyn's "&plot nbnd=,nks=/"
#
# No LLM involved: every label lives in the header line or the filename.
# ─────────────────────────────────────────────────────────────────────────────

def _read_xy_table(path: Path):
    """Parse a commented numeric table into {x_label, series, rows}."""
    header, rows = None, []
    try:
        for ln in path.read_text(errors="ignore").splitlines():
            s = ln.strip()
            if not s:
                continue
            if s.startswith("#"):
                if header is None:
                    header = s.lstrip("#").strip()
                continue
            try:
                rows.append([float(t) for t in s.split()])
            except ValueError:
                pass
    except OSError:
        return None
    if not rows:
        return None
    ncol = min(len(r) for r in rows)
    # QE writes the Fermi level into the same comment line; pull it out so it can
    # be drawn as a reference line instead of being mistaken for a column name.
    fermi = None
    if header:
        m = re.search(r"EFermi\s*=\s*(-?\d+\.\d+)", header)
        if m:
            fermi = float(m.group(1))
            header = header[:m.start()].strip()
    labels = re.findall(r"[A-Za-z][\w ]*\([^)]*\)|[A-Za-z]\w*", header or "")
    x_label = labels[0] if labels else "x"
    series = labels[1:ncol] or [f"col{i}" for i in range(1, ncol)]
    return {"x_label": x_label, "series": series, "e_fermi": fermi,
            "rows": [r[:ncol] for r in rows]}


_PDOS_RE = re.compile(r"pdos_atm#(\d+)\(([A-Za-z]+)\d*\)_wfc#(\d+)\(([spdfSPDF])", re.IGNORECASE)


def parse_dos(run_dir: Path):
    """Total DOS from dos.x, plus projected DOS from projwfc.x grouped by
    (element, orbital) — which is exactly the "separate Mo-d from S-p" view,
    derived from filenames rather than from any per-material code."""
    out = {}

    # Same problem as the band file: the prompt allows fildos='<prefix>.dos' or a
    # fixed 'dos.dat', and only the first matched. Fall back to anything that
    # looks like a DOS table, excluding the projected files which are parsed
    # separately below and would otherwise be mistaken for the total.
    tot = next(iter(sorted(run_dir.glob("*.dos"))), None)
    if tot is None:
        tot = next(
            (f for f in sorted(run_dir.glob("*dos*"))
             if f.is_file() and "pdos" not in f.name.lower()
             and f.suffix.lower() not in (".in", ".out")),
            None,
        )
    if tot is not None:
        t = _read_xy_table(tot)
        if t:
            out["total"] = {
                "x_label": t["x_label"],
                "e_fermi": t["e_fermi"],
                # column 1 is dos(E); column 2 is the running integral, not useful here
                "points": [[r[0], r[1]] for r in t["rows"] if len(r) > 1],
            }

    groups = {}
    for f in sorted(run_dir.glob("*pdos_atm*")):
        m = _PDOS_RE.search(f.name)
        if not m:
            continue
        key = f"{m.group(2)}-{m.group(4).lower()}"
        t = _read_xy_table(f)
        if not t:
            continue
        # ldos(E) is column 1: the total for this atom+shell.
        for r in t["rows"]:
            if len(r) < 2:
                continue
            groups.setdefault(key, {})
            groups[key][round(r[0], 4)] = groups[key].get(round(r[0], 4), 0.0) + r[1]
    if groups:
        out["projected"] = [
            {"label": k, "points": sorted([[e, v] for e, v in pts.items()])}
            for k, pts in sorted(groups.items())
        ]
    return out or None


def parse_phonons(run_dir: Path):
    """Phonon dispersion from matdyn.x flfrq output.

    Format: '&plot nbnd=N, nks=M /' then, per q-point, a coordinate line
    followed by the N frequencies. Same shape as a band structure, so the
    frontend renders it with the band plot.
    """
    src = None
    for pattern in ("*.freq", "*.phbands", "*freq*"):
        cand = [p for p in sorted(run_dir.glob(pattern)) if p.is_file()]
        for c in cand:
            head = ""
            try:
                head = c.read_text(errors="ignore")[:200]
            except OSError:
                continue
            if "&plot" in head and "nbnd" in head:
                src = c
                break
        if src:
            break
    if src is None:
        return None

    try:
        text = src.read_text(errors="ignore")
    except OSError:
        return None
    m = re.search(r"nbnd\s*=\s*(\d+)\s*,\s*nks\s*=\s*(\d+)", text)
    if not m:
        return None
    nbnd, nks = int(m.group(1)), int(m.group(2))

    nums = []
    for ln in text.splitlines()[1:]:
        for tok in ln.split():
            try:
                nums.append(float(tok))
            except ValueError:
                pass
    # Each q-point contributes 3 coordinates followed by nbnd frequencies.
    stride = 3 + nbnd
    if len(nums) < stride:
        return None
    branches = [[] for _ in range(nbnd)]
    x = 0.0
    prev = None
    for i in range(min(nks, len(nums) // stride)):
        chunk = nums[i * stride:(i + 1) * stride]
        q = chunk[:3]
        if prev is not None:
            x += sum((a - b) ** 2 for a, b in zip(q, prev)) ** 0.5
        prev = q
        for b in range(nbnd):
            branches[b].append([x, chunk[3 + b]])
    allf = [v for br in branches for (_, v) in br]
    if not allf:
        return None
    return {
        "bands": branches,
        "n_bands": nbnd,
        "e_min": min(allf),
        "e_max": max(allf),
        "k_min": 0.0,
        "k_max": branches[0][-1][0] if branches[0] else 0.0,
        "e_fermi": None,
        "unit": "cm-1",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scratch cleanup
# ─────────────────────────────────────────────────────────────────────────────

# Quantum ESPRESSO leaves large binary intermediates next to the results. They
# are regenerable, are already excluded from the download bundle (USEFUL_EXTS),
# and none of the parsers above read them — everything surfaced to the user comes
# from the text outputs. On a shared 100Gi PVC with eight workers they are pure
# accumulation, and a phonon run's _ph0 is the worst offender (tens of GB).
_SCRATCH_DIR_PATTERNS = ("_ph0", "*.save")
_SCRATCH_FILE_PATTERNS = ("*.wfc*", "*.mix*", "*.igk*", "*.hub*", "*.dvscf*", "*.bar*", "*.prd*")


def cleanup_scratch(run_dir: Path, enabled: bool = True) -> int:
    """Delete QE binary intermediates from a finished run. Returns bytes freed.

    MUST run only after extract_result(): it is safe for the on-demand parsers
    (bands/DOS/phonons all read top-level text files), but the extraction step
    reads the step outputs, which this never touches.

    `enabled=False` makes this a no-op. The hosted website wants cleanup on — its
    jobs are atomic and nobody resumes them — while the super-user/cluster path
    wants it off, because _ph0 is exactly what ph.x needs for recover=.true. and
    a failed attempt's scratch is evidence.
    """
    if not enabled:
        return 0
    freed = 0
    try:
        for pattern in _SCRATCH_DIR_PATTERNS:
            for d in run_dir.glob(pattern):
                if not d.is_dir() or d.is_symlink():
                    continue
                try:
                    for f in d.rglob("*"):
                        if f.is_file() and not f.is_symlink():
                            freed += f.stat().st_size
                    shutil.rmtree(d, ignore_errors=True)
                except OSError:
                    pass
        for pattern in _SCRATCH_FILE_PATTERNS:
            for f in run_dir.glob(pattern):
                if not f.is_file() or f.is_symlink():
                    continue
                try:
                    freed += f.stat().st_size
                    f.unlink()
                except OSError:
                    pass
    except Exception:
        pass
    return freed


def sweep_stale_scratch(root: str, min_age_s: int) -> tuple:
    """Clean QE intermediates from run directories left behind by dead workers.

    Per-job cleanup covers every normal terminal path, but not a pod that is
    OOM-killed or evicted mid-run — and since each worker's directory is keyed by
    its hostname, a restarted pod never revisits its predecessor's leftovers. So
    this sweeps the whole root.

    Only touches directories older than `min_age_s`. A job cannot outlive
    JOB_TIMEOUT_S (assistant-mode gates auto-continue after 10 minutes each), so
    with a margin this cannot race a run that is still going.

    Returns (dirs_cleaned, bytes_freed).
    """
    import time as _time
    base = Path(root)
    if not base.is_dir():
        return (0, 0)
    cutoff = _time.time() - min_age_s
    cleaned = 0
    freed = 0
    try:
        # layout: <root>/<worker>/<YYYY-MM-DD>/<run>/
        for run_dir in base.glob("*/*/*"):
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            if not (run_dir / "run_meta.json").exists():
                continue
            try:
                if run_dir.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            got = cleanup_scratch(run_dir)
            if got:
                cleaned += 1
                freed += got
    except Exception:
        pass
    return (cleaned, freed)
