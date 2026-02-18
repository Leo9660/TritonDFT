# src/tool/tool_map.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from evaluate.relax_eval import run_relax_metrics, run_relax_metrics_input
from evaluate.scf_eval import run_scf_metrics, run_scf_metrics_input
from prompt.tool_requirements import get_bandsx_requirement, get_dosx_requirement, get_pw_requirement, get_ph_requirement, get_evx_requirement

@dataclass(frozen=True)
class ToolSpec:
    """
    Canonical mapping from a logical 'fn' to an executable and enforced settings.
    - exec: QE binary filename (e.g., 'pw.x', 'bands.x', 'dos.x')
    - mode: Optional calculation mode when exec is pw.x (e.g., 'scf', 'nscf', 'relax', 'vc-relax', 'bands')
    - required: Args that must be present in the step's args (light validation aid)
    - optional: Known optional args (documentation; not enforced here)
    - description: Short human-readable explanation
    """
    exec: str
    mode: Optional[str] = None
    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    description: str = ""
    section: str = ""  # Description of required sections in input file
    requirement_key: Optional[str] = None  # Additional textual requirements for the input file
    parse_requirement_key: Optional[str] = None  # Parsing requirements for the output file

    def __post_init__(self):
        if self.exec == "pw.x":
            if self.mode == "vc-relax":
                object.__setattr__(self, "eval_func", run_relax_metrics)
                object.__setattr__(self, "eval_input", run_relax_metrics_input)
            elif self.mode == "scf":
                object.__setattr__(self, "eval_func", run_scf_metrics)  # To be implemented later
                object.__setattr__(self, "eval_input", run_scf_metrics_input)

# ---- Allowed `fn` set (keep in sync with prompt) ----
ALLOWED_FNS: Set[str] = {
    "pw_scf",
    "pw_nscf",
    "pw_relax",
    "pw_vc_relax",
    "pw_bands",
    "bands_post",
    "dos_post",
    "projwfc_post",
    "pp_post",
    "q2r_post",
    "matdyn_post",
    "pw_phonon_gamma",
}

# ---- Canonical mapping from fn -> ToolSpec ----
FN_MAP: Dict[str, ToolSpec] = {
    # pw.x family
    "pw_scf": ToolSpec(
        exec="pw.x",
        mode="scf",
        required=("conv_thr",),
        optional=("ecutwfc", "ecutrho", "kpoints", "occupations", "smearing", "degauss", "ibrav",
                  "structure", "structure_from", "CELL_PARAMETERS", "ATOMIC_POSITIONS"),
        description="Self-consistent field calculation with pw.x",
        requirement_key="pw",
        parse_requirement_key="scf",
    ),
    "pw_nscf": ToolSpec(
        exec="pw.x",
        mode="nscf",
        required=(),
        optional=("kpoints", "kpath", "kpath_points", "occupations", "ibrav",
                  "structure", "structure_from"),
        description="Non-self-consistent run for DOS/bands preparation",
        requirement_key="pw",
        parse_requirement_key="nscf",
    ),
    "pw_relax": ToolSpec(
        exec="pw.x",
        mode="relax",
        required=(),
        optional=("ion_dynamics", "forc_conv_thr", "press_conv_thr", "ibrav",
                  "structure", "structure_from"),
        description="Ionic relaxation with pw.x",
        requirement_key="pw",
    ),
    "pw_vc_relax": ToolSpec(
        exec="pw.x",
        mode="vc-relax",
        required=(),
        optional=("cell_dynamics", "ion_dynamics", "forc_conv_thr", "press_conv_thr", "ibrav",
                  "structure", "structure_from"),
        description="Variable-cell relaxation with pw.x",
        requirement_key="pw",
        parse_requirement_key="vc-relax",
    ),
    "pw_bands": ToolSpec(
        exec="pw.x",
        mode="bands",
        required=(),
        optional=("kpath", "kpath_points", "ibrav", "structure", "structure_from"),
        description="Bands mode with pw.x (alternative to bands.x postprocessing)",
        requirement_key="pw",
        parse_requirement_key="pw_bands",
    ),

    # post-processing family
    "bands_post": ToolSpec(
        exec="bands.x",
        required=(),
        optional=("input_from", "plot", "emin", "emax"),
        description="Postprocess band structure after pw.x SCF/NSCF",
        requirement_key="bands",
        parse_requirement_key="bandsx",
    ),
    "dos_post": ToolSpec(
        exec="dos.x",
        required=(),
        optional=("input_from", "emin", "emax", "delta_e"),
        description="Total DOS postprocessing",
        requirement_key="dos",
        parse_requirement_key="dosx",
    ),
    "projwfc_post": ToolSpec(
        exec="projwfc.x",
        required=(),
        optional=("input_from", "lsym", "emax", "emin", "delta_e", "kresolved"),
        description="Projected DOS / band character postprocessing",
        section="&projwfc (minimal). Requires wavefunction file(s) from pw.x SCF/NSCF."
    ),
    "pp_post": ToolSpec(
        exec="pp.x",
        required=(),
        optional=("plot_num", "filplot", "input_from"),
        description="Charge density / potential postprocessing",
        section="&inputpp (minimal), &plot (optional). Requires charge density/potential from pw.x."
    ),
    "q2r_post": ToolSpec(
        exec="q2r.x",
        required=(),
        optional=("input_from", "fildyn", "flfrc"),
        description="Fourier transform dynamical matrices to real space",
        section="&input (minimal). Requires dynamical matrices from ph.x."
    ),
    "matdyn_post": ToolSpec(
        exec="matdyn.x",
        required=(),
        optional=("asr", "dos", "q_in_cryst_coord", "q_path", "flfrc", "fldos", "flfrq"),
        description="Phonon frequencies / DOS along paths",
        section="&input (minimal). Reads real-space force constants from q2r.x output."
    ),
    "pw_phonon_gamma": ToolSpec(
        exec="ph.x",
        mode="gamma",
        required=(),
        optional=("tr2_ph", "prefix", "outdir", "fildyn", "asr", "recover", "ldisp", "nq1", "nq2", "nq3", "structure_from"),
        description="Γ-point phonon stability check using DFPT (no Raman, no dispersion)",
        requirement_key="ph",
        parse_requirement_key="ph_gamma",
    ),
    "elastic_post": ToolSpec(
        exec="ev.x",
        required=(),
        optional=("input_from",),
        description="Calculate elastic constants and bulk modulus from vc-relax outputs",
        requirement_key="evx",
        parse_requirement_key="evx",
    ),
}

# ---- Helper functions for runners/validators ----

def is_allowed_fn(fn: str) -> bool:
    return fn in ALLOWED_FNS


def get_spec(fn: str) -> ToolSpec:
    if fn not in FN_MAP:
        raise KeyError(f"Unsupported fn '{fn}'. Allowed: {sorted(ALLOWED_FNS)}")
    return FN_MAP[fn]


def needs_pw_input(fn: str) -> bool:
    """True if the step requires a pw.x-style namelist input (*.pwi)."""
    spec = get_spec(fn)
    return spec.exec == "pw.x"


def enforced_pw_mode(fn: str) -> Optional[str]:
    """Return the calculation mode that must be enforced when writing CONTROL for pw.x."""
    spec = get_spec(fn)
    return spec.mode


def required_args(fn: str) -> Tuple[str, ...]:
    """Return required arg keys (light validation aid; runner may enforce more)."""
    return get_spec(fn).required


def optional_args(fn: str) -> Tuple[str, ...]:
    return get_spec(fn).optional


def binary_name(fn: str) -> str:
    """Return the QE executable name for the logical fn."""
    return get_spec(fn).exec


def is_postproc(fn: str) -> bool:
    """True if the step is a postprocessing executable (not pw.x)."""
    return binary_name(fn) not in {"pw.x"}


def normalize_tool(tool: str) -> str:
    """Normalize tool name aliases."""
    t = (tool or "").strip().lower().replace(" ", "_")
    if t in {"qe", "quantum-espresso", "quantumespresso"}:
        return "quantum_espresso"
    return t or "quantum_espresso"


def build_tool_requirements(fn_spec: ToolSpec, pseudo_dirs) -> str:
    """
    Map a tool's requirement key to its textual requirements.
    """
    if fn_spec.requirement_key == "pw":
        return get_pw_requirement(pseudo_dirs)
    if fn_spec.requirement_key == "bands":
        return get_bandsx_requirement()
    if fn_spec.requirement_key == "dos":
        return get_dosx_requirement()
    if fn_spec.requirement_key == "ph":
        return get_ph_requirement()
    if fn_spec.requirement_key == "evx":
        return get_evx_requirement()
    return ""
