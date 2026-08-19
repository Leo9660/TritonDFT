from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from tool.tool_map import get_spec, is_allowed_fn
from validation.qe_syntax import validate_qe_syntax


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    path: str = ""

    @property
    def blocking(self) -> bool:
        return self.severity in {"fatal", "error"}

    def format(self) -> str:
        location = f" [{Path(self.path).name}]" if self.path else ""
        return f"{self.severity.upper()} {self.code}{location}: {self.message}"


def _requested(query: str, pattern: str) -> bool:
    return bool(re.search(pattern, query or "", re.IGNORECASE))


def normalize_plan(query: str, steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize unambiguous tool choices without selecting scientific parameters."""
    normalized: List[Dict[str, Any]] = []
    for raw in steps:
        step = dict(raw)
        problem = str(step.get("problem") or "")
        tool = str(step.get("tool") or "")
        # A path calculation consumed by bands.x is calculation='bands', not a
        # generic uniform-grid NSCF. This is a tool/syntax invariant, not a
        # material-specific scientific choice.
        if tool == "pw_nscf" and re.search(
            r"high[- ]symmetry|band[- ]?structure|band path|along .*k[- ]?point path",
            problem,
            re.IGNORECASE,
        ):
            step["tool"] = "pw_bands"
            step["normalized_from"] = "pw_nscf"
        normalized.append(step)
    # With the current shared QE save state, ph.x must consume the canonical
    # SCF state before band/DOS PW runs update that state. Stable priorities
    # also keep every postprocessor after its producer.
    priorities = {
        "pw_vc_relax": 10,
        "pw_relax": 10,
        "pw_scf": 20,
        "pw_phonon_gamma": 30,
        "q2r_post": 40,
        "matdyn_post": 50,
        "dynmat_post": 50,
        "pw_bands": 60,
        "bands_post": 70,
        "pw_nscf": 80,
        "dos_post": 90,
        "projwfc_post": 90,
    }
    indexed = list(enumerate(normalized))
    indexed.sort(key=lambda pair: (priorities.get(str(pair[1].get("tool") or ""), 55), pair[0]))
    ordered = [step for _, step in indexed]
    for step_id, step in enumerate(ordered, start=1):
        step["id"] = step_id
    return ordered


def validate_plan(query: str, steps: Sequence[Dict[str, Any]]) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    tools = [str(step.get("tool") or "") for step in steps]
    ids = [step.get("id") for step in steps]
    if len(ids) != len(set(ids)):
        issues.append(ValidationIssue("fatal", "DUPLICATE_STEP_ID", "Plan step IDs must be unique."))
    for index, step in enumerate(steps, 1):
        tool = str(step.get("tool") or "")
        if not tool or not is_allowed_fn(tool):
            issues.append(ValidationIssue("fatal", "UNKNOWN_TOOL", f"Step {index} uses unsupported tool {tool!r}."))
        if not step.get("problem") or not step.get("input"):
            issues.append(ValidationIssue("error", "INCOMPLETE_STEP", f"Step {index} is missing its problem or required input."))
        description = " ".join(str(step.get(key) or "") for key in ("problem", "input", "why"))
        if re.search(r"\boptional(?:ly)?\b", description, re.I):
            issues.append(ValidationIssue(
                "error", "OPTIONAL_STEP_IN_EXECUTABLE_PLAN",
                f"Step {index} is optional; alternatives belong in approval questions, not the executable plan.",
            ))
        # Only an explicit calculation action in the step title constitutes a
        # convergence study. Required-input phrases such as "converged cutoffs"
        # describe production parameters and must not turn vc-relax into a
        # phantom convergence workflow.
        problem = str(step.get("problem") or "")
        convergence_action = re.search(
            r"\b(?:test|study|sweep|converge|optimi[sz]e|determine)\b.{0,50}"
            r"(?:cutoffs?|k[- ]?points?|sampling)",
            problem,
            re.I,
        )
        if convergence_action and not _requested(
            query, r"convergence (?:test|study)|converge (?:the )?(?:cutoff|k[- ]?point)|cutoff convergence|k[- ]?point convergence"
        ):
            issues.append(ValidationIssue(
                "error", "UNREQUESTED_CONVERGENCE_STEP",
                f"Step {index} adds a convergence study that the user did not request.",
            ))

    scf_steps = [step for step in steps if step.get("tool") == "pw_scf"]
    scf_count = len(scf_steps)
    explicit_comparison = _requested(query, r"compare.*(?:soc|scalar)|(?:soc|scalar).*comparison|with and without soc")
    scf_descriptions = [
        " ".join(str(step.get(key) or "") for key in ("problem", "input", "why"))
        for step in scf_steps
    ]
    has_soc_scf = any(re.search(r"\bSOC\b|fully[- ]relativistic|lspinorb", text, re.I) for text in scf_descriptions)
    has_scalar_scf = any(re.search(r"scalar[- ]relativistic|without\s+SOC|SOC\s+(?:off|disabled)", text, re.I) for text in scf_descriptions)
    distinct_scalar_soc_branches = has_soc_scf and has_scalar_scf
    if scf_count > 1 and not explicit_comparison and not distinct_scalar_soc_branches:
        issues.append(ValidationIssue(
            "error", "DUPLICATE_SCF_WORKFLOWS",
            "The plan contains multiple SCF workflows without an explicit scalar-versus-SOC comparison request.",
        ))

    if _requested(query, r"projected\s+dos|\bpdos\b") and "projwfc_post" not in tools:
        issues.append(ValidationIssue("error", "PDOS_STEP_MISSING", "Projected DOS was requested, but the plan has no projwfc.x step."))
    if _requested(query, r"phonon\s+dispersion"):
        for required in ("pw_phonon_gamma", "q2r_post", "matdyn_post"):
            if required not in tools:
                issues.append(ValidationIssue("error", "PHONON_CHAIN_INCOMPLETE", f"Phonon dispersion requires {required}."))
    if _requested(query, r"raman") and "pw_phonon_gamma" not in tools:
        issues.append(ValidationIssue("error", "RAMAN_PH_MISSING", "Raman properties require a ph.x step."))
    if _requested(query, r"raman") and _requested(query, r"phonon\s+dispersion"):
        phonon_steps = [
            step for step in steps if step.get("tool") == "pw_phonon_gamma"
        ]
        has_grid = any(re.search(r"dispersion|q[- ]?point (?:grid|mesh)|uniform.*q", str(s.get("problem") or ""), re.I) for s in phonon_steps)
        has_raman_gamma = any(re.search(r"raman|gamma|Γ", str(s.get("problem") or ""), re.I) for s in phonon_steps)
        if len(phonon_steps) < 2 or not has_grid or not has_raman_gamma:
            issues.append(ValidationIssue(
                "error",
                "RAMAN_DISPERSION_PH_SPLIT_REQUIRED",
                "A combined Raman and phonon-dispersion request requires separate q-grid and Gamma Raman ph.x steps.",
            ))
    if "matdyn_post" in tools and "q2r_post" not in tools:
        issues.append(ValidationIssue("error", "MATDYN_WITHOUT_Q2R", "matdyn.x requires force constants produced by q2r.x."))
    if "q2r_post" in tools and "pw_phonon_gamma" not in tools:
        issues.append(ValidationIssue("error", "Q2R_WITHOUT_PH", "q2r.x requires a preceding q-grid ph.x calculation."))
    if "bands_post" in tools and not any(t in tools for t in ("pw_bands", "pw_nscf")):
        issues.append(ValidationIssue("error", "BANDS_SOURCE_MISSING", "bands.x has no preceding pw.x band-path calculation."))
    if "dos_post" in tools and "pw_nscf" not in tools:
        issues.append(ValidationIssue("error", "DOS_SOURCE_MISSING", "dos.x has no uniform-grid NSCF source."))
    return issues


def _namelist(text: str, name: str) -> str:
    match = re.search(rf"(?mis)^\s*&{re.escape(name)}\b(.*?)^\s*/\s*$", text)
    return match.group(1) if match else ""


def _value(text: str, key: str) -> str:
    match = re.search(
        rf"(?mi)^\s*{re.escape(key)}\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^,!\n/]+))",
        text,
    )
    if not match:
        return ""
    return next((group for group in match.groups() if group is not None), "").strip()


def _card_rows(text: str, card: str) -> tuple[str, List[str]]:
    header = re.search(rf"(?mi)^\s*{re.escape(card)}\b([^\n]*)\n", text)
    if not header:
        return "", []
    rows: List[str] = []
    for line in text[header.end():].splitlines():
        if re.match(r"^\s*(?:&|ATOMIC_(?:SPECIES|POSITIONS|FORCES)|CELL_PARAMETERS|K_POINTS|HUBBARD|OCCUPATIONS|CONSTRAINTS)\b", line, re.I):
            break
        if line.strip() and not line.lstrip().startswith("!"):
            rows.append(line.strip())
    return header.group(1).strip().lower(), rows


def _xc_family(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
    if "pbesol" in normalized:
        return "pbesol"
    if normalized in {"lda", "pz", "pw", "slapwnogxnogc"} or "nogxnogc" in normalized:
        return "lda"
    if "pbe" in normalized:
        return "pbe"
    return ""


def _upf_xc_family(path: Path) -> str:
    if not path.is_file():
        return ""
    header = path.read_text(encoding="utf-8", errors="replace")[:20000]
    match = re.search(r"(?i)functional\s*=\s*['\"]([^'\"]+)['\"]", header)
    return _xc_family(match.group(1)) if match else ""


def _sibling_xc_family(path: str) -> str:
    """The XC family this run is using, read from the pw.x inputs beside a
    ph.x input.

    A ph.x input has no &system and names no pseudopotentials, so it cannot say
    which functional is in play — but every step of a run shares one, and the
    pw.x steps in the same directory do declare it. Prefers an explicit
    input_dft; otherwise reads the functional out of the UPF headers.

    Returns "" when it cannot tell, and callers must treat that as "no opinion"
    rather than as a negative.
    """
    try:
        siblings = sorted(Path(path).resolve().parent.glob("*.in"))
    except OSError:
        return ""
    for f in siblings:
        if str(f) == str(Path(path).resolve()):
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        if not re.search(r"(?mi)^\s*&system\b", text):
            continue
        family = _xc_family(_value(_namelist(text, "system"), "input_dft"))
        if family:
            return family
        pseudo_dir_value = _value(_namelist(text, "control"), "pseudo_dir")
        if not pseudo_dir_value:
            continue
        pseudo_dir = Path(pseudo_dir_value).expanduser()
        if not pseudo_dir.is_absolute():
            pseudo_dir = f.parent / pseudo_dir
        _, species_rows = _card_rows(text, "ATOMIC_SPECIES")
        for row in species_rows:
            tokens = row.split()
            if len(tokens) >= 3:
                family = _upf_xc_family(pseudo_dir / tokens[2])
                if family:
                    return family
    return ""


def _upf_is_fully_relativistic(path: Path) -> bool:
    if not path.is_file():
        return False
    header = path.read_text(encoding="utf-8", errors="replace")[:30000]
    return bool(
        re.search(r"(?i)relativistic\s*=\s*['\"]full['\"]", header)
        or re.search(r"(?i)has_so\s*=\s*['\"]?(?:t|true|\.true\.)", header)
        or re.search(r"(?i)fully?[- ]relativistic", header)
    )


def validate_qe_input(path: str, *, tool: str = "", query: str = "") -> List[ValidationIssue]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    issues: List[ValidationIssue] = []
    add = lambda severity, code, message: issues.append(ValidationIssue(severity, code, message, path))

    for marker in (
        "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN",
        "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_END",
    ):
        for line in text.splitlines():
            if marker in line and line.strip() != f"! {marker}":
                add("fatal", "PLACEHOLDER_EMBEDDED", "The relaxed-structure placeholder is embedded in another input line.")

    exec_name = get_spec(tool).exec if tool and is_allowed_fn(tool) else ""
    for finding in validate_qe_syntax(path, exec_name):
        add("fatal", finding.code, finding.message)
    if exec_name == "pw.x" or re.search(r"(?mi)^\s*&system\b", text):
        for name in ("control", "system", "electrons"):
            if not re.search(rf"(?mi)^\s*&{name}\b", text):
                add("fatal", "NAMELIST_MISSING", f"pw.x input is missing &{name}.")
        for card in ("ATOMIC_SPECIES", "K_POINTS"):
            if not re.search(rf"(?mi)^\s*{card}\b", text):
                add("fatal", "CARD_MISSING", f"pw.x input is missing {card}.")
        calculation = _value(_namelist(text, "control"), "calculation").lower()
        if _value(_namelist(text, "control"), "restart_mode").lower() == "restart":
            add(
                "error",
                "NEW_STEP_RESTART_MODE_INVALID",
                "A newly generated workflow step must use restart_mode='from_scratch'; checkpoint recovery is managed separately.",
            )
        requested_xc = _xc_family(_value(_namelist(text, "system"), "input_dft"))
        pseudo_dir_value = _value(_namelist(text, "control"), "pseudo_dir")
        if pseudo_dir_value:
            pseudo_dir = Path(pseudo_dir_value).expanduser()
            if not pseudo_dir.is_absolute():
                pseudo_dir = Path(path).resolve().parent / pseudo_dir
            _, species_rows = _card_rows(text, "ATOMIC_SPECIES")
            for row in species_rows:
                tokens = row.split()
                if len(tokens) < 3:
                    continue
                pseudo_path = pseudo_dir / tokens[2]
                pseudo_xc = _upf_xc_family(pseudo_path)
                if requested_xc and pseudo_xc and pseudo_xc != requested_xc:
                    add(
                        "fatal",
                        "PSEUDO_XC_MISMATCH",
                        f"input_dft requests {requested_xc.upper()}, but {tokens[2]} declares {pseudo_xc.upper()}.",
                    )
                lspinorb = _value(_namelist(text, "system"), "lspinorb").lower()
                if lspinorb in {".true.", "true"} and not _upf_is_fully_relativistic(pseudo_path):
                    add(
                        "fatal",
                        "SOC_PSEUDO_NOT_FULLY_RELATIVISTIC",
                        f"lspinorb is enabled, but {tokens[2]} is not a fully relativistic UPF.",
                    )
        if tool and is_allowed_fn(tool):
            expected = get_spec(tool).mode
            if expected and calculation and calculation != expected:
                add("error", "CALCULATION_MODE_MISMATCH", f"Tool {tool} requires calculation='{expected}', found '{calculation}'.")

        placeholder = "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN" in text
        pos_kind, positions = _card_rows(text, "ATOMIC_POSITIONS")
        if not placeholder and not positions:
            add("fatal", "POSITIONS_MISSING", "No atomic positions or deferred relaxed structure is present.")
        nat = _value(_namelist(text, "system"), "nat")
        if positions and nat.isdigit() and len(positions) != int(nat):
            add("fatal", "NAT_MISMATCH", f"nat={nat}, but {len(positions)} atomic-position rows were found.")
        if _value(_namelist(text, "system"), "ibrav") == "0" and not placeholder:
            _, cell = _card_rows(text, "CELL_PARAMETERS")
            if len(cell) != 3:
                add("fatal", "CELL_INVALID", "ibrav=0 requires exactly three CELL_PARAMETERS rows.")

        k_kind, k_rows = _card_rows(text, "K_POINTS")
        if "crystal_b" in k_kind:
            if not k_rows or not re.fullmatch(r"\d+", k_rows[0]):
                add("fatal", "KPOINT_COUNT_MISSING", "K_POINTS crystal_b requires an integer path-node count.")
            else:
                declared = int(k_rows[0])
                nodes = k_rows[1:]
                if len(nodes) != declared:
                    add("fatal", "KPOINT_COUNT_MISMATCH", f"K_POINTS declares {declared} nodes, but {len(nodes)} rows were found.")
                for number, row in enumerate(nodes, 1):
                    tokens = row.split()
                    try:
                        valid = len(tokens) == 4 and all(float(x) == float(x) for x in tokens[:3]) and int(tokens[3]) > 0
                    except (ValueError, TypeError):
                        valid = False
                    if not valid:
                        add("fatal", "KPOINT_ROW_INVALID", f"Band-path row {number} must contain three numbers and one positive integer.")
        elif "automatic" in k_kind:
            if len(k_rows) != 1 or not re.fullmatch(r"(?:\d+\s+){5}\d+", k_rows[0]):
                add("fatal", "KPOINT_GRID_INVALID", "K_POINTS automatic requires one row containing six integers.")

    elif exec_name == "ph.x" or re.search(r"(?mi)^\s*&inputph\b", text):
        body = _namelist(text, "inputph")
        if not re.search(r"(?mi)^\s*&inputph\b", text):
            add("fatal", "NAMELIST_MISSING", "ph.x input is missing &inputph.")
        for key in ("prefix", "outdir", "fildyn", "tr2_ph"):
            if not _value(body, key):
                add("fatal", "PH_KEY_MISSING", f"ph.x input is missing {key}.")
        if _requested(query, r"raman"):
            lraman_on = _value(body, "lraman").lower() in {".true.", "true"}
            xc = _sibling_xc_family(path)
            if xc in {"pbe", "pbesol"}:
                # Demanding lraman here would be unsatisfiable, and used to be:
                # with lraman off the step was rejected, and with it on QE
                # aborted with "third order derivatives not implemented with
                # GGA". The agent burned every retry oscillating between the
                # two. Raman tensors in ph.x are a third-order response and QE
                # implements them for LDA only, so say that once and stop.
                add(
                    "fatal",
                    "RAMAN_REQUIRES_LDA",
                    f"Quantum ESPRESSO computes Raman tensors only for LDA; this run uses "
                    f"{xc.upper()}, for which the third-order derivatives are not implemented. "
                    f"Re-run with an LDA functional and an LDA pseudopotential library, or drop "
                    f"the Raman step and report the Gamma-point phonon frequencies alone.",
                )
            elif not lraman_on:
                add("error", "RAMAN_NOT_IN_THIS_PH", "A Raman ph.x step must enable lraman=.true..")

    elif exec_name == "bands.x":
        body = _namelist(text, "bands")
        if not re.search(r"(?mi)^\s*&bands\b", text):
            add("fatal", "NAMELIST_MISSING", "bands.x input is missing &BANDS.")
        for key in ("prefix", "outdir", "filband"):
            if not _value(body, key):
                add("fatal", "BANDS_KEY_MISSING", f"bands.x input is missing {key}.")

    elif exec_name == "dos.x":
        body = _namelist(text, "dos")
        if not re.search(r"(?mi)^\s*&dos\b", text):
            add("fatal", "NAMELIST_MISSING", "dos.x input is missing &DOS.")
        for key in ("prefix", "outdir", "fildos"):
            if not _value(body, key):
                add("fatal", "DOS_KEY_MISSING", f"dos.x input is missing {key}.")
        try:
            emin = float(_value(body, "emin").replace("d", "e").replace("D", "e"))
            emax = float(_value(body, "emax").replace("d", "e").replace("D", "e"))
        except ValueError:
            emin = emax = None
        if emin is not None and emax is not None:
            if emax <= emin:
                add("fatal", "DOS_ENERGY_RANGE_INVALID", "DOS Emax must be greater than Emin.")
            elif emax - emin < 5.0:
                add("error", "DOS_ENERGY_WINDOW_TOO_NARROW", f"The DOS window is only {emax - emin:.3g} eV; use at least 5 eV for a full DOS unless a narrow interval was explicitly requested.")

    elif exec_name == "projwfc.x":
        body = _namelist(text, "projwfc")
        if not re.search(r"(?mi)^\s*&projwfc\b", text):
            add("fatal", "NAMELIST_MISSING", "projwfc.x input is missing &PROJWFC.")
        for key in ("prefix", "outdir", "filpdos"):
            if not _value(body, key):
                add("fatal", "PROJWFC_KEY_MISSING", f"projwfc.x input is missing {key}.")

    elif exec_name in {"q2r.x", "matdyn.x", "dynmat.x"}:
        body = _namelist(text, "input")
        if not re.search(r"(?mi)^\s*&input\b", text):
            add("fatal", "NAMELIST_MISSING", f"{exec_name} input is missing &input.")
        required = {
            "q2r.x": ("fildyn", "flfrc"),
            "matdyn.x": ("flfrc",),
            "dynmat.x": ("fildyn",),
        }[exec_name]
        for key in required:
            if not _value(body, key):
                add("fatal", "POSTPROC_KEY_MISSING", f"{exec_name} input is missing {key}.")

    return issues


def _input_values(path: str) -> Dict[str, str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return {key: _value(text, key).lower() for key in (
        "input_dft", "ecutwfc", "ecutrho", "nspin", "noncolin", "lspinorb",
        "vdw_corr", "london", "dftd3_version", "dftd3_threebody",
        "prefix", "outdir", "fildyn", "flfrc", "bz_sum",
        "occupations", "lraman", "ldisp", "filband", "fildos", "filpdos",
    )}


def _vdw_settings(path: str) -> Dict[str, str]:
    """Extract the complete vdW/D3 model contract from a pw.x &SYSTEM namelist."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    system = _namelist(text, "system")
    settings: Dict[str, str] = {}
    for match in re.finditer(
        r"(?mi)^\s*((?:vdw_corr|london|dftd3_[a-z0-9_]+|ts_vdw_[a-z0-9_]+|xdm_[a-z0-9_]+))\s*=\s*([^,!\n/]+|'[^']*'|\"[^\"]*\")",
        system,
    ):
        key = match.group(1).lower()
        value = match.group(2).strip().strip("'\"").lower()
        settings[key] = value
    return settings


def _species_signature(path: str) -> tuple[tuple[str, str], ...]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    _, rows = _card_rows(text, "ATOMIC_SPECIES")
    signature = []
    for row in rows:
        tokens = row.split()
        if len(tokens) >= 3:
            signature.append((tokens[0].lower(), Path(tokens[2]).name.lower()))
    return tuple(signature)


_SHARED_SYSTEM_KEYS = (
    "input_dft", "ecutwfc", "ecutrho", "nspin", "noncolin", "lspinorb",
    "vdw_corr", "london",
)


def _setting_line(text: str, key: str) -> str:
    match = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*[^\n]+$", text)
    return match.group(0).strip() if match else ""


def _set_system_line(text: str, key: str, source_line: str) -> str:
    existing = re.search(rf"(?mi)^\s*{re.escape(key)}\s*=\s*[^\n]+$", text)
    rendered = "  " + source_line.strip().rstrip(",") + ","
    if existing:
        return text[:existing.start()] + rendered + text[existing.end():]
    system = re.search(r"(?mis)^\s*&system\b.*?^\s*/\s*$", text)
    if not system:
        return text
    slash = text.rfind("/", system.start(), system.end())
    return text[:slash] + rendered + "\n" + text[slash:]


def inherit_shared_pw_settings(packages: Sequence[Dict[str, Any]]) -> None:
    """Carry the model's first PW physical setup into dependent PW steps.

    This does not select values. It prevents later independent generations
    from silently changing the XC, basis, spin, SOC, or vdW model.
    """
    pw_paths = [
        package["input_paths"][0]
        for package in packages
        if package.get("exec_name") == "pw.x" and package.get("input_paths")
    ]
    if not pw_paths:
        return
    baseline_text = Path(pw_paths[0]).read_text(encoding="utf-8")
    baseline_species_span = re.search(
        r"(?mis)^\s*ATOMIC_SPECIES\b[^\n]*\n.*?(?=^\s*(?:ATOMIC_POSITIONS|CELL_PARAMETERS|K_POINTS|HUBBARD|&)|\Z)",
        baseline_text,
    )
    shared_lines = {
        key: _setting_line(baseline_text, key)
        for key in _SHARED_SYSTEM_KEYS
        if _setting_line(baseline_text, key)
    }
    starting_lines = re.findall(r"(?mi)^\s*starting_magnetization\s*\(\s*\d+\s*\)\s*=\s*[^\n]+$", baseline_text)
    hubbard = re.search(r"(?mis)^\s*HUBBARD\b[^\n]*\n.*?(?=^\s*(?:K_POINTS|ATOMIC_|CELL_PARAMETERS|&)|\Z)", baseline_text)
    for path in pw_paths[1:]:
        text = Path(path).read_text(encoding="utf-8")
        for key, line in shared_lines.items():
            text = _set_system_line(text, key, line)
        if starting_lines:
            text = re.sub(r"(?mi)^\s*starting_magnetization\s*\(\s*\d+\s*\)\s*=\s*[^\n]+\n?", "", text)
            for line in starting_lines:
                text = _set_system_line(text, "__starting_placeholder__", line)
                text = text.replace("__starting_placeholder__", "starting_magnetization", 1)
        if hubbard and not re.search(r"(?mi)^\s*HUBBARD\b", text):
            kpoints = re.search(r"(?mi)^\s*K_POINTS\b", text)
            if kpoints:
                text = text[:kpoints.start()] + hubbard.group(0).rstrip() + "\n" + text[kpoints.start():]
        if baseline_species_span:
            current_species = re.search(
                r"(?mis)^\s*ATOMIC_SPECIES\b[^\n]*\n.*?(?=^\s*(?:ATOMIC_POSITIONS|CELL_PARAMETERS|K_POINTS|HUBBARD|&|!)|\Z)",
                text,
            )
            if current_species:
                text = (
                    text[:current_species.start()]
                    + baseline_species_span.group(0).rstrip()
                    + "\n"
                    + text[current_species.end():]
                )
        Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def inherit_vdw_model(packages: Sequence[Dict[str, Any]]) -> None:
    """Carry the complete relaxation vdW model into every later pw.x input.

    This contract is independent of pseudopotential relativity: an SOC branch
    must switch to fully relativistic UPFs and enable SOC, but it must not
    silently drop or change the approved D3/vdW prescription. Only vdW keys are
    copied here so scalar species, spin, and SOC settings never cross branches.
    """
    pw_paths = [
        package["input_paths"][0]
        for package in packages
        if package.get("exec_name") == "pw.x" and package.get("input_paths")
    ]
    if len(pw_paths) < 2:
        return
    baseline = Path(pw_paths[0]).read_text(encoding="utf-8", errors="replace")
    system = _namelist(baseline, "system")
    source_lines: Dict[str, str] = {}
    for match in re.finditer(
        r"(?mi)^\s*((?:vdw_corr|london|dftd3_[a-z0-9_]+|ts_vdw_[a-z0-9_]+|xdm_[a-z0-9_]+))\s*=\s*[^\n]+$",
        system,
    ):
        source_lines[match.group(1).lower()] = match.group(0).strip()
    if not source_lines:
        return
    for path in pw_paths[1:]:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for key, line in source_lines.items():
            text = _set_system_line(text, key, line)
        Path(path).write_text(text.rstrip() + "\n", encoding="utf-8")


def _upf_valence(path: Path) -> float | None:
    if not path.is_file():
        return None
    header = path.read_text(encoding="utf-8", errors="replace")[:30000]
    match = re.search(r"(?i)z_valence\s*=\s*['\"]?\s*([-+0-9.EeDd]+)", header)
    return float(match.group(1).replace("d", "e").replace("D", "E")) if match else None


def _electron_count_from_input(path: Path, text: str) -> float | None:
    _, species_rows = _card_rows(text, "ATOMIC_SPECIES")
    _, position_rows = _card_rows(text, "ATOMIC_POSITIONS")
    pseudo_dir_value = _value(_namelist(text, "control"), "pseudo_dir")
    pseudo_dir = Path(pseudo_dir_value).expanduser() if pseudo_dir_value else path.parent / "pseudos"
    if not pseudo_dir.is_absolute():
        pseudo_dir = (path.parent / pseudo_dir).resolve()
    valences: Dict[str, float] = {}
    for row in species_rows:
        tokens = row.split()
        if len(tokens) < 3:
            continue
        value = _upf_valence(pseudo_dir / tokens[2])
        if value is None:
            return None
        valences[tokens[0]] = value
    if not valences or not position_rows:
        return None
    counts: Dict[str, int] = {}
    for row in position_rows:
        tokens = row.split()
        if len(tokens) >= 4 and tokens[0] in valences:
            counts[tokens[0]] = counts.get(tokens[0], 0) + 1
    if not counts:
        return None
    total_charge_raw = _value(_namelist(text, "system"), "tot_charge") or "0"
    try:
        total_charge = float(total_charge_raw.replace("d", "e").replace("D", "E"))
    except ValueError:
        total_charge = 0.0
    return sum(valences[label] * count for label, count in counts.items()) - total_charge


def recommended_nbnd(path: str | Path, *, extra_bands: int = 16) -> int | None:
    """Occupancy-aware safe band count derived from the selected UPFs."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8", errors="replace")
    electrons = _electron_count_from_input(input_path, text)
    if electrons is None:
        return None
    system = _namelist(text, "system")
    noncollinear = _value(system, "noncolin").lower() in {".true.", "true", "t"}
    spin_orbit = _value(system, "lspinorb").lower() in {".true.", "true", "t"}
    occupied = math.ceil(electrons if (noncollinear or spin_orbit) else electrons / 2.0)
    return occupied + extra_bands


def harmonize_nbnd(packages: Sequence[Dict[str, Any]], *, extra_bands: int = 16) -> None:
    """Raise bands/NSCF nbnd to a safe occupied-plus-empty-state minimum."""
    for package in packages:
        if package.get("exec_name") != "pw.x":
            continue
        for raw_path in package.get("input_paths", []):
            path = Path(raw_path)
            text = path.read_text(encoding="utf-8", errors="replace")
            calculation = _value(_namelist(text, "control"), "calculation").lower()
            if calculation not in {"bands", "nscf"}:
                continue
            minimum = recommended_nbnd(path, extra_bands=extra_bands)
            if minimum is None:
                continue
            current_raw = _value(_namelist(text, "system"), "nbnd")
            try:
                current = int(float(current_raw)) if current_raw else 0
            except ValueError:
                current = 0
            if current < minimum:
                text = _set_system_line(text, "nbnd", f"nbnd={minimum}")
                path.write_text(text.rstrip() + "\n", encoding="utf-8")


def harmonize_dos_integration(
    steps: Sequence[Dict[str, Any]],
    packages: Sequence[Dict[str, Any]],
) -> None:
    """Make dos.x use the integration family selected by its dense NSCF source.

    The two inputs are generated as separate subproblems, but ``occupations``
    and ``bz_sum`` are one producer/consumer contract.  Resolve that contract
    deterministically before approval instead of asking an LLM to repair an
    unambiguous mismatch.
    """
    nscf_path = next((
        package["input_paths"][0]
        for step, package in zip(steps, packages)
        if step.get("tool") == "pw_nscf" and package.get("input_paths")
    ), "")
    if not nscf_path:
        return
    nscf_text = Path(nscf_path).read_text(encoding="utf-8", errors="replace")
    occupations = _value(_namelist(nscf_text, "system"), "occupations").lower()
    target = occupations if occupations in {
        "tetrahedra", "tetrahedra_lin", "tetrahedra_opt"
    } else "smearing"

    for step, package in zip(steps, packages):
        if step.get("tool") != "dos_post" or not package.get("input_paths"):
            continue
        path = Path(package["input_paths"][0])
        text = path.read_text(encoding="utf-8", errors="replace")
        dos_match = re.search(r"(?mis)^\s*&dos\b.*?^\s*/\s*$", text)
        if not dos_match:
            continue
        body = dos_match.group(0)
        rendered = f"  bz_sum='{target}',"
        existing = re.search(r"(?mi)^\s*bz_sum\s*=\s*[^\n]+$", body)
        if existing:
            body = body[:existing.start()] + rendered + body[existing.end():]
        else:
            slash = body.rfind("/")
            body = body[:slash] + rendered + "\n" + body[slash:]
        if target.startswith("tetrahedra"):
            # These are smearing controls. QE may ignore them in tetrahedron
            # mode, but retaining them makes the approved numerical contract
            # ambiguous and has repeatedly caused regeneration drift.
            body = re.sub(r"(?mi)^\s*(?:ngauss|degauss)\s*=\s*[^\n]+\n?", "", body)
        text = text[:dos_match.start()] + body + text[dos_match.end():]
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def harmonize_dos_window(
    steps: Sequence[Dict[str, Any]],
    packages: Sequence[Dict[str, Any]],
    query: str = "",
) -> None:
    """Replace an accidentally tiny full-DOS window with safe eV defaults.

    dos.x expresses Emin/Emax/DeltaE in eV. Model-generated inputs have
    occasionally converted these as though they were Ry, producing windows
    such as +/-0.735 eV. Preserve deliberately narrow spectroscopy requests,
    but make ordinary total-DOS workflows cover the relevant valence and
    conduction states before approval.
    """
    explicit_narrow_request = bool(re.search(
        r"(?i)(?:DOS|density of states).{0,80}(?:between|from|range|window|within)"
        r".{0,40}[-+]?\d+(?:\.\d+)?\s*(?:eV|electron\s*volts?)",
        query,
    ))
    if explicit_narrow_request:
        return

    def number(body: str, key: str) -> Optional[float]:
        raw = _value(body, key)
        if not raw:
            return None
        try:
            return float(raw.replace("d", "e").replace("D", "e"))
        except ValueError:
            return None

    for step, package in zip(steps, packages):
        if step.get("tool") != "dos_post" or not package.get("input_paths"):
            continue
        path = Path(package["input_paths"][0])
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?mis)^\s*&dos\b.*?^\s*/\s*$", text)
        if not match:
            continue
        body = match.group(0)
        emin, emax = number(body, "emin"), number(body, "emax")
        if emin is None or emax is None or emax - emin >= 5.0:
            continue
        replacements = {"emin": "-15.0", "emax": "10.0", "deltae": "0.01"}
        for key, value in replacements.items():
            line = f"  {key.capitalize() if key != 'deltae' else 'DeltaE'}={value},"
            existing = re.search(rf"(?mi)^\s*{key}\s*=\s*[^\n]+$", body)
            if existing:
                body = body[:existing.start()] + line + body[existing.end():]
            else:
                slash = body.rfind("/")
                body = body[:slash] + line + "\n" + body[slash:]
        text = text[:match.start()] + body + text[match.end():]
        path.write_text(text.rstrip() + "\n", encoding="utf-8")


def validate_generated_workflow(
    query: str,
    steps: Sequence[Dict[str, Any]],
    packages: Sequence[Dict[str, Any]],
) -> List[ValidationIssue]:
    issues = validate_plan(query, steps)
    values_by_step: List[Dict[str, str]] = []
    ph_values: List[Dict[str, str]] = []
    for step, package in zip(steps, packages):
        paths = package.get("input_paths", [])
        for path in paths:
            issues.extend(validate_qe_input(
                path,
                tool=str(step.get("tool") or ""),
                query=str(step.get("problem") or ""),
            ))
            vals = _input_values(path)
            values_by_step.append(vals)
            if package.get("exec_name") == "ph.x":
                ph_values.append(vals)

    if _requested(query, r"raman") and not any(v.get("lraman") in {".true.", "true"} for v in ph_values):
        issues.append(ValidationIssue("error", "RAMAN_NOT_IMPLEMENTED", "Raman was requested, but no ph.x input enables lraman=.true.."))

    all_input_text = "\n".join(
        Path(path).read_text(encoding="utf-8", errors="replace")
        for package in packages for path in package.get("input_paths", [])
    )
    if _requested(query, r"\bbulk\b") and re.search(r"assume_isolated|dipfield|tefield", all_input_text, re.I):
        issues.append(ValidationIssue("error", "BULK_BOUNDARY_MISMATCH", "A bulk workflow contains isolated/slab electric-boundary controls."))

    first_pw_text = ""
    for package in packages:
        if package.get("exec_name") == "pw.x" and package.get("input_paths"):
            first_pw_text = Path(package["input_paths"][0]).read_text(encoding="utf-8", errors="replace")
            break
    spin_requested = _requested(query, r"spin[- ]polarized|spin[- ]resolved|local magnetic moment|total magnetic moment")
    if spin_requested:
        nspin = _value(first_pw_text, "nspin")
        noncolin = _value(first_pw_text, "noncolin").lower()
        if nspin != "2" and noncolin not in {".true.", "true"}:
            issues.append(ValidationIssue("error", "SPIN_SETUP_MISSING", "Magnetic/spin-resolved results were requested, but the ground-state setup is not spin polarized."))
        if not re.search(r"(?mi)^\s*starting_magnetization\s*\(", first_pw_text):
            issues.append(ValidationIssue("error", "INITIAL_MOMENTS_MISSING", "The magnetic workflow has no explicit initial site/type magnetization."))

    decision_payload = "\n".join(str(package.get("params_json") or "") for package in packages).lower()
    if _requested(query, r"van der waals|\bvdw\b") and not (
        re.search(r"vdw_corr|london|dftd3", all_input_text, re.I)
        or "vdw" in decision_payload
        or "van der waals" in decision_payload
    ):
        issues.append(ValidationIssue("error", "VDW_DECISION_MISSING", "The request requires an explicit vdW decision, but none was recorded or implemented."))
    if _requested(query, r"dft\s*\+\s*u|hubbard|\bu settings?\b") and not (
        re.search(r"(?mi)^\s*HUBBARD\b|lda_plus_u|Hubbard_U", all_input_text)
        or "dft_plus_u" in decision_payload
        or "hubbard" in decision_payload
    ):
        issues.append(ValidationIssue("error", "HUBBARD_DECISION_MISSING", "The request requires an explicit DFT+U decision, but none was recorded or implemented."))

    # Dependent pw.x steps must not silently change the shared physical model.
    pw_entries: List[tuple[str, str, str, Dict[str, str]]] = []
    for step, package in zip(steps, packages):
        if package.get("exec_name") == "pw.x" and package.get("input_paths"):
            path = package["input_paths"][0]
            model = str(package.get("pseudo_dir") or "default")
            pw_entries.append((str(step.get("tool")), path, model, _input_values(path)))
    if pw_entries:
        baseline = pw_entries[0][3]
        baseline_vdw = _vdw_settings(pw_entries[0][1])
        for tool, path, _model, vals in pw_entries[1:]:
            for key in ("input_dft", "ecutwfc", "ecutrho", "vdw_corr", "london"):
                if baseline.get(key) and vals.get(key) and baseline[key] != vals[key]:
                    issues.append(ValidationIssue("error", "SHARED_SETTING_MISMATCH", f"{tool} changes {key} from {baseline[key]!r} to {vals[key]!r}."))
            current_vdw = _vdw_settings(path)
            if baseline_vdw != current_vdw:
                issues.append(ValidationIssue(
                    "error", "VDW_MODEL_MISMATCH",
                    f"{tool} changes the vdW model from {baseline_vdw!r} to {current_vdw!r}.", path,
                ))
        for model in {entry[2] for entry in pw_entries}:
            branch_entries = [entry for entry in pw_entries if entry[2] == model]
            branch_baseline = branch_entries[0][3]
            baseline_species = _species_signature(branch_entries[0][1])
            for tool, path, _model, vals in branch_entries[1:]:
                for key in ("nspin", "noncolin", "lspinorb"):
                    if branch_baseline.get(key) and vals.get(key) and branch_baseline[key] != vals[key]:
                        issues.append(ValidationIssue(
                            "error", "BRANCH_PHYSICAL_MODEL_MISMATCH",
                            f"{tool} changes {key} within pseudopotential branch {model!r}.", path,
                        ))
                current_species = _species_signature(path)
                if baseline_species and current_species and current_species != baseline_species:
                    issues.append(ValidationIssue(
                        "error", "SPECIES_PSEUDO_MISMATCH",
                        "A dependent pw.x step changes species labels or pseudopotential files within one branch.", path,
                    ))
            geometry_ground_state = [
                entry for entry in branch_entries
                if entry[0] in {"pw_relax", "pw_vc_relax", "pw_scf"}
            ]
            if len(geometry_ground_state) > 1:
                first_kind, first_rows = _card_rows(
                    Path(geometry_ground_state[0][1]).read_text(encoding="utf-8"), "K_POINTS"
                )
                for tool, path, _model, _vals in geometry_ground_state[1:]:
                    kind, rows = _card_rows(Path(path).read_text(encoding="utf-8"), "K_POINTS")
                    if (kind, rows) != (first_kind, first_rows):
                        issues.append(ValidationIssue(
                            "error", "GROUND_STATE_KPOINT_MISMATCH",
                            f"{tool} changes the relaxation/SCF k-point grid within one physical branch.", path,
                        ))

    dos_source = next((
        _input_values(package["input_paths"][0])
        for step, package in zip(steps, packages)
        if step.get("tool") == "pw_nscf" and package.get("input_paths")
    ), {})
    dos_post = next((
        _input_values(package["input_paths"][0])
        for step, package in zip(steps, packages)
        if step.get("tool") == "dos_post" and package.get("input_paths")
    ), {})
    if dos_source and dos_post:
        occupation = dos_source.get("occupations", "")
        bz_sum = dos_post.get("bz_sum", "")
        if "tetrahedra" in occupation and bz_sum and "tetrahedra" not in bz_sum:
            issues.append(ValidationIssue("error", "DOS_INTEGRATION_MISMATCH", "The DOS NSCF uses tetrahedra, but dos.x requests a different integration method."))

    # Validate the explicit phonon filename chain.
    ph_fildyn = next((v["fildyn"] for v in ph_values if v.get("ldisp") in {".true.", "true"} and v.get("fildyn")), "")
    q2r = next((_input_values(p["input_paths"][0]) for s, p in zip(steps, packages) if s.get("tool") == "q2r_post" and p.get("input_paths")), {})
    matdyn = next((_input_values(p["input_paths"][0]) for s, p in zip(steps, packages) if s.get("tool") == "matdyn_post" and p.get("input_paths")), {})
    if ph_fildyn and q2r.get("fildyn") and ph_fildyn != q2r["fildyn"]:
        issues.append(ValidationIssue("error", "PH_Q2R_FILENAME_MISMATCH", "q2r.x does not read the dynamical-matrix series written by ph.x."))
    if q2r.get("flfrc") and matdyn.get("flfrc") and q2r["flfrc"] != matdyn["flfrc"]:
        issues.append(ValidationIssue("error", "Q2R_MATDYN_FILENAME_MISMATCH", "matdyn.x does not read the force constants written by q2r.x."))
    gamma_raman_fildyn = next((v["fildyn"] for v in ph_values if v.get("lraman") in {".true.", "true"} and v.get("fildyn")), "")
    dynmat = next((_input_values(p["input_paths"][0]) for s, p in zip(steps, packages) if s.get("tool") == "dynmat_post" and p.get("input_paths")), {})
    if gamma_raman_fildyn and dynmat.get("fildyn") and gamma_raman_fildyn != dynmat["fildyn"]:
        issues.append(ValidationIssue("error", "RAMAN_DYNMAT_FILENAME_MISMATCH", "dynmat.x does not read the Gamma dynamical matrix written by the Raman ph.x step."))

    for producer_tool, consumer_tools in (
        ("pw_bands", ("bands_post",)),
        ("pw_nscf", ("dos_post", "projwfc_post")),
    ):
        producer = next((_input_values(p["input_paths"][0]) for s, p in zip(steps, packages) if s.get("tool") == producer_tool and p.get("input_paths")), {})
        for consumer_tool in consumer_tools:
            consumer = next((_input_values(p["input_paths"][0]) for s, p in zip(steps, packages) if s.get("tool") == consumer_tool and p.get("input_paths")), {})
            if producer and consumer:
                for key in ("prefix", "outdir"):
                    if producer.get(key) and consumer.get(key) and producer[key] != consumer[key]:
                        issues.append(ValidationIssue("error", "ELECTRONIC_STATE_MISMATCH", f"{consumer_tool} {key} does not match its {producer_tool} producer."))
    return issues


def validate_qe_output(path: str, *, exec_name: str, require_complete: bool = True) -> List[ValidationIssue]:
    output = Path(path)
    if not output.is_file() or output.stat().st_size == 0:
        return [ValidationIssue("fatal", "OUTPUT_MISSING", "Expected output file is missing or empty.", path)]
    text = output.read_text(encoding="utf-8", errors="replace")
    issues: List[ValidationIssue] = []
    if re.search(r"Error in routine|error while reading|MPI_ABORT|%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", text, re.I):
        issues.append(ValidationIssue("fatal", "QE_RUNTIME_ERROR", "QE reported a parser or runtime error.", path))
    if require_complete and exec_name in {"pw.x", "ph.x"} and "JOB DONE." not in text:
        issues.append(ValidationIssue("error", "QE_INCOMPLETE", f"{exec_name} output has no JOB DONE marker.", path))
    if require_complete and exec_name == "pw.x" and re.search(r"calculation\s*=\s*['\"]?scf", text, re.I) and "convergence has been achieved" not in text:
        issues.append(ValidationIssue("error", "SCF_NOT_CONVERGED", "SCF output does not report convergence.", path))
    if exec_name == "ph.x":
        gamma = re.search(
            r"(?is)q\s*=\s*\(\s*0\.0+\s+0\.0+\s+0\.0+\s*\)(.*?)(?=Calculation of q|\Z)",
            text,
        )
        if gamma:
            frequencies = [
                float(value.replace("D", "E").replace("d", "e"))
                for value in re.findall(r"freq\s*\(\s*\d+\s*\).*?=\s*([-+0-9.EeDd]+)\s*\[cm-1\]", gamma.group(1))
            ]
            if len(frequencies) >= 3 and max(abs(value) for value in frequencies[:3]) > 30.0:
                issues.append(ValidationIssue(
                    "error",
                    "GAMMA_ACOUSTIC_MODES_INVALID",
                    "The three Gamma translational modes are not near zero; the dynamical matrix is not safe for q2r/matdyn.",
                    path,
                ))
    return issues
