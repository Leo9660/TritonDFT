import datetime
import json
import math
import re
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator import UnifiedGenerator
from prompt import get_prompt
from tool import fetch_material_info_from_api_snippet
from utils import extract_json_brutal, output_to_log_file
from execute_code.slurm_template import render_slurm_script


VASP_FILE_NAMES = ("POSCAR", "INCAR", "KPOINTS")


@dataclass
class VASPInputSet:
    step_index: int
    title: str
    task: str
    directory: Path
    files: List[str]
    species: List[str]
    output_path: str
    potcar_mode: str = "local"
    slurm_path: str = ""


@dataclass
class VASPRunSettings:
    potcar_functional: str
    vasp_command: str
    reason: str


def _sanitize_name(name: str, max_len: int = 40) -> str:
    name = re.sub(r"[^\w\-]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:max_len] if name else ""


def _extract_query_metadata(query: str) -> Dict[str, str]:
    material = ""
    match = re.search(r"material\s*=\s*([A-Za-z][A-Za-z0-9]*)", query)
    if match:
        material = match.group(1)
    else:
        match = re.search(
            r"\bfor\s+(?:[\w-]+\s+)?([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b",
            query,
        )
        if match:
            material = match.group(1)

    task_patterns = [
        (r"\bvc[_-]relax\b|variable[_-]cell\s+relax", "vc-relax"),
        (r"\bnscf\b|non[_-]self[_-]consistent", "nscf"),
        (r"\bscf\b|self[_-]consistent\s+field", "scf"),
        (r"\brelax(?:ation)?\b", "relax"),
        (r"\bband[\s_-]?(?:structure|gap|calculation)", "bands"),
        (r"\bdos\b|density\s+of\s+states", "dos"),
    ]
    tasks: List[str] = []
    for pattern, task in task_patterns:
        if re.search(pattern, query, re.IGNORECASE) and task not in tasks:
            tasks.append(task)
    if "vc-relax" in tasks and "relax" in tasks:
        tasks.remove("relax")
    return {"material_name": material, "task_type": "+".join(tasks)}


def _normalize_potcar_functional(value: str) -> str:
    normalized = (value or "").strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "PBE": "PBE",
        "PBESOL": "PBEsol",
        "PBE0": "PBE",
        "HSE": "PBE",
        "HSE03": "PBE",
        "HSE06": "PBE",
        "HYBRID": "PBE",
        "LDA": "LDA",
        "PW91": "PW91",
        "GGA": "PW91",
    }
    return aliases.get(normalized, "")


def _infer_potcar_functional(query: str) -> str:
    text = (query or "").lower()
    if re.search(r"\bpbe\s*sol\b|\bpbesol\b", text):
        return "PBEsol"
    if re.search(r"\blda\b|local density approximation", text):
        return "LDA"
    if re.search(r"\bpw91\b", text):
        return "PW91"
    if re.search(r"\bhse(?:03|06)?\b|\bhybrid\b|\bpbe0\b", text):
        return "PBE"
    if re.search(r"\bpbe\b", text):
        return "PBE"
    return ""


def _normalize_vasp_command(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    lower = value.lower()
    if re.fullmatch(r"vasp_(?:std|gam|ncl)", lower):
        return lower
    if lower in {"std", "standard"}:
        return "vasp_std"
    if lower in {"gam", "gamma", "gamma-only", "gamma_only"}:
        return "vasp_gam"
    if lower in {"ncl", "noncollinear", "non-collinear", "soc", "spin-orbit"}:
        return "vasp_ncl"
    return value


def _infer_vasp_command(query: str, steps: List[Dict[str, Any]]) -> str:
    text = (query or "").lower()
    if re.search(r"\bsoc\b|spin[-\s]?orbit|non[-\s]?collinear|\blsorbit\b|\bncl\b", text):
        return "vasp_ncl"
    if re.search(r"gamma[-\s]?only|gamma\s+point\s+only|\bvasp_gam\b", text):
        return "vasp_gam"
    if re.search(r"\bvasp_std\b|\bstandard\s+vasp\b", text):
        return "vasp_std"
    for step in steps:
        task = str(step.get("task") or "").lower()
        if task in {"bands", "dos", "nscf"}:
            return "vasp_std"
    return ""


def _generate_nonempty_text(
    generator,
    prompt: str,
    *,
    max_new_tokens: int,
    attempts: int = 3,
    verbose: bool = False,
    purpose: str = "vasp_generation",
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            result = generator(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
        except Exception as exc:
            last_error = exc
            if verbose:
                print(f"[{purpose}] model call {attempt}/{attempts} failed: {exc}")
        else:
            text = ""
            if result and isinstance(result, list):
                text = result[0].get("generated_text", "") or ""
            if text.strip():
                return text
            if verbose:
                print(f"[{purpose}] model returned an empty response; retrying.")
    if last_error:
        raise RuntimeError(f"{purpose} failed after {attempts} attempts: {last_error}") from last_error
    raise RuntimeError(f"{purpose} returned empty text after {attempts} attempts.")


def _parse_json_object(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    data = extract_json_brutal(text)
    if isinstance(data, dict):
        return data
    match = re.search(r"(?s)\{.*\}", text)
    if match:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    raise ValueError("The model did not return a JSON object.")


def _vasp_plan_prompt(query: str) -> str:
    return f"""You are a senior VASP workflow planner.

Plan a minimal VASP workflow for the user's DFT request. Use only these task labels:
relax, vc-relax, scf, nscf, bands, dos, static.

Rules:
- Prefer a small, executable sequence.
- Use relax or vc-relax before dependent static/scf/bands/dos steps when relaxation is requested.
- For a simple energy or electronic-structure request, include scf/static as appropriate.
- Do not include Quantum ESPRESSO tools or file names.

Return only JSON with this schema:
{{
  "settings": {{
    "potcar_functional": "<PBE|PBEsol|LDA|PW91 or null when not specified>",
    "vasp_command": "<vasp_std|vasp_gam|vasp_ncl or null when not specified>",
    "reason": "<brief reason for these choices>"
  }},
  "steps": [
    {{"title": "<short step title>", "task": "<task label>", "why": "<short reason>"}}
  ]
}}

User request:
{query}
"""


def _vasp_input_prompt(
    *,
    query: str,
    step: Dict[str, Any],
    step_index: int,
    total_steps: int,
    material_info: Dict[str, Any],
    previous_context: str,
    potcar_root: str,
    functional: str,
    vasp_command: str,
) -> str:
    primitive = material_info.get("primitive_structure", "")
    conventional = material_info.get("conventional_structure", "")
    initial = material_info.get("initial_structures", "")
    return f"""You are a computational materials scientist preparing VASP inputs.

Generate input files for only the current VASP step.

VASP facts to obey:
- POSCAR contains the lattice vectors, species names, counts, and ion positions.
- The POSCAR species order must be the same order used to concatenate POTCAR.
- INCAR is tag = value format. Use VASP tags, not Quantum ESPRESSO namelists.
- KPOINTS controls the k-point sampling.
- POTCAR is proprietary and will be assembled by TritonDFT from the user's licensed potential tree. Do not output POTCAR content.
- Use ENCUT in eV. If you do not know the exact POTCAR ENMAX values, set a conservative placeholder ENCUT.
- Use VASP-appropriate tags such as SYSTEM, ENCUT, EDIFF, ISMEAR, SIGMA, IBRION, NSW, ISIF, EDIFFG, LCHARG, LWAVE, NELM, LORBIT, ICHARG, NBANDS, KPAR, NCORE only when relevant.
- Do not write QE tags such as ecutwfc, ecutrho, prefix, outdir, ATOMIC_SPECIES, ATOMIC_POSITIONS, or K_POINTS.

POTCAR setup:
- Functional requested: {functional}
- Licensed POTCAR root, local or cluster path: {potcar_root}

Executable selection:
- VASP executable selected for this workflow: {vasp_command}
- Use INCAR settings consistent with that executable. For spin-orbit or noncollinear calculations,
  include the necessary VASP tags such as LSORBIT, LNONCOLLINEAR, SAXIS, MAGMOM, or related tags
  only when they are relevant to the user's request.

Workflow context:
- Current step {step_index} of {total_steps}: {step}
- Earlier workflow context: {previous_context or "None"}

Available structure data from Materials Project or user query:
Primitive structure:
{primitive}

Conventional structure:
{conventional}

Initial structures:
{initial}

User request:
{query}

Return only JSON with this schema:
{{
  "POSCAR": "<complete POSCAR text>",
  "INCAR": "<complete INCAR text>",
  "KPOINTS": "<complete KPOINTS text>",
  "notes": "<brief internal note about assumptions>"
}}
"""


def _species_from_poscar(poscar_text: str) -> List[str]:
    lines = [line.strip() for line in poscar_text.splitlines() if line.strip()]
    if len(lines) < 7:
        return []
    species_line = lines[5].split()
    count_line = lines[6].split()
    if species_line and all(re.fullmatch(r"[A-Za-z][a-z]?", token) for token in species_line):
        if count_line and all(re.fullmatch(r"\d+", token) for token in count_line):
            return species_line
    return []


def _potential_root_candidates(root: Path, functional: str) -> List[Path]:
    functional = functional.upper()
    names = {
        "PBE": ["potpaw_PBE", "PBE", "pbe"],
        "PBESOL": ["potpaw_PBEsol", "PBEsol", "pbesol", "PBESOL"],
        "LDA": ["potpaw", "LDA", "lda"],
        "PW91": ["potpaw_GGA", "PW91", "pw91"],
    }.get(functional, [functional, functional.lower()])
    candidates = [root]
    candidates.extend(root / name for name in names)
    return candidates


def _find_potcar_for_species(root: Path, species: str, functional: str) -> Path:
    variants = [
        species,
        f"{species}_pv",
        f"{species}_sv",
        f"{species}_GW",
    ]
    for base in _potential_root_candidates(root, functional):
        for variant in variants:
            candidate = base / variant / "POTCAR"
            if candidate.exists():
                return candidate
    matches: List[Path] = []
    for base in _potential_root_candidates(root, functional):
        if base.exists():
            matches.extend(base.glob(f"{species}*/POTCAR"))
    if matches:
        return sorted(matches, key=lambda p: (len(p.parent.name), p.parent.name))[0]
    raise FileNotFoundError(
        f"Could not find a POTCAR for species '{species}' under {root} "
        f"using functional {functional}."
    )


def _extract_enmax(potcar_text: str) -> List[float]:
    values: List[float] = []
    for match in re.finditer(r"ENMAX\s*=\s*([0-9.]+)", potcar_text, re.IGNORECASE):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            pass
    return values


def _set_or_add_incar_tag(incar_text: str, tag: str, value: str) -> str:
    pattern = re.compile(rf"(?mi)^(\s*{re.escape(tag)}\s*=\s*).*$")
    if pattern.search(incar_text):
        return pattern.sub(rf"\g<1>{value}", incar_text)
    return incar_text.rstrip() + f"\n{tag} = {value}\n"


def _assemble_potcar(
    *,
    poscar_path: Path,
    incar_path: Path,
    destination: Path,
    potcar_root: str,
    functional: str,
) -> List[str]:
    root = Path(potcar_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(
            f"VASP POTCAR root does not exist: {root}. Set CLUSTER_AGENT_VASP_POTCAR_ROOT."
        )
    species = _species_from_poscar(poscar_path.read_text(encoding="utf-8"))
    if not species:
        raise ValueError(f"Could not read species order from {poscar_path}.")

    potcar_parts: List[str] = []
    source_paths: List[str] = []
    for element in species:
        src = _find_potcar_for_species(root, element, functional)
        potcar_parts.append(src.read_text(encoding="utf-8", errors="replace").rstrip())
        source_paths.append(str(src))
    potcar_text = "\n".join(potcar_parts).rstrip() + "\n"
    destination.write_text(potcar_text, encoding="utf-8")

    enmax_values = _extract_enmax(potcar_text)
    if enmax_values:
        recommended = int(math.ceil(max(enmax_values) / 10.0) * 10)
        incar_text = incar_path.read_text(encoding="utf-8")
        match = re.search(r"(?mi)^\s*ENCUT\s*=\s*([0-9.]+)", incar_text)
        current = float(match.group(1)) if match else 0.0
        if current < recommended:
            incar_path.write_text(
                _set_or_add_incar_tag(incar_text, "ENCUT", str(recommended)).rstrip() + "\n",
                encoding="utf-8",
            )
    return source_paths


def _functional_root_names(functional: str) -> List[str]:
    functional = functional.upper()
    return {
        "PBE": ["potpaw_PBE", "PBE", "pbe"],
        "PBESOL": ["potpaw_PBEsol", "PBEsol", "pbesol", "PBESOL"],
        "LDA": ["potpaw", "LDA", "lda"],
        "PW91": ["potpaw_GGA", "PW91", "pw91"],
    }.get(functional, [functional, functional.lower()])


def _write_remote_potcar_assembler(
    *,
    destination: Path,
    species: List[str],
    potcar_root: str,
    functional: str,
) -> str:
    root_names = " ".join(shlex.quote(name) for name in _functional_root_names(functional))
    species_names = " ".join(shlex.quote(name) for name in species)
    script = f"""#!/bin/bash
set -euo pipefail

POTCAR_ROOT={shlex.quote(potcar_root)}
FUNCTIONAL={shlex.quote(functional)}
SPECIES=({species_names})
ROOT_NAMES=({root_names})

rm -f POTCAR
for element in "${{SPECIES[@]}}"; do
  found=""
  BASES=("$POTCAR_ROOT")
  for root_name in "${{ROOT_NAMES[@]}}"; do
    BASES+=("$POTCAR_ROOT/$root_name")
  done
  for base in "${{BASES[@]}}"; do
    for variant in "$element" "${{element}}_pv" "${{element}}_sv" "${{element}}_GW"; do
      candidate="$base/$variant/POTCAR"
      if [ -f "$candidate" ]; then
        found="$candidate"
        break 2
      fi
    done
    for candidate in "$base"/"$element"*/POTCAR; do
      if [ -f "$candidate" ]; then
        found="$candidate"
        break 2
      fi
    done
  done
  if [ -z "$found" ]; then
    echo "Could not find POTCAR for $element under $POTCAR_ROOT using $FUNCTIONAL" >&2
    exit 1
  fi
  echo "$found" >> POTCAR.sources
  cat "$found" >> POTCAR
done
"""
    destination.write_text(script, encoding="utf-8")
    destination.chmod(0o755)
    return str(destination)


def _validation_errors(input_set: VASPInputSet) -> List[str]:
    errors: List[str] = []
    required = ["POSCAR", "INCAR", "KPOINTS"]
    if input_set.potcar_mode == "local":
        required.append("POTCAR")
    else:
        required.append("assemble_potcar.sh")
    for name in required:
        path = input_set.directory / name
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{name} is missing or empty")
    poscar = input_set.directory / "POSCAR"
    if poscar.exists() and not _species_from_poscar(poscar.read_text(encoding="utf-8")):
        errors.append("POSCAR does not contain a readable species/count block")
    incar = input_set.directory / "INCAR"
    if incar.exists():
        incar_text = incar.read_text(encoding="utf-8")
        for bad in ("&control", "&system", "ecutwfc", "ATOMIC_SPECIES", "K_POINTS"):
            if re.search(rf"(?i){re.escape(bad)}", incar_text):
                errors.append(f"INCAR appears to contain Quantum ESPRESSO syntax: {bad}")
        if not re.search(r"(?mi)^\s*ENCUT\s*=", incar_text):
            errors.append("INCAR is missing ENCUT")
    return errors


def _approve_vasp_inputs_popup(plan: str, input_sets: List[VASPInputSet]) -> bool:
    flat_files: List[Path] = []
    for item in input_sets:
        flat_files.extend(item.directory / name for name in ("POSCAR", "INCAR", "KPOINTS"))
        if item.potcar_mode == "remote":
            flat_files.append(item.directory / "assemble_potcar.sh")

    print("\n[approval] All VASP inputs are ready.")
    print("[approval] Opening the TritonDFT approval window; execution is paused until approval.")
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk

        root = tk.Tk()
        root.title("TritonDFT VASP plan and input approval")
        root.geometry("1100x760")
        root.update_idletasks()
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()

        def release_topmost() -> None:
            try:
                root.attributes("-topmost", False)
                root.lift()
                root.focus_force()
            except tk.TclError:
                pass

        root.after(750, release_topmost)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        plan_text = tk.Text(notebook, wrap="word", font=("Menlo", 12))
        plan_text.insert("1.0", plan)
        plan_text.configure(state="disabled")
        notebook.add(plan_text, text="Complete plan")

        editors: Dict[Path, Any] = {}
        for path in flat_files:
            editor = tk.Text(notebook, wrap="none", undo=True, font=("Menlo", 11))
            editor.insert("1.0", path.read_text(encoding="utf-8"))
            notebook.add(editor, text=f"{path.parent.name}/{path.name}")
            editors[path] = editor

        validation_text = tk.Text(notebook, wrap="word", font=("Menlo", 11))
        validation_text.configure(state="disabled")
        notebook.add(validation_text, text="Validation")

        decision = {"approved": False}

        def save_edits() -> None:
            for path, editor in editors.items():
                path.write_text(editor.get("1.0", "end-1c").rstrip() + "\n", encoding="utf-8")

        def refresh_validation() -> List[str]:
            save_edits()
            messages: List[str] = []
            for item in input_sets:
                errors = _validation_errors(item)
                if errors:
                    messages.append(f"{item.directory.name}:\n  - " + "\n  - ".join(errors))
            report = "\n\n".join(messages) if messages else "Basic VASP input validation passed."
            validation_text.configure(state="normal")
            validation_text.delete("1.0", "end")
            validation_text.insert("1.0", report)
            validation_text.configure(state="disabled")
            return messages

        def close_window() -> None:
            try:
                root.withdraw()
                root.update_idletasks()
                root.destroy()
            except tk.TclError:
                pass

        def approve() -> None:
            messages = refresh_validation()
            if messages:
                notebook.select(validation_text)
                messagebox.showerror("Input validation failed", "Fix the listed issues before approval.")
                return
            decision["approved"] = True
            root.after_idle(close_window)

        def cancel() -> None:
            if messagebox.askyesno("Cancel workflow", "Cancel without submitting any cluster jobs?"):
                root.after_idle(close_window)

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(controls, text="Review or edit generated VASP input files before cluster submission.").pack(side="left")
        ttk.Button(controls, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
        ttk.Button(controls, text="Approve & Run", command=approve).pack(side="right")
        ttk.Button(controls, text="Validate", command=refresh_validation).pack(side="right", padx=(0, 8))
        root.protocol("WM_DELETE_WINDOW", cancel)
        refresh_validation()
        root.mainloop()
        return decision["approved"]
    except Exception as exc:
        print(f"[approval] GUI unavailable ({exc}). Falling back to terminal approval.")
        print("\n" + plan)
        for path in flat_files:
            print(f"  - {path}")
        while True:
            answer = input("Type 'approve' to run, 'edit' after editing files, or 'cancel': ").strip().lower()
            if answer in {"approve", "yes", "y"}:
                return True
            if answer in {"cancel", "no", "n"}:
                return False
            if answer == "edit":
                print("Edit the files in your editor, then return here to approve or cancel.")


class VASPAgent:
    def __init__(
        self,
        *,
        model: str,
        backend: str,
        work_dir: str,
        potcar_root: str,
        vasp_command: str = "",
        functional: str = "",
        slurm_template_path: str = "",
        max_new_tokens: int = 4096,
        verbose: bool = False,
        need_query_info: bool = False,
        output_log: bool = False,
        output_log_file: str = "remote_cluster_agent.log",
    ):
        self.model = model
        self.backend = backend
        self.work_dir_root = Path(work_dir).expanduser().resolve()
        self.work_dir_root.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.work_dir_root
        self.potcar_root = potcar_root
        self.default_vasp_command = vasp_command.strip()
        self.default_functional = functional.strip().upper()
        self.vasp_command = ""
        self.functional = ""
        self.slurm_template_path = slurm_template_path
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.need_query_info = need_query_info
        self.output_log = output_log
        self.output_log_file = output_log_file
        self.generator = UnifiedGenerator(
            backend=backend,
            model=model,
            default_max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=1234,
            verbose=verbose,
        )

    def _prepare_run_directory(
        self,
        query: str,
        *,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
    ) -> Path:
        if query and (not material_name or not task_type):
            extracted = _extract_query_metadata(query)
            material_name = material_name or extracted["material_name"]
            task_type = task_type or extracted["task_type"]
        now = datetime.datetime.now()
        date_dir = self.work_dir_root / now.strftime("%Y-%m-%d")
        parts = [_sanitize_name(p) for p in (material_name, task_type) if p]
        if not parts:
            parts.append("vasp_run")
        parts.extend([now.strftime("%H%M%S"), uuid.uuid4().hex[:8]])
        self.work_dir = date_dir / "_".join(parts)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "run_id": run_id,
            "category": category,
            "query": query,
            "model": self.model,
            "dft_tool": "vasp",
            "created_at": now.isoformat(),
            "directory": str(self.work_dir),
            "potcar_root": self.potcar_root,
            "default_potcar_functional": self.default_functional,
            "default_vasp_command": self.default_vasp_command,
        }
        (self.work_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return self.work_dir

    def info_query(self, query: str) -> Dict[str, Any]:
        if not self.need_query_info:
            return {}
        api_messages = get_prompt(prompt_type="api_call", query=query)
        api_out = self.generator(
            api_messages[0]["content"],
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        snippet = api_out[0]["generated_text"]
        info = fetch_material_info_from_api_snippet(snippet, limit=25, verbose=self.verbose)
        if self.output_log:
            output_to_log_file(self.work_dir_root, self.output_log_file, f"[vasp info_query] {info.get('material_ids', ['N/A'])[0]}")
        return info

    def plan(self, query: str) -> List[Dict[str, Any]]:
        text = _generate_nonempty_text(
            self.generator,
            _vasp_plan_prompt(query),
            max_new_tokens=self.max_new_tokens,
            attempts=3,
            verbose=self.verbose,
            purpose="vasp_plan",
        )
        data = _parse_json_object(text)
        steps = data.get("steps", [])
        if not isinstance(steps, list) or not steps:
            steps = [{"title": "Static VASP calculation", "task": "static", "why": "Default single-step VASP calculation."}]
        settings = data.get("settings", {})
        valid_steps = [step for step in steps if isinstance(step, dict)]
        for step in valid_steps:
            step.setdefault("_workflow_settings", settings if isinstance(settings, dict) else {})
        return valid_steps

    def _select_run_settings(self, query: str, steps: List[Dict[str, Any]]) -> VASPRunSettings:
        settings: Dict[str, Any] = {}
        if steps and isinstance(steps[0].get("_workflow_settings"), dict):
            settings = steps[0].get("_workflow_settings", {})

        requested_functional = str(settings.get("potcar_functional") or "").strip()
        requested_command = str(settings.get("vasp_command") or "").strip()

        functional = (
            _infer_potcar_functional(query)
            or _normalize_potcar_functional(requested_functional)
            or _normalize_potcar_functional(self.default_functional)
            or "PBE"
        )
        command = (
            _infer_vasp_command(query, steps)
            or _normalize_vasp_command(requested_command)
            or _normalize_vasp_command(self.default_vasp_command)
            or "vasp_std"
        )
        reason = str(settings.get("reason") or "").strip()
        if not reason:
            reason = (
                f"Selected {functional} POTCARs and {command} from the user request, "
                "with configured values used only as fallbacks."
            )
        return VASPRunSettings(potcar_functional=functional, vasp_command=command, reason=reason)

    def generate_inputs(
        self,
        query: str,
        *,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
    ) -> Dict[str, Any]:
        self._prepare_run_directory(
            query,
            run_id=run_id,
            category=category,
            task_type=task_type,
            material_name=material_name,
        )
        material_info = self.info_query(query)
        steps = self.plan(query)
        run_settings = self._select_run_settings(query, steps)
        self.functional = run_settings.potcar_functional
        self.vasp_command = run_settings.vasp_command
        self._update_run_meta(
            {
                "potcar_functional": self.functional,
                "vasp_command": self.vasp_command,
                "vasp_settings_reason": run_settings.reason,
            }
        )
        plan_text = self._plan_text(steps)
        (self.work_dir / "workflow_plan.txt").write_text(plan_text, encoding="utf-8")
        (self.work_dir / "workflow_plan.json").write_text(json.dumps(steps, indent=2) + "\n", encoding="utf-8")

        previous_context = ""
        input_sets: List[VASPInputSet] = []
        for index, step in enumerate(steps, start=1):
            print(f"[vasp-agent] generating input set {index}/{len(steps)}: {step.get('title') or step.get('task')}")
            prompt = _vasp_input_prompt(
                query=query,
                step=step,
                step_index=index,
                total_steps=len(steps),
                material_info=material_info,
                previous_context=previous_context,
                potcar_root=self.potcar_root,
                functional=self.functional,
                vasp_command=self.vasp_command,
            )
            generated = _generate_nonempty_text(
                self.generator,
                prompt,
                max_new_tokens=max(self.max_new_tokens, 8192),
                attempts=3,
                verbose=self.verbose,
                purpose=f"vasp_input_{index}",
            )
            data = _parse_json_object(generated)
            step_dir = self.work_dir / f"step_{index:02d}_{_sanitize_name(step.get('task', 'vasp'), 24)}"
            step_dir.mkdir(parents=True, exist_ok=True)
            files: List[str] = []
            for name in VASP_FILE_NAMES:
                content = str(data.get(name, "")).strip()
                if not content:
                    raise ValueError(f"VASP generation omitted {name} for step {index}.")
                path = step_dir / name
                path.write_text(content.rstrip() + "\n", encoding="utf-8")
                files.append(str(path))
            species = _species_from_poscar((step_dir / "POSCAR").read_text(encoding="utf-8"))
            potcar_root_path = Path(self.potcar_root).expanduser()
            if potcar_root_path.exists():
                potcar_mode = "local"
                potcar_sources = _assemble_potcar(
                    poscar_path=step_dir / "POSCAR",
                    incar_path=step_dir / "INCAR",
                    destination=step_dir / "POTCAR",
                    potcar_root=self.potcar_root,
                    functional=self.functional,
                )
                (step_dir / "POTCAR.sources.json").write_text(
                    json.dumps(
                        {
                            "mode": "local",
                            "sources": potcar_sources,
                            "functional": self.functional,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                files.append(str(step_dir / "POTCAR"))
            else:
                potcar_mode = "remote"
                assemble_path = _write_remote_potcar_assembler(
                    destination=step_dir / "assemble_potcar.sh",
                    species=species,
                    potcar_root=self.potcar_root,
                    functional=self.functional,
                )
                (step_dir / "POTCAR.sources.json").write_text(
                    json.dumps(
                        {
                            "mode": "remote",
                            "remote_root": self.potcar_root,
                            "functional": self.functional,
                            "species": species,
                            "assembler": "assemble_potcar.sh",
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                files.append(assemble_path)
            input_set = VASPInputSet(
                step_index=index,
                title=str(step.get("title") or f"VASP step {index}"),
                task=str(step.get("task") or "static"),
                directory=step_dir,
                files=files,
                species=species,
                output_path=str(step_dir / "vasp.out"),
                potcar_mode=potcar_mode,
            )
            errors = _validation_errors(input_set)
            if errors:
                raise ValueError(f"Generated VASP input set failed validation: {'; '.join(errors)}")
            input_sets.append(input_set)
            previous_context += (
                f"\nStep {index}: {input_set.title} ({input_set.task}); "
                f"species order {' '.join(species)}; directory {step_dir.name}."
            )

        manifest = {
            "query": query,
            "plan": steps,
            "vasp_settings": {
                "potcar_functional": self.functional,
                "vasp_command": self.vasp_command,
                "reason": run_settings.reason,
            },
            "input_sets": [
                {
                    "step": item.step_index,
                    "title": item.title,
                    "task": item.task,
                    "directory": str(item.directory),
                    "files": item.files,
                    "species": item.species,
                    "potcar_mode": item.potcar_mode,
                    "output_path": item.output_path,
                }
                for item in input_sets
            ],
            "status": "awaiting_approval",
        }
        (self.work_dir / "approval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"plan_text": plan_text, "plan": steps, "input_sets": input_sets, "manifest": manifest}

    def _update_run_meta(self, values: Dict[str, Any]) -> None:
        path = self.work_dir / "run_meta.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            data = {}
        data.update(values)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _plan_text(steps: List[Dict[str, Any]]) -> str:
        lines = ["TritonDFT VASP execution plan", "============================="]
        for idx, step in enumerate(steps, start=1):
            lines.extend(
                [
                    "",
                    f"{idx}. {step.get('title') or step.get('task') or 'VASP step'}",
                    f"   Task: {step.get('task') or 'static'}",
                    f"   Why: {step.get('why') or 'This step contributes to the requested VASP workflow.'}",
                ]
            )
        lines.extend(
            [
                "",
                "Approval gate",
                "-------------",
                "POSCAR, KPOINTS, and INCAR are generated before execution.",
                "POTCAR is either assembled locally when available or assembled on the cluster before VASP runs.",
                "No cluster job is submitted until you approve the files.",
            ]
        )
        return "\n".join(lines) + "\n"


class RemoteClusterVASPAgent:
    def __init__(
        self,
        *,
        vasp_agent: VASPAgent,
        transport,
        approval_callback=None,
        parallel_np: int = 1,
        vasp_command: str = "",
        slurm_template_path: str = "",
    ):
        self.agent = vasp_agent
        self.transport = transport
        self.approval_callback = approval_callback or _approve_vasp_inputs_popup
        self.parallel_np = max(1, parallel_np or 1)
        self.default_vasp_command = vasp_command.strip()
        self.slurm_template_path = slurm_template_path or vasp_agent.slurm_template_path

    def run(
        self,
        query: str,
        *,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
    ) -> Dict[str, Any]:
        generated = self.agent.generate_inputs(
            query,
            run_id=run_id,
            category=category,
            task_type=task_type,
            material_name=material_name,
        )
        plan_text = generated["plan_text"]
        input_sets: List[VASPInputSet] = generated["input_sets"]
        print("\n" + plan_text)

        manifest_path = self.agent.work_dir / "approval_manifest.json"
        manifest = generated["manifest"]
        if not self.approval_callback(plan_text, input_sets):
            manifest["status"] = "cancelled"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return {
                "status": "cancelled",
                "run_dir": str(self.agent.work_dir),
                "plan": generated["plan"],
                "input_sets": [str(item.directory) for item in input_sets],
            }

        manifest["status"] = "approved"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self._write_slurm_scripts(input_sets)
        self.transport.ensure_connection()

        latest_contcar: Optional[Path] = None
        for item in input_sets:
            if latest_contcar and latest_contcar.exists():
                shutil_path = item.directory / "POSCAR"
                shutil_path.write_text(latest_contcar.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
                print(f"[vasp-agent] using relaxed CONTCAR from the previous step as {item.directory.name}/POSCAR")

            remote_dir = f"{self.transport.remote_root}/{self.agent.work_dir.name}/{item.directory.name}"
            self.transport.upload_directory(item.directory, remote_dir)
            job = self.transport.submit(remote_dir, Path(item.slurm_path).name)
            print(f"[vasp-agent] submitted {Path(item.slurm_path).name}: {job.submit_output}")
            self.transport.wait_for_job(job)
            self.transport.fetch_directory(remote_dir, item.directory)
            contcar = item.directory / "CONTCAR"
            if item.task in {"relax", "vc-relax"} and contcar.exists() and contcar.stat().st_size > 0:
                latest_contcar = contcar

        analysis = self._summarize_outputs(query, input_sets)
        (self.agent.work_dir / "analysis.json").write_text(
            json.dumps({"query": query, "analysis": analysis}, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "run_dir": str(self.agent.work_dir),
            "input_sets": [
                {
                    "step": item.step_index,
                    "directory": str(item.directory),
                    "output_path": item.output_path,
                    "slurm_path": item.slurm_path,
                }
                for item in input_sets
            ],
            "analysis": analysis,
        }

    def _write_slurm_scripts(self, input_sets: List[VASPInputSet]) -> None:
        for item in input_sets:
            command = self._vasp_command()
            command_line = f"{command} > $OUTPUT"
            if item.potcar_mode == "remote":
                command_line = "./assemble_potcar.sh\n" + command_line
            script = render_slurm_script(
                exec_path=self._vasp_executable_name(),
                input_path="POSCAR",
                output_path="vasp.out",
                command_line=command_line,
                nodes=1,
                tasks_per_node=self.parallel_np,
                work_dir=".",
                time_limit="01:00:00",
                template_path=self.slurm_template_path,
            )
            path = item.directory / "run_vasp.slurm"
            path.write_text(script.rstrip() + "\n", encoding="utf-8")
            item.slurm_path = str(path)

    def _vasp_command(self) -> str:
        command = (self.agent.vasp_command or self.default_vasp_command).strip()
        if not command:
            command = "vasp_std"
        if re.search(r"\b(mpirun|mpiexec|srun)\b", command):
            return command
        return f"mpirun -np {self.parallel_np} $exe"

    def _vasp_executable_name(self) -> str:
        command = (self.agent.vasp_command or self.default_vasp_command).strip() or "vasp_std"
        if re.search(r"\b(mpirun|mpiexec|srun)\b", command):
            parts = shlex.split(command)
            for token in reversed(parts):
                if not token.startswith("-") and "=" not in token:
                    return token
            return "vasp_std"
        return command

    def _summarize_outputs(self, query: str, input_sets: List[VASPInputSet]) -> str:
        snippets: List[str] = []
        for item in input_sets:
            out_path = Path(item.output_path)
            if out_path.exists():
                text = out_path.read_text(encoding="utf-8", errors="replace")
                snippets.append(f"Step {item.step_index} {item.task} output tail:\n{text[-6000:]}")
            else:
                snippets.append(f"Step {item.step_index} {item.task}: output file missing at {out_path}")
        prompt = f"""Summarize the result of this VASP workflow for the user.

User request:
{query}

Output snippets:
{chr(10).join(snippets)}

Return a concise scientific summary. Mention if any output file is missing or incomplete.
"""
        try:
            text = _generate_nonempty_text(
                self.agent.generator,
                prompt,
                max_new_tokens=self.agent.max_new_tokens,
                attempts=2,
                verbose=self.agent.verbose,
                purpose="vasp_summary",
            )
            return text.strip()
        except Exception as exc:
            return f"VASP workflow completed, but automatic result summarization failed: {exc}"
