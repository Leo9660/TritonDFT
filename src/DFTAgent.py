import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import datetime
import time

# 假设这些模块存在于你的项目中
from prompt import get_prompt
from prompt.tool_requirements import get_parse_requirement
from config import Config
from generator import UnifiedGenerator
from execute_code.slurm import SlurmLauncher
from tool import get_spec, fetch_material_info_from_api_snippet, build_tool_requirements
from utils import get_qe_prefix, parse_scripts_block, write_inputs, \
parse_plan_string, patch_qe_input_file, enforce_qe_parameters_from_guess, get_qe_result, preprocess_output_list, extract_json_brutal, output_to_log_file
from executor import run_qe_inputs
from evaluate.compare import compare_evaluation

class DFTAgent:
    """
    DFTAgent: Minimal framework
    - run(user_query): entry point for the workflow
    - Internally: plan -> execute -> parse
    """

    def __init__(
        self,
        model: str,
        dft_tool: str = "quantum espresso",
        verbose: bool = False,
        work_dir: str = "tmp",
        max_new_tokens: int = 2048,
        backend: str = "auto",
        temperature: float = 0.0,
        top_p: float = 1.0,
        vllm_tensor_parallel_size: int = None,
        openai_api_key: str = None,
        openai_base_url: str = None,
        need_query_info: bool = False,
        auto_parallel: bool = False,
        parallel_exec: bool = False,
        parallel_np: int = 1,
        run_mode: str = "mpirun", # "mpirun", "local", "slurm"
        auto_confirm: bool = False,
        hardware_description: Optional[str] = None,
        benchmark: bool = False,
        benchmark_file: str = "benchmark.csv",
        evaluation_mode: bool = False,
        output_log: bool = False,
        output_log_file: str = "dft_agent_log.txt",
        config_name: Optional[str] = None,
        script_only: bool = False,
        mpid_output_file: Optional[str] = None,
        self_judge_max_loops: int = 5,
        pre_qe_review_loops: int = 5,
        use_category_reference_sample: bool = True,
    ):
        self.config_name = config_name or "config.yaml"
        self.config = Config.load(self.config_name)
        self.model = model
        self.dft_tool = dft_tool
        if self.dft_tool != "quantum espresso":
            raise ValueError("Currently only 'quantum espresso' is supported as dft_tool.")
        self.verbose = verbose
        self.work_dir_root = Path(work_dir).expanduser().resolve()
        self.work_dir_root.mkdir(parents=True, exist_ok=True)
        self.work_dir = self.work_dir_root

        self.max_new_tokens = max_new_tokens
        self.need_query_info = need_query_info

        self.auto_parallel = auto_parallel
        self.parallel_exec = parallel_exec
        self.parallel_np = parallel_np
        self.hardware_description = hardware_description
        self.benchmark = benchmark
        self.benchmark_file = benchmark_file
        valid_run_modes = {"mpirun", "local", "slurm"}
        if run_mode not in valid_run_modes:
            raise ValueError(f"run_mode must be one of {valid_run_modes}.")
        self.run_mode = run_mode
        self.auto_confirm = auto_confirm
        self.pseudo_dirs = self.config.pseudo
        self.pseudo_dir = self.config.pseudo.PBE
        self.qe_bin_prefix = self.config.qe_bin_dir
        
        self.generator = UnifiedGenerator(
            backend=backend,
            model=model,
            default_max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=1234,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            verbose=self.verbose,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
        )

        self.slurm_launcher = SlurmLauncher(
            generator=self.generator,
            max_new_tokens=self.max_new_tokens,
            verbose=self.verbose,
            auto_confirm=self.auto_confirm,
        )

        self.out_dir = "./"
        self.evaluation_mode = evaluation_mode
        self.output_log = output_log
        self.output_log_file = output_log_file
        self.script_only = script_only
        self.mpid_output_file = str(mpid_output_file) if mpid_output_file else None
        self.system_num = 0
        self.self_judge_max_loops = max(1, int(self_judge_max_loops))
        self.pre_qe_review_loops = max(1, int(pre_qe_review_loops))
        self.use_category_reference_sample = bool(use_category_reference_sample)
        self.current_category = "unknown"
        self.category_reference_samples = self._load_category_reference_samples()

        if self.verbose:
            print(f"[DFTAgent] Initialized with model={model}, dft_tool={dft_tool}, work_dir_root={self.work_dir_root}")

    @staticmethod
    def _sanitize_name(name: str, max_len: int = 40) -> str:
        """Sanitize a string for safe use as a directory component."""
        name = re.sub(r'[^\w\-]', '_', name)
        name = re.sub(r'_+', '_', name).strip('_')
        return name[:max_len] if name else ""

    _TASK_PATTERNS: list[tuple[str, str]] = [
        (r'\bvc[_-]relax\b|variable[_-]cell\s+relax', 'vc-relax'),
        (r'\bnscf\b|non[_-]self[_-]consistent', 'nscf'),
        (r'\bscf\b|self[_-]consistent\s+field', 'scf'),
        (r'\brelax(?:ation)?\b', 'relax'),
        (r'\bband[\s_-]?(?:structure|gap|calculation)', 'bands'),
        (r'\bphonon', 'phonon'),
        (r'\bmolecular[\s_-]dynamics\b', 'md'),
        (r'\bdos\b|density\s+of\s+states', 'dos'),
    ]

    @classmethod
    def _extract_query_metadata(cls, query: str) -> dict:
        """
        Extract material name and task type(s) from a free-form query string.

        Returns ``{"material_name": str, "task_type": str}``.
        ``task_type`` joins multiple detected types with ``+``
        (e.g. ``"vc-relax+scf"``).
        """
        material = ""

        # "material = Si", "material=BaTiO3"
        m = re.search(r'material\s*=\s*([A-Za-z][A-Za-z0-9]*)', query)
        if m:
            material = m.group(1)
        else:
            # "for [adjective] <ChemFormula>", e.g. "for tetragonal BaTiO3"
            m = re.search(
                r'\bfor\s+(?:[\w-]+\s+)?'
                r'([A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*)\b',
                query,
            )
            if m:
                material = m.group(1)

        tasks: list[str] = []
        for pattern, name in cls._TASK_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE) and name not in tasks:
                tasks.append(name)
        if 'vc-relax' in tasks and 'relax' in tasks:
            tasks.remove('relax')

        return {"material_name": material, "task_type": "+".join(tasks)}

    @staticmethod
    def _json_to_text(obj: Any) -> str:
        return json.dumps(obj, indent=2, ensure_ascii=False)

    @staticmethod
    def _normalize_category_name(category: str) -> str:
        return (category or "").strip().lower().replace(" ", "_").replace("-", "_")

    def _load_category_reference_samples(self) -> Dict[str, Any]:
        sample_path = Path(__file__).resolve().parents[1] / "benchmark" / "category_reference_samples.json"
        if not sample_path.exists():
            return {}
        try:
            return json.loads(sample_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @classmethod
    def _infer_category_from_query(cls, query: str) -> str:
        q = (query or "").lower()
        keyword_map = [
            ("topological", "topological"),
            ("semiconductor", "semiconductor"),
            ("insulator", "insulator"),
            ("ferroelectric", "ferroelectric"),
            ("piezoelectric", "piezoelectric"),
            ("magnetic", "magnetic"),
            ("thermoelectric", "thermoelectric"),
            ("superconductor", "superconductor"),
            ("optical", "optical"),
            ("metal", "metal"),
        ]
        for needle, category in keyword_map:
            if needle in q:
                return category
        return "unknown"

    def _format_category_reference_sample(self, category: str) -> str:
        category_key = self._normalize_category_name(category)
        sample = self.category_reference_samples.get(category_key)
        if not isinstance(sample, dict):
            return ""

        tiers = sample.get("tiers", {})
        lines = [
            f"Category: {category_key}",
            f"Exemplar material: {sample.get('exemplar_material', 'unknown')}",
            f"Notes: {sample.get('notes', '')}",
        ]
        tier_order = [
            ("1_mev_atom", "1 meV/atom"),
            ("10_mev_atom", "10 meV/atom"),
            ("20_mev_atom", "20 meV/atom"),
        ]
        for tier_key, tier_label in tier_order:
            tier_data = tiers.get(tier_key, {})
            if not isinstance(tier_data, dict):
                continue
            ecut = tier_data.get("ecutwfc_ry", "unknown")
            kpts = tier_data.get("k_points", [])
            if isinstance(kpts, list) and len(kpts) == 3:
                kpt_text = "x".join(str(v) for v in kpts)
            else:
                kpt_text = "unknown"
            lines.append(f"{tier_label}: ecutwfc ~ {ecut} Ry, exemplar k-point ~ {kpt_text}")
        return "\n".join(lines)

    def _get_category_reference_sample_text(self, query: str = "", category: str = "") -> str:
        if not self.use_category_reference_sample:
            return ""
        resolved = self._normalize_category_name(category)
        if not resolved or resolved == "unknown":
            resolved = self._infer_category_from_query(query)
        if not resolved or resolved == "unknown":
            resolved = self._normalize_category_name(self.current_category)
        return self._format_category_reference_sample(resolved)

    @staticmethod
    def _infer_accuracy_tier_key(query: str) -> str:
        q = (query or "").lower()
        if "1 mev/atom" in q or "1mev/atom" in q:
            return "1_mev_atom"
        if "10 mev/atom" in q or "10mev/atom" in q:
            return "10_mev_atom"
        if "20 mev/atom" in q or "20mev/atom" in q:
            return "20_mev_atom"
        if "strict energy accuracy target of 1 mev/atom" in q or "strict" in q or "high accuracy" in q:
            return "1_mev_atom"
        if "medium accuracy" in q:
            return "10_mev_atom"
        if "loose accuracy" in q:
            return "20_mev_atom"
        return ""

    @staticmethod
    def _parse_mesh_value(mesh: Any) -> Optional[Tuple[int, int, int]]:
        if isinstance(mesh, (list, tuple)) and len(mesh) >= 3:
            try:
                return (int(mesh[0]), int(mesh[1]), int(mesh[2]))
            except Exception:
                return None
        if mesh is None:
            return None
        text = str(mesh).strip().lower().replace("×", "x")
        parts = re.findall(r"\d+", text)
        if len(parts) >= 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except Exception:
                return None
        return None

    @staticmethod
    def _format_mesh_value(mesh: Tuple[int, int, int]) -> str:
        return f"{mesh[0]}x{mesh[1]}x{mesh[2]}"

    @staticmethod
    def _get_param_k_mesh(params_obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[Tuple[int, int, int]]]:
        guessed = params_obj.get("parameter_guesses", {}) if isinstance(params_obj, dict) else {}
        for key in ("k_points", "kpoint_mesh"):
            if key in guessed:
                parsed = DFTAgent._parse_mesh_value(guessed.get(key))
                return key, parsed
        return None, None

    @staticmethod
    def _set_param_k_mesh(params_obj: Dict[str, Any], mesh: Tuple[int, int, int], preferred_key: str = "k_points") -> Dict[str, Any]:
        updated = DFTAgent._coerce_param_guess_dict(params_obj)
        guessed = updated.setdefault("parameter_guesses", {})
        key = preferred_key if preferred_key in {"k_points", "kpoint_mesh"} else "k_points"
        if "k_points" not in guessed and "kpoint_mesh" in guessed:
            key = "kpoint_mesh"
        guessed[key] = DFTAgent._format_mesh_value(mesh)
        return updated

    def _get_category_reference_tier(self, query: str = "", category: str = "") -> Dict[str, Any]:
        if not self.use_category_reference_sample:
            return {}
        resolved = self._normalize_category_name(category)
        if not resolved or resolved == "unknown":
            resolved = self._infer_category_from_query(query)
        if not resolved or resolved == "unknown":
            resolved = self._normalize_category_name(self.current_category)
        sample = self.category_reference_samples.get(resolved, {})
        if not isinstance(sample, dict):
            return {}
        tier_key = self._infer_accuracy_tier_key(query)
        if not tier_key:
            return {}
        tiers = sample.get("tiers", {})
        tier = tiers.get(tier_key, {})
        return tier if isinstance(tier, dict) else {}

    @staticmethod
    def _guess_functional_from_query(query: str) -> str:
        q = (query or "").lower()
        if "pbesol" in q:
            return "PBESOL"
        if "lda" in q:
            return "LDA"
        return "PBE"

    @staticmethod
    def _truthy_qe_flag(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        return text in {".true.", "true", "1", "yes"}

    def _select_pseudo_dir_for_guess(self, params_obj: Dict[str, Any], query: str = "") -> str:
        guessed = params_obj.get("parameter_guesses", {}) if isinstance(params_obj, dict) else {}
        functional = self._guess_functional_from_query(query)
        use_soc = self._truthy_qe_flag(guessed.get("lspinorb")) or self._truthy_qe_flag(guessed.get("noncolin"))

        if use_soc:
            if functional == "PBESOL":
                return self.pseudo_dirs.PBESOL_FR
            if functional == "PBE":
                return self.pseudo_dirs.PBE_FR

        if functional == "LDA":
            return self.pseudo_dirs.LDA
        if functional == "PBESOL":
            return self.pseudo_dirs.PBESOL
        return self.pseudo_dirs.PBE

    def _apply_category_reference_floor(
        self,
        params_obj: Dict[str, Any],
        *,
        query: str,
        category: str = "",
    ) -> Dict[str, Any]:
        tier = self._get_category_reference_tier(query=query, category=category)
        if not tier:
            return self._coerce_param_guess_dict(params_obj)

        updated = self._coerce_param_guess_dict(params_obj)
        guessed = updated.setdefault("parameter_guesses", {})

        sample_mesh = self._parse_mesh_value(tier.get("k_points"))
        sample_ecut = tier.get("ecutwfc_ry")
        mesh_key, current_mesh = self._get_param_k_mesh(updated)

        tier_key = self._infer_accuracy_tier_key(query)
        if tier_key in {"1_mev_atom", "10_mev_atom", "20_mev_atom"} and sample_mesh is not None:
            if current_mesh is None:
                updated = self._set_param_k_mesh(updated, sample_mesh, preferred_key=mesh_key or "k_points")
            else:
                guarded_mesh = (
                    max(current_mesh[0], sample_mesh[0]),
                    max(current_mesh[1], sample_mesh[1]),
                    max(current_mesh[2], sample_mesh[2]),
                )
                updated = self._set_param_k_mesh(updated, guarded_mesh, preferred_key=mesh_key or "k_points")

        if tier_key in {"1_mev_atom", "10_mev_atom", "20_mev_atom"} and sample_ecut is not None:
            try:
                current_ecut = guessed.get("ecutwfc")
                current_num_match = re.search(r"[-+]?\d*\.?\d+", str(current_ecut)) if current_ecut is not None else None
                current_num = float(current_num_match.group(0)) if current_num_match else None
                sample_ecut_num = float(sample_ecut)
                if current_num is None or current_num < sample_ecut_num:
                    guessed["ecutwfc"] = f"{int(sample_ecut_num) if sample_ecut_num.is_integer() else sample_ecut_num} Ry"
            except Exception:
                pass

        return updated

    @staticmethod
    def _canonical_signature(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _ensure_json_dict(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if raw is None:
            return {}
        if isinstance(raw, str):
            return extract_json_brutal(raw)
        raise TypeError(f"Expected dict-like JSON payload, got {type(raw)!r}")

    @staticmethod
    def _coerce_param_guess_dict(raw: Any) -> Dict[str, Any]:
        obj = DFTAgent._ensure_json_dict(raw)
        if "parameter_guesses" not in obj:
            obj = {
                "material": obj.get("material", ""),
                "structure": obj.get("structure", {}),
                "parameter_guesses": obj,
            }
        if not isinstance(obj.get("parameter_guesses"), dict):
            obj["parameter_guesses"] = {}
        return obj

    @staticmethod
    def _merge_param_guess(
        current_guess: Dict[str, Any],
        new_guess: Any,
    ) -> Dict[str, Any]:
        merged = json.loads(json.dumps(DFTAgent._coerce_param_guess_dict(current_guess)))
        if not new_guess:
            return merged

        incoming = DFTAgent._ensure_json_dict(new_guess)

        if "material" in incoming and incoming["material"]:
            merged["material"] = incoming["material"]
        if "structure" in incoming and isinstance(incoming["structure"], dict):
            base_structure = merged.get("structure", {})
            if not isinstance(base_structure, dict):
                base_structure = {}
            base_structure.update(incoming["structure"])
            merged["structure"] = base_structure

        if "parameter_guesses" in incoming and isinstance(incoming["parameter_guesses"], dict):
            patch = incoming["parameter_guesses"]
        else:
            patch = {
                k: v for k, v in incoming.items()
                if k not in {"material", "structure"}
            }

        merged.setdefault("parameter_guesses", {})
        for key, value in patch.items():
            if value is not None:
                merged["parameter_guesses"][key] = value

        return merged

    @staticmethod
    def _summarize_attempt_for_history(
        attempt_index: int,
        params_obj: Dict[str, Any],
        parsed_results: List[Dict[str, Any]],
        judge_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "attempt": attempt_index,
            "parameter_guesses": params_obj.get("parameter_guesses", {}),
            "results": parsed_results,
        }
        if judge_json:
            summary["judge_status"] = judge_json.get("status", "")
            summary["failure_mode"] = judge_json.get("failure_mode", "")
            summary["underestimate"] = judge_json.get("underestimate", "")
            summary["overcost"] = judge_json.get("overcost", "")
            summary["root_cause"] = judge_json.get("root_cause", "")
            summary["judge_desc"] = judge_json.get("desc", "")
            if judge_json.get("new_param_guess"):
                summary["proposed_revision"] = judge_json.get("new_param_guess")
        return summary

    def _write_attempt_history_file(
        self,
        work_dir: Path,
        attempt_history: List[Dict[str, Any]],
    ) -> Path:
        history_path = Path(work_dir) / "attempt_history.json"
        history_path.write_text(
            json.dumps(attempt_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return history_path

    def _write_pre_attempt_history_file(
        self,
        work_dir: Path,
        attempt_history: List[Dict[str, Any]],
    ) -> Path:
        history_path = Path(work_dir) / "pre_attempt_history.json"
        history_path.write_text(
            json.dumps(attempt_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return history_path

    @staticmethod
    def _failure_text(judge_json: Optional[Dict[str, Any]]) -> str:
        if not judge_json:
            return ""
        return (
            f"{judge_json.get('failure_mode', '')} "
            f"{judge_json.get('desc', '')}"
        ).lower()

    @staticmethod
    def _is_sampling_related_failure(failure_text: str) -> bool:
        keywords = (
            "k-point", "kpoint", "mesh", "ecut", "cutoff",
            "underestimate", "over-dense", "overdense",
            "sampling", "brillouin", "reciprocal-space",
        )
        return any(key in failure_text for key in keywords)

    @staticmethod
    def _is_structure_related_failure(failure_text: str) -> bool:
        keywords = (
            "atom position", "atomic position", "atoms #", "lattice vector",
            "overlap", "overlapping", "structure", "stoichiometry",
            "cell_parameters", "cell parameters", "namelist", "&ions",
            "input reading", "check_atoms",
        )
        return any(key in failure_text for key in keywords)

    @staticmethod
    def _is_vdw_related_failure(failure_text: str) -> bool:
        keywords = ("vdw", "dft-d3", "dispersion", "functional name unknown")
        return any(key in failure_text for key in keywords)

    @staticmethod
    def _is_soc_pseudo_related_failure(failure_text: str) -> bool:
        keywords = (
            "spin-orbit", "lspinorb", "noncolin",
            "fully relativistic", "scalar-relativistic",
            "pseudopotential", "pseudo_dir",
        )
        return any(key in failure_text for key in keywords)

    @classmethod
    def _stabilize_revision(
        cls,
        current_guess: Dict[str, Any],
        revised_guess: Dict[str, Any],
        judge_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current = cls._coerce_param_guess_dict(current_guess)
        revised = cls._coerce_param_guess_dict(revised_guess)
        failure_text = cls._failure_text(judge_json)

        current_params = current.get("parameter_guesses", {})
        revised_params = revised.get("parameter_guesses", {})

        freeze_keys: set[str] = set()

        if cls._is_structure_related_failure(failure_text):
            freeze_keys.update({
                "k_points", "ecutwfc", "ecutrho", "vdw_corr",
                "noncolin", "lspinorb", "smearing", "degauss",
                "occupations", "spin_polarization", "hubbard_u",
            })
        elif cls._is_vdw_related_failure(failure_text):
            freeze_keys.update({
                "k_points", "ecutwfc", "ecutrho",
                "noncolin", "lspinorb", "smearing", "degauss",
                "occupations",
            })
        elif cls._is_soc_pseudo_related_failure(failure_text):
            freeze_keys.update({
                "k_points", "ecutwfc", "ecutrho", "vdw_corr",
                "smearing", "degauss", "occupations",
            })
        elif not cls._is_sampling_related_failure(failure_text):
            freeze_keys.update({"k_points", "ecutwfc", "ecutrho"})

        for key in freeze_keys:
            if key in current_params:
                revised_params[key] = current_params[key]

        revised["parameter_guesses"] = revised_params
        return revised

    def _generate_parameter_guess(
        self,
        *,
        subproblem_text: str,
        tool_name: str,
        query: str,
        total_memory: str = "",
        attempt_history_text: str = "",
        category_reference_sample_text: str = "",
    ) -> Tuple[str, Dict[str, Any]]:
        prompt_type = "parameter_gemini" if self.generator.backend == "gemini" else "parameter"
        prompt = get_prompt(
            prompt_type=prompt_type,
            subproblem=subproblem_text,
            fn=tool_name,
            tool=self.dft_tool,
            query=query,
            previous_memory=total_memory,
            attempt_history=attempt_history_text,
            category_reference_sample=category_reference_sample_text,
        )
        params_out = self.generator(
            prompt[0]["content"],
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        params_json_text = params_out[0]["generated_text"]
        if self.generator.backend == "gemini":
            params_obj = self._parse_gemini_parameter_plaintext(params_json_text)
        else:
            params_obj = self._coerce_param_guess_dict(params_json_text)
        return self._json_to_text(params_obj), params_obj

    def _pre_judge_parameter_guess(
        self,
        *,
        query: str,
        subproblem_text: str,
        params_json: str,
        attempt_history_text: str = "",
        category_reference_sample_text: str = "",
    ) -> Dict[str, Any]:
        prompt_type = "parameter_self_judge_gemini" if self.generator.backend == "gemini" else "parameter_self_judge"
        messages = get_prompt(
            prompt_type=prompt_type,
            query=query,
            subproblem=subproblem_text,
            param_json=params_json,
            attempt_history=attempt_history_text,
            category_reference_sample=category_reference_sample_text,
        )
        result = self.generator(
            messages[0]["content"],
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        raw_text = result[0]["generated_text"]
        if self.generator.backend == "gemini":
            return self._parse_gemini_pre_judge_plaintext(raw_text)
        return extract_json_brutal(raw_text)

    @staticmethod
    def _extract_plaintext_field(raw_text: str, key: str, next_keys: List[str]) -> str:
        if next_keys:
            next_pattern = "|".join(re.escape(k) for k in next_keys)
            pattern = rf"(?ms)^{re.escape(key)}:\s*(.*?)(?=^(?:{next_pattern}):|\Z)"
        else:
            pattern = rf"(?ms)^{re.escape(key)}:\s*(.*)\Z"
        match = re.search(pattern, raw_text.strip())
        return match.group(1).strip() if match else ""

    @staticmethod
    def _coerce_plaintext_scalar(value: str) -> Any:
        text = value.strip()
        if not text:
            return ""
        lower = text.lower()
        if lower == "true":
            return True
        if lower == "false":
            return False
        if lower in {"null", "none"}:
            return None
        if re.fullmatch(r"[-+]?\d+", text):
            try:
                return int(text)
            except Exception:
                pass
        if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", text) or re.fullmatch(r"[-+]?\d+(?:[eE][-+]?\d+)", text):
            try:
                return float(text)
            except Exception:
                pass
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except Exception:
                return text
        return text

    @classmethod
    def _parse_plaintext_mapping_block(cls, block_text: str) -> Dict[str, Any]:
        mapping: Dict[str, Any] = {}
        for raw_line in block_text.splitlines():
            line = raw_line.strip()
            if not line or line.lower() == "none":
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            mapping[key.strip()] = cls._coerce_plaintext_scalar(value)
        return mapping

    @classmethod
    def _parse_gemini_parameter_plaintext(cls, raw_text: str) -> Dict[str, Any]:
        material = cls._extract_plaintext_field(raw_text, "MATERIAL", ["STRUCTURE_PROTOTYPE", "PARAMETER_GUESSES"])
        prototype = cls._extract_plaintext_field(raw_text, "STRUCTURE_PROTOTYPE", ["PARAMETER_GUESSES"])
        guesses_raw = cls._extract_plaintext_field(raw_text, "PARAMETER_GUESSES", ["END_PARAMETER_GUESSES"])
        guesses = cls._parse_plaintext_mapping_block(guesses_raw)
        return {
            "material": material,
            "structure": {"prototype": prototype},
            "parameter_guesses": guesses,
        }

    @classmethod
    def _parse_gemini_pre_judge_plaintext(cls, raw_text: str) -> Dict[str, Any]:
        status = cls._extract_plaintext_field(raw_text, "STATUS", ["UNDERESTIMATE", "OVERCOST", "ROOT_CAUSE", "KEEP_FIXED", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"]).lower()
        underestimate = cls._extract_plaintext_field(raw_text, "UNDERESTIMATE", ["OVERCOST", "ROOT_CAUSE", "KEEP_FIXED", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"]).lower()
        overcost = cls._extract_plaintext_field(raw_text, "OVERCOST", ["ROOT_CAUSE", "KEEP_FIXED", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"]).lower()
        root_cause = cls._extract_plaintext_field(raw_text, "ROOT_CAUSE", ["KEEP_FIXED", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"])
        keep_fixed_raw = cls._extract_plaintext_field(raw_text, "KEEP_FIXED", ["NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"])
        new_param_guess_raw = cls._extract_plaintext_field(raw_text, "NEW_PARAM_GUESS", ["END_NEW_PARAM_GUESS", "DESC"])
        desc = cls._extract_plaintext_field(raw_text, "DESC", [])

        keep_fixed = []
        if keep_fixed_raw and keep_fixed_raw.lower() != "none":
            keep_fixed = [part.strip() for part in keep_fixed_raw.split(",") if part.strip()]

        new_param_guess = cls._parse_plaintext_mapping_block(new_param_guess_raw)

        return {
            "status": status,
            "underestimate": underestimate,
            "overcost": overcost,
            "root_cause": root_cause,
            "keep_fixed": keep_fixed,
            "new_param_guess": new_param_guess,
            "desc": desc,
        }

    @classmethod
    def _parse_gemini_result_judge_plaintext(cls, raw_text: str) -> Dict[str, Any]:
        status = cls._extract_plaintext_field(raw_text, "STATUS", ["UNDERESTIMATE", "OVERCOST", "ROOT_CAUSE", "FAILURE_MODE", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"]).lower()
        underestimate = cls._extract_plaintext_field(raw_text, "UNDERESTIMATE", ["OVERCOST", "ROOT_CAUSE", "FAILURE_MODE", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"]).lower()
        overcost = cls._extract_plaintext_field(raw_text, "OVERCOST", ["ROOT_CAUSE", "FAILURE_MODE", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"]).lower()
        root_cause = cls._extract_plaintext_field(raw_text, "ROOT_CAUSE", ["FAILURE_MODE", "NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"])
        failure_mode = cls._extract_plaintext_field(raw_text, "FAILURE_MODE", ["NEW_PARAM_GUESS", "END_NEW_PARAM_GUESS", "DESC"])
        new_param_guess_raw = cls._extract_plaintext_field(raw_text, "NEW_PARAM_GUESS", ["END_NEW_PARAM_GUESS", "DESC"])
        desc = cls._extract_plaintext_field(raw_text, "DESC", [])

        new_param_guess = cls._parse_plaintext_mapping_block(new_param_guess_raw)

        return {
            "status": status,
            "underestimate": underestimate,
            "overcost": overcost,
            "root_cause": root_cause,
            "new_param_guess": new_param_guess,
            "failure_mode": failure_mode,
            "desc": desc,
        }

    @classmethod
    def _parse_gemini_result_parse_plaintext(cls, raw_text: str) -> Dict[str, Any]:
        prefix = cls._extract_plaintext_field(raw_text, "PREFIX", ["SUCCESS", "KEY_FINDINGS", "CONCLUSION"])
        success_raw = cls._extract_plaintext_field(raw_text, "SUCCESS", ["KEY_FINDINGS", "CONCLUSION"])
        findings_raw = cls._extract_plaintext_field(raw_text, "KEY_FINDINGS", ["END_KEY_FINDINGS", "CONCLUSION"])
        conclusion = cls._extract_plaintext_field(raw_text, "CONCLUSION", [])
        success_val = cls._coerce_plaintext_scalar(success_raw)
        findings = cls._parse_plaintext_mapping_block(findings_raw)
        return {
            "prefix": None if prefix.lower() in {"", "null", "none"} else prefix,
            "success": bool(success_val),
            "key_findings": findings,
            "conclusion": conclusion,
        }

    @staticmethod
    def _looks_like_qe_input(text: str) -> bool:
        lowered = (text or "").lower()
        required_markers = ("&control", "&system", "atomic_species", "k_points")
        return all(marker in lowered for marker in required_markers)

    def _parse_generated_scripts(self, generated_scripts: str) -> List[str]:
        try:
            return parse_scripts_block(generated_scripts)
        except Exception:
            if self.generator.backend != "gemini":
                raise

        text = (generated_scripts or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
            text = re.sub(r"\n```$", "", text).strip()

        scripts_wrapper = re.search(r"(?is)<scripts>(.*)</scripts>", text)
        if scripts_wrapper:
            inner = scripts_wrapper.group(1).strip()
            if inner and self._looks_like_qe_input(inner):
                return [inner]

        first_qe_marker = re.search(r"(?is)(&control|&system)", text)
        if first_qe_marker:
            text = text[first_qe_marker.start():].strip()

        trailing_note = re.search(r"(?im)^(note|explanation|comment|summary)\s*:", text)
        if trailing_note:
            text = text[:trailing_note.start()].strip()

        if self._looks_like_qe_input(text):
            return [text]

        raise ValueError("No <script>...</script> blocks found in model output.")

    @staticmethod
    def _stabilize_pre_judge_revision(
        current_guess: Dict[str, Any],
        revised_guess: Dict[str, Any],
        judge_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stabilized = DFTAgent._merge_param_guess(current_guess, revised_guess)
        current = DFTAgent._coerce_param_guess_dict(current_guess)
        stabilized = DFTAgent._coerce_param_guess_dict(stabilized)
        current_params = current.get("parameter_guesses", {})
        stabilized_params = stabilized.get("parameter_guesses", {})

        keep_fixed = judge_json.get("keep_fixed", []) if judge_json else []
        if not isinstance(keep_fixed, list):
            keep_fixed = []
        for key in keep_fixed:
            if key in current_params:
                stabilized_params[key] = current_params[key]

        root_cause = (judge_json.get("root_cause", "") if judge_json else "").lower()
        if root_cause in {"structure", "stoichiometry", "advanced_parameters", "vdw_compatibility", "pseudopotential_soc"}:
            for key in ("k_points", "ecutwfc", "ecutrho"):
                if key in current_params:
                    stabilized_params[key] = current_params[key]

        stabilized["parameter_guesses"] = stabilized_params
        return stabilized

    def _run_pre_qe_review_cycle(
        self,
        *,
        query: str,
        subproblem: Dict[str, Any],
        params_obj: Dict[str, Any],
        work_dir: Path,
        category_reference_sample_text: str = "",
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        review_history: List[Dict[str, Any]] = []
        no_progress_count = 0
        params_obj = self._apply_category_reference_floor(
            params_obj,
            query=query,
            category=self.current_category,
        )

        for round_idx in range(1, self.pre_qe_review_loops + 1):
            params_json = self._json_to_text(params_obj)
            review = self._pre_judge_parameter_guess(
                query=query,
                subproblem_text=subproblem["problem"],
                params_json=params_json,
                attempt_history_text=self._json_to_text(review_history) if review_history else "",
                category_reference_sample_text=category_reference_sample_text,
            )

            review_entry = {
                "round": round_idx,
                "parameter_guesses": params_obj.get("parameter_guesses", {}),
                "status": review.get("status", ""),
                "underestimate": review.get("underestimate", ""),
                "overcost": review.get("overcost", ""),
                "root_cause": review.get("root_cause", ""),
                "keep_fixed": review.get("keep_fixed", []),
                "desc": review.get("desc", ""),
                "proposed_revision": review.get("new_param_guess", {}),
            }
            review_history.append(review_entry)
            self._write_pre_attempt_history_file(work_dir, review_history)

            if review.get("status") == "ready":
                break

            revised = self._stabilize_pre_judge_revision(
                params_obj,
                review.get("new_param_guess", {}),
                review,
            )
            revised = self._apply_category_reference_floor(
                revised,
                query=query,
                category=self.current_category,
            )
            if self._canonical_signature(revised) == self._canonical_signature(params_obj):
                no_progress_count += 1
            else:
                no_progress_count = 0
                params_obj = revised

            if no_progress_count >= 2:
                break

        return params_obj, review_history

    def _prepare_run_directory(
        self,
        query: str = "",
        material_name: str = "",
        task_type: str = "",
        run_id: int = 0,
        category: str = "",
    ) -> Path:
        """
        Build a structured run directory under *work_dir_root*.

        If *material_name* or *task_type* are empty, they are auto-extracted
        from *query*.  Layout::

            work_dir_root/
            └── YYYY-MM-DD/
                └── <material>_<task>_<HHMMSS>_<uuid8>/
                    └── run_meta.json

        Falls back to ``run_<HHMMSS>_<uuid8>`` when nothing can be inferred.
        """
        if query and (not material_name or not task_type):
            extracted = self._extract_query_metadata(query)
            material_name = material_name or extracted["material_name"]
            task_type = task_type or extracted["task_type"]

        now = datetime.datetime.now()
        date_dir = self.work_dir_root / now.strftime("%Y-%m-%d")

        parts: list[str] = []
        if material_name:
            parts.append(self._sanitize_name(material_name))
        if task_type:
            parts.append(self._sanitize_name(task_type))
        if not parts:
            parts.append("run")
        parts.append(now.strftime("%H%M%S"))
        parts.append(uuid.uuid4().hex[:8])

        run_dir = date_dir / "_".join(parts)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = run_dir

        meta = {
            "run_id": run_id,
            "material_name": material_name,
            "task_type": task_type,
            "category": category,
            "query": query,
            "model": self.model,
            "dft_tool": self.dft_tool,
            "created_at": now.isoformat(),
            "directory": str(run_dir),
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if self.verbose:
            print(f"[DFTAgent] Using run directory: {self.work_dir}")
        return run_dir

    def info_query(self, query: str) -> Any:
        if self.verbose:
            print(f"[info_query] Querying material information for: {query}")
        
        api_messages = get_prompt(prompt_type="api_call", query=query)
        api_text = api_messages[0]["content"]

        api_call_snippet_out = self.generator(api_text, max_new_tokens=self.max_new_tokens, return_full_text=False)
        api_call_snippet_out = api_call_snippet_out[0]['generated_text']

        if self.verbose:
            print(f"[info_query] API call snippet received: {api_call_snippet_out}")

        fetch_result = fetch_material_info_from_api_snippet(api_call_snippet_out, limit=25, verbose=self.verbose)
    
        if self.output_log:
            output_to_log_file(self.work_dir_root, self.output_log_file, f"[info_query] Retrieved material information: {fetch_result.get('material_ids', ['N/A'])[0]}")

        if self.verbose and fetch_result.get('initial_structures'):
             print(f"[info_query] Retrieved material information: {fetch_result['initial_structures'][0][0]}")

        return fetch_result

    def plan(self, query: str) -> List[Dict[str, Any]]:
        if self.verbose:
            print(f"[plan] Generating plan for query: {query}")

        messages = get_prompt(prompt_type="planner", question=query, tool=self.dft_tool)
        try:
            prompt_text = messages[0]["content"]
            raw_out = self.generator(prompt_text, max_new_tokens=self.max_new_tokens, return_full_text=False)
            if self.verbose:
                print(f"[plan] LLM raw output received: {raw_out[0]['generated_text']}")
        except Exception as e:
            if self.verbose:
                print(f"[plan][error] model call failed: {e}")
            return None

        plan_dict = parse_plan_string(raw_out[0]["generated_text"])
        if self.verbose:
            print(f"[plan] Parsed {len(plan_dict)} steps.")

        return plan_dict

    def solve_sub_problem(self, subproblem: Dict[str, Any], problem_id: int = 0, query: str = "", total_memory: str = "", material_info: Dict = []) -> Any:
        if self.verbose:
            print(f"[solve_sub_problem] Solving subproblem: {subproblem['problem']}")
            print(f"[solve_sub_problem] Using tool: {subproblem['tool']}")

        # --- Subproblem Timing Accumulators ---
        # These must accumulate over the potential loops/retries
        acc_script_gen_time = 0.0
        acc_parse_validate_time = 0.0
        acc_dft_run_time = 0.0

        total_result_json = ""
        error_code = ""
        attempt_history: List[Dict[str, Any]] = []
        pre_attempt_history: List[Dict[str, Any]] = []
        no_progress_count = 0

        fn_spec = get_spec(subproblem['tool'])
        if fn_spec is None:
            raise ValueError(f"Unknown function/tool: {subproblem['tool']}")

        # 1. Parameter Generation
        t0 = time.perf_counter()
        category_reference_sample_text = self._get_category_reference_sample_text(
            query=query,
            category=self.current_category,
        )
        try:
            params_json, params_obj = self._generate_parameter_guess(
                subproblem_text=subproblem['problem'],
                tool_name=subproblem['tool'],
                query=query,
                total_memory=total_memory,
                category_reference_sample_text=category_reference_sample_text,
            )
            if self.verbose:
                print(f"[solve_sub_problem] parameter output received: {params_json}")
        except Exception as e:
            if self.verbose:
                print(f"[solve_sub_problem][error] subproblem solve failed: {e}")
            # Even if it fails early, count this time
            acc_script_gen_time += (time.perf_counter() - t0)
            return {
                "status": "failed",
                "timing": {
                    "script_gen_s": acc_script_gen_time,
                    "parse_validate_s": 0.0,
                    "dft_run_s": 0.0
                }
            }
        
        acc_script_gen_time += (time.perf_counter() - t0)

        initial_structures = material_info.get("initial_structures", [])
        conventional_structures = material_info.get("conventional_structure", [])
        primitive_structures = material_info.get("primitive_structure", [])

        params_obj, pre_attempt_history = self._run_pre_qe_review_cycle(
            query=query,
            subproblem=subproblem,
            params_obj=params_obj,
            work_dir=Path(self.work_dir),
            category_reference_sample_text=category_reference_sample_text,
        )
        params_json = self._json_to_text(params_obj)

        loop_count = 0
        MAX_LOOPS = self.self_judge_max_loops
        judge_json: Dict[str, Any] = {}

        while True:
            loop_count += 1
            current_attempt_history_text = self._json_to_text(attempt_history) if attempt_history else ""
            
            # --- Script Generation Phase ---
            t_script_start = time.perf_counter()
            
            tool_requirements = build_tool_requirements(fn_spec, self.pseudo_dirs)
            
            # Construct Prompt
            if loop_count > 1:
                script_prompt = get_prompt(prompt_type="script_fixed_gemini" if self.generator.backend == "gemini" else "script_fixed",
                    bin_tool=fn_spec.exec,
                    tool_mode=fn_spec.mode if fn_spec.mode else "standard",
                    params_json=params_json,
                    upf_dir=self.pseudo_dir,
                    previous_run=error_code,
                    previous_memory=total_memory,
                    attempt_history=current_attempt_history_text,
                    fn_section=fn_spec.section,
                    query=query,
                    initial_structures=initial_structures,
                    conventional_structure=conventional_structures,
                    primitive_structure=primitive_structures,
                    subproblem=subproblem['problem'],
                    tool_requirements=tool_requirements
                )
            else:
                script_prompt = get_prompt(prompt_type="script_gemini" if self.generator.backend == "gemini" else "script",
                    bin_tool=fn_spec.exec,
                    tool_mode=fn_spec.mode if fn_spec.mode else "standard",
                    params_json=params_json,
                    upf_dir=self.pseudo_dir,
                    previous_memory=total_memory,
                    attempt_history=current_attempt_history_text,
                    fn_section=fn_spec.section,
                    query=query,
                    initial_structures=initial_structures,
                    conventional_structure=conventional_structures,
                    primitive_structure=primitive_structures,
                    subproblem=subproblem['problem'],
                    tool_requirements=tool_requirements
                )

            script_out = self.generator(script_prompt[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            generated_scripts = script_out[0]['generated_text']
            if self.verbose:
                print(f"[solve_sub_problem] Script generated (Loop {loop_count})")

            scripts = self._parse_generated_scripts(generated_scripts)
            
            # Work Dir setup & Write Inputs
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
            subproblem_id = subproblem.get("id", problem_id)
            input_paths = write_inputs(work_dir, scripts, prefix="input", suffix=".in", subproblem_id=subproblem_id)
            
            # Patch Inputs
            for i, path in enumerate(input_paths):
                exec_id = f"{problem_id}_{loop_count}_{i}"
                selected_pseudo_dir = self._select_pseudo_dir_for_guess(params_obj, query=query)
                patch_qe_input_file(path, new_pseudo_dir=selected_pseudo_dir, new_outdir=self.out_dir, new_prefix=f"subproblem_{exec_id}", pp_dir_clean=True)
                enforce_qe_parameters_from_guess(path, params_obj)

            # Input Eval (if enabled)
            if self.evaluation_mode and hasattr(fn_spec, "eval_input") and fn_spec.eval_input:
                for input_path in input_paths:
                    fn_spec.eval_input(input_path)

            acc_script_gen_time += (time.perf_counter() - t_script_start)

            if self.script_only:
                return {
                    "status": "script_only",
                    "result_json": "",
                    "result_judge": "script_only",
                    "details": f"Generated {len(input_paths)} inputs.",
                    "timing": {
                        "script_gen_s": acc_script_gen_time,
                        "parse_validate_s": acc_parse_validate_time,
                        "dft_run_s": acc_dft_run_time,
                    },
                    "evaluation": None,
                }

            # --- DFT Execution Phase ---
            t_dft_start = time.perf_counter()
            
            qe_prefix = get_qe_prefix(self)
            output_paths = [os.path.join(work_dir, f"output_{subproblem_id}_{idx}.out") for idx in range(1, len(input_paths) + 1)]
            auto_parallel = self.auto_parallel and fn_spec.mode == "vc-relax"

            try:
                retcodes, output_paths = run_qe_inputs(
                    exec_name=fn_spec.exec,
                    qe_prefix=qe_prefix,
                    input_paths=input_paths,
                    work_dir=work_dir,
                    verbose=self.verbose,
                    parallel_exec=self.parallel_exec,
                    parallel_np=self.parallel_np,
                    auto_parallel=auto_parallel,
                    hardware_description=self.hardware_description,
                    run_mode=self.run_mode,
                    slurm_launcher=self.slurm_launcher.launch if self.run_mode == "slurm" else None,
                    auto_parallel_generator=self.generator,
                    max_new_tokens=self.max_new_tokens,
                    auto_confirm=self.auto_confirm,
                    output_paths=output_paths,
                )
                # Successful execution block end
                acc_dft_run_time += (time.perf_counter() - t_dft_start)

            except TimeoutError as exc:
                # Capture time even on failure
                acc_dft_run_time += (time.perf_counter() - t_dft_start)
                return {
                    "status": "timeout",
                    "result_json": "",
                    "result_judge": "timeout",
                    "details": f"QE execution timed out: {exc}",
                    "timing": {
                        "script_gen_s": acc_script_gen_time,
                        "parse_validate_s": acc_parse_validate_time,
                        "dft_run_s": acc_dft_run_time,
                    },
                    "evaluation": None,
                }
            except Exception as e:
                # Capture time even on crash
                acc_dft_run_time += (time.perf_counter() - t_dft_start)
                raise e

            # Handle Probe Failure (Auto Parallel)
            if retcodes == "probe_failed":
                if loop_count > MAX_LOOPS:
                    raise ValueError("Could not solve subproblem: Probe Failed multiple times.")
                if self.verbose:
                    print(f"[solve_sub_problem][error] Auto-parallel probing failed. Outputs: {output_paths}")
                
                # Append error info and retry
                for input_script, err_code in zip(scripts, output_paths):
                    error_code += f"Input Script: {input_script}, Error: {err_code}\n\n"
                continue # Retry loop

            # --- Parsing & Validation Phase ---
            t_parse_start = time.perf_counter()

            input_list, output_list = get_qe_result(
                work_dir=work_dir,
                input_paths=input_paths,
                verbose=self.verbose,
                subproblem_id=subproblem_id,
            )
            output_list = preprocess_output_list(output_list, verbose=self.verbose)

            current_parsed_results: List[Dict[str, Any]] = []

            # Result Parsing
            for i, (input_file, output_file) in enumerate(zip(input_list, output_list)):
                parse_requirement = get_parse_requirement(fn_spec.parse_requirement_key)
                prompt_type = "result_parse_gemini" if self.generator.backend == "gemini" else "result_parse"
                messages = get_prompt(
                    prompt_type=prompt_type,
                    input_json=params_json,
                    input_file=input_file,
                    output_text=output_file,
                    fn=fn_spec.exec,
                    parse_requirement=parse_requirement,
                )
                try:
                    result_out = self.generator(
                        messages[0]['content'],
                        max_new_tokens=self.max_new_tokens,
                        return_full_text=False,
                    )
                    if self.generator.backend == "gemini":
                        parsed_json = self._parse_gemini_result_parse_plaintext(result_out[0]['generated_text'])
                    else:
                        parsed_json = extract_json_brutal(result_out[0]['generated_text'])
                except Exception as e:
                    if self.verbose:
                        print(f"[solve_sub_problem][error] result parsing failed: {e}")
                    parsed_json = {
                        "prefix": None,
                        "success": False,
                        "key_findings": {},
                        "conclusion": f"Parsing failed: {e}",
                    }
                current_parsed_results.append(parsed_json)

            current_result_json = self._json_to_text(current_parsed_results)

            # Result Judging
            prompt_type = "result_judge_gemini" if self.generator.backend == "gemini" else "result_judge"
            messages = get_prompt(
                prompt_type=prompt_type,
                query=query,
                subproblem=subproblem['problem'],
                param_json=params_json,
                result_json=current_result_json,
                attempt_history=current_attempt_history_text,
            )
            judge_out = self.generator(
                messages[0]['content'],
                max_new_tokens=self.max_new_tokens,
                return_full_text=False,
            )
            if self.generator.backend == "gemini":
                judge_json = self._parse_gemini_result_judge_plaintext(judge_out[0]['generated_text'])
            else:
                judge_json = extract_json_brutal(judge_out[0]['generated_text'])
            total_result_json = current_result_json
            
            acc_parse_validate_time += (time.perf_counter() - t_parse_start)

            attempt_record = self._summarize_attempt_for_history(
                attempt_index=loop_count,
                params_obj=params_obj,
                parsed_results=current_parsed_results,
                judge_json=judge_json,
            )
            attempt_history.append(attempt_record)
            self._write_attempt_history_file(Path(work_dir), attempt_history)

            if judge_json.get("status") == "done":
                if self.verbose:
                    print(f"[solve_sub_problem] Finished: {subproblem['problem']}")
                break
            elif loop_count >= MAX_LOOPS:
                raise ValueError("Could not solve the subproblem! Max iterations reached.")
            else:
                # Prepare for next loop
                revised_params_obj = self._merge_param_guess(
                    params_obj,
                    judge_json.get("new_param_guess", {}),
                )
                revised_params_obj = self._stabilize_revision(
                    params_obj,
                    revised_params_obj,
                    judge_json,
                )
                revised_params_obj = self._apply_category_reference_floor(
                    revised_params_obj,
                    query=query,
                    category=self.current_category,
                )
                revised_signature = self._canonical_signature(revised_params_obj)
                current_signature = self._canonical_signature(params_obj)

                if revised_signature == current_signature:
                    rethink_memory = total_memory + "\n" + current_attempt_history_text
                    params_json, revised_params_obj = self._generate_parameter_guess(
                        subproblem_text=subproblem['problem'],
                        tool_name=subproblem['tool'],
                        query=query,
                        total_memory=rethink_memory,
                        attempt_history_text=self._json_to_text(attempt_history),
                        category_reference_sample_text=category_reference_sample_text,
                    )
                    revised_params_obj = self._stabilize_revision(
                        params_obj,
                        revised_params_obj,
                        judge_json,
                    )
                    revised_params_obj = self._apply_category_reference_floor(
                        revised_params_obj,
                        query=query,
                        category=self.current_category,
                    )
                    revised_signature = self._canonical_signature(revised_params_obj)

                if revised_signature == current_signature:
                    no_progress_count += 1
                else:
                    no_progress_count = 0

                if no_progress_count >= 2:
                    raise ValueError("Could not solve the subproblem! Self-judgment cycle made no progress.")

                params_obj = revised_params_obj
                params_json = self._json_to_text(params_obj)
                error_code += (
                    f"Attempt {loop_count} script(s): {generated_scripts}\n"
                    f"Judge feedback: {judge_out[0]['generated_text']}\n\n"
                )
                if self.verbose:
                    print(f"[solve_sub_problem] Retrying... New params: {params_json}")

        # --- Final Evaluation Phase (Post-Success) ---
        t_eval_start = time.perf_counter()
        eval_result = None
        if self.evaluation_mode:
            for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
                if hasattr(fn_spec, "eval_func") and fn_spec.eval_func:
                    eval_result = fn_spec.eval_func(input_path, output_path)
                    if self.output_log:
                        output_to_log_file(self.work_dir_root, self.output_log_file, f"[Output Evaluation] {i}:\n {eval_result}")
        
        # Counting eval time into parse_validate for simplicity, or separate if needed.
        # Here adding to parse_validate to match signature.
        acc_parse_validate_time += (time.perf_counter() - t_eval_start)

        result = {
            "status": "success",
            "result_json": f"{total_result_json}",
            "result_judge": f"{judge_out[0]['generated_text']}",
            "details": f"Executed {subproblem['tool']}!",
            "pre_attempt_history": pre_attempt_history,
            "pre_attempt_history_file": str(Path(work_dir) / "pre_attempt_history.json"),
            "attempt_history": attempt_history,
            "attempt_history_file": str(Path(work_dir) / "attempt_history.json"),
            "timing": {
                "script_gen_s": acc_script_gen_time,
                "parse_validate_s": acc_parse_validate_time,
                "dft_run_s": acc_dft_run_time,
            },
            "evaluation": eval_result,
        }

        return result

    def run(
        self,
        query: str,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
        work_dir: Optional[str] = None,
    ) -> Any:
        
        # --- Global Timer Start ---
        run_start_time = time.perf_counter()
        
        if self.benchmark and hasattr(self.generator, "reset_token_counters"):
            self.generator.reset_token_counters()

        if work_dir:
            self.work_dir = Path(work_dir).expanduser().resolve()
            self.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._prepare_run_directory(
                query=query,
                material_name=material_name,
                task_type=task_type,
                run_id=run_id,
                category=category,
            )
        self.current_category = category or self._infer_category_from_query(query) or "unknown"
            
        if self.output_log:
            output_to_log_file(self.work_dir_root, self.output_log_file, f"###[Starting new run for query]: {query}\n", new=False)

        # --- Phase 1: Info Query & Plan ---
        t_plan_query_start = time.perf_counter()
        
        # Token Counters
        info_query_prompt_tokens = 0
        info_query_output_tokens = 0
        
        material_info = {}
        if self.need_query_info:
            pt_before = getattr(self.generator, "total_prompt_tokens", 0)
            ot_before = getattr(self.generator, "total_output_tokens", 0)
            
            material_info = self.info_query(query)
            
            info_query_prompt_tokens = getattr(self.generator, "total_prompt_tokens", 0) - pt_before
            info_query_output_tokens = getattr(self.generator, "total_output_tokens", 0) - ot_before
            
            if self.mpid_output_file:
                material_ids = material_info.get("material_ids") or []
                material_id = material_ids[0] if material_ids else ""
                mpid_path = Path(self.mpid_output_file)
                mpid_path.parent.mkdir(parents=True, exist_ok=True)
                with open(mpid_path, "a", encoding="utf-8") as f:
                    f.write(f"{self.model},{run_id},{category},{task_type},{material_name},{material_id}\n")

        pt_before_plan = getattr(self.generator, "total_prompt_tokens", 0)
        ot_before_plan = getattr(self.generator, "total_output_tokens", 0)
        
        subproblems = self.plan(query=query)
        
        plan_prompt_tokens = getattr(self.generator, "total_prompt_tokens", 0) - pt_before_plan
        plan_output_tokens = getattr(self.generator, "total_output_tokens", 0) - ot_before_plan

        plan_and_query_time = time.perf_counter() - t_plan_query_start

        if not subproblems:
            if self.verbose:
                print("[run] No valid plan generated. Exiting.")
            return None

        # --- Phase 2: Subproblem Execution ---
        total_memory = ""
        
        # Lists for CSV (per subproblem)
        subproblem_dft_times = []
        subproblem_script_times = []
        subproblem_parse_validate_times = []
        subproblem_prompt_tokens = []
        subproblem_output_tokens = []

        # Global Accumulators
        total_script_time = 0.0
        total_parse_validate_time = 0.0
        total_dft_time = 0.0
        
        last_sub_problem_res = None
        
        for i, step in enumerate(subproblems):
            if self.verbose:
                print(f"[run] Executing step {i+1}/{len(subproblems)}: {step['problem']}")
            
            pt_before_sub = getattr(self.generator, "total_prompt_tokens", 0)
            ot_before_sub = getattr(self.generator, "total_output_tokens", 0)
            
            sub_problem_res = self.solve_sub_problem(
                step, 
                problem_id=i+1, 
                query=query, 
                total_memory=total_memory,
                material_info=material_info
            )
            
            # Token Tracking
            subproblem_prompt_tokens.append(getattr(self.generator, "total_prompt_tokens", 0) - pt_before_sub)
            subproblem_output_tokens.append(getattr(self.generator, "total_output_tokens", 0) - ot_before_sub)

            if sub_problem_res and sub_problem_res.get("status") == "timeout":
                return sub_problem_res
            
            last_sub_problem_res = sub_problem_res
            
            # Extract Timing
            timing = sub_problem_res.get("timing", {})
            t_script = timing.get("script_gen_s", 0.0)
            t_parse = timing.get("parse_validate_s", 0.0)
            t_dft = timing.get("dft_run_s", 0.0)

            # Update Lists
            subproblem_script_times.append(t_script)
            subproblem_parse_validate_times.append(t_parse)
            subproblem_dft_times.append(t_dft)

            # Update Totals
            total_script_time += t_script
            total_parse_validate_time += t_parse
            total_dft_time += t_dft

            if self.script_only:
                # Stop here if script only
                return sub_problem_res

            # Update Memory
            total_memory += f" Subproblem {i+1}:\n System Results:\n {sub_problem_res.get('result_json','')} \n"
            total_memory += f" Conclusion of Subproblem {i+1}: {sub_problem_res.get('result_judge','')} \n\n"

        # --- Benchmark Recording ---
        total_run_time = time.perf_counter() - run_start_time

        if self.benchmark:
            prompt_tokens = getattr(self.generator, "total_prompt_tokens", 0)
            output_tokens = getattr(self.generator, "total_output_tokens", 0)
            
            ground_truth = {}
            if material_info and isinstance(material_info.get("ground_truth"), dict):
                ground_truth = material_info.get("ground_truth", {})
            
            evaluation = {}
            if last_sub_problem_res and isinstance(last_sub_problem_res.get("evaluation"), dict):
                evaluation = last_sub_problem_res.get("evaluation", {})
            
            max_rel_error, all_exact_match = compare_evaluation(ground_truth, evaluation)
            
            if self.verbose:
                print(f"[benchmark] Max relative error: {max_rel_error}")
                print(f"[benchmark] Exact match: {all_exact_match}")
            
            benchmark_path = Path(self.benchmark_file)
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            
            header = (
                "model_name,run_id,category,task_type,material_name,prompt_tokens,output_tokens,"
                "info_query_prompt_tokens,info_query_output_tokens,plan_prompt_tokens,plan_output_tokens,"
                "subproblem_prompt_tokens,subproblem_output_tokens,"
                "subproblem_dft_times,subproblem_script_times,subproblem_parse_validate_times,"
                "total_run_time,plan_and_query_time,total_dft_time,total_script_time,"
                "total_parse_validate_time,max_rel_error,all_exact_match\n"
            )
            
            need_header = not benchmark_path.exists() or benchmark_path.stat().st_size == 0
            
            with open(benchmark_path, "a", encoding="utf-8") as f:
                if need_header:
                    f.write(header)
                
                # Careful constructing the CSV line:
                # Lists are dumped as JSON strings.
                # Scalars are formatted floats.
                f.write(
                    f"{self.model},{run_id},{category},{task_type},{material_name},{prompt_tokens},{output_tokens},"
                    f"{info_query_prompt_tokens},{info_query_output_tokens},{plan_prompt_tokens},{plan_output_tokens},"
                    f"\"{json.dumps(subproblem_prompt_tokens)}\",\"{json.dumps(subproblem_output_tokens)}\","
                    f"\"{json.dumps(subproblem_dft_times)}\",\"{json.dumps(subproblem_script_times)}\",\"{json.dumps(subproblem_parse_validate_times)}\","
                    f"{total_run_time:.6f},{plan_and_query_time:.6f},{total_dft_time:.6f},"
                    f"{total_script_time:.6f},{total_parse_validate_time:.6f},"
                    f"{max_rel_error},{all_exact_match}\n"
                )

        return last_sub_problem_res
