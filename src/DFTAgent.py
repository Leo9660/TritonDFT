import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import shutil
import datetime
import time

# 假设这些模块存在于你的项目中
from prompt import get_prompt
from prompt.tool_requirements import get_parse_requirement
from config import Config
from generator import UnifiedGenerator
from execute_code.slurm import SlurmLauncher
from tool import get_spec, fetch_material_info_from_api_snippet, build_tool_requirements, is_allowed_fn
from utils import get_qe_prefix, parse_scripts_block, write_inputs, \
parse_plan_string, patch_qe_input_file, get_qe_result, preprocess_output_list, extract_json_brutal, output_to_log_file, \
validate_pseudos_exist, read_qe_cutoffs
from executor import run_qe_inputs
from evaluate.compare import compare_evaluation


class JobCancelled(Exception):
    """Raised inside the agent when an approval gate reports the job was
    cancelled by the user. The worker treats this as a clean stop, not a crash."""
    pass


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
        qe_timeout_seconds: int = 600,
        force_vc_relax: bool = True,
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
        self.qe_timeout_seconds = qe_timeout_seconds
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
        # Default ON: MP gives initial (un-relaxed) structures, so the planner
        # always prepends a pw_vc_relax step. Turn off to let the planner decide.
        self.force_vc_relax = force_vc_relax
        self.mpid_output_file = str(mpid_output_file) if mpid_output_file else None
        self.system_num = 0
        # Every step of a run shares one QE prefix — see patch_qe_input_file.
        # Each run gets its own work_dir, so a constant can't collide.
        self._run_prefix = "qerun"
        # Filled from the first pw.x input of a run; pinned onto every later step.
        self._run_cutoffs: Dict[str, str] = {}
        # The literal input files earlier steps of this run generated, fed into
        # later steps' script prompts so shared settings stay consistent.
        self._run_inputs: List[str] = []
        # Per-step manifest (binary, inputs, description) for run_all.sh.
        self._run_steps: List[Dict[str, Any]] = []

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

        self._write_structure_files(fetch_result)

        return fetch_result

    def _write_structure_files(self, material_info: Dict[str, Any]) -> None:
        """Persist the fetched structures as CIF + a summary JSON in the run dir.

        These used to be print()-ed in full into the streamed log. Writing them
        as files keeps them available (the artifacts endpoint already whitelists
        .cif/.json) without flooding the transcript.
        """
        try:
            summary = material_info.get("summary") or {}
            if summary:
                (self.work_dir / "material.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            for key, name in (("primitive_structure", "structure_primitive.cif"),
                              ("conventional_structure", "structure_conventional.cif")):
                items = material_info.get(key) or []
                if not items:
                    continue
                try:
                    (self.work_dir / name).write_text(items[0].to(fmt="cif"), encoding="utf-8")
                except Exception:
                    pass

            initial = material_info.get("initial_structures") or []
            if initial:
                # Already CIF text (see tool_mp), not a Structure object.
                (self.work_dir / "structure_initial.cif").write_text(
                    str(initial[0]), encoding="utf-8")
        except Exception as e:
            if self.verbose:
                print(f"[info_query][warn] could not write structure files: {e}")

    def plan(self, query: str) -> List[Dict[str, Any]]:
        if self.verbose:
            print(f"[plan] Generating plan for query: {query}")

        messages = get_prompt(prompt_type="planner", question=query, tool=self.dft_tool,
                              force_vc_relax=self.force_vc_relax)
        try:
            prompt_text = messages[0]["content"]
            raw_out = self.generator(prompt_text, max_new_tokens=self.max_new_tokens, return_full_text=False)
        except Exception as e:
            if self.verbose:
                print(f"[plan][error] model call failed: {e}")
            return None

        plan_dict = parse_plan_string(raw_out[0]["generated_text"])
        self._log_plan(plan_dict)
        return plan_dict

    @staticmethod
    def _step_meta(tool: str) -> Dict[str, str]:
        """Resolve a logical tool name to its executable + mode, for display.

        Never raises: an LLM (or a user editing the plan) can produce an unknown
        tool name, and a display helper must not take the run down.
        """
        try:
            spec = get_spec(tool)
        except (KeyError, ValueError):
            return {"exec": "", "mode": "", "description": "", "valid": False}
        return {
            "exec": spec.exec,
            "mode": spec.mode or "",
            "description": spec.description or "",
            "valid": True,
        }

    @classmethod
    def plan_payload(cls, subproblems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Structured, UI-ready view of a plan.

        The API stores this on the job row so the frontend renders real fields
        instead of scraping `<subproblem>` tags out of the stdout blob.
        """
        out = []
        for i, s in enumerate(subproblems or []):
            tool = (s.get("tool") or "").strip()
            meta = cls._step_meta(tool)
            out.append({
                "id": s.get("id", i + 1),
                "index": i + 1,
                "problem": (s.get("problem") or "").strip(),
                "tool": tool,
                "input": (s.get("input") or "").strip(),
                "exec": meta["exec"],
                "mode": meta["mode"],
                "description": meta["description"],
                "valid": meta["valid"],
            })
        return out

    def _log_plan(self, subproblems: List[Dict[str, Any]]) -> None:
        """Emit one readable line per step.

        Deliberately does NOT dump the raw <subproblem> blocks: that blob was the
        single noisiest thing in the streamed log, and everything useful in it is
        already in the structured fields below.
        """
        steps = self.plan_payload(subproblems)
        print(f"[plan] Parsed {len(steps)} steps.")
        for s in steps:
            binary = s["exec"] or "?"
            if s["mode"]:
                binary += f" · {s['mode']}"
            suffix = "" if s["valid"] else "  [unknown tool]"
            print(f"[plan] {s['index']}/{len(steps)} · {s['tool']} ({binary}) — {s['problem']}{suffix}")

    def refine_plan(self, subproblems: List[Dict[str, Any]], query: str, suggestion: str):
        """Re-plan from a user's natural-language instruction.

        Deliberately regenerates the WHOLE plan rather than patching one step:
        a suggestion like "drop the relaxation" or "add a DOS step" changes the
        step list, not just one field.
        """
        current = "\n".join(
            f"<subproblem{s['index']}>\n"
            f"Problem: {s['problem']}\n"
            f"Tool: {s['tool']}\n"
            f"Required input: {s['input']}\n"
            f"</subproblem{s['index']}>"
            for s in self.plan_payload(subproblems)
        )
        messages = get_prompt(prompt_type="plan_refine", question=query, tool=self.dft_tool,
                              current_plan=current, suggestion=suggestion)
        try:
            raw_out = self.generator(messages[0]["content"], max_new_tokens=self.max_new_tokens,
                                     return_full_text=False)
            revised = parse_plan_string(raw_out[0]["generated_text"])
        except Exception as e:
            # Keep the previous plan rather than failing the run — the user can
            # edit it by hand or just approve it.
            print(f"[plan][error] revision failed, keeping the previous plan: {e}")
            return subproblems
        print(f"[plan] Revised the plan per your suggestion.")
        self._log_plan(revised)
        return revised

    def _review_plan(self, subproblems: List[Dict[str, Any]], query: str, plan_gate) -> List[Dict[str, Any]]:
        """Human-in-the-loop gate on the PLAN, before any script is generated.

        Returns the (possibly edited) plan. Raises JobCancelled if the user
        cancels. Loops so the user can iterate: suggest → review → suggest → run.
        """
        MAX_ROUNDS = 10
        for _ in range(MAX_ROUNDS):
            decision = plan_gate({
                "query": query,
                "steps": self.plan_payload(subproblems),
            }) or {}
            action = decision.get("action", "approve")

            if action == "cancel":
                raise JobCancelled()

            if action == "suggest":
                suggestion = str(decision.get("suggestion", "")).strip()
                if suggestion:
                    subproblems = self.refine_plan(subproblems, query, suggestion)
                continue

            edited = decision.get("steps")
            if edited:
                merged = self._apply_plan_edits(subproblems, edited)
                if merged is not None:
                    subproblems = merged
                    print("[plan] Running your edited plan.")
                    self._log_plan(subproblems)
            return subproblems

        print("[plan][warn] too many revision rounds; running the current plan.")
        return subproblems

    @staticmethod
    def _apply_plan_edits(subproblems, edited):
        """Rebuild the plan from user-edited steps.

        Drops steps whose tool isn't in the allowed set — a bad tool name would
        otherwise blow up later in get_spec() with the run already half-done.
        """
        rebuilt = []
        for i, item in enumerate(edited or []):
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool", "")).strip()
            if not is_allowed_fn(tool):
                print(f"[plan][warn] dropping step {i + 1}: unknown tool '{tool}'")
                continue
            rebuilt.append({
                "id": i + 1,
                "problem": str(item.get("problem", "")).strip(),
                "tool": tool,
                "input": str(item.get("input", "")).strip(),
                "sweep": None,
            })
        if not rebuilt:
            print("[plan][warn] edited plan had no valid steps; keeping the original.")
            return None
        return rebuilt

    def solve_sub_problem(self, subproblem: Dict[str, Any], problem_id: int = 0, query: str = "", total_memory: str = "", material_info: Dict = [], approval_gate=None) -> Any:
        # No announcement here: run() already prints "[run] Executing step N/M:
        # <problem>" with the same text, and the plan card shows the tool. Both
        # lines used to be emitted and read as duplicates once the streamed log
        # stopped concatenating everything onto one line.

        # --- Subproblem Timing Accumulators ---
        # These must accumulate over the potential loops/retries
        acc_script_gen_time = 0.0
        acc_parse_validate_time = 0.0
        acc_dft_run_time = 0.0

        total_result_json = ""
        error_code = ""
        # Where this step's entries start in the run-wide input log (see below).
        inputs_mark = len(self._run_inputs)
        
        # 1. Parameter Generation
        t0 = time.perf_counter()
        prompt = get_prompt(prompt_type="parameter", subproblem=subproblem['problem'],
                            fn=subproblem['tool'], tool=self.dft_tool, query=query, previous_memory=total_memory)
        try:
            params_out = self.generator(prompt[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            params_json = params_out[0]['generated_text']
            # Don't dump the raw parameter JSON into the streamed log — it is a
            # multi-line LLM blob the user can't act on, and the values that
            # matter end up in the generated input file anyway (downloadable).
            if self.verbose:
                print(f"[solve_sub_problem] Parameters ready ({len(params_json)} chars).")
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

        fn_spec = get_spec(subproblem['tool'])
        if fn_spec is None:
            raise ValueError(f"Unknown function/tool: {subproblem['tool']}")

        initial_structures = material_info.get("initial_structures", [])
        conventional_structures = material_info.get("conventional_structure", [])
        primitive_structures = material_info.get("primitive_structure", [])

        loop_count = 0
        MAX_LOOPS = 3

        while True:
            loop_count += 1
            
            # --- Script Generation Phase ---
            t_script_start = time.perf_counter()
            
            tool_requirements = build_tool_requirements(fn_spec, self.pseudo_dirs)
            
            # Construct Prompt
            if loop_count > 1:
                script_prompt = get_prompt(prompt_type="script_fixed",
                    bin_tool=fn_spec.exec,
                    tool_mode=fn_spec.mode if fn_spec.mode else "standard",
                    params_json=params_json,
                    upf_dir=self.pseudo_dir,
                    previous_run=error_code,
                    previous_memory=total_memory,
                    previous_inputs="\n\n".join(self._run_inputs),
                    fn_section=fn_spec.section,
                    query=query,
                    initial_structures=initial_structures,
                    conventional_structure=conventional_structures,
                    primitive_structure=primitive_structures,
                    subproblem=subproblem['problem'],
                    tool_requirements=tool_requirements
                )
            else:
                script_prompt = get_prompt(prompt_type="script",
                    bin_tool=fn_spec.exec,
                    tool_mode=fn_spec.mode if fn_spec.mode else "standard",
                    params_json=params_json,
                    upf_dir=self.pseudo_dir,
                    previous_memory=total_memory,
                    previous_inputs="\n\n".join(self._run_inputs),
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

            scripts = parse_scripts_block(generated_scripts)
            
            # Work Dir setup & Write Inputs
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)
            subproblem_id = subproblem.get("id", problem_id)
            input_paths = write_inputs(work_dir, scripts, prefix="input", suffix=".in", subproblem_id=subproblem_id)
            
            # Patch Inputs
            missing_pseudo_err: Optional[str] = None
            for path in input_paths:
                patch_qe_input_file(path, new_pseudo_dir=self.pseudo_dir, new_outdir=self.out_dir,
                                    new_prefix=self._run_prefix if fn_spec.takes_prefix else None,
                                    pp_dir_clean=True, new_cutoffs=self._run_cutoffs)
                # The first pw.x step of a run fixes the cutoffs every later step
                # must reuse — they all read the same charge density.
                if not self._run_cutoffs and fn_spec.exec == "pw.x":
                    self._run_cutoffs = read_qe_cutoffs(path)
                    if self.verbose and self._run_cutoffs:
                        print(f"[solve_sub_problem] Run cutoffs pinned: {self._run_cutoffs}")
                # Fail fast (within the retry loop) if a pseudopotential the
                # LLM requested is missing — otherwise pw.x crashes with an
                # obscure mpirun rc=132.
                _, err = validate_pseudos_exist(path)
                if err:
                    missing_pseudo_err = err
                    if self.verbose:
                        print(f"[solve_sub_problem][warn] {err}")
                    break

            if missing_pseudo_err:
                if loop_count >= MAX_LOOPS:
                    raise ValueError(f"Could not solve the subproblem! {missing_pseudo_err}")
                error_code += f"Pseudo missing: {missing_pseudo_err}\n\n"
                params_json = json.dumps({"hint": missing_pseudo_err})
                continue

            # Input Eval (if enabled)
            if self.evaluation_mode and hasattr(fn_spec, "eval_input") and fn_spec.eval_input:
                for input_path in input_paths:
                    fn_spec.eval_input(input_path)

            acc_script_gen_time += (time.perf_counter() - t_script_start)

            # --- Human-in-the-loop approval gate ---
            # In assistant mode the worker passes a gate callback. We surface the
            # freshly generated (and patched) input file(s) and block until the
            # user approves / edits / suggests a revision / cancels.
            if approval_gate is not None:
                current_scripts = []
                for p in input_paths:
                    try:
                        with open(p, "r") as f:
                            current_scripts.append({"filename": os.path.basename(p), "content": f.read()})
                    except OSError:
                        pass
                decision = approval_gate({
                    "step_index": problem_id,
                    "step_id": subproblem_id,
                    "total_steps": getattr(self, "_total_steps", None),
                    "problem": subproblem.get("problem", ""),
                    "tool": subproblem.get("tool", ""),
                    "attempt": loop_count,
                }, current_scripts) or {}
                action = decision.get("action", "approve")

                if action == "cancel":
                    raise JobCancelled()

                if action == "suggest":
                    # Regenerate the script incorporating the user's feedback. The
                    # next loop uses the "script_fixed" prompt (loop_count > 1),
                    # which already folds in `error_code`.
                    suggestion = str(decision.get("suggestion", "")).strip()
                    error_code += f"\n[User feedback] Revise the input as requested: {suggestion}\n\n"
                    continue

                # approve (optionally with user-edited scripts) → run it.
                edited = decision.get("scripts")
                if edited:
                    by_name = {os.path.basename(p): p for p in input_paths}
                    for item in edited:
                        path = by_name.get(item.get("filename"))
                        if not path:
                            continue
                        with open(path, "w") as f:
                            f.write(str(item.get("content", "")).rstrip() + "\n\n")
                    # Re-apply path/prefix patching so QE still finds pseudos & outdir.
                    for path in input_paths:
                        patch_qe_input_file(path, new_pseudo_dir=self.pseudo_dir, new_outdir=self.out_dir,
                                            new_prefix=self._run_prefix if fn_spec.takes_prefix else None,
                                            pp_dir_clean=True, new_cutoffs=self._run_cutoffs)

            # Record this step's FINAL inputs (post-patch, post-user-edit) so later
            # steps can see them. Truncating to the entry-time mark first keeps a
            # retry from stacking several attempts of the same step.
            del self._run_inputs[inputs_mark:]
            for path in input_paths:
                try:
                    with open(path, "r") as f:
                        self._run_inputs.append(
                            f"# --- step {problem_id} ({subproblem.get('tool','')}), "
                            f"{os.path.basename(path)} ---\n{f.read().strip()}")
                except OSError:
                    pass

            # Step manifest, used to emit a runnable run_all.sh for script-only runs.
            self._run_steps = [s for s in self._run_steps if s["step"] != problem_id]
            self._run_steps.append({
                "step": problem_id,
                "tool": subproblem.get("tool", ""),
                "exec": fn_spec.exec,
                "mode": fn_spec.mode or "",
                "problem": subproblem.get("problem", ""),
                "inputs": [os.path.basename(p) for p in input_paths],
            })

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

            # Branch off the pristine SCF state rather than whatever the previous
            # step left in <prefix>.save (see _snapshot_scf_state).
            if fn_spec.exec == "ph.x" or (
                fn_spec.exec == "pw.x" and fn_spec.mode in self._SCF_BRANCH_MODES
            ):
                self._restore_scf_state()

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
                    timeout_seconds=self.qe_timeout_seconds,
                )
                # Successful execution block end
                acc_dft_run_time += (time.perf_counter() - t_dft_start)

                # The SCF is what every later branch starts from — checkpoint it
                # while it is still pristine.
                if fn_spec.exec == "pw.x" and fn_spec.mode == "scf":
                    self._snapshot_scf_state()

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

            input_list, output_list = get_qe_result(work_dir=work_dir, input_paths=input_paths, verbose=self.verbose, subproblem_id=subproblem_id)
            output_list = preprocess_output_list(output_list, verbose=self.verbose)
            
            # Result Parsing
            for i, (input_file, output_file) in enumerate(zip(input_list, output_list)):
                parse_requirement = get_parse_requirement(fn_spec.parse_requirement_key)
                messages = get_prompt(prompt_type="result_parse", input_json=params_json,
                                      input_file=input_file, output_text=output_file, fn=fn_spec.exec, parse_requirement=parse_requirement)
                try:
                    result_out = self.generator(messages[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
                    total_result_json += result_out[0]['generated_text']
                except Exception as e:
                    if self.verbose:
                        print(f"[solve_sub_problem][error] result parsing failed: {e}")
                    break

            # Result Judging
            messages = get_prompt(prompt_type="result_judge", query=query, subproblem=subproblem['problem'],
                                  param_json=params_json, result_json=total_result_json)
            judge_out = self.generator(messages[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            judge_json = extract_json_brutal(judge_out[0]['generated_text'])
            
            acc_parse_validate_time += (time.perf_counter() - t_parse_start)

            if judge_json.get("status") == "done":
                if self.verbose:
                    print(f"[solve_sub_problem] Finished: {subproblem['problem']}")
                break
            elif loop_count >= MAX_LOOPS:
                raise ValueError("Could not solve the subproblem! Max iterations reached.")
            else:
                # Prepare for next loop
                params_json = json.dumps(judge_json.get("new_param_guess", {}))
                error_code += f"Input script: {generated_scripts}. Error code: {judge_out[0]['generated_text']}\n\n"
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
            "timing": {
                "script_gen_s": acc_script_gen_time,
                "parse_validate_s": acc_parse_validate_time,
                "dft_run_s": acc_dft_run_time,
            },
            "evaluation": eval_result,
        }

        return result

    @staticmethod
    def _judge_prose(judge) -> str:
        """The agent's result_judge is often a JSON blob like
        {"status":"done","desc":"..."} — extract the human-readable sentence so
        the Analysis section reads as prose, not raw JSON."""
        if not judge or not isinstance(judge, str):
            return ""
        s = judge.strip()
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s).strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("desc", "description", "conclusion", "summary", "answer", "result"):
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                parts = [v.strip() for v in obj.values() if isinstance(v, str) and v.strip()]
                if parts:
                    return " ".join(parts)
        except Exception:
            pass
        return s

    # ── SCF state checkpointing ───────────────────────────────────────────────
    # Every step of a run shares one prefix, so they also share one <prefix>.save.
    # That is required for chaining (bands.x must read the bands run's
    # wavefunctions) but it breaks BRANCHING workflows: a DOS-oriented nscf with
    # occupations='tetrahedra' rewrites the saved state, and a later ph.x reading
    # it dies with "DFPT with the Blochl correction is not implemented".
    #
    # So the SCF state is treated as a checkpoint. Steps that branch off the SCF
    # (nscf, bands, ph.x) restore it first, instead of inheriting whatever the
    # previous step happened to leave behind. Post-processing steps (bands.x,
    # dos.x, q2r.x, ...) consume the step immediately before them and must NOT
    # restore.
    _SCF_BRANCH_MODES = {"nscf", "bands"}

    def _scf_state_paths(self):
        save = self.work_dir / f"{self._run_prefix}.save"
        xml = self.work_dir / f"{self._run_prefix}.xml"
        ckpt = self.work_dir / ".scf_checkpoint"
        return save, xml, ckpt

    def _snapshot_scf_state(self) -> None:
        save, xml, ckpt = self._scf_state_paths()
        if not save.is_dir():
            return
        try:
            if ckpt.exists():
                shutil.rmtree(ckpt)
            ckpt.mkdir(parents=True)
            shutil.copytree(save, ckpt / save.name)
            if xml.exists():
                shutil.copy2(xml, ckpt / xml.name)
            if self.verbose:
                print("[solve_sub_problem] SCF state checkpointed.")
        except OSError as e:
            if self.verbose:
                print(f"[solve_sub_problem][warn] could not checkpoint SCF state: {e}")

    def _restore_scf_state(self) -> None:
        save, xml, ckpt = self._scf_state_paths()
        src_save = ckpt / save.name
        if not src_save.is_dir():
            return
        try:
            if save.exists():
                shutil.rmtree(save)
            shutil.copytree(src_save, save)
            src_xml = ckpt / xml.name
            if src_xml.exists():
                shutil.copy2(src_xml, xml)
            if self.verbose:
                print("[solve_sub_problem] Restored the SCF state for this branch.")
        except OSError as e:
            if self.verbose:
                print(f"[solve_sub_problem][warn] could not restore SCF state: {e}")

    # Steps whose input embeds a geometry that only exists AFTER an earlier
    # relaxation has actually been run. In a script-only bundle nothing was run,
    # so the generated file still carries the Materials Project starting guess and
    # the user has to paste the relaxed cell in themselves.
    _RELAXING_MODES = {"relax", "vc-relax"}

    def _write_runner_script(self, query: str = "") -> Optional[str]:
        """Emit run_all.sh: every generated input, in order, with the manual
        substitutions called out.

        Script-only users get inputs but no results, so the bundle has to be
        self-explanatory — which binary runs which file, in what order, and where
        the chain genuinely cannot be automated.
        """
        steps = sorted(self._run_steps, key=lambda s: s["step"])
        if not steps:
            return None

        relax_steps = [s for s in steps if s["mode"] in self._RELAXING_MODES]
        L = [
            "#!/usr/bin/env bash",
            "#",
            "# Quantum ESPRESSO workflow generated by TritonDFT.",
            f"# Query: {query}".rstrip(),
            "#",
            "# Run from inside this directory:   bash run_all.sh",
            "#",
            "# Before running, set these to match your machine:",
            "#   QE_BIN     directory holding pw.x / bands.x / dos.x / ...",
            "#   PSEUDO_DIR directory holding the .upf pseudopotential files",
            "#   NP         number of MPI ranks",
            "#",
        ]
        if relax_steps:
            first = relax_steps[0]
            # Only pw.x inputs carry a geometry — a bands.x / dos.x namelist has
            # no CELL_PARAMETERS to paste into, so listing it would just confuse.
            later = [s for s in steps if s["step"] > first["step"] and s["exec"] == "pw.x"]
            L += [
                "# ---------------------------------------------------------------",
                "# MANUAL STEP — READ THIS FIRST",
                "# ---------------------------------------------------------------",
                f"# Step {first['step']} ({first['tool']}) relaxes the structure. Every later step",
                "# must start from the RELAXED geometry, but these inputs were generated",
                "# without running anything, so they still contain the unrelaxed starting",
                "# structure from the Materials Project.",
                "#",
                f"# After step {first['step']} finishes, open its output:",
                f"#     output_{first['step']}_1.out",
                "# find the LAST 'CELL_PARAMETERS' and 'ATOMIC_POSITIONS' blocks in it, and",
                "# paste them over the corresponding blocks in:",
            ]
            for s in later:
                L.append(f"#     {', '.join(s['inputs'])}   (step {s['step']}, {s['tool']})")
            L += [
                "#",
                "# The script pauses after the relaxation so you can do this.",
                "# ---------------------------------------------------------------",
                "#",
            ]
        L += [
            "set -euo pipefail",
            "",
            'QE_BIN="${QE_BIN:-/opt/qe/bin}"',
            'PSEUDO_DIR="${PSEUDO_DIR:-./pseudo}"',
            'NP="${NP:-4}"',
            "",
            "# Point every input at your pseudopotential directory.",
            'sed -i.bak "s|pseudo_dir *=.*|pseudo_dir = \'$PSEUDO_DIR\'|" *.in',
            "",
            "run() {  # run <binary> <input> <output>",
            '  echo "==> $1  $2"',
            '  mpirun -np "$NP" "$QE_BIN/$1" -in "$2" > "$3"',
            "}",
            "",
        ]
        for s in steps:
            L.append(f"# Step {s['step']}/{len(steps)} — {s['tool']}: {s['problem']}")
            for idx, name in enumerate(s["inputs"], start=1):
                out = f"output_{s['step']}_{idx}.out"
                L.append(f'run {s["exec"]} "{name}" "{out}"')
            if relax_steps and s["step"] == relax_steps[0]["step"] and s["step"] != steps[-1]["step"]:
                L += [
                    "",
                    'echo',
                    'echo "=============================================================="',
                    f'echo "Relaxation done. Copy the final CELL_PARAMETERS and"',
                    f'echo "ATOMIC_POSITIONS from output_{s["step"]}_1.out into the inputs"',
                    'echo "listed at the top of this script, then press Enter."',
                    'echo "=============================================================="',
                    'read -r _',
                ]
            L.append("")
        L.append('echo "All steps finished."')

        path = self.work_dir / "run_all.sh"
        try:
            path.write_text("\n".join(L) + "\n", encoding="utf-8")
            os.chmod(path, 0o755)
        except OSError as e:
            if self.verbose:
                print(f"[script_only][warn] could not write run_all.sh: {e}")
            return None
        return path.name

    def _write_analysis(self, analysis: str, query: str = "") -> None:
        """Persist the run's natural-language conclusion to ``analysis.json`` in
        the run directory so the API can surface it as the answer to the user's
        question (separate from the raw streamed log)."""
        try:
            payload = {"query": query, "analysis": (analysis or "").strip()}
            (self.work_dir / "analysis.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            if self.verbose:
                print(f"[analysis] write failed: {e}")

    def run(
        self,
        query: str,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
        work_dir: Optional[str] = None,
        approval_gate=None,
        plan_gate=None,
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

        # Publish the plan (and, in assistant mode, block for the user's review /
        # edits) before any script is generated. Runs in both modes: the auto-mode
        # hook just records the plan and approves immediately.
        if plan_gate is not None:
            subproblems = self._review_plan(subproblems, query, plan_gate)

        # --- Phase 2: Subproblem Execution ---
        total_memory = ""
        self._run_inputs = []   # per-run, not per-agent — the worker reuses the agent
        
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
        conclusions = []   # per-subproblem natural-language judgments → analysis.json
        self._total_steps = len(subproblems)   # surfaced to the approval gate

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
                material_info=material_info,
                approval_gate=approval_gate,
            )
            
            # Token Tracking
            subproblem_prompt_tokens.append(getattr(self.generator, "total_prompt_tokens", 0) - pt_before_sub)
            subproblem_output_tokens.append(getattr(self.generator, "total_output_tokens", 0) - ot_before_sub)

            if sub_problem_res and sub_problem_res.get("status") == "timeout":
                self._write_analysis(
                    "\n\n".join(conclusions) or "The run timed out before completing.", query)
                return sub_problem_res

            last_sub_problem_res = sub_problem_res
            judge = sub_problem_res.get("result_judge", "") if isinstance(sub_problem_res, dict) else ""
            prose = self._judge_prose(judge)
            if prose and prose not in ("script_only", "timeout"):
                label = f"Step {i+1}: " if len(subproblems) > 1 else ""
                conclusions.append(f"{label}{prose}".strip())
            
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
                # Walk EVERY step, in both modes. Script-only is the only thing
                # non-privileged accounts get, and stopping after step 1 handed
                # them a single vc-relax input for a five-step workflow — not
                # something they could actually run. No DFT executed, so there are
                # no results to carry forward; later steps still see earlier
                # INPUTS via previous_inputs.
                continue

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

        if self.script_only:
            runner = self._write_runner_script(query)
            n_inputs = sum(len(s["inputs"]) for s in self._run_steps)
            self._write_analysis(
                f"Script-only run: generated {n_inputs} Quantum ESPRESSO input file(s) "
                f"covering all {len(self._run_steps)} planned steps, without executing "
                f"them on CPU. Download the bundle below — "
                + (f"`{runner}` runs them in order and its comments mark the two places "
                   f"you must edit by hand (the relaxed geometry, which only exists once "
                   f"you have actually run the relaxation)."
                   if runner else
                   "run the inputs in numeric order."), query)
        else:
            self._write_analysis("\n\n".join(conclusions), query)
        return last_sub_problem_res