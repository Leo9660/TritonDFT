import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
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
parse_plan_string, patch_qe_input_file, get_qe_result, preprocess_output_list, extract_json_brutal, output_to_log_file
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
        backend: str = "hf",
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

        if self.verbose:
            print(f"[DFTAgent] Initialized with model={model}, dft_tool={dft_tool}, work_dir_root={self.work_dir_root}")

    def _prepare_run_directory(self) -> Path:
        timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.work_dir_root / f"job_{timestamp_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = run_dir
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
        
        # 1. Parameter Generation
        t0 = time.perf_counter()
        prompt = get_prompt(prompt_type="parameter", subproblem=subproblem['problem'],
                            fn=subproblem['tool'], tool=self.dft_tool, query=query, previous_memory=total_memory)
        try:
            params_out = self.generator(prompt[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            params_json = params_out[0]['generated_text']
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
            for i, path in enumerate(input_paths):
                exec_id = f"{problem_id}_{loop_count}_{i}"
                patch_qe_input_file(path, new_pseudo_dir=self.pseudo_dir, new_outdir=self.out_dir, new_prefix=f"subproblem_{exec_id}", pp_dir_clean=True)

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
            self._prepare_run_directory()
            
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