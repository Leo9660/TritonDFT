import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List
import os
import datetime

from prompt import get_prompt
from tool import get_spec, fetch_initial_structures_from_api_snippet
from generator import UnifiedGenerator
from slurm_launcher import SlurmLauncher
from utils import get_qe_prefix, parse_scripts_block, write_inputs, \
parse_plan_string, patch_qe_input_file, get_qe_result, extract_json_brutal, output_to_log_file
from executor import run_qe_inputs

class DFTAgent:
    """
    DFTAgent: Minimal framework
    - run(user_query): entry point for the workflow
    - Internally: plan → execute → parse (mocked here)
    """

    def __init__(
        self,
        model: str,
        dft_tool: str = "quantum espresso",
        verbose: bool = False,
        work_dir: str = "tmp",
        max_new_tokens: int = 2048,
        backend: str = "hf",  # default to HF, will change to openai later
        temperature: float = 0.0,
        top_p: float = 1.0,
        vllm_tensor_parallel_size: int = None,
        openai_api_key: str = None,  # New parameter for OpenAI API Key
        openai_base_url: str = None,  # New parameter for OpenAI base URL
        need_query_info: bool = False,  # New parameter to control info query
        parallel_exec: bool = False,
        parallel_np: int = 1,
        run_mode: str = "mpirun",
        slurm_auto_confirm: bool = False, # whether the Slurm job submission is bypassing human confirmation, False stands for manual confirmation
        evaluation_mode: bool = False, # Evaluate results of each subproblem
        output_log: bool = False,
        output_log_file: str = "dft_agent_log.txt"
    ):
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

        self.parallel_exec = parallel_exec
        self.parallel_np = parallel_np
        valid_run_modes = {"mpirun", "local", "slurm"}
        if run_mode not in valid_run_modes:
            raise ValueError(f"run_mode must be one of {valid_run_modes}.")
        self.run_mode = run_mode
        self.slurm_auto_confirm = slurm_auto_confirm
        
        # Updated to support OpenAI API calls
        self.generator = UnifiedGenerator(
            backend=backend,
            model=model,
            default_max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=1234,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            verbose=self.verbose,
            openai_api_key=openai_api_key,  # Passing OpenAI API key
            openai_base_url=openai_base_url,  # Passing OpenAI base URL
        )

        self.slurm_launcher = SlurmLauncher(
            generator=self.generator,
            max_new_tokens=self.max_new_tokens,
            verbose=self.verbose,
            auto_confirm=self.slurm_auto_confirm,
        )

        # Tool setup params
        self.qe_bin_prefix = "../QuantumE/bin/"
        self.pseudo_dir = "../SSSP_clean/"
        self.out_dir = "./"

        # Evaluation and logging
        self.evaluation_mode = evaluation_mode
        self.output_log = output_log
        self.output_log_file = output_log_file

        # Numbers of SCF systems
        self.system_num = 0

        if self.verbose:
            print(
                f"[DFTAgent] Initialized with model={model}, dft_tool={dft_tool}, work_dir_root={self.work_dir_root}"
            )

    def _prepare_run_directory(self) -> Path:
        """Create a unique working directory for each run."""
        timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.work_dir_root / f"job_{timestamp_tag}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = run_dir
        if self.verbose:
            print(f"[DFTAgent] Using run directory: {self.work_dir}")
        return run_dir

    def info_query(self, query: str) -> Any:
        """
        Query material information from external databases (e.g., Material Project).
        """
        if self.verbose:
            print(f"[info_query] Querying material information for: {query}")
        
        
        api_messages = get_prompt(prompt_type="api_call", query=query)
        api_text = api_messages[0]["content"]

        # if self.verbose:
        #     print(f"[info_query] API call prompt: {api_text}")

        api_call_snippet_out = self.generator(api_text, max_new_tokens=self.max_new_tokens, return_full_text=False)
        api_call_snippet_out = api_call_snippet_out[0]['generated_text']

        if self.verbose:
            print(f"[info_query] API call snippet received: {api_call_snippet_out}")

        fetch_result = fetch_initial_structures_from_api_snippet(api_call_snippet_out, limit=25, verbose=self.verbose)
    
        if self.output_log:
            output_to_log_file(self.work_dir_root, self.output_log_file, f"[info_query] Retrieved material information: {fetch_result['material_ids'][0]}")

        if self.verbose:
            print(f"[info_query] Retrieved material information: {fetch_result['initial_structures'][0][0]}")

        return fetch_result

    def plan(self, query: str) -> List[Dict[str, Any]]:
        """
        Planner: convert user query into a sequence of steps using an LLM.
        1) Build a decomposition prompt (each subproblem = EXACTLY one tool call)
        2) Call the model to get a JSON plan
        3) Parse & validate -> normalize into a list of executable steps
        4) Fallback to a minimal plan when the LLM output is unusable
        """
        if self.verbose:
            print(f"[plan] Generating plan for query: {query}")

        # 1) build planner prompt (messages-like, but we will flatten for text-generation)
        messages = get_prompt(prompt_type="planner", question=query, tool=self.dft_tool)

        # 2) call model
        try:
            prompt_text = messages[0]["content"]
            raw_out = self.generator(prompt_text, max_new_tokens=self.max_new_tokens, return_full_text=False)
            # raw_out = raw_out[0]["generated_text"][len(prompt_text) :]
            if self.verbose:
                print(f"[plan] LLM raw output received: {raw_out[0]['generated_text']}")
        except Exception as e:
            if self.verbose:
                print(f"[plan][error] model call failed: {e}")
            return None

        # 3) extract json
        plan_dict = parse_plan_string(raw_out[0]["generated_text"])
        if self.verbose:
            print(f"[plan] Parsed {len(plan_dict)} steps.")

        return plan_dict

    def solve_sub_problem(self, subproblem: Dict[str, Any], problem_id: int = 0, query: str = "", total_memory: str = "", initial_structures: str = "") -> Any:
        """
        Solve a single subproblem using the specified DFT tool and function.
        """
        if self.verbose:
            print(f"[solve_sub_problem] Solving subproblem: {subproblem['problem']}")
            print(f"[solve_sub_problem] Using tool: {subproblem['tool']}")
        
        prompt = get_prompt(prompt_type="parameter", subproblem=subproblem['problem'],
        fn=subproblem['tool'], tool=self.dft_tool, query=query, previous_memory=total_memory)

        loop_time = 0

        try:
            params_out = self.generator(prompt[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            params_json = params_out[0]['generated_text']
            if self.verbose:
                print(f"[solve_sub_problem] parameter output received: {params_json}")
        except Exception as e:
            if self.verbose:
                print(f"[solve_sub_problem][error] subproblem solve failed: {e}")

        fn_spec = get_spec(subproblem['tool'])
        if fn_spec is None:
            raise ValueError(f"Unknown function/tool: {subproblem['tool']}")

        total_result_json = ""
        error_code = ""

        if self.verbose and total_memory != "":
            print(f"[solve_sub_problem] Previous memory received: {total_memory}")

        while (True):

            loop_time += 1

            # Previous run failed
            if loop_time > 1:
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
                subproblem=subproblem['problem'],
                tool_requirements=fn_spec.requirement
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
                subproblem=subproblem['problem'],
                tool_requirements=fn_spec.requirement
                )

            script_out = self.generator(script_prompt[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            if self.verbose:
                print(f"[solve_sub_problem] Script of tool call received: {script_out[0]['generated_text']}")

            generated = script_out[0]['generated_text']
            scripts = parse_scripts_block(generated)

            # work_dir setting
            work_dir = self.work_dir
            os.makedirs(work_dir, exist_ok=True)

            # write input files
            input_paths = write_inputs(work_dir, scripts, prefix="input", suffix=".in")
            if self.verbose:
                print(f"[solve_sub_problem] Wrote {len(input_paths)} input files to: {work_dir}")

            # patch input files
            for i, path in enumerate(input_paths):
                exec_id = f"{problem_id}_{loop_time}_{i}"
                patch_qe_input_file(path, new_pseudo_dir=self.pseudo_dir,
                new_outdir=self.out_dir,
                new_prefix=f"subproblem_{exec_id}",
                pp_dir_clean=True
            )

            # Evaluate the input scripts
            if self.evaluation_mode:
                for i, input_path in enumerate(input_paths):
                    if hasattr(fn_spec, "eval_input") and fn_spec.eval_input is not None:
                        eval_result = fn_spec.eval_input(input_path)
                        if self.verbose:
                            print(f"[solve_sub_problem] Input Evaluation result: {eval_result}")
                        if self.output_log:
                            output_to_log_file(self.work_dir_root, self.output_log_file, f"[Input check] {i}:\n {eval_result}")
                    else:
                        print(f"[Warning] No input evaluation function! Skipping input evaluation for {fn_spec.exec}.")

            # execute qe
            qe_prefix = get_qe_prefix(self)
            retcodes, output_paths = run_qe_inputs(
                exec_name=fn_spec.exec,
                qe_prefix=qe_prefix,
                input_paths=input_paths,
                work_dir=work_dir,
                verbose=self.verbose,
                parallel_exec=self.parallel_exec,
                parallel_np=self.parallel_np,
                run_mode=self.run_mode,
                slurm_launcher=self.slurm_launcher.launch if self.run_mode == "slurm" else None
            )

            input_list, output_list = get_qe_result(work_dir=work_dir, input_paths=input_paths, verbose=self.verbose)

            # Parse the output results
            for i, (input_file, output_file) in enumerate(zip(input_list, output_list)):
                messages = get_prompt(prompt_type="result_parse", input_json=params_json,
                input_file=input_file, output_text=output_file, fn=fn_spec.exec, parse_requirement=fn_spec.parse_requirement)

                try:
                    result_out = self.generator(messages[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
                    if self.verbose:
                        print(f"[solve_sub_problem] Result parsing output received: {result_out[0]['generated_text']}")
                    total_result_json += result_out[0]['generated_text']
                except Exception as e:
                    if self.verbose:
                        print(f"[solve_sub_problem][error] result parsing failed: {e}")
                    break

            # Judge the result of the final subproblem answer
            messages = get_prompt(prompt_type="result_judge", query=query, subproblem=subproblem['problem'],
            param_json=params_json, result_json=total_result_json)
            judge_out = self.generator(messages[0]['content'], max_new_tokens=self.max_new_tokens, return_full_text=False)
            if self.verbose:
                print(f"[solve_sub_problem] Result judging output received: {judge_out[0]['generated_text']}")
            
            judge_json = extract_json_brutal(judge_out[0]['generated_text'])
            
            if judge_json["status"] == "done":
                if self.verbose:
                    print(f"[solve_sub_problem] Finished: {subproblem['problem']}")
                break  # Exit the while loop if successful
            # error when two many iterations
            elif loop_time > 5:
                raise ValueError("Could not solve the subproblem!")
            else:
                params_json = json.dumps(judge_json["new_param_guess"])
                error_code += judge_out[0]['generated_text']
                if self.verbose:
                    print(f"[solve_sub_problem] new parameter output received: {params_json}")

        # Evaluate the output result
        if self.evaluation_mode:
            for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
                if hasattr(fn_spec, "eval_func") and fn_spec.eval_func is not None:
                    eval_result = fn_spec.eval_func(input_path, output_path)
                    if self.verbose:
                        print(f"[solve_sub_problem] Evaluation result: {eval_result}")
                else:
                    print(f"[Warning] No evaluation function! Skipping evaluation for {fn_spec.exec}.")

                if self.output_log:
                    input_str = Path(input_path).read_text()
                    output_to_log_file(self.work_dir_root, self.output_log_file, f"[Input Structure] {i}:\n{input_str}\n")
                    output_to_log_file(self.work_dir_root, self.output_log_file, f"[Output Evaluation] {i}:\n {eval_result}")


        # Placeholder for actual execution logic
        result = {
            "status": "success",
            "result_json": f"{total_result_json}",
            "result_judge": f"{judge_out[0]['generated_text']}",
            "details": f"Executed {subproblem['tool']}!"
        }
        return result

    def run(self, query: str) -> Any:
        """
        Main entry point: run the full workflow for a user query.
        1) Plan
        2) Execute each step (mocked here)
        3) Parse & aggregate results (mocked here)
        """
        self._prepare_run_directory()
        if self.verbose:
            print(f"[run] Starting workflow for query: {query}")
        
        if self.output_log:
            output_to_log_file(self.work_dir_root, self.output_log_file, f"###[Starting new run for query]: {query}\n", new=False)
            
        # 0) Query material information (Material Project)
        initial_structures = None
        if self.need_query_info:
            material_info = self.info_query(query)
            # comment: This is a temporary solution, need to be fixed later
            initial_structures = material_info['initial_structures']
            
            if self.verbose:
                print("[run] Initial structures obtained from Material Project.")

        # 1) Plan
        subproblems = self.plan(query = query)
        if len(subproblems) == 0:
            if self.verbose:
                print("[run] No valid plan generated. Exiting.")
            return None

        total_memory = ""

        # 2) Execute each step
        for i, step in enumerate(subproblems):
            if self.verbose:
                print(f"[run] Executing step {i+1}/{len(subproblems)}: {step['problem']}")
            
            # try:
            sub_problem_res = self.solve_sub_problem(step, problem_id=i+1, query=query, total_memory=total_memory, initial_structures=initial_structures)
            if self.verbose:
                print(f"[run] Subproblem {i+1} solved!")
            total_memory += f" Subproblem {i+1}:\n System Results:\n {sub_problem_res['result_json']} \n"
            total_memory += f" Conculsion of Subproblem {i+1}: {sub_problem_res['result_judge']} \n\n"
