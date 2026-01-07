import os
from typing import Callable, List, Optional, Tuple

from execute_code.mpi_run import run_mpirun_probe, run_with_mpirun
from execute_code.command_line import _run_direct
SlurmLauncher = Callable[
    [str, str, List[str], str, bool, bool, int, bool, Optional[str]],
    Tuple[List[int], List[str]],
]


def run_qe_inputs(
    exec_name: str,
    qe_prefix: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool = False,
    parallel_exec: bool = False,
    parallel_np: int = 1,
    run_mode: str = "mpirun",
    slurm_launcher: Optional[SlurmLauncher] = None,
    auto_parallel: bool = False,
    auto_parallel_generator: Optional[Callable[[str], List[dict]]] = None,
    max_new_tokens: int = 1024,
    hardware_description: Optional[str] = None,
    auto_confirm: bool = False,
) -> Tuple[List[int], List[str]]:
    """
    Execute QE inputs sequentially using the selected execution mode.
    """
    exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
    if run_mode == "mpirun":
        if auto_parallel:
            commands = run_mpirun_probe(
                exec_path,
                input_paths,
                work_dir,
                verbose,
                parallel_np,
                auto_parallel_generator,
                max_new_tokens,
                hardware_description,
            )
            if commands:
                retcodes: List[int] = []
                output_paths: List[str] = []
                for idx, (input_path, command) in enumerate(zip(input_paths, commands), start=1):
                    if not command:
                        raise RuntimeError(f"Auto-parallel command missing for input {idx}.")
                    if verbose:
                        print(f"[auto_parallel] Running recommended command for input {idx}: {command}")
                    if not _confirm_auto_parallel_run(auto_confirm):
                        raise RuntimeError("Auto-parallel execution cancelled by user.")
                    out_path = os.path.join(work_dir, f"output_{idx}.out")
                    run_codes, run_outputs = run_with_mpirun(
                        exec_path,
                        [input_path],
                        work_dir,
                        verbose,
                        parallel_np,
                        command=command,
                        output_paths=[out_path],
                    )
                    retcodes.extend(run_codes)
                    output_paths.extend(run_outputs)
                return retcodes, output_paths
        return run_with_mpirun(exec_path, input_paths, work_dir, verbose, parallel_np)
    elif run_mode == "local":
        return _run_direct(exec_path, input_paths, work_dir, verbose)
    elif run_mode == "slurm":
        if slurm_launcher is None:
            raise ValueError("Slurm launcher callback is required for slurm run mode.")
        return slurm_launcher(
            exec_name,
            qe_prefix,
            input_paths,
            work_dir,
            verbose,
            parallel_exec,
            parallel_np,
            auto_parallel,
            hardware_description,
        )
    else:
        raise ValueError(f"Unknown run mode: {run_mode}")


def _confirm_auto_parallel_run(auto_confirm: bool) -> bool:
    if auto_confirm:
        print("[auto_parallel] Auto-confirm enabled, proceeding with execution.")
        return True
    print("should I run this? (type 'yes' to confirm): ", end="")
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "n"
    return answer == "yes"
