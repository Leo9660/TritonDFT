import os
import shlex
import subprocess
from typing import Callable, List, Optional, Tuple


SlurmLauncher = Callable[
    [str, str, List[str], str, bool, bool, int],
    Tuple[List[int], List[str]],
]


def _run_commands(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    build_cmd: Callable[[str, str, str], str],
) -> Tuple[List[int], List[str]]:
    output_paths: List[str] = []
    retcodes: List[int] = []

    for idx, in_path in enumerate(input_paths, start=1):
        out_path = os.path.join(work_dir, f"output_{idx}.out")
        output_paths.append(out_path)
        cmd = build_cmd(exec_path, os.path.basename(in_path), os.path.basename(out_path))

        if verbose:
            print(f"[runner] Running: {cmd} (cwd={work_dir})")

        completed = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        retcodes.append(completed.returncode)

        if verbose:
            print(f"[runner] Return code: {completed.returncode}")
            if completed.stderr:
                print(f"[runner][stderr]\n{completed.stderr}")

    return retcodes, output_paths


def _run_with_mpirun(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    parallel_np: int,
) -> Tuple[List[int], List[str]]:
    return _run_commands(
        exec_path,
        input_paths,
        work_dir,
        verbose,
        lambda e, inp, out: (
            f"mpirun --allow-run-as-root -np {parallel_np} "
            f"{shlex.quote(e)} -in {shlex.quote(inp)} | tee {shlex.quote(out)}"
        ),
    )


def _run_direct(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
) -> Tuple[List[int], List[str]]:
    return _run_commands(
        exec_path,
        input_paths,
        work_dir,
        verbose,
        lambda e, inp, out: (
            f"{shlex.quote(e)} -in {shlex.quote(inp)} | tee {shlex.quote(out)}"
        ),
    )


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
) -> Tuple[List[int], List[str]]:
    """
    Execute QE inputs sequentially using the selected execution mode.
    """
    exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
    if run_mode == "mpirun":
        return _run_with_mpirun(exec_path, input_paths, work_dir, verbose, parallel_np)
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
        )
    else:
        raise ValueError(f"Unknown run mode: {run_mode}")
