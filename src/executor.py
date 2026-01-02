import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from execute_code.mpi_run import run_with_mpirun
from execute_code.command_line import _run_direct
from prompt.auto_parallel import auto_parallel_prompt
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
) -> Tuple[List[int], List[str]]:
    """
    Execute QE inputs sequentially using the selected execution mode.
    """
    exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
    if run_mode == "mpirun":
        if auto_parallel:
            command = _run_probe_scripts(
                exec_path,
                input_paths,
                work_dir,
                verbose,
                parallel_np,
                auto_parallel_generator,
                max_new_tokens,
                hardware_description,
            )
            if command:
                if verbose:
                    print(f"[auto_parallel] Running recommended command: {command}")
                if not _confirm_auto_parallel_run():
                    raise RuntimeError("Auto-parallel execution cancelled by user.")
                return run_with_mpirun(
                    exec_path,
                    input_paths,
                    work_dir,
                    verbose,
                    parallel_np,
                    command=command,
                    output_paths=[
                        os.path.join(work_dir, f"output_{idx}.out")
                        for idx in range(1, len(input_paths) + 1)
                    ],
                )
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


def _run_probe_scripts(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    parallel_np: int,
    generator: Optional[Callable[[str], List[dict]]],
    max_new_tokens: int,
    hardware_description: Optional[str],
) -> Optional[str]:
    probe_paths = [_create_probe_script(path) for path in input_paths]
    probe_output_paths = [
        os.path.join(work_dir, f"output_{idx}_probe.out")
        for idx in range(1, len(probe_paths) + 1)
    ]
    run_with_mpirun(
        exec_path,
        probe_paths,
        work_dir,
        verbose,
        parallel_np,
        output_paths=probe_output_paths,
    )
    summaries = []
    output_paths = []
    for idx, probe_output in enumerate(probe_output_paths, start=1):
        summaries.append(_extract_probe_summary(probe_output))
        output_paths.append(os.path.join(work_dir, f"output_{idx}.out"))
    if not summaries or not generator:
        return None

    first_input = os.path.basename(input_paths[0]) if input_paths else ""
    first_output = output_paths[0] if output_paths else os.path.join(work_dir, "output_1.out")
    hw_desc = hardware_description or f"Environment with up to {parallel_np} MPI ranks available on shared nodes."
    prompt_text = auto_parallel_prompt.format(
        exec_path=exec_path,
        input_script="",
        hardware_description=hw_desc,
        probe_output="\n\n".join(summaries),
        input_filename=first_input,
        output_filename=first_output,
    )
    if verbose:
        print("[auto_parallel] Querying LLM for auto-parallel plan.")
    try:
        result = generator(
            prompt_text,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
    except Exception as exc:
        print(f"[auto_parallel] Failed to invoke generator: {exc}")
        return None
    if not result:
        return None
    command = result[0].get("generated_text", "").strip()
    return command or None


def _create_probe_script(input_path: str) -> str:
    original = Path(input_path)
    probe_path = original.with_name(f"{original.stem}_probe{original.suffix}")
    content = original.read_text()
    content = _ensure_parameter(content, "max_seconds", "30")
    content = _ensure_parameter(content, "verbosity", "'high'")
    probe_path.write_text(content, encoding="utf-8")
    return str(probe_path)


def _ensure_parameter(content: str, key: str, value: str) -> str:
    block_pattern = re.compile(r"^\s*&control\b", re.IGNORECASE)
    lines = content.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if block_pattern.match(line):
            start_idx = idx
            break

    if start_idx is None:
        insertion = f"&control\n{key} = {value}\n/\n"
        return insertion + content

    key_pattern = re.compile(rf"^\s*{key}\s*=", re.IGNORECASE)
    for idx in range(start_idx + 1, len(lines)):
        if lines[idx].strip().startswith("&"):
            insert_idx = idx
            break
        if key_pattern.match(lines[idx]):
            lines[idx] = f"{key} = {value}"
            return "\n".join(lines)
    else:
        insert_idx = len(lines)

    lines.insert(start_idx + 1, f"{key} = {value}")
    return "\n".join(lines)


def _extract_probe_summary(probe_output: str) -> str:
    text = Path(probe_output).read_text()
    marker = "Self-consistent Calculation"
    if marker in text:
        return text.split(marker, 1)[0].strip()
    return text[:1000].strip()


def _confirm_auto_parallel_run() -> bool:
    print("should I run this? (type 'yes' to confirm): ", end="")
    try:
        answer = input().strip().lower()
    except EOFError:
        answer = "n"
    return answer == "yes"
