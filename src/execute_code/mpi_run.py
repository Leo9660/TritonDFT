import os
import re
import shlex
import subprocess
from typing import List, Optional, Tuple

from execute_code.command_line import _run_commands


def run_with_mpirun(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    parallel_np: int,
    command: Optional[str] = None,
    output_paths: Optional[List[str]] = None,
) -> Tuple[List[int], List[str]]:
    if command:
        return _run_mpirun_command(command, input_paths, work_dir, verbose, output_paths)
    return _run_commands(
        exec_path,
        input_paths,
        work_dir,
        verbose,
        lambda e, inp, out: (
            f"mpirun --allow-run-as-root -np {parallel_np} "
            f"{shlex.quote(e)} -in {shlex.quote(inp)} | tee {shlex.quote(out)}"
        ),
        output_paths=output_paths,
    )


def _run_mpirun_command(
    command: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    output_paths: Optional[List[str]] = None,
) -> Tuple[List[int], List[str]]:
    placeholders = ("{input_filename}" in command) or ("{output_filename}" in command)
    inputs = input_paths if input_paths else [os.path.join(work_dir, "input_1.in")]
    retcodes: List[int] = []
    resolved_outputs: List[str] = []

    if not placeholders and len(inputs) > 1 and verbose:
        print("[runner] Warning: command has no input/output placeholders; running once for the first input.")

    targets = inputs if placeholders else inputs[:1]
    for idx, input_path in enumerate(targets, start=1):
        input_name = os.path.basename(input_path)
        output_name = (
            os.path.basename(output_paths[idx - 1])
            if output_paths and idx - 1 < len(output_paths)
            else f"output_{idx}.out"
        )
        cmd = command
        if placeholders:
            cmd = command.replace("{input_filename}", input_name).replace("{output_filename}", output_name)
        if output_paths and idx - 1 < len(output_paths):
            output_path = output_paths[idx - 1]
        else:
            output_path = _extract_output_path(cmd, work_dir)

        if verbose:
            print(f"[runner] Running: {cmd} (cwd={work_dir})")
        completed = subprocess.run(
            ["bash", "-lc", cmd],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if verbose:
            print(f"[runner] Return code: {completed.returncode}")
            if completed.stderr:
                print(f"[runner][stderr]\n{completed.stderr}")
        retcodes.append(completed.returncode)
        resolved_outputs.append(output_path)

    return retcodes, resolved_outputs


def _extract_output_path(command: str, work_dir: str) -> str:
    match = re.search(r"\|\s*tee\s+([^\s]+)", command)
    if not match:
        return os.path.join(work_dir, "output_1.out")
    output_name = match.group(1).strip().strip("'\"")
    if os.path.isabs(output_name):
        return output_name
    return os.path.join(work_dir, output_name)
