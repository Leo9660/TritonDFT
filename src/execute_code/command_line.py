import os
import shlex
import subprocess
from typing import Callable, List, Tuple


def _run_commands(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    build_cmd: Callable[[str, str, str], str],
    output_paths: List[str] | None = None,
) -> Tuple[List[int], List[str]]:
    resolved_outputs: List[str] = []
    retcodes: List[int] = []

    for idx, in_path in enumerate(input_paths, start=1):
        out_path = (
            output_paths[idx - 1]
            if output_paths and idx - 1 < len(output_paths)
            else os.path.join(work_dir, f"output_{idx}.out")
        )
        resolved_outputs.append(out_path)
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

    return retcodes, resolved_outputs


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
