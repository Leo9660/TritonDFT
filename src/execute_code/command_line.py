import os
import shlex
import signal
import subprocess
from typing import Callable, List, Tuple


def _run_bash_command(
    cmd: str,
    work_dir: str,
    verbose: bool,
    timeout_seconds: int | None = None,
) -> tuple[int, str, str, bool]:
    process = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        timed_out = False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        timed_out = True
    if verbose and timed_out and timeout_seconds:
        print(f"[runner] Timeout: exceeded {timeout_seconds}s, terminated.")
    return process.returncode, stdout or "", stderr or "", timed_out


def _run_commands(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    build_cmd: Callable[[str, str, str], str],
    output_paths: List[str] | None = None,
    timeout_seconds: int | None = None,
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
        cmd = f"set -o pipefail; {cmd}"

        if verbose:
            print(f"[runner] Running: {cmd} (cwd={work_dir})")

        rc, stdout, stderr, timed_out = _run_bash_command(
            cmd, work_dir, verbose, timeout_seconds=timeout_seconds
        )
        retcodes.append(rc)
        if timed_out:
            raise TimeoutError(
                f"Command timed out after {timeout_seconds}s: {cmd}"
            )

        if verbose:
            print(f"[runner] Return code: {rc}")
            if timed_out and not stderr:
                stderr = f"[runner] Command exceeded {timeout_seconds}s and was terminated."
            if stderr:
                print(f"[runner][stderr]\n{stderr}")

    return retcodes, resolved_outputs


def _run_direct(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    output_paths: List[str] | None = None,
    timeout_seconds: int | None = None,
) -> Tuple[List[int], List[str]]:
    return _run_commands(
        exec_path,
        input_paths,
        work_dir,
        verbose,
        lambda e, inp, out: (
            f"{shlex.quote(e)} -in {shlex.quote(inp)} | tee {shlex.quote(out)}"
        ),
        output_paths=output_paths,
        timeout_seconds=timeout_seconds,
    )
