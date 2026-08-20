import os
import re
import shlex
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from execute_code.command_line import _run_bash_command, _run_commands
from prompt.auto_parallel import auto_parallel_prompt


def _kpool_flag(exec_path: str, parallel_np: int) -> str:
    """`-nk` pools for pw.x / ph.x, when it helps and divides the rank count.

    Plane-wave parallelism saturates: measured on 6-atom MoS2, 8 -> 32 ranks was
    only 2.1x on its own, but 3.8x with pools, because the k-points parallelise
    almost linearly while the FFT work does not. Set QE_NK=0 to disable.

    Only pw.x and ph.x accept it; the post-processors do not, and passing it
    there is an error rather than a no-op.
    """
    import os as _os

    try:
        nk = int(_os.environ.get("QE_NK", "0"))
    except ValueError:
        return ""
    if nk <= 1 or parallel_np < 2:
        return ""
    binary = _os.path.basename(exec_path)
    if binary not in ("pw.x", "ph.x"):
        return ""
    # A pool count that does not divide the ranks makes QE abort.
    while nk > 1 and parallel_np % nk != 0:
        nk -= 1
    return f" -nk {nk}" if nk > 1 else ""


def _describe_cmd(cmd: str) -> str:
    """A short, user-facing description of a QE invocation.

    "mpirun --allow-run-as-root --bind-to none --oversubscribe -np 8
     /workspace/QuantumE/bin/pw.x -in input_3_1.in | tee output_3_1.out"
    -> "pw.x on 8 MPI ranks (input_3_1.in)"
    """
    binary = ""
    m = re.search(r"(\b\w+\.x)\b", cmd)
    if m:
        binary = m.group(1)
    np = ""
    m = re.search(r"-np\s+(\d+)", cmd)
    if m:
        np = f" on {m.group(1)} MPI ranks"
    src = ""
    m = re.search(r"-(?:in|inp|input_file)\s+(\S+)", cmd)
    if m:
        src = f" ({os.path.basename(m.group(1))})"
    return f"{binary or 'command'}{np}{src}"


def _log_full_cmd(cmd: str, work_dir: str) -> None:
    """Full invocation to the real stdout only — kept out of the user-facing
    stream (which the worker hijacks) but still present in the pod logs."""
    try:
        stream = sys.__stdout__
        if stream is not None:
            stream.write(f"[runner] cmd: {cmd} (cwd={work_dir})\n")
            stream.flush()
    except Exception:
        pass


def run_with_mpirun(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    parallel_np: int,
    command: Optional[str] = None,
    output_paths: Optional[List[str]] = None,
    timeout_seconds: int = 60000,
) -> Tuple[List[int], List[str]]:
    if command:
        return _run_mpirun_command(
            command,
            input_paths,
            work_dir,
            verbose,
            output_paths,
            timeout_seconds=timeout_seconds,
        )
    return _run_commands(
        exec_path,
        input_paths,
        work_dir,
        verbose,
        lambda e, inp, out: (
            # ev.x is an interactive prompt-driven tool: it reads its answers
            # from STDIN and does not understand -in. Given -in it saw an empty
            # stdin and died with "Fortran runtime error: End of file" at
            # ev.f90:80, every time. It is also serial, so mpirun is not just
            # unnecessary here but launches eight copies of a prompt reader.
            f"{shlex.quote(e)} < {shlex.quote(inp)} > {shlex.quote(out)} 2>&1"
            if os.path.basename(e) == "ev.x" else
            # --bind-to none + --oversubscribe: K8s CPU cgroups rarely expose a
            # clean set of bindable cores, so Open MPI's default core binding
            # aborts with "no available cpus" on busy/opportunistic nodes. Don't
            # bind to cores and allow oversubscription so the run launches (and
            # time-shares) instead of failing.
            f"mpirun --allow-run-as-root --bind-to none --oversubscribe -np {parallel_np} "
            f"{shlex.quote(e)}{_kpool_flag(e, parallel_np)} -in {shlex.quote(inp)} | tee {shlex.quote(out)}"
        ),
        output_paths=output_paths,
        timeout_seconds=timeout_seconds,
    )


def _run_mpirun_command(
    command: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    output_paths: Optional[List[str]] = None,
    timeout_seconds: int = 600,
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
            # The user-facing line names the binary, not the whole mpirun
            # invocation with its flags and cwd — that is debugging detail, and
            # the full command is still visible in the pod logs below.
            print(f"[runner] Running {_describe_cmd(cmd)}")
        _log_full_cmd(cmd, work_dir)
        rc, stdout, stderr, timed_out = _run_bash_command(
            cmd, work_dir, verbose, timeout_seconds=timeout_seconds
        )
        if verbose:
            print(f"[runner] Return code: {rc}")
            if timed_out and not stderr:
                stderr = f"[runner] Command exceeded {timeout_seconds}s and was terminated."
            if stderr:
                print(f"[runner][stderr]\n{stderr}")
        retcodes.append(rc)
        resolved_outputs.append(output_path)
        if timed_out:
            raise TimeoutError(
                f"mpirun command timed out after {timeout_seconds}s: {cmd}"
            )

    return retcodes, resolved_outputs


def _extract_output_path(command: str, work_dir: str) -> str:
    match = re.search(r"\|\s*tee\s+([^\s]+)", command)
    if not match:
        return os.path.join(work_dir, "output_1.out")
    output_name = match.group(1).strip().strip("'\"")
    if os.path.isabs(output_name):
        return output_name
    return os.path.join(work_dir, output_name)


def run_mpirun_probe(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    parallel_np: int,
    generator: Optional[Callable[[str], List[dict]]],
    max_new_tokens: int,
    hardware_description: Optional[str],
    output_paths: Optional[List[str]] = None,
) -> Tuple[int, List[str]]:
    """
    Run probe calculations and ask the LLM to generate mpirun commands.

    A generated command is considered VALID iff, after stripping leading
    whitespace, it starts with 'mpirun'.

    Return:
        rc:
            0  -> all generated commands are valid
           -1  -> at least one generated command is invalid
        commands:
            raw generated command strings (never None)
    """

    probe_paths = [_create_probe_script(path) for path in input_paths]

    probe_output_paths: List[str] = []
    for idx in range(1, len(probe_paths) + 1):
        if output_paths and idx - 1 < len(output_paths):
            base = output_paths[idx - 1]
            stem, ext = os.path.splitext(base)
            probe_output_paths.append(f"{stem}_probe{ext or '.out'}")
        else:
            probe_output_paths.append(os.path.join(work_dir, f"output_{idx}_probe.out"))

    # Run probe jobs with default parallel settings
    run_with_mpirun(
        exec_path,
        probe_paths,
        work_dir,
        verbose,
        parallel_np,
        output_paths=probe_output_paths,
    )

    summaries = []
    resolved_outputs: List[str] = []
    for idx, probe_output in enumerate(probe_output_paths, start=1):
        summaries.append(_extract_probe_summary(probe_output))
        if output_paths and idx - 1 < len(output_paths):
            resolved_outputs.append(output_paths[idx - 1])
        else:
            resolved_outputs.append(os.path.join(work_dir, f"output_{idx}.out"))

    if not summaries or not generator:
        return -1, []

    hw_desc = (
        hardware_description
        or f"Environment with up to {parallel_np} MPI ranks available on shared nodes."
    )

    commands: List[str] = []
    has_error = False

    for idx, summary in enumerate(summaries, start=1):
        input_name = (
            os.path.basename(input_paths[idx - 1])
            if idx - 1 < len(input_paths)
            else ""
        )
        output_name = (
            os.path.basename(resolved_outputs[idx - 1])
            if idx - 1 < len(resolved_outputs)
            else f"output_{idx}.out"
        )

        prompt_text = auto_parallel_prompt.format(
            exec_path=exec_path,
            input_script="",
            hardware_description=hw_desc,
            probe_output=summary,
            input_filename=input_name,
            output_filename=output_name,
        )

        if verbose:
            print(f"[auto_parallel] Querying LLM for auto-parallel plan (input {idx}).")

        try:
            result = generator(
                prompt_text,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
            )
        except Exception as exc:
            # Generator failure is treated as an invalid command
            commands.append(f"<generator exception>: {exc}")
            has_error = True
            continue

        raw_text = result[0].get("generated_text", "") if result else ""
        text = (raw_text or "").lstrip()  # only strip leading spaces

        commands.append(text)

        # Validation rule:
        # A command is valid iff it starts with 'mpirun'
        if text.startswith("Error"):
            has_error = True

    rc = 0 if not has_error else -1
    return rc, commands


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
