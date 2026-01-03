import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from execute_code.command_line import _run_commands
from prompt.auto_parallel import auto_parallel_prompt


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


def run_mpirun_probe(
    exec_path: str,
    input_paths: List[str],
    work_dir: str,
    verbose: bool,
    parallel_np: int,
    generator: Optional[Callable[[str], List[dict]]],
    max_new_tokens: int,
    hardware_description: Optional[str],
) -> List[Optional[str]]:
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
        return []

    hw_desc = hardware_description or f"Environment with up to {parallel_np} MPI ranks available on shared nodes."
    commands: List[Optional[str]] = []
    for idx, summary in enumerate(summaries, start=1):
        input_name = os.path.basename(input_paths[idx - 1]) if idx - 1 < len(input_paths) else ""
        output_name = os.path.basename(output_paths[idx - 1]) if idx - 1 < len(output_paths) else f"output_{idx}.out"
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
            print(f"[auto_parallel] Failed to invoke generator: {exc}")
            commands.append(None)
            continue
        if not result:
            commands.append(None)
            continue
        command = result[0].get("generated_text", "").strip()
        commands.append(command or None)
    return commands


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
