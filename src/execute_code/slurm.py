import os
import re
import subprocess
from pathlib import Path
from typing import Any, List, Optional, Tuple

from prompt import get_prompt
from prompt.auto_parallel import auto_parallel_prompt
from execute_code.slurm_template import render_slurm_script


class SlurmLauncher:
    def __init__(self, generator: Any, max_new_tokens: int, verbose: bool = False, auto_confirm: bool = False):
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.auto_confirm = auto_confirm

    def launch(
        self,
        exec_name: str,
        qe_prefix: str,
        input_paths: List[str],
        work_dir: str,
        verbose: bool,
        parallel_exec: bool,
        parallel_np: int,
        auto_parallel: bool = False,
        hardware_description: Optional[str] = None,
    ) -> Tuple[List[int], List[str]]:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

        if auto_parallel:
            commands = self._run_probe_scripts_and_generate_auto_parallel_commands(
                exec_name=exec_name,
                qe_prefix=qe_prefix,
                input_paths=input_paths,
                work_dir=work_dir,
                verbose=verbose,
                parallel_exec=parallel_exec,
                parallel_np=parallel_np,
                hardware_description=hardware_description,
            )
            if commands:
                return self._run_auto_parallel_command(
                    exec_name=exec_name,
                    qe_prefix=qe_prefix,
                    input_paths=input_paths,
                    work_dir=work_dir,
                    verbose=verbose,
                    parallel_exec=parallel_exec,
                    parallel_np=parallel_np,
                    commands=commands,
                )

        retcodes: List[int] = []
        output_paths: List[str] = []

        for idx, input_path in enumerate(input_paths, start=1):
            output_path = os.path.join(str(work_dir_path), f"output_{idx}.out")
            script_text = render_slurm_script(
                exec_path=os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name,
                input_path=str(input_path),
                output_path=str(output_path),
                command_line=f"mpirun -np {parallel_np} $exe -in $INPUT > $OUTPUT",
                tasks_per_node=parallel_np,
            )
            script_path = work_dir_path / f"slurm_job_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            # print(f"[slurm] Generated script ({script_path}):\n{script_content}")
            if not self._confirm_slurm_run():
                raise RuntimeError("Slurm execution cancelled by user.")

            completed = subprocess.run(
                ["bash", str(script_path)],
                cwd=str(work_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if verbose:
                if completed.stdout:
                    print(f"[slurm][stdout]\n{completed.stdout}")
                if completed.stderr:
                    print(f"[slurm][stderr]\n{completed.stderr}")

            retcodes.append(completed.returncode)
            output_paths.append(output_path)

        return retcodes, output_paths

    def _generate_slurm_script(
        self,
        exec_name: str,
        qe_prefix: str,
        input_path: str,
        output_path: str,
        work_dir: str,
        parallel_exec: bool,
        parallel_np: int,
        command_line: Optional[str] = None,
    ) -> str:
        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
        try:
            content_lines = Path(input_path).read_text().splitlines()
        except (FileNotFoundError, OSError):
            content_lines = []
        input_context = "\n".join(content_lines[:5]) or "No content preview."

        if command_line is None:
            messages = get_prompt(
                prompt_type="slurm",
                exec_name=exec_name,
                exec_path=exec_path,
                work_dir=str(work_dir),
                input_dir=str(input_path),
                output_dir=str(output_path),
                parallel_exec=str(parallel_exec).lower(),
                parallel_np=parallel_np,
                input_context=input_context,
            )
            try:
                script_out = self.generator(
                    messages[0]["content"],
                    max_new_tokens=self.max_new_tokens,
                    return_full_text=False,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[slurm] Command generation failed: {exc}")
                raise
            command_line = script_out[0]["generated_text"].strip()

        if not command_line:
            raise RuntimeError("Empty command line generated for Slurm script.")

        return render_slurm_script(
            exec_path=exec_path,
            input_path=str(input_path),
            output_path=str(output_path),
            command_line=command_line,
            tasks_per_node=parallel_np if parallel_exec else 1,
        )

    def _confirm_slurm_run(self) -> bool:
        if self.auto_confirm:
            print("[slurm] Auto-confirm enabled, proceeding with execution.")
            return True
        print("should I run this? (type 'yes' to confirm): ", end="")
        try:
            answer = input().strip().lower()
        except EOFError:
            answer = "n"
        return answer == "yes"

    def _run_probe_scripts_and_generate_auto_parallel_commands(
        self,
        exec_name: str,
        qe_prefix: str,
        input_paths: List[str],
        work_dir: str,
        verbose: bool,
        parallel_exec: bool,
        parallel_np: int,
        hardware_description: Optional[str],
    ) -> List[Optional[str]]:
        work_dir_path = Path(work_dir)
        probe_paths = [_create_probe_script(path) for path in input_paths]
        probe_outputs = [
            work_dir_path / f"output_{idx}_probe.out"
            for idx in range(1, len(probe_paths) + 1)
        ]
        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name

        for idx, (probe_path, probe_output) in enumerate(zip(probe_paths, probe_outputs), start=1):
            script_text = render_slurm_script(
                exec_path=exec_path,
                input_path=str(probe_path),
                output_path=str(probe_output),
                command_line=f"mpirun -np {parallel_np} $exe -in $INPUT > $OUTPUT",
                tasks_per_node=parallel_np,
            )

            script_path = work_dir_path / f"slurm_probe_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            print(f"[slurm] Generated probe script ({script_path}):\n{script_content}")
            if not self._confirm_slurm_run():
                raise RuntimeError("Slurm probe execution cancelled by user.")

            completed = subprocess.run(
                ["bash", str(script_path)],
                cwd=str(work_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if verbose:
                if completed.stdout:
                    print(f"[slurm][stdout]\n{completed.stdout}")
                if completed.stderr:
                    print(f"[slurm][stderr]\n{completed.stderr}")

        summaries = [_extract_probe_summary(str(path)) for path in probe_outputs]
        if not summaries:
            return []

        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
        hw_desc = hardware_description or f"Environment with up to {parallel_np} MPI ranks available on shared nodes."
        commands: List[Optional[str]] = []
        for idx, summary in enumerate(summaries, start=1):
            input_name = os.path.basename(input_paths[idx - 1]) if idx - 1 < len(input_paths) else ""
            output_name = f"output_{idx}.out"
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
                result = self.generator(
                    prompt_text,
                    max_new_tokens=self.max_new_tokens,
                    return_full_text=False,
                )
            except Exception as exc:
                if self.verbose:
                    print(f"[slurm] Auto-parallel generation failed: {exc}")
                commands.append(None)
                continue
            if not result:
                commands.append(None)
                continue
            command = result[0].get("generated_text", "").strip()
            commands.append(command or None)
        return commands

    def _run_auto_parallel_command(
        self,
        exec_name: str,
        qe_prefix: str,
        input_paths: List[str],
        work_dir: str,
        verbose: bool,
        parallel_exec: bool,
        parallel_np: int,
        commands: List[Optional[str]],
    ) -> Tuple[List[int], List[str]]:
        work_dir_path = Path(work_dir)
        retcodes: List[int] = []
        output_paths: List[str] = []
        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name

        for idx, input_path in enumerate(input_paths, start=1):
            command = commands[idx - 1] if idx - 1 < len(commands) else None
            if not command:
                raise RuntimeError(f"Auto-parallel command missing for input {idx}.")
            input_name = os.path.basename(input_path)
            output_path = os.path.join(str(work_dir_path), f"output_{idx}.out")
            output_name = os.path.basename(output_path)
            cmd = _render_auto_parallel_command(
                command=command,
                exec_path=exec_path,
                input_name=input_name,
                output_name=output_name,
            )
            script_text = render_slurm_script(
                exec_path=exec_path,
                input_path=str(input_path),
                output_path=str(output_path),
                command_line=cmd,
                tasks_per_node=parallel_np,
            )
            script_path = work_dir_path / f"slurm_job_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            if not self._confirm_slurm_run():
                raise RuntimeError("Slurm execution cancelled by user.")

            completed = subprocess.run(
                ["bash", str(script_path)],
                cwd=str(work_dir_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if verbose:
                if completed.stdout:
                    print(f"[slurm][stdout]\n{completed.stdout}")
                if completed.stderr:
                    print(f"[slurm][stderr]\n{completed.stderr}")

            retcodes.append(completed.returncode)
            output_paths.append(output_path)

        return retcodes, output_paths


def _render_auto_parallel_command(
    *,
    command: str,
    exec_path: str,
    input_name: str,
    output_name: str,
) -> str:
    cmd = command
    if "{exec_path}" in cmd:
        cmd = cmd.replace("{exec_path}", "$exe")
    cmd = cmd.replace(exec_path, "$exe")
    if "{input_filename}" in cmd:
        cmd = cmd.replace("{input_filename}", "$INPUT")
    else:
        cmd = cmd.replace(input_name, "$INPUT")
    if "{output_filename}" in cmd:
        cmd = cmd.replace("{output_filename}", "$OUTPUT")
    else:
        cmd = cmd.replace(output_name, "$OUTPUT")
    return cmd


def _create_probe_script(input_path: str) -> str:
    original = Path(input_path)
    probe_path = original.with_name(f"{original.stem}_probe{original.suffix}")
    content = original.read_text()
    content = _ensure_parameter(content, "max_seconds", "120")
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
