import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt import get_prompt
from prompt.auto_parallel import auto_parallel_prompt
from execute_code.slurm_template import render_slurm_script


class SlurmLauncher:
    def __init__(
        self,
        generator: Any,
        max_new_tokens: int,
        verbose: bool = False,
        auto_confirm: bool = False,
        default_time_limit: Optional[str] = None,
        template_path: Optional[str] = None,
    ):
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.verbose = verbose
        self.auto_confirm = auto_confirm
        self.default_time_limit = default_time_limit
        self.template_path = template_path or os.environ.get("TRITONDFT_SLURM_TEMPLATE", "")

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
        output_paths: Optional[List[str]] = None,
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
                output_paths=output_paths,
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
                    output_paths=output_paths,
                )

        retcodes: List[int] = []
        output_paths: List[str] = []

        for idx, input_path in enumerate(input_paths, start=1):
            resolved_np = self._parallel_np_for(exec_name, str(input_path), parallel_np)
            output_path = (
                output_paths[idx - 1]
                if output_paths and idx - 1 < len(output_paths)
                else os.path.join(str(work_dir_path), f"output_{idx}.out")
            )
            script_text = render_slurm_script(
                exec_path=os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name,
                input_path=str(input_path),
                output_path=str(output_path),
                command_line=f"mpirun -np {resolved_np} $exe -in $INPUT > $OUTPUT",
                nodes=1,
                tasks_per_node=resolved_np,
                work_dir=str(work_dir_path),
                time_limit=self._time_limit_for(exec_name, str(input_path), resolved_np),
                template_path=self.template_path,
            )
            script_path = work_dir_path / f"slurm_job_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            print(f"[slurm] Generated script ({script_path}):\n{script_content}")
            if not self._confirm_slurm_run():
                raise RuntimeError("Slurm execution cancelled by user.")

            completed = subprocess.run(
                ["sbatch", "--wait", str(script_path)],
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

    def package(
        self,
        exec_name: str,
        qe_prefix: str,
        input_paths: List[str],
        work_dir: str,
        parallel_exec: bool,
        parallel_np: int,
        output_paths: Optional[List[str]] = None,
        hardware_description: Optional[str] = None,
        probe_output_paths: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate Slurm scripts next to already-created QE inputs without
        submitting them. This is the local-desktop half of a future
        SSH/cluster workflow: build files locally, then let a transport layer
        upload this directory and run ``sbatch`` on the remote cluster.
        """
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

        script_paths: List[str] = []
        for idx, input_path in enumerate(input_paths, start=1):
            input_name = os.path.basename(str(input_path))
            output_name = (
                os.path.basename(str(output_paths[idx - 1]))
                if output_paths and idx - 1 < len(output_paths)
                else f"output_{idx}.out"
            )
            try:
                plan = self._generate_slurm_auto_parallel_plan(
                    exec_name=exec_name,
                    qe_prefix=qe_prefix,
                    input_path=str(input_path),
                    input_name=input_name,
                    output_name=output_name,
                    parallel_np=parallel_np,
                    hardware_description=hardware_description,
                    probe_output_path=(
                        probe_output_paths[idx - 1]
                        if probe_output_paths and idx - 1 < len(probe_output_paths)
                        else None
                    ),
                )
            except Exception as exc:
                print(
                    "[slurm] Auto-parallel resource planning failed; "
                    f"using deterministic fallback settings ({exc})."
                )
                plan = self._fallback_slurm_plan(
                    exec_name=exec_name,
                    input_path=str(input_path),
                    parallel_np=parallel_np,
                )
            script_text = self._generate_slurm_script(
                exec_name=exec_name,
                qe_prefix=qe_prefix,
                input_path=input_name,
                output_path=output_name,
                work_dir=".",
                parallel_exec=parallel_exec,
                parallel_np=plan["mpi_ranks"],
                command_line=plan["command_line"],
                time_limit=plan["time_limit"],
                nodes=plan["nodes"],
                tasks_per_node=plan["tasks_per_node"],
            )
            input_stem = Path(input_path).stem
            script_path = work_dir_path / f"slurm_job_{input_stem}.sh"
            script_path.write_text(f"{script_text.rstrip()}\n", encoding="utf-8")
            script_path.chmod(0o755)
            script_paths.append(str(script_path))

            if self.verbose:
                print(f"[slurm] Packaged script: {script_path}")

        return script_paths

    def package_probe(
        self,
        *,
        exec_name: str,
        qe_prefix: str,
        input_paths: List[str],
        work_dir: str,
        parallel_np: int,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Create short probe inputs/scripts for remote execution."""
        work_dir_path = Path(work_dir)
        probe_inputs: List[str] = []
        probe_outputs: List[str] = []
        probe_scripts: List[str] = []
        probe_np = max(1, parallel_np)
        for input_path in input_paths:
            probe_input = _create_probe_script(input_path, exec_name=exec_name)
            input_stem = Path(input_path).stem
            probe_output = work_dir_path / f"{input_stem}_probe.out"
            probe_script = work_dir_path / f"slurm_probe_{input_stem}.sh"
            script_text = render_slurm_script(
                exec_path=os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name,
                input_path=Path(probe_input).name,
                output_path=probe_output.name,
                command_line=(
                    f"export OMP_NUM_THREADS=1; mpirun --allow-run-as-root -np {probe_np} "
                    "$exe -in $INPUT > $OUTPUT"
                ),
                nodes=1,
                tasks_per_node=probe_np,
                work_dir=".",
                time_limit="00:10:00",
                template_path=self.template_path,
            )
            probe_script.write_text(script_text.rstrip() + "\n", encoding="utf-8")
            probe_script.chmod(0o755)
            probe_inputs.append(str(probe_input))
            probe_outputs.append(str(probe_output))
            probe_scripts.append(str(probe_script))
        return probe_inputs, probe_outputs, probe_scripts

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
        time_limit: Optional[str] = None,
        nodes: int = 1,
        tasks_per_node: Optional[int] = None,
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
            nodes=max(1, nodes),
            tasks_per_node=max(1, tasks_per_node or parallel_np),
            work_dir=str(work_dir),
            time_limit=time_limit or self._time_limit_for(exec_name, str(input_path), parallel_np),
            template_path=self.template_path,
        )

    def _generate_slurm_auto_parallel_plan(
        self,
        *,
        exec_name: str,
        qe_prefix: str,
        input_path: str,
        input_name: str,
        output_name: str,
        parallel_np: int,
        hardware_description: Optional[str],
        probe_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
        try:
            input_script = Path(input_path).read_text(encoding="utf-8")
        except OSError:
            input_script = ""

        hw_desc = hardware_description or _default_slurm_hardware_description(parallel_np)
        if probe_output_path and Path(probe_output_path).exists():
            probe_output = _extract_probe_summary(probe_output_path)
        else:
            probe_output = (
                "No probe output is available. Decide from the QE input, executable, and "
                "hardware description. Be conservative only when the workload is genuinely small."
            )
        prompt_text = auto_parallel_prompt.format(
            exec_path="$exe",
            input_script=input_script,
            hardware_description=hw_desc,
            probe_output=probe_output,
            input_filename="$INPUT",
            output_filename="$OUTPUT",
        )
        prompt_text += """

# Slurm packaging requirement
Use the same QE parallelization reasoning above, but also choose the Slurm resources for this batch job.
Do not use a fixed default for all jobs. Choose walltime, nodes, and tasks_per_node from the calculation size,
calculation type, k-points, and requested executable.

After the normal Analysis and Command lines, add exactly one final machine-readable line:
Slurm: {"nodes": <int>, "tasks_per_node": <int>, "time_limit": "HH:MM:SS"}

The Command line must use $exe, $INPUT, and $OUTPUT exactly, so it can be inserted into a generated Slurm script.
"""
        if self.verbose:
            print("[slurm] Querying auto-parallel prompt for Slurm resources.")

        result = self.generator(
            prompt_text,
            max_new_tokens=self.max_new_tokens,
            return_full_text=False,
        )
        generated = result[0].get("generated_text", "").strip() if result else ""
        if not generated:
            raise RuntimeError("Auto-parallel prompt returned an empty Slurm plan.")
        if self.verbose:
            print(f"[slurm][auto_parallel]\n{generated}")

        command_line = _extract_auto_parallel_command(generated)
        if not command_line:
            raise RuntimeError("Auto-parallel prompt did not return a Command line for Slurm packaging.")
        command_line = _render_auto_parallel_command(
            command=command_line,
            exec_path=exec_path,
            input_name=input_name,
            output_name=output_name,
        )

        resources = _extract_slurm_resource_line(generated)
        mpi_ranks = _extract_mpi_ranks(command_line) or parallel_np or 1
        cores_per_node = _slurm_cores_per_node(parallel_np)
        nodes = int(resources.get("nodes") or max(1, math.ceil(mpi_ranks / cores_per_node)))
        tasks_per_node = int(resources.get("tasks_per_node") or max(1, math.ceil(mpi_ranks / nodes)))
        time_limit = _normalize_slurm_time_limit(resources.get("time_limit") or self.default_time_limit)
        if not time_limit:
            raise RuntimeError("Auto-parallel prompt did not return a Slurm time_limit.")

        return {
            "command_line": command_line,
            "mpi_ranks": max(1, mpi_ranks),
            "nodes": max(1, nodes),
            "tasks_per_node": max(1, tasks_per_node),
            "time_limit": time_limit,
        }

    def _fallback_slurm_plan(
        self,
        *,
        exec_name: str,
        input_path: str,
        parallel_np: int,
    ) -> Dict[str, Any]:
        """Build a safe package when the optional LLM resource planner fails."""
        mpi_ranks = self._parallel_np_for(exec_name, input_path, parallel_np)
        if exec_name in {"bands.x", "dos.x", "q2r.x", "matdyn.x", "dynmat.x", "ev.x"}:
            mpi_ranks = 1

        try:
            input_text = Path(input_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            input_text = ""

        calculation_match = re.search(
            r"(?mi)^\s*calculation\s*=\s*['\"]([^'\"]+)['\"]",
            input_text,
        )
        calculation = calculation_match.group(1).strip().lower() if calculation_match else ""
        if self.default_time_limit:
            time_limit = self.default_time_limit
        elif exec_name == "ph.x":
            time_limit = "12:00:00"
        elif calculation in {"vc-relax", "relax"}:
            time_limit = "04:00:00"
        elif calculation in {"nscf", "bands"}:
            time_limit = "02:00:00"
        elif exec_name == "pw.x":
            time_limit = "01:00:00"
        else:
            time_limit = "00:30:00"

        return {
            "command_line": (
                f"export OMP_NUM_THREADS=1; mpirun --allow-run-as-root -np {mpi_ranks} "
                "$exe -in $INPUT > $OUTPUT"
            ),
            "mpi_ranks": mpi_ranks,
            "nodes": 1,
            "tasks_per_node": mpi_ranks,
            "time_limit": time_limit,
        }

    def _time_limit_for(self, exec_name: str, input_path: str, parallel_np: int) -> str:
        return self.default_time_limit or "00:10:00"

    def _parallel_np_for(self, exec_name: str, input_path: str, parallel_np: int) -> int:
        if parallel_np and parallel_np > 0:
            return parallel_np
        return 1

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
        output_paths: Optional[List[str]] = None,
    ) -> List[Optional[str]]:
        work_dir_path = Path(work_dir)
        probe_paths = [_create_probe_script(path, exec_name=exec_name) for path in input_paths]
        probe_outputs: List[Path] = []
        for idx in range(1, len(probe_paths) + 1):
            if output_paths and idx - 1 < len(output_paths):
                base = Path(output_paths[idx - 1])
                probe_outputs.append(base.with_name(f"{base.stem}_probe{base.suffix or '.out'}"))
            else:
                probe_outputs.append(work_dir_path / f"output_{idx}_probe.out")
        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name

        for idx, (probe_path, probe_output) in enumerate(zip(probe_paths, probe_outputs), start=1):
            script_text = render_slurm_script(
                exec_path=exec_path,
                input_path=str(probe_path),
                output_path=str(probe_output),
                command_line=f"mpirun -np {parallel_np} $exe -in $INPUT > $OUTPUT",
                nodes=1,
                tasks_per_node=parallel_np,
                work_dir=str(work_dir_path),
                time_limit=self._time_limit_for(exec_name, str(probe_path), parallel_np),
                template_path=self.template_path,
            )

            script_path = work_dir_path / f"slurm_probe_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            print(f"[slurm] Generated probe script ({script_path}):\n{script_content}")
            if not self._confirm_slurm_run():
                raise RuntimeError("Slurm probe execution cancelled by user.")

            completed = subprocess.run(
                ["sbatch", "--wait", str(script_path)],
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
            if output_paths and idx - 1 < len(output_paths):
                output_name = os.path.basename(output_paths[idx - 1])
            else:
                output_name = f"output_{idx}.out"
            try:
                input_script = Path(input_paths[idx - 1]).read_text()
            except OSError:
                input_script = ""
            prompt_text = auto_parallel_prompt.format(
                exec_path=exec_path,
                input_script=input_script,
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
        output_paths: Optional[List[str]] = None,
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
            output_path = (
                output_paths[idx - 1]
                if output_paths and idx - 1 < len(output_paths)
                else os.path.join(str(work_dir_path), f"output_{idx}.out")
            )
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
                nodes=1,
                tasks_per_node=parallel_np,
                work_dir=str(work_dir_path),
                time_limit=self._time_limit_for(exec_name, str(input_path), parallel_np),
                template_path=self.template_path,
            )
            script_path = work_dir_path / f"slurm_job_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            print(f"[slurm] Generated script ({script_path}):\n{script_content}")
            if not self._confirm_slurm_run():
                raise RuntimeError("Slurm execution cancelled by user.")

            completed = subprocess.run(
                ["sbatch", "--wait", str(script_path)],
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
    cmd = re.sub(r"^Command:\s*", "", cmd.strip(), flags=re.IGNORECASE)
    cmd = cmd.strip("`").strip()
    if "$OUTPUT" not in cmd:
        cmd = f"{cmd} > $OUTPUT"
    return cmd


def _extract_auto_parallel_command(generated: str) -> str:
    command_match = re.search(
        r"^\s*Command:\s*(.+)$",
        generated,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if command_match:
        return command_match.group(1).strip()

    for line in generated.splitlines():
        stripped = line.strip().strip("`")
        if re.search(r"\b(mpirun|mpiexec|srun)\b", stripped):
            return stripped
    return ""


def _extract_slurm_resource_line(generated: str) -> Dict[str, Any]:
    match = re.search(r"^\s*Slurm:\s*(\{.*\})\s*$", generated, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _extract_mpi_ranks(command_line: str) -> int:
    patterns = [
        r"\b(?:mpirun|mpiexec)\b.*?(?:-np|-n)\s+(\d+)",
        r"\bsrun\b.*?(?:-n|--ntasks(?:=|\s+))(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, command_line)
        if match:
            return int(match.group(1))
    return 0


def _normalize_slurm_time_limit(value: Optional[Any]) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", text):
        hours, minutes, seconds = text.split(":")
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    if re.match(r"^\d{1,2}:\d{2}$", text):
        minutes, seconds = text.split(":")
        return f"00:{int(minutes):02d}:{int(seconds):02d}"
    if text.isdigit():
        return _format_slurm_minutes(int(text))
    return ""


def _default_slurm_hardware_description(parallel_np: int) -> str:
    cores_per_node = _slurm_cores_per_node(parallel_np)
    return (
        f"Remote Slurm cluster. Assume shared/compute nodes with {cores_per_node} physical cores per node. "
        "Choose one or more nodes only when the QE workload can benefit from that scale. "
        "The Slurm script can set #SBATCH --nodes and #SBATCH --tasks-per-node."
    )


def _slurm_cores_per_node(parallel_np: int) -> int:
    for name in ("TRITONDFT_SLURM_CORES_PER_NODE", "TRITONDFT_MAX_MPI_RANKS"):
        try:
            value = int(os.environ.get(name, ""))
        except ValueError:
            value = 0
        if value > 0:
            return value
    return parallel_np if parallel_np and parallel_np > 0 else 64


def _format_slurm_minutes(minutes: int) -> str:
    minutes = max(1, int(minutes))
    hours, mins = divmod(minutes, 60)
    return f"{hours:02d}:{mins:02d}:00"


def _create_probe_script(input_path: str, *, exec_name: str = "pw.x") -> str:
    original = Path(input_path)
    probe_path = original.with_name(f"{original.stem}_probe{original.suffix}")
    content = original.read_text()
    namelist = "inputph" if exec_name == "ph.x" else "control"
    content = _ensure_parameter(content, "max_seconds", "120", namelist=namelist)
    if exec_name == "pw.x":
        content = _ensure_parameter(content, "verbosity", "'high'", namelist="control")
    probe_path.write_text(content, encoding="utf-8")
    return str(probe_path)


def _ensure_parameter(
    content: str,
    key: str,
    value: str,
    *,
    namelist: str = "control",
) -> str:
    block_pattern = re.compile(rf"^\s*&{re.escape(namelist)}\b", re.IGNORECASE)
    lines = content.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if block_pattern.match(line):
            start_idx = idx
            break

    if start_idx is None:
        insertion = f"&{namelist}\n{key} = {value}\n/\n"
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
