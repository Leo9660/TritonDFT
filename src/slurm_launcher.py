import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, List, Tuple

from prompt import get_prompt


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
    ) -> Tuple[List[int], List[str]]:
        work_dir_path = Path(work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)

        retcodes: List[int] = []
        output_paths: List[str] = []

        for idx, input_path in enumerate(input_paths, start=1):
            output_path = os.path.join(work_dir, f"output_{idx}.out")
            output_paths.append(output_path)
            script_text = self._generate_slurm_script(
                exec_name=exec_name,
                qe_prefix=qe_prefix,
                input_path=input_path,
                output_name=output_path,
                input_index=idx,
                work_dir=work_dir,
                parallel_exec=parallel_exec,
                parallel_np=parallel_np,
            )
            script_path = work_dir_path / f"slurm_job_{idx}.sh"
            script_content = f"{script_text.rstrip()}\n"
            script_path.write_text(script_content, encoding="utf-8")
            script_path.chmod(0o755)

            print(f"[slurm] Generated script ({script_path}):\n{script_content}")
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

        return retcodes, output_paths

    def _generate_slurm_script(
        self,
        exec_name: str,
        qe_prefix: str,
        input_path: str,
        output_name: str,
        input_index: int,
        work_dir: str,
        parallel_exec: bool,
        parallel_np: int,
    ) -> str:
        exec_path = os.path.join(qe_prefix, exec_name) if qe_prefix else exec_name
        try:
            content_lines = Path(input_path).read_text().splitlines()
        except (FileNotFoundError, OSError):
            content_lines = []
        input_context = "\n".join(content_lines[:]) or "No content preview."

        example_command = (
            f"{exec_path} -in {os.path.basename(input_path)} | tee {output_name}"
        )

        messages = get_prompt(
            prompt_type="slurm",
            exec_name=exec_name,
            exec_path=exec_path,
            work_dir=str(work_dir),
            input_dir=input_path,
            output_dir=output_name,
            num_inputs=1,
            parallel_exec=str(parallel_exec).lower(),
            parallel_np=parallel_np,
            example_command=example_command,
            input_context=input_context,
        )

        print("[debug] prompt is : {}".format(messages[0]["content"]))

        try:
            script_out = self.generator(
                messages[0]["content"],
                max_new_tokens=self.max_new_tokens,
                return_full_text=False,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[slurm] Script generation failed: {exc}")
            raise

        header_text = script_out[0]["generated_text"].strip()
        script_lines = ["#!/bin/bash"]
        if header_text:
            script_lines.extend(line for line in header_text.splitlines() if line.strip())
        script_lines.append("")
        script_lines.append(f"echo Running {os.path.basename(input_path)}")
        output_name = f"output_{input_index}.out"
        script_lines.append(
            self._build_slurm_command(
                exec_path=exec_path,
                input_name=os.path.basename(input_path),
                output_name=output_name,
                parallel_exec=parallel_exec,
                parallel_np=parallel_np,
            )
        )
        return "\n".join(script_lines).rstrip()

    def _build_slurm_command(
        self,
        exec_path: str,
        input_name: str,
        output_name: str,
        parallel_exec: bool,
        parallel_np: int,
    ) -> str:
        if parallel_exec:
            return (
                f"mpirun --allow-run-as-root -np {parallel_np} "
                f"{shlex.quote(exec_path)} -in {shlex.quote(input_name)} | "
                f"tee {shlex.quote(output_name)}"
            )
        return (
            f"{shlex.quote(exec_path)} -in {shlex.quote(input_name)} | "
            f"tee {shlex.quote(output_name)}"
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
