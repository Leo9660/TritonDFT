import os
import re
from pathlib import Path

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --partition=shared
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={tasks_per_node}
#SBATCH -t {time_limit}
#SBATCH -o {log_out}
#SBATCH -e {log_err}
#SBATCH --export=ALL
#SBATCH --job-name=tritondft-qe
module load openmpi
module load quantum-espresso
# Set the executable and input file
exe={exec_path}
INPUT={input_path}
OUTPUT={output_path}
{command_line}
"""


def render_slurm_script(
    *,
    exec_path: str,
    input_path: str,
    output_path: str,
    command_line: str,
    nodes: int,
    tasks_per_node: int,
    work_dir: str,
    time_limit: str = "00:10:00",
    template_path: str = "",
) -> str:
    if template_path:
        template_text = _read_template(template_path)
        if template_text:
            return _render_from_example(
                template_text,
                exec_path=exec_path,
                input_path=input_path,
                output_path=output_path,
                command_line=command_line,
                nodes=nodes,
                tasks_per_node=tasks_per_node,
                work_dir=work_dir,
                time_limit=time_limit,
            )

    return SLURM_TEMPLATE.format(
        exec_path=exec_path,
        input_path=input_path,
        output_path=output_path,
        command_line=command_line,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        time_limit=time_limit,
        log_out=os.path.join(work_dir, "qe.out"),
        log_err=os.path.join(work_dir, "qe.err"),
    ).rstrip()


def _read_template(template_path: str) -> str:
    path = Path(template_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _render_from_example(
    template_text: str,
    *,
    exec_path: str,
    input_path: str,
    output_path: str,
    command_line: str,
    nodes: int,
    tasks_per_node: int,
    work_dir: str,
    time_limit: str,
) -> str:
    lines = template_text.splitlines()
    lines = _replace_or_add_sbatch(lines, ["--nodes"], str(nodes))
    lines = _replace_or_add_sbatch(lines, ["--tasks-per-node"], str(tasks_per_node))
    lines = _replace_or_add_sbatch(lines, ["-t", "--time"], time_limit, preferred="-t")
    lines = _replace_or_add_sbatch(lines, ["-o", "--output"], os.path.join(work_dir, "qe.out"), preferred="-o")
    lines = _replace_or_add_sbatch(lines, ["-e", "--error"], os.path.join(work_dir, "qe.err"), preferred="-e")

    lines = _strip_old_qe_command_block(lines)
    block = [
        "",
        "# TritonDFT generated execution block",
        f"exe={exec_path}",
        f"INPUT={input_path}",
        f"OUTPUT={output_path}",
        command_line,
    ]
    return "\n".join(lines + block).strip()


def _replace_or_add_sbatch(
    lines: list[str],
    options: list[str],
    value: str,
    *,
    preferred: str = "",
) -> list[str]:
    preferred = preferred or options[0]
    pattern = re.compile(
        rf"^\s*#SBATCH\s+(?:{'|'.join(re.escape(option) for option in options)})(?:=|\s+).*$"
    )
    replacement = f"#SBATCH {preferred}={value}" if preferred.startswith("--") else f"#SBATCH {preferred} {value}"
    replaced = False
    out: list[str] = []
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    for idx, line in enumerate(lines):
        if pattern.match(line):
            if not replaced:
                out.append(replacement)
                replaced = True
            continue
        out.append(line)
        if line.strip().startswith("#SBATCH"):
            insert_at = len(out)
    if not replaced:
        out.insert(insert_at, replacement)
    return out


def _strip_old_qe_command_block(lines: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next_command = False
    command_patterns = [
        re.compile(r"^\s*exe\s*=", re.IGNORECASE),
        re.compile(r"^\s*INPUT\s*=", re.IGNORECASE),
        re.compile(r"^\s*OUTPUT\s*=", re.IGNORECASE),
        re.compile(r"^\s*(mpirun|mpiexec|srun)\b.*\s-in\s+", re.IGNORECASE),
        re.compile(r"^\s*(mpirun|mpiexec|srun)\b.*\$(exe|INPUT|OUTPUT)\b", re.IGNORECASE),
        re.compile(r"^\s*(pw\.x|bands\.x|dos\.x|projwfc\.x|ph\.x|pp\.x)\b.*\s-in\s+", re.IGNORECASE),
    ]
    for line in lines:
        if "tritondft generated execution block" in line.lower():
            skip_next_command = True
            continue
        if skip_next_command and any(p.match(line) for p in command_patterns):
            continue
        skip_next_command = False
        if any(p.match(line) for p in command_patterns):
            continue
        cleaned.append(line.rstrip())
    while cleaned and not cleaned[-1].strip():
        cleaned.pop()
    return cleaned
