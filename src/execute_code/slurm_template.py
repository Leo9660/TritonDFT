import os

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node={tasks_per_node}
#SBATCH -t 1:00:00
#SBATCH -o {log_out}
#SBATCH -e {log_err}
#SBATCH -p compute
#SBATCH --export=ALL
#SBATCH --account=TG-PHY250365
#SBATCH --job-name=qe
#SBATCH --mem=0
module reset
module load cpu/0.15.4
module load gcc/9.2.0
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
    tasks_per_node: int,
    work_dir: str,
) -> str:
    return SLURM_TEMPLATE.format(
        exec_path=exec_path,
        input_path=input_path,
        output_path=output_path,
        command_line=command_line,
        tasks_per_node=tasks_per_node,
        log_out=os.path.join(work_dir, "qe.out"),
        log_err=os.path.join(work_dir, "qe.err"),
    ).rstrip()
