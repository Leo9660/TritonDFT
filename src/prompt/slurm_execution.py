slurm_execution_prompt = {
    "role": "user",
    "content": """I have a Slurm call to invoke `{exec_name}` for the input files located in `{input_dir}`.
    I want the output to be `{output_dir}`.
Please choose a suitable Slurm submission pattern, select partitions/nodes/tasks-per-node, and configure parallel resources
based on {parallel_np} ranks when {parallel_exec} is true.
Use the command below as the execution example:
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH -t 20:00:00
#SBATCH -o qe.out
#SBATCH -e qe.err
#SBATCH -p compute
#SBATCH --export=ALL
#SBATCH --account=TG-PHY250365
#SBATCH --job-name=ph
#SBATCH --mem=0
module reset
module load cpu/0.15.4
module load gcc/9.2.0
module load openmpi
module load quantum-espresso
# Set the executable and input file
exe=`pw.x`
INPUT={input_dir}
OUTPUT={output_dir}
export OMP_NUM_THREADS=1
mpirun $exe -in $INPUT > $OUTPUT

Input details:
{input_context}

Return ONLY the Slurm batch directives or environment setup (lines beginning with `#SBATCH` or module commands).
Do NOT include `#!/bin/bash`, loops, or per-input commands—those will be appended afterwards.
Keep the output minimal so it can be inserted header-first into our script template.
"""
}
