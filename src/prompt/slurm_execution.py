slurm_execution_prompt = {
    "role": "user",
    "content": """I have a Slurm call to invoke `{exec_name}` for the input files located in `{input_dir}`.
    I want the output to be `{output_dir}`. You can use `which {fn}` to find the executable path.
Please choose a suitable Slurm submission pattern, select partitions/nodes/tasks-per-node.
Use on {parallel_np} ranks for mpirun.

### Below is the execution script file template:
#!/bin/bash
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH -t 1:00:00
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
exe=`which pw.x`
INPUT={input_dir}
OUTPUT={output_dir}
export OMP_NUM_THREADS=1
mpirun -np {parallel_np} $exe -in $INPUT > $OUTPUT

### Input details:
{input_context}

IMPORTANT OUTPUT CONSTRAINTS (STRICT):
- Output MUST be raw bash script text only. DO NOT use Markdown formatting of any kind.
- You MUST output raw plain-text only. Rendering, formatting, or wrapping is strictly forbidden. This includes but is not limited to Markdown, code fences (``` or ```bash), HTML, XML, or any other presentation format.
- The first character of the output MUST be `#` from `#!/bin/bash`.
- The last character of the output MUST be the last character of the final command.
- If you violate any of these rules, the output is considered incorrect."""
}
