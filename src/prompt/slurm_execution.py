slurm_execution_prompt = {
    "role": "user",
    "content": """I have a Slurm call to invoke `{exec_path}` for the input files located in `{input_dir}`.
    I want the output to be `{output_dir}`.
Please choose a suitable Slurm submission pattern, select partitions/nodes/tasks-per-node.
Use on {parallel_np} ranks for mpirun.

### Below is the execution script file template:
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
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
exe=`pw.x`
INPUT={input_dir}
OUTPUT={output_dir}
export OMP_NUM_THREADS=1
mpirun -np {parallel_np} $exe -in $INPUT > $OUTPUT

### Input details:
{input_context}

Important:
Please use mpirun instead of srun or command-line execution. Only output the final script without any explanation.
"""
}
