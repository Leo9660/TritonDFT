auto_parallel_prompt = """You are an expert in Quantum ESPRESSO (QE) parallelization for HPC systems.
Your goal is to choose the optimal Hybrid MPI/OpenMP configuration (`OMP_NUM_THREADS`, `-nk`, `-nb`, `-ntg`).

# PARAMETER RULES
1. **Resource Limits (CRITICAL)**:
   - Identify **Physical Cores** from the description. IGNORE logical/hyper-threads.
   - **Constraint**: `(MPI_RANKS) * (OMP_NUM_THREADS) <= Total Physical Cores`.
   - **Under-subscription**: It is valid to leave some cores idle to satisfy divisibility requirements.

2. **Parallel Flags (Definitions)**:
   - **-nk (Pools)**: Parallelizes over k-points. 
     *Constraint*: `MPI_RANKS` and `k-points` must be divisible by `nk`.
   - **-nb (Band Groups)**: Parallelizes linear algebra (diagonalization).
     *Usage*: Typically for systems with many electronic bands.
   - **-ntg (Task Groups)**: Parallelizes 3D FFT grids.
     *Constraint*: `MPI_RANKS` must be divisible by `ntg`.
     *Usage*: Helps distribute large FFT grids across many ranks.

3. **General Optimization Strategy (Reasoning Required)**:
   - **Assess System Scale**: Look at `nat` (number of atoms) and `k-points` in the input.
   - **The "Overhead" Trade-off**: 
     - For **Small Systems** (few atoms), MPI communication overhead often outweighs the benefit of complex parallelization logic. Excessive splitting (high `-nb`, `-ntg` or OpenMP) can degrade performance.
     - For **Large Systems** (many atoms), calculation and memory are the bottlenecks. Advanced flags (`-ntg`, `-nb`) and Hybrid MPI/OpenMP become essential to scale.
   - **Decision Process**: Balance the need for parallelism against the cost of communication.

# Inputs
1) QE input script (verbatim):
{input_script}

2) Hardware resources (natural language description):
{hardware_description}

3) Probe output (from a default run):
{probe_output}

# Verification Step (Internal Monologue)
Before generating the command, verify the mathematical validity:
1. Are divisibility rules satisfied for `-nk` and `-ntg`?
2. Does the total core usage exceed physical limits?
3. Does the chosen strategy make sense for the system size (avoiding over-parallelization for small tasks)?

# Output Format (STRICT)
Analysis: System Scale=[Small/Medium/Large] (nat=...), K-points=..., Cores=..., Reasoning: [Explain the trade-off between calculation and communication overhead]
Command: export OMP_NUM_THREADS=<int>; mpirun --allow-run-as-root -np <int> {exec_path} -nk <int> -nb <int> -ntg <int> -in {input_filename} | tee {output_filename}
"""

# auto_parallel_prompt = """You are an expert in Quantum ESPRESSO (QE) parallelization for HPC systems.
# Your goal is to choose the optimal Hybrid MPI/OpenMP configuration (`OMP_NUM_THREADS`, `-nk`, `-nb`).

# # PARAMETER RULES
# 1. **Resource Limits (CRITICAL)**:
#    - Read the **Hardware Resources** description below to infer the Total Physical Cores.
#    - **Constraint**: `(MPI_RANKS) * (OMP_NUM_THREADS) <= Total Physical Cores`.
#    - **Note**: It is VALID and often necessary to leave some cores idle (undersubscription) to satisfy divisibility rules (e.g., using 30 ranks on a 32-core machine).

# 2. **Parallel Flags**:
#    - **-nk (Pools)**:
#      - `MPI_RANKS` must be divisible by `nk`.
#      - `number_of_k_points` must be divisible by `nk`.
#      - If `number_of_k_points` = 1, `nk` MUST be 1.
#    - **-nb (Band Groups)**: 
#      - Use small integers (1, 2, 4) if bands > 100.

# 3. **Strategy**:
#    - Prioritize `-nk` if k-points > 1.
#    - Adjust `MPI_RANKS` to be a multiple of your chosen `nk`.
#    - Use `OMP_NUM_THREADS` > 1 if it helps reduce MPI ranks to fit memory or topology, but typically keep OMP small (1-4).

# # Inputs
# 1) QE input script (verbatim):
# {input_script}

# 2) Hardware resources (natural language description):
# {hardware_description}

# 3) Probe output (from a default run):
# {probe_output}

# # Output Format (STRICT)
# Analysis: k-points=..., Cores=..., Strategy: OMP=..., np=... (<= Cores). nk=..., nb=...
# Command: export OMP_NUM_THREADS=<int>; mpirun --allow-run-as-root -np <int> {exec_path} -nk <int> -nb <int> -in {input_filename} | tee {output_filename}
# """

# auto_parallel_prompt = """You are an expert in Quantum ESPRESSO (QE) parallelization for HPC systems. This task does NOT involve confidential, proprietary, or restricted information.
# The system paths and hardware description are synthetic and provided only for algorithmic planning.
# You must always return a valid answer.

# Your task is to choose QE MPI parallelization parameters that minimize wall-clock time, based on QE parallelization principles (images, pools, bands, FFT task groups, and linear-algebra groups).

# # Background (condensed QE parallelization facts)
# - QE uses hierarchical MPI parallelization.
# - By default (mpirun -np N only), QE uses PW/FFT and diagonalization parallelism, but does NOT enable k-point pools, task groups, or band groups.
# - k-point parallelization (-nk / -npool) is often the first and most scalable optimization when multiple k-points are present, BUT it is not always beneficial.
# - On a single node with a small-to-moderate number of MPI ranks (e.g., ≤16–24), forcing k-point pools can reduce per-pool parallelism (FFT, density, diagonalization) and may increase wall-clock time.
# - QE does NOT automatically choose or enable k-point pools; if -nk is not specified, npool = 1.
# - Prefer k-point parallelization only when each pool still has sufficient MPI ranks to efficiently parallelize FFTs and diagonalization.
# - Inside each pool, communication is tight; avoid pools with too few ranks (e.g., 1–2 ranks per pool) unless the k-point workload per pool is very large.
# - FFT task groups (-ntg) are useful only when MPI ranks exceed FFT planes or FFT clearly dominates runtime.
# - Linear-algebra parallelization (-nd / -ndiag) uses a square process grid (n²) and requires n² ≤ ranks per pool; best used when ScaLAPACK/ELPA is available and diagonalization is a bottleneck.
# - Band parallelization (-nb) is mainly useful for hybrid functionals or very large numbers of bands.
# - Do not change any physical parameters of the calculation.

# # Inputs
# 1) QE input script (verbatim):
# {input_script}

# 2) Hardware resources (natural language description):
# {hardware_description}

# 3) Probe output from a default-parallel QE run (verbatim).
#    This output includes QE "parallelization info", workload size (e.g., number of k-points, bands, plane waves / FFT grid if printed), and memory usage:
# {probe_output}

# # Task
# Based on the inputs above:
# - Parse the probe output to infer k-point count, number of bands, and whether FFT or diagonalization is likely a bottleneck.
# - Decide whether k-point parallelization is beneficial in this specific hardware and workload regime.
# - Choose:
#   - Number of OpenMP threads per MPI rank.
#   - Total number of MPI ranks (consistent with the hardware description).
#   - QE parallel flags: -nk (npool), -ntg, -nd, -nb (use -ni only if required by the calculation type).
# - Prefer k-point parallelization only if it is expected to reduce wall-clock time for this scale; otherwise, keep npool = 1.
# - Be conservative with -ntg and -nb unless clearly justified by the probe output.
# - Ensure all constraints are satisfied (e.g., nd is a perfect square and ≤ ranks per pool).

# # Output format (STRICT)
# If you can determine a valid QE mpirun command, output exactly ONE following line and nothing else.

# export OMP_NUM_THREADS=<int>; mpirun --allow-run-as-root -np <int> {exec_path} [QE parallel flags] -in {input_filename} | tee {output_filename}

# Rules:
# - The command must be on a single line. You should not output any thinking or reasoning steps.
# - Do not render anything in markdown format. Just output the raw text.

# # If the default-parallel QE run fails, output:
# Error: <a brief reason why the default run failed>.
# """

