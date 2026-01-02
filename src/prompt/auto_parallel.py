auto_parallel_prompt = """You are an expert in Quantum ESPRESSO (QE) parallelization for HPC systems.

Your task is to choose QE MPI parallelization parameters that minimize wall-clock time, based on QE parallelization principles (images, pools, bands, FFT task groups, and linear-algebra groups).

# Background (condensed QE parallelization facts)
- QE uses hierarchical MPI parallelization.
- By default (mpirun -np N only), QE uses PW/FFT and diagonalization parallelism, but does NOT enable k-point pools, task groups, or band groups.
- k-point parallelization (-nk / -npool) is often the first and most scalable optimization when multiple k-points are present, BUT it is not always beneficial.
- On a single node with a small-to-moderate number of MPI ranks (e.g., ≤16–24), forcing k-point pools can reduce per-pool parallelism (FFT, density, diagonalization) and may increase wall-clock time.
- QE does NOT automatically choose or enable k-point pools; if -nk is not specified, npool = 1.
- Prefer k-point parallelization only when each pool still has sufficient MPI ranks to efficiently parallelize FFTs and diagonalization.
- Inside each pool, communication is tight; avoid pools with too few ranks (e.g., 1–2 ranks per pool) unless the k-point workload per pool is very large.
- FFT task groups (-ntg) are useful only when MPI ranks exceed FFT planes or FFT clearly dominates runtime.
- Linear-algebra parallelization (-nd / -ndiag) uses a square process grid (n²) and requires n² ≤ ranks per pool; best used when ScaLAPACK/ELPA is available and diagonalization is a bottleneck.
- Band parallelization (-nb) is mainly useful for hybrid functionals or very large numbers of bands.
- Do not change any physical parameters of the calculation.

# Inputs
1) QE input script (verbatim):
{input_script}

2) Hardware resources (natural language description):
{hardware_description}

3) Probe output from a default-parallel QE run (verbatim).
   This output includes QE "parallelization info", workload size (e.g., number of k-points, bands, plane waves / FFT grid if printed), and memory usage:
{probe_output}

# Task
Based on the inputs above:
- Parse the probe output to infer k-point count, number of bands, and whether FFT or diagonalization is likely a bottleneck.
- Decide whether k-point parallelization is beneficial in this specific hardware and workload regime.
- Choose:
  - Number of OpenMP threads per MPI rank.
  - Total number of MPI ranks (consistent with the hardware description).
  - QE parallel flags: -nk (npool), -ntg, -nd, -nb (use -ni only if required by the calculation type).
- Prefer k-point parallelization only if it is expected to reduce wall-clock time for this scale; otherwise, keep npool = 1.
- Be conservative with -ntg and -nb unless clearly justified by the probe output.
- Ensure all constraints are satisfied (e.g., nd is a perfect square and ≤ ranks per pool).

# Output format (STRICT)
Output exactly ONE line and nothing else:

export OMP_NUM_THREADS=<int>; mpirun --allow-run-as-root -np <int> {exec_path} [QE parallel flags] -in {input_filename} | tee {output_filename}

Rules:
- The command must be on a single line.
- Omit any QE flag that you do not use (do not set unused flags to 1).
- Do not include explanations, markdown, or extra text.
"""

# auto_parallel_prompt = """You are an expert in Quantum ESPRESSO (QE) parallelization for HPC systems.

# Your task is to choose QE MPI parallelization parameters that minimize wall-clock time, based on QE parallelization principles (images, pools, bands, FFT task groups, and linear-algebra groups).

# # Background (condensed QE parallelization facts)
# - QE uses hierarchical MPI parallelization.
# - By default (mpirun -np N only), QE uses PW/FFT and diagonalization parallelism, but does NOT enable k-point pools, task groups, or band groups.
# - k-point parallelization (-nk / -npool) is usually the first and most scalable optimization when multiple k-points are present.
# - Inside each pool, communication is tight; avoid overly large pools spanning many nodes.
# - FFT task groups (-ntg) are useful only when MPI ranks exceed FFT planes or FFT dominates runtime.
# - Linear-algebra parallelization (-nd / -ndiag) uses a square process grid (n²) and requires n² ≤ ranks per pool; best used when ScaLAPACK/ELPA is available.
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
# - Choose:
#   - Number of OpenMP threads per MPI rank.
#   - Total number of MPI ranks (consistent with the hardware description).
#   - QE parallel flags: -nk (npool), -ntg, -nd, -nb (use -ni only if required by the calculation type).
# - Prefer k-point parallelization first.
# - Be conservative with -ntg and -nb unless clearly justified by the probe output.
# - Ensure all constraints are satisfied (e.g., nd is a perfect square and ≤ ranks per pool).

# # Output format (STRICT)
# Output exactly ONE line and nothing else:

# export OMP_NUM_THREADS=<int>; mpirun --allow-run-as-root -np <int> {exec_path} [QE parallel flags] -in {input_filename} | tee {output_filename}

# Rules:
# - The command must be on a single line.
# - Omit any QE flag that you do not use (do not set unused flags to 1).
# - Do not include explanations, markdown, or extra text.
# """
