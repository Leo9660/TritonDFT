# Kuntal development version
This branch is for testing changes before merging into the main TritonDFT code.

## Cluster/super-user workflow changes - 2026-06-17

### Goal
This version is being prepared for TritonDFT super users who already have access
to a remote cluster or supercomputer. The local desktop should generate Quantum
ESPRESSO inputs, package all required files, upload them to the cluster, submit
Slurm jobs there, fetch outputs back, and let the agent continue multi-step
workflows such as:

1. `vc-relax`
2. `scf`
3. `nscf`
4. band path / `bands`
5. post-processing such as `bands.x` or `dos.x`

The local machine should not need a local Quantum ESPRESSO installation for this
cluster workflow.

### New files
- `src/cluster_agent.py`
  - New interactive terminal agent for super users.
  - Shows the TritonDFT welcome banner.
  - Runs first-time cluster setup when `.env.cluster` is missing/incomplete.
  - Checks/reuses `~/.ssh/config` entries and creates one only when needed.
  - Keeps a persistent SSH ControlMaster session when possible.
  - Uploads local run directories with `rsync`.
  - Submits generated Slurm jobs with `sbatch`.
  - Polls `squeue`.
  - Fetches cluster outputs back into the local run directory.
  - Parses remote results and feeds them into the next agent step.

- `scripts/run_cluster_agent.sh`
  - Convenience launcher.
  - Sets `PYTHONPATH=src`.
  - Uses `.venv/bin/python` automatically if present.
  - Sets a local Matplotlib cache directory under `tmp/matplotlib`.
  - Suppresses the noisy `pkg_resources is deprecated` warning from dependencies.

- `.env.cluster.example`
  - Template for super-user local configuration.
  - Stores placeholders for `OPENAI_API_KEY`, `MP_API_KEY`, SSH target, remote
    working directory, model/backend, and Slurm template path.
  - Real `.env.cluster` is ignored by Git.

- `example_slurm_job_file.txt`
  - User-editable example Slurm submit file.
  - Super users should put their real cluster-specific Slurm settings here:
    partition, account/allocation, module loads, environment setup.
  - TritonDFT uses this file as the safe base and replaces only:
    nodes, tasks-per-node, walltime, output/error files, executable, input,
    output, and launch command.

### Modified files
- `.gitignore`
  - Added `.env.cluster` so API keys and local cluster settings are not committed.

- `README.md`
  - Added instructions for the desktop-to-cluster workflow.
  - Added instructions for running `bash scripts/run_cluster_agent.sh`.
  - Documented that `.env.cluster` is ignored and contains super-user settings.
  - Documented that `example_slurm_job_file.txt` controls cluster-specific Slurm
    settings.

- `src/DFTAgent.py`
  - Added support for `run_mode="cluster_package"`.
  - In cluster package mode, the agent generates QE inputs and Slurm scripts but
    does not run QE locally.
  - Returns metadata needed by `cluster_agent.py` for remote parsing:
    params JSON, executable name, parse requirement key, subproblem id, work dir,
    input paths, output paths, Slurm paths, and packaged pseudo paths.
  - Saves raw failed script-generation output as
    `script_generation_failed_<step>_<loop>.txt` if parsing generated input fails.
  - If script generation returns empty text or text without a usable QE input,
    the agent now feeds that parser failure back into the normal retry loop
    instead of immediately stopping the cluster-agent session.

- `src/utils.py`
  - Added `package_pseudos_for_remote()`.
    - Copies only referenced `.upf` files into `run_dir/pseudos/`.
    - Patches QE input files to use `pseudo_dir='./pseudos'`.
    - Prevents remote cluster jobs from referencing local Mac paths such as
      `/Users/.../PseudoDojo/...`.
  - Made `parse_scripts_block()` more robust.
    - Still supports `<script>...</script>`.
    - Also accepts raw QE inputs.
    - Also accepts fenced code blocks.
    - Also accepts QE inputs embedded in prose or JSON strings.

- `src/execute_code/slurm_template.py`
  - Changed Slurm rendering to support a user-provided template file.
  - If `TRITONDFT_SLURM_TEMPLATE` is set, TritonDFT reads that file and preserves
    cluster-specific settings.
  - Replaces duplicate/old QE execution blocks with a fresh TritonDFT-generated
    execution block.
  - Replaces only safe dynamic Slurm fields:
    `--nodes`, `--tasks-per-node`, `-t/--time`, `-o/--output`, `-e/--error`.
  - Removed unsafe fallback defaults such as duplicate `-p compute`, hardcoded
    account, and `--mem=0`.

- `src/execute_code/slurm.py`
  - Added `SlurmLauncher.package()` to generate scripts without submitting.
  - Added support for template path through `TRITONDFT_SLURM_TEMPLATE`.
  - Cluster-package Slurm scripts now use the existing auto-parallel prompt
    instead of a separate Slurm-only estimator.
  - The auto-parallel prompt decides:
    - MPI command and QE parallel flags such as `-np`, `-nk`, `-nb`, and `-ntg`.
    - Slurm `--nodes`.
    - Slurm `--tasks-per-node`.
    - Slurm walltime `-t`.
  - The generated command is normalized to use `$exe`, `$INPUT`, and `$OUTPUT`
    so the same decision can be inserted safely into the batch template.
  - If no hardware description is provided, the cluster-package path describes a
    generic Slurm cluster with a configurable core count instead of asking the
    user to choose MPI ranks manually.

### Important behavior changes
- Cluster workflow now creates self-contained run directories:

```text
run_dir/
  input_*.in
  slurm_job_*.sh
  pseudos/
    *.upf
  run_meta.json
```

- Generated QE inputs for cluster mode now use:

```fortran
pseudo_dir = './pseudos'
```

- Super users should edit `example_slurm_job_file.txt` instead of editing Python
  code for cluster-specific Slurm settings.

- The agent should not ask users to manually choose MPI ranks during normal use.
  For cluster-package Slurm jobs, parallel settings are chosen through the
  existing auto-parallel prompt from the generated QE input, executable type,
  hardware description, and requested calculation.

### Testing performed
- Python compile checks:

```bash
python -m py_compile src/cluster_agent.py
python -m py_compile src/DFTAgent.py src/utils.py
python -m py_compile src/execute_code/slurm.py src/execute_code/slurm_template.py
```

- Smoke tested Slurm rendering from `example_slurm_job_file.txt`.
- Smoke tested cluster-package Slurm generation with a fake auto-parallel model
  response:
  - Prompt-selected `mpirun -np 36` produced `#SBATCH --tasks-per-node=36`.
  - Prompt-selected walltime `00:45:00` produced `#SBATCH -t 00:45:00`.
- Smoke tested pseudopotential packaging:
  - Input was patched to `pseudo_dir='./pseudos'`.
  - Required `.upf` file copied into `pseudos/`.
- Smoke tested script parser fallback for:
  - `<script>...</script>`
  - raw QE input
  - prose-wrapped QE input
  - JSON-wrapped QE input
- Smoke tested that empty script text still raises a parser error, which is now
  handled by `DFTAgent` as a retryable script-generation failure.

### Current run command
Use:

```bash
bash scripts/run_cluster_agent.sh
```

The agent reads `.env.cluster` and launches the interactive `DFT request>` prompt.

