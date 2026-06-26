# TritonDFT

## 1. Setup
```
pip install -r requirements.txt
git config --global --add safe.directory '*'
git submodule update --init --recursive
cd QuantumE && ./configure && make all -j$(nproc)
cd ..
```

## 2. Run
```
export OPENAI_API_KEY=xxx
export MP_API_KEY=xxx
python test/new_test.py
```

<br><br>

## Introduction
DFTagent automates the setup of Quantum ESPRESSO (QE) calculations by letting a language-model agent interpret natural-language DFT requests, generate QE inputs, run relaxation/SCF workflows, and parse the resulting structures/energies. The repo contains the Python agent plus a local checkout of QE so everything can run from one workspace.

## Python environment
- Use Python 3.10+ and create an isolated environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
- Upgrade pip and install the required packages: `pip install --upgrade pip && pip install -r requirements.txt`. The list covers `numpy`, `torch`, `transformers`, `openai`, `mp-api`, and `pymatgen`. Install `vllm` separately if you plan to use the vLLM backend.

## Quantum ESPRESSO setup
0. [Optional] OpenMPI and MKL for High-Performance Execution
   apt-get install -y libopenmpi-dev openmpi-bin
   conda install -c conda-forge mkl mkl-devel mkl-include

1. Initialize/update the bundled QE submodule (this populates `./QuantumE/`):
   ```
   git submodule update --init --recursive
   cd QuantumE
   ```
   When QE needs refreshing later, either run `git submodule update --remote QuantumE` or enter the submodule and `git pull` the desired tag/branch, then commit the new submodule pointer in the main repo.
2. Follow the “Quick installation instructions for CPU-based machines” from upstream (summarized here). For GPU execution, consult `README_GPU.md` inside the QE tree.

**Using `make`** (`[]` = optional arguments):
```
./configure [options]
make all
```
Running `make` with no target lists all available targets. `make -jN` enables parallel builds on `N` processors. Finished binaries appear under `bin/`.

**Using CMake** (requires CMake ≥ 3.14):
```
mkdir ./build
cd ./build
cmake -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_COMPILER=mpicc \
      [-DCMAKE_INSTALL_PREFIX=/path/to/install] ..
make [-jN]
[make install]
```
Even though CMake can guess compilers, explicitly set `CMAKE_Fortran_COMPILER` and `CMAKE_C_COMPILER` (or their MPI wrappers). Targets end up under `build/bin`, and `make install` places them beneath `CMAKE_INSTALL_PREFIX`.

Refer to the QE documentation in `Doc/`, the package-specific `*/Doc/` folders, and https://www.quantum-espresso.org/ for more background. Technical notes for users/developers live on the QE GitLab wiki.

## Running the agent
1. Export the APIs you need (OpenAI + Materials Project). Either run `export OPENAI_API_KEY=...` and `export MP_API_KEY=...` manually or reuse the commands you placed in `test/env_setup.sh`.
2. For local execution, ensure the QE binaries you built reside in `QuantumE/bin` (the default path used inside `DFTAgent.py`). Update `self.qe_bin_prefix`/`self.pseudo_dir` there if your layout differs. Desktop-to-cluster package mode does not require local QE binaries.
3. Execute the sample workflow: `python test/new_test.py`. The script initializes `DFTAgent`, submits a Si relaxation → SCF → NSCF request, and logs outputs to `evaluation.log` so you can verify that the full stack is wired correctly.

### Desktop-to-cluster workflow
For users who should not install Quantum ESPRESSO locally, initialize the agent with `run_mode="cluster_package"`. In this mode the agent still generates QE input files on the local desktop, but it does not call `pw.x`, `mpirun`, or `sbatch` locally. Instead it writes Slurm scripts beside the generated inputs in the run directory:

```python
agent = DFTAgent(
    model="gpt-4o",
    run_mode="cluster_package",
)
```

The generated `slurm_job_*.sh` files assume the cluster environment can expose QE with `module load quantum-espresso`. If your cluster requires an explicit remote binary directory, set `remote_qe_bin_dir` in `config/config.yaml`; otherwise leave it empty so the script runs `pw.x`, `bands.x`, etc. from the cluster `PATH`.

After this package step, an SSH transport layer can upload the run directory to the cluster, run `sbatch slurm_job_*.sh` remotely, and fetch `output_*.out` back for parsing.

### Interactive SSH cluster agent
Use `src/cluster_agent.py` when you want the agent to keep talking to the
cluster between workflow steps. It first prints and saves the complete plan,
including why every step is needed, then generates all requested QE inputs
before running anything. A tabbed approval window shows the plan and every
editable input file. No SSH connection or Slurm submission starts until
`Approve & Run` is selected.

Slurm scripts are deliberately not generated during this approval phase.
Immediately before each approved `pw.x` or `ph.x` step runs, TritonDFT uploads
a short, time-limited probe input, executes it on the remote cluster, fetches
the reported k-point/band/FFT parallelization information, and then generates
the real Slurm script from that runtime evidence. Lightweight post-processing
executables are packaged at execution time without an unnecessary probe.

For workflows beginning with `vc-relax`, later `pw.x` files show a
`TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER` during review. After relaxation,
TritonDFT replaces that marker with the final `CELL_PARAMETERS` and
`ATOMIC_POSITIONS` immediately before the dependent file is uploaded. The
approved pre-execution versions are retained under `approved_inputs/` in the
run directory.

```bash
bash scripts/run_cluster_agent.sh
```

On the first run, TritonDFT asks for your cluster username, login address, and
remote working directory. It creates or reuses an SSH config entry, writes the
cluster defaults to `.env.cluster`, then asks you to add `OPENAI_API_KEY` and
`MP_API_KEY` there before continuing.

Edit `example_slurm_job_file.txt` with your cluster's normal submission header
and module commands. TritonDFT uses that file as the safe site-specific base and
only replaces the walltime, Slurm task count, executable, input, output, and QE
launch command for each generated job.

The `.env.cluster` file is ignored by Git because it can contain API keys. The
`CLUSTER_AGENT_SSH_TARGET` value can be an alias from `~/.ssh/config` or
`user@hostname`.
The agent asks for the remote parent directory at startup and before each DFT
request; use a scratch/project path where your cluster user has write
permission, such as `/scratch/$USER/qe_jobs`.
The agent opens a persistent SSH ControlMaster connection by default so repeated
uploads, submissions, queue checks, and downloads reuse the same login session.
If `module load quantum-espresso` does not expose `pw.x` on the cluster `PATH`,
add `--remote-qe-bin-dir /path/to/qe/bin`.
The agent chooses the Slurm walltime and parallel launch settings from the
generated QE input.
Any command-line option still overrides the value from `.env.cluster`.
Type `exit` or `quit` at the `DFT request>` prompt to stop the loop.

## [Optional] Deploy Backend in Dokcer
```
docker run -it -v $(pwd):/workspace  \
-e OPENAI_API_KEY=sk-proj-xxx \
-e MP_API_KEY=xxx \
--name triton-dft-lyc -p 8000:8000 triton-dft /bin/bash

docker start -ai triton-dft-lyc

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```
and then, find docs on: http://localhost:8000/docs

## Prompt:
```
For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. Lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, and conv_thr = 1.0e-8. Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for ecutwfc and the k-point. After the vc-relax finishes, run a self-consistent field (scf) calculation on the relaxed structure with consistent settings, then perform a non-self-consistent field (nscf) calculation and compute the band gap from that electronic structure.
```
