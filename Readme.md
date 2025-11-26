### TritonDFT

DFTagent automates the setup of Quantum ESPRESSO (QE) calculations by letting a language-model agent interpret natural-language DFT requests, generate QE inputs, run relaxation/SCF workflows, and parse the resulting structures/energies. The repo contains the Python agent plus a local checkout of QE so everything can run from one workspace.

## Python environment
- Use Python 3.10+ and create an isolated environment (e.g., `python -m venv .venv && source .venv/bin/activate`).
- Upgrade pip and install the required packages: `pip install --upgrade pip && pip install -r requirements.txt`. The list covers `numpy`, `torch`, `transformers`, `openai`, `mp-api`, and `pymatgen`. Install `vllm` separately if you plan to use the vLLM backend.

## Quantum ESPRESSO setup
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
2. Ensure the QE binaries you built reside in `QuantumE/bin` (the default path used inside `DFTAgent.py`). Update `self.qe_bin_prefix`/`self.pseudo_dir` there if your layout differs.
3. Execute the sample workflow: `python test/new_test.py`. The script initializes `DFTAgent`, submits a Si relaxation → SCF → NSCF request, and logs outputs to `evaluation.log` so you can verify that the full stack is wired correctly.


```
pip install -r requirements.txt
git config --global --add safe.directory '*'
git submodule update --init --recursive
cd QuantumE && ./configure && make all -j$(nproc)
cd ..
```

```
docker run -it -v $(pwd):/workspace  \
-e OPENAI_API_KEY=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 \
-e MP_API_KEY=3eLwNGQn4F5tMLgLmoRErmGbY6k2KjLv \
--name triton-dft-lyc -p 8000:8000 triton-dft /bin/bash


docker start -ai triton-dft-lyc

uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```