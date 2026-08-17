# TritonDFT cluster installation guide

This is the tested installation procedure for TritonDFT on the LBL cluster
environment. Each section explains why the step is needed and ends with a
checkpoint. Do not continue when a checkpoint fails: save the error and fix
that stage first. This prevents repeated downloads and unnecessary rebuilding.

Values such as `ktalit`, `/home/ktalit`, API keys, remote directories, and SSH
hosts are examples. Every user must replace them with their own values.

## Before starting

Log in to the Linux machine where the TritonDFT client will run and enter the
existing repository checkout:

```bash
cd /path/to/TritonDFT
```

Confirm that this is the correct directory:

```bash
pwd
git status --short
git branch --show-current
```

If `git status --short` shows somebody's unfinished work, preserve it before
continuing. Do not delete or overwrite files that you do not recognize.

## 1. Load the supported Python module

```bash
module load python/3.11.6-gcc-11.4.0
```

Why: TritonDFT currently pins `pymatgen==2023.12.18`. During the previous
installation, system Python 3.12 could not obtain a compatible pymatgen wheel.
Pip attempted to compile pymatgen and failed with:

```text
error: command 'gcc' failed: No such file or directory
```

Using the cluster's Python 3.11 module allowed pip to download the prebuilt
`cp311` wheel. Therefore, installing GCC manually is not the first solution.
The `gcc-11.4.0` text in the module name describes how the cluster built that
Python module.

On another cluster, run `module avail python` and select its Python 3.11 module.
Do not assume the exact mhenson module name exists everywhere.

### Checkpoint 1

```bash
python --version
which python
```

The known-working mhenson output is:

```text
Python 3.11.6
/global/software/rocky-8.x86_64/python-3.11.6/python-3.11.6/python/3.11.6-gcc/bin/python
```

Stop if the version is Python 3.12 or if `which python` points to an unexpected
system or Conda installation.

## 2. Create and activate an isolated environment

Create the environment once:

```bash
python -m venv .venv
source .venv/bin/activate
```

Why: `.venv` keeps TritonDFT packages separate from system Python and from
other users' projects. The launcher automatically uses `.venv/bin/python` when
this environment exists.

If `.venv` already exists, activate it and inspect its Python version. Do not
recreate a working environment after every `git pull`. If it was created with
Python 3.12, rename it as a backup and recreate it after loading Python 3.11.

### Checkpoint 2

```bash
python --version
python -m pip --version
```

Python must still be 3.11, and pip's path must be inside the repository's
`.venv` directory.

## 3. Update the Python packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

Why: `pip` resolves and downloads dependencies, `setuptools` supports legacy
Python packages, and `wheel` installs binary wheels. Updating them before the
requirements avoids several old-package installation problems. Using
`python -m pip` ensures that packages go into the active `.venv`.

### Checkpoint 3

```bash
python -m pip --version
```

The displayed path must remain inside `.venv`.

## 4. Configure the Git branch and update the code

The working branch is `kuntal-version`. Set its upstream once:

```bash
git branch --set-upstream-to=origin/kuntal-version kuntal-version
```

Expected output:

```text
branch 'kuntal-version' set up to track 'origin/kuntal-version'.
```

Then update without creating an accidental merge commit:

```bash
git pull --ff-only
```

Why: setting the upstream allows later `git pull` commands to know which remote
branch to use. `--ff-only` stops safely if local and remote histories have
diverged.

If the upstream command says the remote branch is unknown, run
`git fetch origin kuntal-version` and retry. If pull says an untracked file
would be overwritten, move that file to a named backup outside the repository
and retry. The previous installation encountered this with a manually created
`tritondft-cluster` file.

### Checkpoint 4

```bash
git branch -vv
git status --short
git rev-parse HEAD
```

The current branch should show `[origin/kuntal-version]`. Record the commit ID
when reporting an installation problem.

## 5. Install the project requirements

```bash
mkdir -p install-checkpoints
python -m pip install -r requirements.txt 2>&1 | tee install-checkpoints/pip-install.txt
```

Why: this installs the exact Python dependencies required by TritonDFT and
saves the output for troubleshooting.

Do not repeatedly restart this command after a failure. If pip says it is
building pymatgen from source or reports `command 'gcc' failed`, return to
Checkpoint 1—the wrong Python is probably active.

### Checkpoint 5

```bash
PYTHONPATH=src python -c "import cluster_agent; print('cluster_agent import OK')"
python -m pip check
```

The first command must print `cluster_agent import OK`; the second should print
`No broken requirements found`.

If an old checkout reports `ModuleNotFoundError: pkg_resources`, first verify
that the Git update in Step 4 succeeded. Do not keep reinstalling packages
without checking the code version.

## 6. Verify the launcher

The repository should now contain the launcher:

```bash
ls -l tritondft-cluster
```

The tracked file should contain:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/scripts/run_cluster_agent.sh" "$@"
```

If the file exists but is not executable:

```bash
chmod +x tritondft-cluster
```

Why: this wrapper finds the repository directory and delegates to the standard
launcher, which configures `PYTHONPATH` and selects `.venv/bin/python`.

If the file is completely missing after a successful pull, do not immediately
create an untracked replacement. Recheck the branch and commit first. Creating
it manually can block a future pull when Git later supplies the tracked file.

### Checkpoint 6

```bash
./tritondft-cluster --help
```

This must print the command-line help without an import traceback.

## 7. Create the private per-user configuration directory

```bash
mkdir -p ~/.tritondft
chmod 700 ~/.tritondft
```

Why: this directory contains API keys and user-specific cluster configuration.
Permission `700` allows only its owner to access it.

The directory must ultimately contain at least:

```text
~/.tritondft/
  .env.cluster
  example_qe_slurm_job_file.txt
```

VASP users will also need `example_vasp_slurm_job_file.txt` and access to their
licensed POTCAR files.

The first interactive launch can create starter files automatically. They can
also be initialized from the repository examples:

```bash
cp -n .env.cluster.example ~/.tritondft/.env.cluster
cp -n example_slurm_job_file.txt ~/.tritondft/example_qe_slurm_job_file.txt
chmod 600 ~/.tritondft/.env.cluster
```

`cp -n` preserves an existing configuration instead of overwriting it.

### Checkpoint 7

```bash
ls -la ~/.tritondft
```

Confirm that `.env.cluster` is visible with `ls -la` and is readable only by
its owner. Never commit or paste a real API key into Git, an issue report, or an
installation log.

## 8. Configure `.env.cluster`

Edit `~/.tritondft/.env.cluster`. The following is an example for a user whose
TritonDFT client runs locally and submits to LRC:

```dotenv
# TritonDFT super-user cluster configuration.
# This file is ignored by Git because it can contain API keys.

OPENAI_API_KEY=replace_with_your_key
CLUSTER_AGENT_SSH_TARGET=lrc
CLUSTER_AGENT_REMOTE_ROOT=/global/home/users/YOUR_USER/Triton-jobs
CLUSTER_AGENT_MODEL=gpt-5.5
CLUSTER_AGENT_BACKEND=openai
CLUSTER_AGENT_WORK_DIR=tmp
CLUSTER_AGENT_POLL_SECONDS=30
TRITONDFT_SLURM_TEMPLATE=~/.tritondft/example_qe_slurm_job_file.txt
TRITONDFT_QE_SLURM_TEMPLATE=~/.tritondft/example_qe_slurm_job_file.txt
TRITONDFT_VASP_SLURM_TEMPLATE=~/.tritondft/example_vasp_slurm_job_file.txt
CLUSTER_AGENT_REMOTE_QE_BIN_DIR=
CLUSTER_AGENT_NO_QUERY_INFO=true
```

Why each setting exists:

- `OPENAI_API_KEY` authenticates the selected OpenAI backend.
- `CLUSTER_AGENT_SSH_TARGET` must match a `Host` entry in `~/.ssh/config`.
- `CLUSTER_AGENT_REMOTE_ROOT` is where remote run directories are created. It
  must exist or be creatable and writable by that cluster user.
- `CLUSTER_AGENT_MODEL` and `CLUSTER_AGENT_BACKEND` select the model provider.
- `CLUSTER_AGENT_WORK_DIR` stores local generated inputs and downloaded output.
- `CLUSTER_AGENT_POLL_SECONDS` controls how often Slurm status is checked.
- The template variables identify the per-user Slurm templates. A path under
  `~/.tritondft` is recommended; `/opt/...` is appropriate only when an
  administrator maintains a verified shared, read-only template there.
- `CLUSTER_AGENT_REMOTE_QE_BIN_DIR` stays empty when the Slurm module commands
  put `pw.x` on `PATH`; otherwise set the remote QE `bin` directory.
- `CLUSTER_AGENT_NO_QUERY_INFO=true` disables Materials Project lookup. An
  `MP_API_KEY` is needed only when this is changed to `false`.

### Checkpoint 8

```bash
grep -E '^(CLUSTER_AGENT_SSH_TARGET|CLUSTER_AGENT_REMOTE_ROOT|CLUSTER_AGENT_MODEL|TRITONDFT_QE_SLURM_TEMPLATE)=' ~/.tritondft/.env.cluster
```

Inspect the output for placeholders. Do not print the API-key line.

## 9. Create the SSH configuration

Create `~/.ssh` if needed and protect it:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/config
chmod 600 ~/.ssh/config
```

Add only the hosts that the user can access. For example:

```sshconfig
Host expanse
  HostName login.expanse.sdsc.edu
  User YOUR_EXPANSE_USER
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPath ~/.ssh/control-%r@%h:%p
  ControlPersist 4h

Host lrc
  HostName lrc-login.lbl.gov
  User YOUR_LRC_USER
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPath ~/.ssh/control-%r@%h:%p
  ControlPersist 4h
```

Why: TritonDFT uses the short alias from `CLUSTER_AGENT_SSH_TARGET`. Persistent
SSH control connections avoid requesting a password, OTP, or Duo approval for
every upload, queue check, and download. TritonDFT does not store the password.

### Checkpoint 9

Test the same alias configured in `.env.cluster`:

```bash
ssh lrc
```

On the remote system, verify the intended job root is writable:

```bash
mkdir -p /global/home/users/YOUR_USER/Triton-jobs
test -w /global/home/users/YOUR_USER/Triton-jobs && echo "remote job directory is writable"
```

Do not proceed until SSH works and the success message appears.

## 10. Configure and validate the Slurm template

Edit `~/.tritondft/example_qe_slurm_job_file.txt`. Replace the sample account,
partition, and module commands with values that are valid on the target
compute cluster.

On that cluster, identify the available software:

```bash
module avail quantum-espresso
module avail openmpi
which sbatch
```

Load the site's recommended modules and confirm:

```bash
which pw.x
pw.x --version
```

Why: repository examples may contain settings from a different cluster, such
as `module load gcc/9.2.0` or a specific allocation. Those values are not
portable. A Python or GCC reinstall cannot fix Slurm's
`Invalid account or account/partition combination` error.

### Checkpoint 10

Confirm that:

- `pw.x` resolves after the template's module commands;
- `sbatch` exists;
- the `#SBATCH --account` and `#SBATCH --partition` combination is valid; and
- the remote job root is writable.

Save a successful module listing for later users:

```bash
module list 2>&1 | tee ~/tritondft-module-list.txt
```

## 11. Add an optional convenience alias

Test the full command first:

```bash
cd /home/YOUR_USER/TritonDFT
source .venv/bin/activate
./tritondft-cluster --env-file ~/.tritondft/.env.cluster
```

After it works, add an alias to `~/.bashrc` (or the startup file used by the
login shell):

```bash
alias tritondft='cd /home/YOUR_USER/TritonDFT && source .venv/bin/activate && ./tritondft-cluster --env-file ~/.tritondft/.env.cluster'
```

Reload the file:

```bash
source ~/.bashrc
```

Why: the alias always enters the correct checkout, activates its environment,
and explicitly selects the private env file. This avoids the previous problem
where TritonDFT read a similarly named file from the wrong directory and then
reported a missing API key.

### Checkpoint 11

```bash
type tritondft
tritondft
```

Choose the DFT code, confirm that the `DFT request>` prompt appears, then type
`exit`. No real calculation is needed for this checkpoint.

## 12. Run one minimal end-to-end test

Start TritonDFT:

```bash
tritondft
```

Request one very small calculation. Review the generated input and Slurm script
before selecting **Approve & Run**. Confirm again that the script contains the
verified account, partition, and module commands.

### Final checkpoint

The installation is complete only when:

1. the TritonDFT approval screen works;
2. the probe job is accepted by `sbatch`;
3. the job finishes on the compute cluster; and
4. TritonDFT fetches the output back successfully.

Record the Git commit ID, pip installation log, module list, and successful job
ID. Do not record passwords, OTP values, or API keys.

## Routine use and updates

Normal startup requires only:

```bash
module load python/3.11.6-gcc-11.4.0
tritondft
```

For a code update:

```bash
cd /home/YOUR_USER/TritonDFT
module load python/3.11.6-gcc-11.4.0
source .venv/bin/activate
git status --short
git pull --ff-only
python -m pip install -r requirements.txt
PYTHONPATH=src python -c "import cluster_agent; print('update OK')"
```

Do not recreate `.venv`, reinstall GCC, or repeat large downloads when the
corresponding checkpoint already passes.
