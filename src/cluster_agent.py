import argparse
import copy
import difflib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from DFTAgent import DFTAgent
from prompt import get_prompt
from prompt.tool_requirements import get_parse_requirement
from tool import get_spec
from validation import (
    normalize_plan,
    harmonize_dos_integration,
    validate_generated_workflow,
    validate_plan,
    validate_qe_input,
    validate_qe_output,
)
from utils import (
    extract_json_brutal,
    get_qe_result,
    output_to_log_file,
    package_pseudos_for_remote,
    preprocess_output_list,
    parse_scripts_block,
)
from vasp_agent import RemoteClusterVASPAgent, VASPAgent
from results import extract_magnetic_moments, generate_electronic_plots
from structure_paths import materialize_relaxed_band_path
from workflow_state import AttemptCheckpoint, WorkflowCheckpoint, create_checkpoint, file_sha256, infer_branches
from workflow_context import WorkflowContext, create_workflow_context


DEFAULT_USER_QE_SLURM_TEMPLATE = "~/.tritondft/example_qe_slurm_job_file.txt"
DEFAULT_USER_VASP_SLURM_TEMPLATE = "~/.tritondft/example_vasp_slurm_job_file.txt"
DEFAULT_USER_SLURM_TEMPLATE = DEFAULT_USER_QE_SLURM_TEMPLATE


WELCOME_BANNER = r"""
--------------------------------------------------------------------------------

                                 WELCOME TO
                    ______     _ __              ____  _____________
                   /_  __/____(_) /_____  ____  / __ \/ ____/_   __/
                    / / / ___/ / __/ __ \/ __ \/ / / / /_    / / 
                   / / / /  / / /_/ /_/ / / / / /_/ / __/   / /
                  /_/ /_/  /_/\__/\____/_/ /_/_____/_/     /_/

                      super user interface (local version)     
--------------------------------------------------------------------------------
"""


def _load_env_file(path: str, *, override: bool = False) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def _read_env_file(path: str) -> Dict[str, str]:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return {}

    data: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _write_env_file(path: str, values: Dict[str, str]) -> None:
    env_path = Path(path).expanduser()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_env_file(str(env_path))
    existing.update({k: v for k, v in values.items() if v is not None})

    ordered_keys = [
        "OPENAI_API_KEY",
        "MP_API_KEY",
        "CLUSTER_AGENT_SSH_TARGET",
        "CLUSTER_AGENT_REMOTE_ROOT",
        "CLUSTER_AGENT_MODEL",
        "CLUSTER_AGENT_BACKEND",
        "CLUSTER_AGENT_WORK_DIR",
        "CLUSTER_AGENT_POLL_SECONDS",
        "TRITONDFT_SLURM_TEMPLATE",
        "TRITONDFT_QE_SLURM_TEMPLATE",
        "TRITONDFT_VASP_SLURM_TEMPLATE",
        "CLUSTER_AGENT_REMOTE_QE_BIN_DIR",
        "CLUSTER_AGENT_REMOTE_VASP_COMMAND",
        "CLUSTER_AGENT_VASP_POTCAR_ROOT",
        "CLUSTER_AGENT_VASP_FUNCTIONAL",
        "CLUSTER_AGENT_NO_QUERY_INFO",
    ]
    lines = [
        "# TritonDFT super-user cluster configuration.",
        "# This file is ignored by Git because it can contain API keys.",
        "",
    ]
    for key in ordered_keys:
        if key in existing:
            lines.append(f"{key}={existing[key]}")
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _env_missing_cluster_setup(path: str) -> bool:
    data = _read_env_file(path)
    ssh_target = data.get("CLUSTER_AGENT_SSH_TARGET", "").strip()
    remote_root = data.get("CLUSTER_AGENT_REMOTE_ROOT", "").strip()
    if not ssh_target or not remote_root:
        return True
    if _ssh_target_needs_user_config(ssh_target):
        print(
            f"[cluster-agent] SSH target '{ssh_target}' is not configured for "
            f"this Linux user ({Path.home()})."
        )
        return True
    return False


def _env_missing_api_keys(path: str) -> bool:
    data = _read_env_file(path)
    return not data.get("OPENAI_API_KEY") or not data.get("MP_API_KEY")


def _ensure_env_defaults(path: str) -> None:
    data = _read_env_file(path)
    defaults: Dict[str, str] = {}
    qe_template_value = (
        data.get("TRITONDFT_QE_SLURM_TEMPLATE", "").strip()
        or data.get("TRITONDFT_SLURM_TEMPLATE", "").strip()
    )
    if not qe_template_value or qe_template_value == "example_slurm_job_file.txt":
        defaults["TRITONDFT_QE_SLURM_TEMPLATE"] = DEFAULT_USER_QE_SLURM_TEMPLATE
        defaults["TRITONDFT_SLURM_TEMPLATE"] = DEFAULT_USER_QE_SLURM_TEMPLATE
    elif not data.get("TRITONDFT_QE_SLURM_TEMPLATE", "").strip():
        defaults["TRITONDFT_QE_SLURM_TEMPLATE"] = qe_template_value
    if not data.get("TRITONDFT_VASP_SLURM_TEMPLATE", "").strip():
        defaults["TRITONDFT_VASP_SLURM_TEMPLATE"] = DEFAULT_USER_VASP_SLURM_TEMPLATE
    if defaults:
        _write_env_file(path, defaults)
    refreshed = _read_env_file(path)
    _ensure_user_qe_slurm_template(
        refreshed.get("TRITONDFT_QE_SLURM_TEMPLATE", "")
        or refreshed.get("TRITONDFT_SLURM_TEMPLATE", "")
    )
    _ensure_user_vasp_slurm_template(refreshed.get("TRITONDFT_VASP_SLURM_TEMPLATE", ""))


def _ensure_user_qe_slurm_template(template_path: str) -> None:
    """Create a per-user QE Slurm example template when the default path is used."""
    if not template_path:
        return
    destination = Path(template_path).expanduser()
    if destination.exists():
        return
    if destination != Path(DEFAULT_USER_QE_SLURM_TEMPLATE).expanduser():
        return

    shared_template = Path(__file__).resolve().parent.parent / "example_slurm_job_file.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shared_template.exists():
        shutil.copy2(shared_template, destination)
    else:
        destination.write_text(
            "#!/bin/bash\n"
            "# Edit this file for your cluster account, partition, and modules.\n"
            "#SBATCH --partition=shared\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --tasks-per-node=1\n"
            "#SBATCH -t 00:10:00\n"
            "#SBATCH -o qe.out\n"
            "#SBATCH -e qe.err\n"
            "#SBATCH --export=ALL\n"
            "#SBATCH --job-name=tritondft-qe\n\n"
            "module load openmpi\n"
            "module load quantum-espresso\n\n"
            "# TritonDFT generated execution block\n"
            "exe=pw.x\n"
            "INPUT=input.in\n"
            "OUTPUT=output.out\n"
            "mpirun -np 1 $exe -in $INPUT > $OUTPUT\n",
            encoding="utf-8",
        )
    destination.chmod(0o600)
    print(f"[cluster-agent] Created user QE Slurm template: {destination}")


def _ensure_user_vasp_slurm_template(template_path: str) -> None:
    """Create a per-user VASP Slurm example template when the default path is used."""
    if not template_path:
        return
    destination = Path(template_path).expanduser()
    if destination.exists():
        return
    if destination != Path(DEFAULT_USER_VASP_SLURM_TEMPLATE).expanduser():
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "#!/bin/bash\n"
        "# Edit this file for your cluster account, partition, and VASP module.\n"
        "# TritonDFT replaces nodes, tasks-per-node, time, output/error files,\n"
        "# and the execution block below while preserving your account/module lines.\n"
        "#SBATCH --partition=shared\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH --tasks-per-node=1\n"
        "#SBATCH -t 01:00:00\n"
        "#SBATCH -o vasp.slurm.out\n"
        "#SBATCH -e vasp.slurm.err\n"
        "#SBATCH --export=ALL\n"
        "#SBATCH --job-name=tritondft-vasp\n\n"
        "module load openmpi\n"
        "module load vasp\n\n"
        "# TritonDFT generated execution block\n"
        "exe=vasp_std\n"
        "INPUT=POSCAR\n"
        "OUTPUT=vasp.out\n"
        "mpirun -np 1 $exe > $OUTPUT\n",
        encoding="utf-8",
    )
    destination.chmod(0o600)
    print(f"[cluster-agent] Created user VASP Slurm template: {destination}")


def _run_interactive(command: str, cwd: Optional[str] = None, verbose: bool = True) -> None:
    if verbose:
        print(f"\n[cluster] running: {command}\n")
    completed = subprocess.run(command, shell=True, cwd=cwd, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with return code {completed.returncode}: {command}")


def _run_capture(command: str, cwd: Optional[str] = None, verbose: bool = True) -> str:
    if verbose:
        print(f"\n[cluster] running: {command}\n")
    completed = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr)
        raise RuntimeError(f"Command failed with return code {completed.returncode}: {command}")
    return completed.stdout.strip()


def _ssh_config_path() -> Path:
    return Path.home() / ".ssh" / "config"


def _parse_ssh_config_hosts() -> List[Dict[str, str]]:
    config_path = _ssh_config_path()
    if not config_path.exists():
        return []

    hosts: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    for raw in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        key = parts[0].lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        if key == "host":
            current = {"host": value.split()[0] if value else ""}
            hosts.append(current)
        elif current is not None:
            current[key] = value
    return hosts


def _find_matching_ssh_host(username: str, hostname: str) -> str:
    for host in _parse_ssh_config_hosts():
        if (
            host.get("hostname", "").lower() == hostname.lower()
            and host.get("user", "") == username
            and host.get("host")
        ):
            return host["host"]
    return ""


def _ssh_host_alias_exists(alias: str) -> bool:
    return any(host.get("host") == alias for host in _parse_ssh_config_hosts())


def _ssh_target_needs_user_config(target: str) -> bool:
    """Return True when an env SSH target is an alias missing from this user's config."""
    target = target.strip()
    if not target:
        return True
    if "@" in target or "." in target or ":" in target:
        return False
    return not _ssh_host_alias_exists(target)


def _default_alias_for(hostname: str) -> str:
    parts = [p for p in hostname.split(".") if p]
    if len(parts) > 1 and parts[0].lower() in {"login", "logon", "ssh"}:
        return re.sub(r"[^A-Za-z0-9_-]+", "-", parts[1]).strip("-") or "cluster"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", parts[0]).strip("-") if parts else "cluster"


def _ensure_ssh_config_host(username: str, hostname: str, preferred_alias: str = "") -> str:
    existing = _find_matching_ssh_host(username, hostname)
    if existing:
        print(f"Found existing SSH config entry: Host {existing}")
        return existing

    alias = preferred_alias or _default_alias_for(hostname)
    if _ssh_host_alias_exists(alias):
        alias = input(f"SSH alias '{alias}' already exists. Enter a new alias: ").strip() or alias

    answer = input(f"No SSH config entry found for {username}@{hostname}. Create Host {alias}? [Y/n] ").strip().lower()
    if answer in {"n", "no"}:
        return f"{username}@{hostname}"

    ssh_dir = Path.home() / ".ssh"
    config_path = _ssh_config_path()
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    if config_path.exists():
        backup_path = config_path.with_name(f"config.backup_{int(time.time())}")
        backup_path.write_text(config_path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        print(f"Backup created: {backup_path}")
    else:
        config_path.touch(mode=0o600)

    block = f"""

Host {alias}
  HostName {hostname}
  User {username}
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPath ~/.ssh/control-%r@%h:%p
  ControlPersist 4h
"""
    with config_path.open("a", encoding="utf-8") as f:
        f.write(block)
    config_path.chmod(0o600)
    print(f"Added SSH config entry: Host {alias}")
    return alias


def _prompt_with_default(label: str, default: str = "") -> str:
    if default:
        entered = input(f"{label} [{default}]: ").strip()
        return entered or default
    return input(f"{label}: ").strip()


def _run_super_user_setup(env_file: str) -> None:
    existing = _read_env_file(env_file)
    current_target = existing.get("CLUSTER_AGENT_SSH_TARGET", "").strip()
    current_remote_root = existing.get("CLUSTER_AGENT_REMOTE_ROOT", "").strip()
    default_alias = current_target if current_target and "@" not in current_target and "." not in current_target else ""

    print("TritonDFT needs a per-user remote cluster setup for this Linux account.\n")
    print("This will create/update:")
    print("  - ~/.tritondft/.env.cluster")
    print("  - ~/.tritondft/example_qe_slurm_job_file.txt")
    print("  - ~/.tritondft/example_vasp_slurm_job_file.txt")
    print("  - ~/.ssh/config, if you choose to create an SSH alias\n")

    print("\nPlease provide the remote cluster connection details.")
    alias = _prompt_with_default(
        "Cluster nickname / SSH alias, e.g. expanse",
        default_alias,
    )
    hostname = _prompt_with_default(
        "Cluster login hostname, e.g. login.expanse.sdsc.edu",
        "",
    )
    username = _prompt_with_default("Your user id on that cluster", os.environ.get("USER", ""))

    print("\nPassword/OTP:")
    print("  TritonDFT does not store your password. SSH will ask for your password/OTP when the connection opens.")

    remote_root = _prompt_with_default(
        "\nRemote working directory on the cluster, e.g. /scratch/$USER/tritondft_runs",
        current_remote_root,
    ).rstrip("/")

    if not username or not hostname or not remote_root:
        raise ValueError("Cluster nickname, user id, login hostname, and remote working directory are required.")

    ssh_target = _ensure_ssh_config_host(username=username, hostname=hostname, preferred_alias=alias)
    _write_env_file(
        env_file,
        {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
            "MP_API_KEY": os.environ.get("MP_API_KEY", ""),
            "CLUSTER_AGENT_SSH_TARGET": ssh_target,
            "CLUSTER_AGENT_REMOTE_ROOT": remote_root,
            "CLUSTER_AGENT_MODEL": os.environ.get("CLUSTER_AGENT_MODEL", "gpt-4o"),
            "CLUSTER_AGENT_BACKEND": os.environ.get("CLUSTER_AGENT_BACKEND", "openai"),
            "CLUSTER_AGENT_WORK_DIR": os.environ.get("CLUSTER_AGENT_WORK_DIR", "tmp"),
            "CLUSTER_AGENT_POLL_SECONDS": os.environ.get("CLUSTER_AGENT_POLL_SECONDS", "30"),
            "TRITONDFT_SLURM_TEMPLATE": os.environ.get(
                "TRITONDFT_SLURM_TEMPLATE",
                DEFAULT_USER_QE_SLURM_TEMPLATE,
            ),
            "TRITONDFT_QE_SLURM_TEMPLATE": os.environ.get(
                "TRITONDFT_QE_SLURM_TEMPLATE",
                os.environ.get("TRITONDFT_SLURM_TEMPLATE", DEFAULT_USER_QE_SLURM_TEMPLATE),
            ),
            "TRITONDFT_VASP_SLURM_TEMPLATE": os.environ.get(
                "TRITONDFT_VASP_SLURM_TEMPLATE",
                DEFAULT_USER_VASP_SLURM_TEMPLATE,
            ),
            "CLUSTER_AGENT_REMOTE_QE_BIN_DIR": os.environ.get("CLUSTER_AGENT_REMOTE_QE_BIN_DIR", ""),
            "CLUSTER_AGENT_REMOTE_VASP_COMMAND": os.environ.get("CLUSTER_AGENT_REMOTE_VASP_COMMAND", ""),
            "CLUSTER_AGENT_VASP_POTCAR_ROOT": os.environ.get("CLUSTER_AGENT_VASP_POTCAR_ROOT", ""),
            "CLUSTER_AGENT_VASP_FUNCTIONAL": os.environ.get("CLUSTER_AGENT_VASP_FUNCTIONAL", ""),
            "CLUSTER_AGENT_NO_QUERY_INFO": os.environ.get("CLUSTER_AGENT_NO_QUERY_INFO", "true"),
        },
    )
    _ensure_env_defaults(env_file)
    print(f"\nSetup complete. Cluster settings were written to {env_file}.")
    print(f"Per-user QE Slurm template: {Path(DEFAULT_USER_QE_SLURM_TEMPLATE).expanduser()}")
    print(f"Per-user VASP Slurm template: {Path(DEFAULT_USER_VASP_SLURM_TEMPLATE).expanduser()}")


@dataclass
class ClusterJob:
    job_id: str
    script_name: str
    remote_dir: str
    submit_output: str


class SSHClusterTransport:
    """
    Shell-based SSH/rsync transport for cluster workflows.

    It intentionally uses local ``ssh`` and ``rsync`` commands rather than a
    Python SSH library so password, OTP, Duo, and cluster banner flows remain
    visible in the user's terminal.
    """

    def __init__(
        self,
        ssh_target: str,
        remote_root: str,
        *,
        poll_seconds: int = 30,
        keep_master: bool = True,
        verbose: bool = True,
    ):
        self.ssh_target = ssh_target
        self.remote_root = remote_root.rstrip("/")
        self.poll_seconds = poll_seconds
        self.keep_master = keep_master
        self.verbose = verbose

    def set_remote_root(self, remote_root: str) -> None:
        remote_root = remote_root.strip().rstrip("/")
        if not remote_root:
            raise ValueError("Remote root directory cannot be empty.")
        self.remote_root = remote_root

    def ensure_connection(self) -> None:
        if not self.keep_master:
            self.test_connection()
            return

        check = subprocess.run(
            f"ssh -O check {shlex.quote(self.ssh_target)}",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if check.returncode == 0:
            if self.verbose:
                print("[cluster] persistent SSH connection is active.")
            return

        print("[cluster] opening persistent SSH connection. You may be prompted for password/OTP.")
        _run_interactive(f"ssh -MNf {shlex.quote(self.ssh_target)}", verbose=self.verbose)
        self.test_connection()

    def close_connection(self) -> None:
        if not self.keep_master:
            return
        subprocess.run(
            f"ssh -O exit {shlex.quote(self.ssh_target)}",
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_connection(self) -> None:
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} 'echo SSH connection successful'",
            verbose=self.verbose,
        )

    def remote_dir_for(self, local_run_dir: Path) -> str:
        if local_run_dir.parent.name == "branches":
            run_name = local_run_dir.parent.parent.name
            return f"{self.remote_root}/{run_name}/branches/{local_run_dir.name}"
        return f"{self.remote_root}/{local_run_dir.name}"

    def remote_attempt_dir(self, run_dir: Path, step_id: int, task_name: str, attempt: int) -> str:
        safe_task = re.sub(r"[^a-z0-9]+", "-", task_name.lower()).strip("-") or "step"
        return f"{self.remote_root}/{run_dir.name}/steps/{step_id:02d}-{safe_task}/attempt_{attempt:03d}"

    def clone_remote_directory(self, source_dir: str, destination_dir: str) -> None:
        """Seed a branch with QE state only, never with jobs, logs, or inputs."""
        source_q = shlex.quote(source_dir)
        destination_q = shlex.quote(destination_dir)
        marker_q = shlex.quote(f"{destination_dir}/.tritondft_branch_initialized")
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} '"
            f"if [ ! -f {marker_q} ]; then mkdir -p {destination_q} && "
            f"for item in tritondft_workflow.save tritondft_workflow.xml; do "
            f"if [ -e {source_q}/$item ]; then cp -a {source_q}/$item {destination_q}/; fi; "
            f"done && touch {marker_q}; fi'",
            verbose=self.verbose,
        )

    def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        remote_q = shlex.quote(remote_dir)
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} 'mkdir -p {remote_q}'",
            verbose=self.verbose,
        )

    def upload_step_files(
        self,
        local_dir: Path,
        remote_dir: str,
        paths: List[str],
    ) -> None:
        """Upload only current executable inputs/scripts and branch pseudopotentials."""
        remote_q = shlex.quote(remote_dir)
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} 'mkdir -p {remote_q}'",
            verbose=self.verbose,
        )
        selected = [Path(path) for path in paths if Path(path).is_file()]
        pseudo_dir = local_dir / "pseudos"
        if pseudo_dir.is_dir():
            selected.append(pseudo_dir)
        for source in selected:
            _run_interactive(
                "rsync -az --progress -e ssh "
                f"{shlex.quote(str(source))} "
                f"{shlex.quote(self.ssh_target + ':' + remote_dir + '/')}",
                verbose=self.verbose,
            )

    def submit(self, remote_dir: str, script_name: str) -> ClusterJob:
        remote_dir_q = shlex.quote(remote_dir)
        script_q = shlex.quote(script_name)
        out = _run_capture(
            f"ssh {shlex.quote(self.ssh_target)} 'cd {remote_dir_q} && sbatch {script_q}'",
            verbose=self.verbose,
        )
        job_id = self._extract_job_id(out)
        if not job_id:
            raise RuntimeError(f"Could not detect Slurm job id from sbatch output: {out}")
        return ClusterJob(job_id=job_id, script_name=script_name, remote_dir=remote_dir, submit_output=out)

    def archive_failure_markers(self, remote_dir: str, attempt: int) -> None:
        """Move stale diagnostics into a recoverable attempt history before retry."""
        directory_q = shlex.quote(remote_dir)
        history_q = shlex.quote(f"{remote_dir}/.tritondft_history/attempt_{attempt}")
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} 'mkdir -p {history_q}; "
            f"for item in CRASH qe.err qe.out; do "
            f"if [ -f {directory_q}/$item ]; then mv {directory_q}/$item {history_q}/$item; fi; done'",
            verbose=self.verbose,
        )

    def fresh_start_branch(
        self,
        source_dir: str,
        destination_dir: str,
        attempt: int,
    ) -> str:
        """Archive a failed branch and recreate it from clean completed parent state."""
        source_q = shlex.quote(source_dir)
        destination_q = shlex.quote(destination_dir)
        archive_dir = f"{destination_dir}.failed_attempt_{attempt}_{int(time.time())}"
        archive_q = shlex.quote(archive_dir)
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} '"
            f"if [ -d {destination_q} ]; then mv {destination_q} {archive_q}; fi; "
            f"mkdir -p {destination_q}; "
            f"for item in tritondft_workflow.save tritondft_workflow.xml; do "
            f"if [ -e {source_q}/$item ]; then cp -a {source_q}/$item {destination_q}/; fi; done; "
            f"touch {destination_q}/.tritondft_branch_initialized'",
            verbose=self.verbose,
        )
        return archive_dir

    def wait_for_job(self, job: ClusterJob) -> None:
        print(f"[cluster] waiting for Slurm job {job.job_id}. Press Ctrl-C to stop monitoring.")
        while True:
            try:
                out = _run_capture(
                    f"ssh {shlex.quote(self.ssh_target)} 'squeue -j {shlex.quote(job.job_id)} -h'",
                    verbose=False,
                )
            except RuntimeError as exc:
                print(f"[cluster] squeue check failed: {exc}")
                out = ""

            if not out.strip():
                print(f"[cluster] job {job.job_id} no longer appears in squeue.")
                return

            if self.verbose:
                print(out)
            time.sleep(self.poll_seconds)

    def fetch_directory(self, remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        _run_interactive(
            "rsync -az --progress -e ssh "
            f"{shlex.quote(self.ssh_target + ':' + remote_dir + '/')} "
            f"{shlex.quote(str(local_dir) + '/')}",
            verbose=self.verbose,
        )

    def fetch_files(self, remote_dir: str, local_dir: Path, names: List[str]) -> List[str]:
        """Fetch explicitly named small control/result files, ignoring absent optional files."""
        local_dir.mkdir(parents=True, exist_ok=True)
        fetched: List[str] = []
        for name in dict.fromkeys(Path(name).name for name in names if name):
            remote_file = f"{remote_dir}/{name}"
            exists = _run_capture(
                f"ssh {shlex.quote(self.ssh_target)} 'test -f {shlex.quote(remote_file)} && echo yes || true'",
                verbose=False,
            )
            if exists.strip() != "yes":
                continue
            _run_interactive(
                "rsync -az -e ssh "
                f"{shlex.quote(self.ssh_target + ':' + remote_file)} "
                f"{shlex.quote(str(local_dir) + '/')} ",
                verbose=self.verbose,
            )
            fetched.append(str(local_dir / name))
        return fetched

    def remote_files_ok(self, remote_dir: str, names: List[str]) -> List[str]:
        """Return missing/empty remote artifacts without downloading them."""
        problems: List[str] = []
        for name in dict.fromkeys(Path(name).name for name in names if name):
            remote_file = f"{remote_dir}/{name}"
            result = _run_capture(
                f"ssh {shlex.quote(self.ssh_target)} 'test -s {shlex.quote(remote_file)} && echo ok || echo missing'",
                verbose=False,
            )
            if result.strip() != "ok":
                problems.append(f"Remote artifact missing/empty: {remote_file}")
        return problems

    def remote_qe_state_ok(self, remote_dir: str, prefix: str = "tritondft_workflow") -> List[str]:
        save_q = shlex.quote(f"{remote_dir}/{prefix}.save")
        try:
            result = _run_capture(
                f"ssh {shlex.quote(self.ssh_target)} 'test -d {save_q} && "
                f"test -s {save_q}/data-file-schema.xml'",
                verbose=False,
            )
            _ = result
            return []
        except RuntimeError:
            return [f"Reusable QE state is missing or incomplete: {remote_dir}/{prefix}.save"]

    def list_remote_result_files(self, remote_dir: str) -> List[str]:
        """List processed result files while explicitly excluding QE save trees."""
        directory_q = shlex.quote(remote_dir)
        output = _run_capture(
            f"ssh {shlex.quote(self.ssh_target)} 'cd {directory_q} && "
            "find . -maxdepth 1 -type f \\( -name \"*.dos\" -o -name \"*.gnu\" "
            "-o -name \"*pdos*\" -o -name \"*.freq\" -o -name \"*.modes\" "
            "-o -name \"*.dyn*\" -o -name \"*.fc\" \\) -print'",
            verbose=False,
        )
        return [line.strip().removeprefix("./") for line in output.splitlines() if line.strip()]

    @staticmethod
    def _extract_job_id(sbatch_output: str) -> str:
        match = re.search(r"\b(\d+)\b", sbatch_output or "")
        return match.group(1) if match else ""


RELAXED_STRUCTURE_PLACEHOLDER = """! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN
! Relaxed CELL_PARAMETERS and ATOMIC_POSITIONS from the vc-relax step will be inserted here before execution.
! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_END"""


def _plan_text(
    subproblems: List[Dict[str, Any]],
    packages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    lines = ["TritonDFT execution plan", "========================"]
    for index, step in enumerate(subproblems):
        tool_name = step.get("tool") or "unknown"
        step_id = int(step.get("id", index + 1))
        why = (step.get("why") or "").strip()
        if not why:
            try:
                why = get_spec(tool_name).description
            except Exception:
                why = "This step contributes to the requested workflow result."
        lines.extend(
            [
                "",
                f"Step {step_id}: {step.get('problem') or tool_name}",
                f"   Tool: {tool_name}",
                *(
                    ["   Input file(s): " + ", ".join(
                        Path(path).name for path in packages[index].get("input_paths", [])
                    )]
                    if packages and index < len(packages)
                    else []
                ),
                f"   Required input: {step.get('input') or 'Generated from the user request and workflow context'}",
                f"   Why: {why}",
            ]
        )
    lines.extend(
        [
            "",
            "Approval gate",
            "-------------",
            "All input files are generated before execution. No cluster job is submitted until you approve them.",
            "Files after vc-relax contain a visible relaxed-structure placeholder; it is replaced with the final",
            "vc-relax CELL_PARAMETERS and ATOMIC_POSITIONS immediately before that file is submitted.",
        ]
    )
    return "\n".join(lines) + "\n"


def _card_block_span(text: str, card: str) -> Optional[tuple[int, int]]:
    header = re.search(rf"(?mi)^[ \t]*{re.escape(card)}(?:\s*\([^)]+\))?[^\n]*\n?", text)
    if not header:
        return None
    next_header = re.search(
        r"(?mi)^[ \t]*(?:ATOMIC_SPECIES|ATOMIC_POSITIONS|CELL_PARAMETERS|K_POINTS|"
        r"ATOMIC_FORCES|OCCUPATIONS|CONSTRAINTS|HUBBARD|&[A-Z_]+)\b",
        text[header.end():],
    )
    end = header.end() + next_header.start() if next_header else len(text)
    return header.start(), end


def _add_relaxed_structure_placeholder(path: str) -> bool:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN" in text:
        return True

    spans = [
        span
        for span in (
            _card_block_span(text, "ATOMIC_POSITIONS"),
            _card_block_span(text, "CELL_PARAMETERS"),
        )
        if span is not None
    ]
    if not spans:
        return False

    for start, end in sorted(spans, reverse=True):
        text = text[:start] + text[end:]

    text = re.sub(
        r"(?mi)^([ \t]*ibrav\s*=\s*)[-+]?\d+(\s*,?.*)$",
        r"\g<1>0\2",
        text,
    )
    text = re.sub(
        r"(?mi)^[ \t]*(?:celldm\s*\(\s*[1-6]\s*\)|A|B|C|cosAB|cosAC|cosBC)\s*=.*\n?",
        "",
        text,
    )
    # Card removal changes every later character offset. Recompute a semantic
    # insertion point instead of reusing an offset from the original text (the
    # old behavior could place this marker in the middle of a K_POINTS row).
    kpoints = re.search(r"(?mi)^[ \t]*K_POINTS\b", text)
    if not kpoints:
        return False
    marker = RELAXED_STRUCTURE_PLACEHOLDER + "\n"
    text = text[:kpoints.start()] + marker + text[kpoints.start():]
    input_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return True


def _set_workflow_prefix(path: str, prefix: str = "tritondft_workflow") -> None:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    updated = re.sub(
        r"(?mi)^([ \t]*prefix\s*=\s*)['\"][^'\"]+['\"](\s*,?.*)$",
        rf"\g<1>'{prefix}'\2",
        text,
    )
    if updated != text:
        input_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _normalize_namelist_final_commas(path: str) -> None:
    """Canonicalize final namelist assignments for older/site-patched QE parsers."""
    input_path = Path(path)
    lines = input_path.read_text(encoding="utf-8").splitlines()
    inside = False
    previous_assignment: Optional[int] = None
    changed = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("&"):
            inside = True
            previous_assignment = None
            continue
        if inside and stripped == "/":
            if previous_assignment is not None:
                code, separator, comment = lines[previous_assignment].partition("!")
                if not code.rstrip().endswith(","):
                    code = code.rstrip() + ","
                    lines[previous_assignment] = code + (f" !{comment}" if separator else "")
                    changed = True
            inside = False
            previous_assignment = None
            continue
        if inside and "=" in line and not stripped.startswith("!"):
            previous_assignment = index
    if changed:
        input_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _enforce_workflow_artifact_names(step: Dict[str, Any], path: str) -> None:
    """Keep producer and consumer filenames canonical during generation/repair."""
    tool = str(step.get("tool") or "")
    problem = str(step.get("problem") or "").lower()
    replacements: Dict[str, str] = {}
    if tool == "pw_phonon_gamma":
        is_grid = any(term in problem for term in ("dispersion", "q-grid", "q grid", "q-point mesh", "uniform phonon"))
        replacements["fildyn"] = "tritondft_workflow.dyn" if is_grid else "tritondft_workflow.dynG"
    elif tool == "q2r_post":
        replacements = {"fildyn": "tritondft_workflow.dyn", "flfrc": "tritondft_workflow.fc"}
    elif tool == "matdyn_post":
        replacements = {"flfrc": "tritondft_workflow.fc"}
    elif tool == "dynmat_post":
        replacements = {"fildyn": "tritondft_workflow.dynG"}
    if not replacements:
        return
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = re.sub(
            rf"(?mi)^(\s*{re.escape(key)}\s*=\s*)['\"][^'\"]+['\"]",
            rf"\g<1>'{value}'",
            text,
        )
    input_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _force_ph_fresh_start(path: str) -> None:
    """Disable ph.x recovery when a branch has been deliberately rebuilt cleanly."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    if not re.search(r"(?mi)^\s*&inputph\b", text):
        return
    if re.search(r"(?mi)^\s*recover\s*=", text):
        text = re.sub(
            r"(?mi)^(\s*recover\s*=\s*)(?:\.true\.|true|\.false\.|false)(\s*,?.*)$",
            r"\g<1>.false.,",
            text,
        )
    else:
        match = re.search(r"(?mis)^\s*&inputph\b.*?^\s*/\s*$", text)
        if not match:
            return
        slash = text.rfind("/", match.start(), match.end())
        text = text[:slash] + "  recover=.false.,\n" + text[slash:]
    input_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _extract_relaxed_structure(output_path: str) -> str:
    from evaluate.relax_eval import (
        _extract_cell,
        _extract_pos,
        _normalize_cell_to_angstrom,
        _slice_final_window,
    )

    output_text = Path(output_path).read_text(encoding="utf-8", errors="replace")
    final_window = _slice_final_window(output_text)
    if not final_window:
        raise RuntimeError(f"No final relaxed coordinates found in {output_path}.")
    cell = _normalize_cell_to_angstrom(_extract_cell(final_window)).rstrip()
    positions = _extract_pos(final_window).rstrip()
    return f"{cell}\n{positions}\n"


def _insert_relaxed_structure(path: str, structure_block: str) -> bool:
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?ms)^! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN\n"
        r".*?"
        r"^! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_END\n?"
    )
    if not pattern.search(text):
        return False
    updated = pattern.sub(structure_block.rstrip() + "\n", text, count=1)
    input_path.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return True


def _input_validation_errors(path: str) -> List[str]:
    return [issue.message for issue in validate_qe_input(path) if issue.blocking]


_WORKFLOW_REPAIR_TOOLS = {
    "DOS_INTEGRATION_MISMATCH": {"dos_post"},
    "RAMAN_NOT_IMPLEMENTED": {"pw_phonon_gamma"},
    "RAMAN_DYNMAT_FILENAME_MISMATCH": {"dynmat_post"},
    "PH_Q2R_FILENAME_MISMATCH": {"q2r_post"},
    "Q2R_MATDYN_FILENAME_MISMATCH": {"matdyn_post"},
    "ELECTRONIC_STATE_MISMATCH": {"bands_post", "dos_post", "projwfc_post"},
    "SPIN_SETUP_MISSING": {"pw_vc_relax", "pw_relax", "pw_scf"},
    "INITIAL_MOMENTS_MISSING": {"pw_vc_relax", "pw_relax", "pw_scf"},
    "VDW_DECISION_MISSING": {"pw_vc_relax", "pw_relax", "pw_scf"},
    "HUBBARD_DECISION_MISSING": {"pw_vc_relax", "pw_relax", "pw_scf"},
    "SPECIES_PSEUDO_MISMATCH": {"pw_scf", "pw_bands", "pw_nscf"},
    "SHARED_SETTING_MISMATCH": {"pw_scf", "pw_bands", "pw_nscf"},
    "BULK_BOUNDARY_MISMATCH": {"pw_vc_relax", "pw_relax", "pw_scf"},
}


def _workflow_repair_indices(
    issues,
    steps: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
) -> List[int]:
    """Choose the smallest safe set of generated steps to ask the LLM to repair."""
    selected = set()
    path_to_index = {
        str(Path(path).resolve()): index
        for index, package in enumerate(packages)
        for path in package.get("input_paths", [])
    }
    for issue in issues:
        if issue.path:
            index = path_to_index.get(str(Path(issue.path).resolve()))
            if index is not None:
                selected.add(index)
                continue
        tools = _WORKFLOW_REPAIR_TOOLS.get(issue.code, set())
        selected.update(
            index for index, step in enumerate(steps) if step.get("tool") in tools
        )
    if selected:
        return sorted(selected)
    # Unknown cross-step failures are safer to repair by regenerating all
    # non-relaxation inputs. Never discard a valid initial structure merely
    # because the validator did not yet have a targeted mapping.
    fallback = [
        index for index, step in enumerate(steps)
        if step.get("tool") not in {"pw_vc_relax", "pw_relax"}
    ]
    return fallback or list(range(len(steps)))


def _isolate_branch_packages(
    run_dir: Path,
    steps: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
) -> List[str]:
    """Move generated inputs into branch-local workspaces and update packages."""
    branches_root = run_dir / "branches"
    branches_root.mkdir(parents=True, exist_ok=True)
    branches = infer_branches(steps)
    all_inputs: List[str] = []
    used_stems: Dict[str, int] = {}
    for index, (package, branch) in enumerate(zip(packages, branches), start=1):
        branch_dir = branches_root / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        base_stem = _task_file_stem(steps[index - 1])
        used_stems[base_stem] = used_stems.get(base_stem, 0) + 1
        if used_stems[base_stem] > 1:
            base_stem = f"{base_stem}-{used_stems[base_stem]}"
        relocated_inputs: List[str] = []
        input_count = len(package.get("input_paths", []))
        for input_index, input_path in enumerate(package.get("input_paths", []), start=1):
            source = Path(input_path)
            suffix = f"-{input_index}" if input_count > 1 else ""
            destination = branch_dir / f"{base_stem}{suffix}.in"
            if source.resolve() != destination.resolve():
                if destination.exists():
                    destination.unlink()
                shutil.move(str(source), destination)
            relocated_inputs.append(str(destination))
        package["input_paths"] = relocated_inputs
        output_count = len(package.get("output_paths", []))
        package["output_paths"] = [
            str(branch_dir / f"{base_stem}{f'-{output_index}' if output_count > 1 else ''}.out")
            for output_index, _path in enumerate(package.get("output_paths", []), start=1)
        ]
        package["work_dir"] = str(branch_dir)
        package["branch"] = branch
        all_inputs.extend(relocated_inputs)
    return all_inputs


def _task_file_stem(step: Dict[str, Any]) -> str:
    """Return a concise scientific task name for a generated QE input."""
    tool = str(step.get("tool") or "")
    problem = str(step.get("problem") or "").lower()
    if tool == "pw_vc_relax":
        return "vc-relax"
    if tool == "pw_relax":
        return "relax"
    if tool == "pw_scf":
        return "scf"
    if tool == "pw_nscf":
        return "nscf"
    if tool == "pw_bands":
        return "bands"
    if tool == "bands_post":
        return "bands-post"
    if tool == "dos_post":
        return "dos"
    if tool == "projwfc_post":
        return "pdos"
    if tool == "pw_phonon_gamma":
        is_grid = any(term in problem for term in ("dispersion", "q-grid", "q grid", "q-point mesh"))
        return "phonon-grid" if is_grid else "phonon-gamma-raman" if "raman" in problem else "phonon-gamma"
    if tool == "q2r_post":
        return "q2r"
    if tool == "matdyn_post":
        return "phonon-dispersion"
    if tool == "dynmat_post":
        return "raman-analysis" if "raman" in problem else "dynmat"
    cleaned = re.sub(r"[^a-z0-9]+", "-", tool.lower()).strip("-")
    return cleaned or f"step-{step.get('id', 'unknown')}"


def _approve_inputs_popup(
    plan: str,
    input_paths: List[str],
    workflow_validator=None,
    *,
    review_stage: str = "inputs",
) -> Any:
    plan_only = review_stage == "plan"
    print("\n[approval] Scientific assessment and execution plan are ready." if plan_only else "\n[approval] All inputs are ready.")
    print(
        "[approval] Opening the TritonDFT plan-review window; input generation is paused."
        if plan_only
        else "[approval] Opening the TritonDFT approval window; execution is paused until you choose Approve & Run or Cancel."
    )
    print("[approval] If the window is behind your IDE, use Cmd-Tab to select Python/TritonDFT.")
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk

        root = tk.Tk()
        root.title("TritonDFT scientific assessment and plan review" if plan_only else "TritonDFT plan and input approval")
        root.geometry("1100x760")
        root.update_idletasks()
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.focus_force()

        def release_topmost() -> None:
            try:
                root.attributes("-topmost", False)
                root.lift()
                root.focus_force()
            except tk.TclError:
                pass

        root.after(750, release_topmost)
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        plan_text = tk.Text(notebook, wrap="word", font=("Menlo", 12))
        plan_text.insert("1.0", plan)
        plan_text.configure(state="disabled")
        notebook.add(plan_text, text="Complete plan")

        editors: Dict[str, Any] = {}
        for path in input_paths:
            editor = tk.Text(notebook, wrap="none", undo=True, font=("Menlo", 11))
            editor.insert("1.0", Path(path).read_text(encoding="utf-8"))
            notebook.add(editor, text=Path(path).name)
            editors[path] = editor

        validation_text = tk.Text(notebook, wrap="word", font=("Menlo", 11))
        validation_text.configure(state="disabled")
        notebook.add(validation_text, text="Validation")

        decision = {"action": "cancel", "revision": ""}

        revision_frame = ttk.Frame(notebook)
        ttk.Label(
            revision_frame,
            text=(
                "Describe anything missing or incorrect. TritonDFT will create a new audited "
                "draft and reopen this window without submitting a job."
            ),
            wraplength=1000,
        ).pack(fill="x", padx=8, pady=8)
        revision_text = tk.Text(revision_frame, wrap="word", height=12, font=("Menlo", 11))
        revision_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        notebook.add(revision_frame, text="Request revision")

        def close_window() -> None:
            """Destroy Tk from inside its event loop before execution continues."""
            try:
                root.withdraw()
                root.update_idletasks()
                root.destroy()
            except tk.TclError:
                pass

        def save_edits() -> None:
            for path, editor in editors.items():
                Path(path).write_text(editor.get("1.0", "end-1c").rstrip() + "\n", encoding="utf-8")

        def refresh_validation() -> List[str]:
            save_edits()
            messages: List[str] = []
            for path in input_paths:
                errors = _input_validation_errors(path)
                if errors:
                    messages.append(f"{Path(path).name}:\n  - " + "\n  - ".join(errors))
            if workflow_validator is not None:
                workflow_messages = list(workflow_validator())
                if workflow_messages:
                    messages.append("Cross-step workflow validation:\n  - " + "\n  - ".join(workflow_messages))
            report = (
                "\n\n".join(messages)
                if messages
                else "Deterministic input and cross-step workflow validation passed."
            )
            validation_text.configure(state="normal")
            validation_text.delete("1.0", "end")
            validation_text.insert("1.0", report)
            validation_text.configure(state="disabled")
            return messages

        def approve() -> None:
            messages = refresh_validation()
            if messages:
                notebook.select(validation_text)
                messagebox.showerror(
                    "Input validation failed",
                    "One or more inputs are incomplete. Fix the listed issues before approval.",
                )
                return
            decision["action"] = "approve"
            # Schedule destruction inside Tk's own event loop. mainloop() must
            # return before any SSH/probe/cluster work starts.
            root.withdraw()
            root.update_idletasks()
            root.after_idle(close_window)

        def cancel() -> None:
            if messagebox.askyesno("Cancel workflow", "Cancel without submitting any cluster jobs?"):
                root.withdraw()
                root.update_idletasks()
                root.after_idle(close_window)

        def revise() -> None:
            comment = revision_text.get("1.0", "end-1c").strip()
            if not comment:
                notebook.select(revision_frame)
                messagebox.showerror("Revision comment required", "Describe what must change in the plan or inputs.")
                return
            decision["action"] = "revise"
            decision["revision"] = comment
            root.withdraw()
            root.update_idletasks()
            root.after_idle(close_window)

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            controls,
            text=(
                "Review the assessment and plan. Continue generates inputs locally; no SSH job is submitted."
                if plan_only
                else "Review or edit any tab. Approve & Run saves the edits and unlocks cluster submission."
            ),
        ).pack(side="left")
        ttk.Button(controls, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
        ttk.Button(
            controls,
            text="Approve Plan & Generate Inputs" if plan_only else "Approve & Run",
            command=approve,
        ).pack(side="right")
        ttk.Button(controls, text="Revise Plan & Inputs", command=revise).pack(side="right", padx=(0, 8))
        ttk.Button(controls, text="Validate", command=refresh_validation).pack(side="right", padx=(0, 8))
        root.protocol("WM_DELETE_WINDOW", cancel)
        refresh_validation()
        root.mainloop()
        return decision
    except Exception as exc:
        print(f"[approval] GUI unavailable ({exc}). Falling back to terminal approval.")
        print("\n" + plan)
        print("Generated input files:")
        for path in input_paths:
            print(f"  - {path}")
        while True:
            validation_messages: List[str] = []
            if workflow_validator is not None:
                validation_messages = list(workflow_validator())
                if validation_messages:
                    print("Validation errors:")
                    for message in validation_messages:
                        print(f"  - {message}")
            answer = input(
                "Type 'approve', 'revise', 'edit', or 'cancel': "
            ).strip()
            normalized = answer.lower()
            if normalized in {"approve", "yes", "y"}:
                if validation_messages:
                    print("Approval is blocked until the validation errors are fixed.")
                    continue
                return {"action": "approve", "revision": ""}
            if normalized in {"cancel", "no", "n"}:
                return {"action": "cancel", "revision": ""}
            if normalized == "revise":
                comment = input("Revision request: ").strip()
                if comment:
                    return {"action": "revise", "revision": comment}
            if normalized == "edit":
                print("Edit the files in your editor, then return here to approve or cancel.")


def _approval_result(value: Any) -> tuple[str, str]:
    """Normalize modern approval decisions and legacy boolean test callbacks."""
    if isinstance(value, dict):
        return str(value.get("action", "cancel")), str(value.get("revision", "")).strip()
    return ("approve", "") if value else ("cancel", "")


def _prompt_download_scope(event: str, run_dir: str) -> str:
    label = "completed" if event == "completed" else "paused after an error"
    print(f"\n[download] Workflow {label}. Remote data remain on the cluster.")
    print("  1) summary  - text outputs and compact summary only")
    print("  2) results  - processed numerical data and plots (recommended)")
    print("  3) all      - complete branch directories including QE save data")
    print("  4) none     - keep data on the cluster")
    try:
        answer = input("Download choice [results]: ").strip().lower()
    except (EOFError, OSError):
        return "none"
    aliases = {
        "1": "summary", "summary": "summary",
        "2": "results", "result": "results", "results": "results", "": "results",
        "3": "all", "all": "all",
        "4": "none", "none": "none", "no": "none", "n": "none",
    }
    return aliases.get(answer, "none")


def _prompt_failure_action(step_label: str, error: str) -> str:
    print(f"\n[recovery] {step_label} failed:\n{error}")
    print("  1) Fresh-start the failed branch from its completed parent (recommended for invalid results)")
    print("  2) Edit the failed input yourself, validate it, and run a new attempt")
    print("  3) Retry the unchanged input in a new attempt")
    print("  4) Choose files to download")
    print("  5) Stop and keep remote data/checkpoint")
    try:
        answer = input("Recovery choice [stop]: ").strip().lower()
    except (EOFError, OSError):
        return "stop"
    return {
        "1": "fresh", "fresh": "fresh",
        "2": "edit", "edit": "edit",
        "3": "retry", "retry": "retry",
        "4": "download", "download": "download",
        "5": "stop", "stop": "stop", "": "stop",
    }.get(answer, "stop")


def _launch_workflow_monitor(run_dir: str | Path) -> None:
    """Open a non-blocking status/validation window that remains alive during execution."""
    monitor = Path(__file__).with_name("workflow_monitor.py")
    try:
        subprocess.Popen(
            [sys.executable, str(monitor), str(Path(run_dir).resolve())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        print(f"[monitor] live workflow window unavailable: {exc}")


class RemoteClusterDFTAgent:
    """
    Orchestrates DFTAgent generation locally and QE execution remotely.

    The full plan and every QE input are generated locally first. The user can
    edit them in an approval window, and no cluster job is submitted until the
    complete set is approved. Downstream pw.x inputs use a placeholder that is
    replaced with the final vc-relax structure immediately before submission.
    """

    def __init__(
        self,
        dft_agent: DFTAgent,
        transport: SSHClusterTransport,
        approval_callback=None,
        download_callback=None,
        failure_callback=None,
    ):
        self.agent = dft_agent
        self.transport = transport
        self.approval_callback = approval_callback or _approve_inputs_popup
        self._uses_default_approval = approval_callback is None
        self.download_callback = download_callback or _prompt_download_scope
        self.failure_callback = failure_callback or _prompt_failure_action
        self.agent.run_mode = "cluster_input"

    def _download_workflow_artifacts(
        self,
        workflow_state: WorkflowCheckpoint,
        scope: str,
    ) -> None:
        if scope == "none":
            return
        seen: set[tuple[str, str]] = set()
        for checkpoint in workflow_state.steps:
            if not checkpoint.remote_dir:
                continue
            local_dir = Path(workflow_state.run_dir) / "branches" / checkpoint.branch
            key = (checkpoint.remote_dir, str(local_dir))
            if key in seen:
                continue
            seen.add(key)
            if scope == "all":
                self.transport.fetch_directory(checkpoint.remote_dir, local_dir)
                continue
            names = [Path(path).name for path in checkpoint.output_paths]
            if scope == "results":
                names.extend(self.transport.list_remote_result_files(checkpoint.remote_dir))
            self.transport.fetch_files(checkpoint.remote_dir, local_dir, names)

    @staticmethod
    def _failed_step(state: WorkflowCheckpoint):
        return next(
            (step for step in state.steps if step.status in {"awaiting_user", "failed", "running", "submitted"}),
            None,
        )

    def _propose_recovery_edit(
        self,
        run_dir: str,
        comment: str,
        error: str,
    ) -> bool:
        """Generate a reviewable input proposal; never submit or overwrite before approval."""
        state = WorkflowCheckpoint.load(run_dir)
        failed = self._failed_step(state)
        if failed is None or not failed.input_paths:
            print("[recovery] No editable failed input was found.")
            return False
        step_index = next(i for i, step in enumerate(state.plan) if int(step["id"]) == failed.id)
        original_texts = [Path(path).read_text(encoding="utf-8") for path in failed.input_paths]
        version = getattr(self.agent, "remote_qe_version", "") or "unknown"
        prompt = (
            "You are preparing a proposed correction to failed Quantum ESPRESSO input files. "
            "Return exactly one <scripts> block containing one <script> element per input, in the "
            "same order. Each script must contain the complete corrected input and no Markdown. "
            "Make only changes necessary to address the user's recovery comment and runtime error. "
            "Preserve prefix, outdir, pseudopotential/XC choices, physical task, and artifact filenames. "
            "Do not invent unsupported executable keywords. Nothing will run automatically; the user "
            "will review the exact proposal.\n\n"
            f"Remote QE version: {version}\n"
            f"Failed step: {failed.id} {failed.tool} — {failed.problem}\n"
            f"User recovery comment: {comment}\n"
            f"Runtime/validation error:\n{error[-6000:]}\n\n"
            + "\n\n".join(
                f"CURRENT INPUT {Path(path).name}:\n{text}"
                for path, text in zip(failed.input_paths, original_texts)
            )
        )
        response = self.agent.generator(
            prompt,
            max_new_tokens=max(self.agent.max_new_tokens, 8192),
            return_full_text=False,
        )
        generated = response[0].get("generated_text", "") if response else ""
        scripts = parse_scripts_block(generated)
        if len(scripts) != len(failed.input_paths):
            raise RuntimeError(
                f"Recovery proposal returned {len(scripts)} input(s); expected {len(failed.input_paths)}."
            )

        proposal_number = len(list((Path(run_dir) / "recovery_proposals").glob("proposal_*"))) + 1
        proposal_dir = Path(run_dir) / "recovery_proposals" / f"proposal_{proposal_number:03d}"
        proposal_dir.mkdir(parents=True, exist_ok=False)
        pseudo_source = Path(state.packages[step_index]["work_dir"]) / "pseudos"
        if pseudo_source.is_dir():
            shutil.copytree(pseudo_source, proposal_dir / "pseudos", dirs_exist_ok=True)
        proposal_paths: List[str] = []
        diff_sections: List[str] = []
        for original, old_text, new_text in zip(failed.input_paths, original_texts, scripts):
            proposal = proposal_dir / Path(original).name
            proposal.write_text(new_text.rstrip() + "\n", encoding="utf-8")
            proposal_paths.append(str(proposal))
            diff_sections.extend(difflib.unified_diff(
                old_text.splitlines(), new_text.splitlines(),
                fromfile=f"current/{Path(original).name}",
                tofile=f"proposal/{Path(original).name}",
                lineterm="",
            ))

        proposed_packages = copy.deepcopy(state.packages)
        proposed_packages[step_index]["input_paths"] = proposal_paths

        def proposal_validation() -> List[str]:
            issues = validate_generated_workflow(state.query, state.plan, proposed_packages)
            return [issue.format() for issue in issues if issue.blocking]

        review = (
            f"Recovery proposal for step {failed.id}\n\nUser comment:\n{comment}\n\n"
            f"Reported error:\n{error}\n\nExact proposed diff:\n"
            + ("\n".join(diff_sections) or "No changes were proposed.")
        )
        proposal_decision = _approve_inputs_popup(review, proposal_paths, proposal_validation)
        proposal_action, _ = _approval_result(proposal_decision)
        approved = proposal_action == "approve"
        metadata = {
            "step_id": failed.id,
            "comment": comment,
            "error": error,
            "approved": bool(approved),
            "proposal_paths": proposal_paths,
        }
        (proposal_dir / "proposal.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if not approved:
            print("[recovery] Proposal was not applied.")
            return False
        for source, destination in zip(proposal_paths, failed.input_paths):
            shutil.copy2(source, destination)
        print(f"[recovery] Approved proposal {proposal_number:03d} applied; no job has been submitted yet.")
        return True

    def recovery_console(self, run_dir: str, error: str) -> Optional[Dict[str, Any]]:
        """Keep the terminal attached to a paused workflow until the user explicitly leaves it."""
        current_error = error
        while True:
            state = WorkflowCheckpoint.load(run_dir)
            failed = self._failed_step(state)
            if failed is None:
                return self.resume(run_dir) if state.status != "completed" else {"status": "success", "run_dir": run_dir}
            print(
                f"\n[recovery] Workflow {Path(run_dir).name} is paused at step {failed.id} "
                f"({failed.tool}).\n"
                "Enter a plain-language correction for an agent-generated proposal, or use:\n"
                "  edit | retry | fresh | status | download | new | stop"
            )
            try:
                instruction = input(f"Recovery step {failed.id}> ").strip()
            except (EOFError, OSError):
                return None
            command = instruction.lower()
            if command in {"stop", "exit", "quit", "q"}:
                return None
            if command == "new":
                return {"status": "new_request", "run_dir": run_dir}
            if command == "status":
                print(f"State: {failed.status}\nAttempts: {failed.attempts}\nLast error:\n{failed.last_error or current_error}")
                continue
            if command == "download":
                scope = self.download_callback("failed", run_dir)
                self._download_workflow_artifacts(state, scope)
                continue
            try:
                if command == "edit":
                    print("Edit these file(s):")
                    for path in failed.input_paths:
                        print(f"  - {path}")
                    input("Press Enter after saving your edits to validate and resume...")
                elif command == "fresh":
                    self.fresh_start_step(run_dir, failed.id)
                elif command == "retry":
                    pass
                elif instruction:
                    if not self._propose_recovery_edit(run_dir, instruction, failed.last_error or current_error):
                        continue
                else:
                    continue
                result = self.resume(run_dir)
                if result.get("status") == "awaiting_user":
                    current_error = WorkflowCheckpoint.load(run_dir).step(failed.id).last_error
                    continue
                return result
            except KeyboardInterrupt:
                print("\n[recovery] Attempt interrupted; checkpoint preserved.")
                current_error = "Attempt interrupted by user."
            except Exception as exc:
                current_error = str(exc)
                print(f"[recovery] Attempt paused again: {exc}")

    def recover_interactively(self, run_dir: str, error: str) -> Optional[Dict[str, Any]]:
        """Offer checkpoint-preserving repair/edit/retry choices after a runtime failure."""
        state = WorkflowCheckpoint.load(run_dir)
        failed = next(
            (step for step in state.steps if step.status in {"awaiting_user", "failed", "running", "submitted"}),
            None,
        )
        if failed is None:
            return None
        if failed.remote_dir:
            package = next(
                package for step, package in zip(state.plan, state.packages)
                if int(step["id"]) == failed.id
            )
            self.transport.fetch_files(
                failed.remote_dir,
                Path(package["work_dir"]),
                [
                    *(Path(path).name for path in package.get("output_paths", [])),
                    "CRASH", "qe.err", "qe.out",
                ],
            )
        diagnostic_excerpt = ""
        failed_work_dir = Path(state.run_dir) / "branches" / failed.branch
        for name in ("CRASH", "qe.err", "qe.out"):
            path = failed_work_dir / name
            if path.is_file() and path.stat().st_size:
                diagnostic_excerpt += f"\n--- {name} ---\n{path.read_text(encoding='utf-8', errors='replace')[-4000:]}"
        action = self.failure_callback(
            f"Step {failed.id} ({failed.tool})",
            (failed.last_error or error) + diagnostic_excerpt,
        )
        if action == "download":
            scope = self.download_callback("failed", state.run_dir)
            self._download_workflow_artifacts(state, scope)
            return None
        if action == "fresh":
            self.fresh_start_step(state.run_dir, failed.id)
            return self.resume(state.run_dir)
        if action == "edit":
            print("Edit these file(s):")
            for path in failed.input_paths:
                print(f"  - {path}")
            try:
                input("Press Enter after saving your edits to validate and resume...")
            except (EOFError, OSError):
                return None
            return self.resume(state.run_dir)
        if action == "retry":
            return self.resume(state.run_dir)
        return None

    def fresh_start_step(self, run_dir: str, step_id: int) -> None:
        """Schedule a clean immutable attempt without changing prior attempts or approved input."""
        state = WorkflowCheckpoint.load(run_dir)
        checkpoint = state.step(step_id)
        completed_parents = [
            state.step(parent_id) for parent_id in checkpoint.depends_on
            if state.step(parent_id).status == "completed" and state.step(parent_id).remote_dir
        ]
        parent = next(iter(completed_parents), None)
        if checkpoint.depends_on and parent is None:
            raise RuntimeError(
                f"Step {step_id} has no completed parent to seed a clean attempt."
            )
        checkpoint.job_ids = []
        checkpoint.last_error = (
            f"Clean attempt requested; it will be seeded from completed step {parent.id}."
            if parent else "Clean first-step attempt requested."
        )
        checkpoint.set_status("ready")
        state.invalidate_descendants(step_id)
        state.status = "approved"
        state.save()
        print(f"[recovery] step {step_id} is ready for a new immutable attempt.")

    def run(
        self,
        query: str,
        *,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
        reuse_run_dir: str = "",
    ) -> Dict[str, Any]:
        parent_state: Optional[WorkflowCheckpoint] = None
        parent_scf = None
        parent_context_data: Dict[str, Any] = {}
        if reuse_run_dir:
            parent_state = WorkflowCheckpoint.load(reuse_run_dir)
            problems = parent_state.verify_completed_inputs()
            if problems:
                raise RuntimeError("Cannot extend the selected workflow:\n" + "\n".join(problems))
            parent_scf = next(
                (step for step in reversed(parent_state.steps) if step.tool == "pw_scf" and step.status == "completed" and step.remote_dir),
                None,
            )
            if parent_scf is None:
                raise RuntimeError("The selected workflow has no completed SCF step with reusable remote state.")
            parent_context_path = Path(parent_state.run_dir) / "workflow_context.json"
            if not parent_context_path.is_file():
                raise RuntimeError("The selected parent has no immutable workflow_context.json.")
            parent_context_data = json.loads(parent_context_path.read_text(encoding="utf-8"))
        self.agent._prepare_run_directory(
            query=query,
            material_name=material_name,
            task_type=task_type,
            run_id=run_id,
            category=category,
        )
        scientific_assessment: Dict[str, Any] = {}
        if hasattr(self.agent, "analyze_workflow_intent"):
            print("[scientific-assessment] reviewing the complete request before planning...")
            scientific_assessment = self.agent.analyze_workflow_intent(query)
            (self.agent.work_dir / "scientific_assessment.json").write_text(
                json.dumps(scientific_assessment, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"[scientific-assessment] {scientific_assessment.get('summary', '')}")
        # Freeze the assessed baseline XC/relativity library for this workflow.
        if hasattr(self.agent, "select_pseudo_dir"):
            try:
                self.agent.select_pseudo_dir(
                    query, scientific_assessment, target_scope="baseline"
                )
            except TypeError:
                self.agent.select_pseudo_dir(query)
        if parent_state is not None and not re.search(
            r"\b(?:lda|pbe|pbesol|pbe\s*sol|local[ -]density approximation)\b",
            query,
            re.I,
        ):
            self.agent.pseudo_dir = parent_context_data.get("pseudo_library", self.agent.pseudo_dir)
        if parent_state is not None:
            (self.agent.work_dir / "extension_source.json").write_text(
                json.dumps({
                    "parent_run_dir": parent_state.run_dir,
                    "parent_query": parent_state.query,
                    "reused_step_ids": [
                        step.id for step in parent_state.steps
                        if step.status == "completed" and step.tool in {"pw_relax", "pw_vc_relax", "pw_scf"}
                    ],
                    "seed_scf_remote_dir": parent_scf.remote_dir,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            parent_relax = next(
                (step for step in reversed(parent_state.steps) if step.tool in {"pw_relax", "pw_vc_relax"} and step.status == "completed" and step.attempt_history),
                None,
            )
            if parent_relax:
                source = Path(parent_relax.attempt_history[-1].local_dir) / "relaxed_structure.in"
                if source.is_file():
                    shutil.copy2(source, self.agent.work_dir / "imported_relaxed_structure.in")

        material_info: Dict[str, Any] = {}
        if self.agent.need_query_info:
            material_info = self.agent.info_query(query)

        subproblems = []
        plan_feedback = ""
        last_proposed: List[Dict[str, Any]] = []
        blocking_plan_issues = []
        for plan_attempt in range(1, 4):
            planner_query = query
            if scientific_assessment:
                planner_query += (
                    "\n\nA scientific assessment was completed before planning. Treat it as the "
                    "auditable workflow strategy; preserve invariants while respecting stage/branch "
                    "scope:\n" + json.dumps(scientific_assessment, indent=2)
                )
            if parent_state is not None:
                planner_query += (
                    "\n\nThis is an extension of a verified completed workflow. The relaxed structure "
                    "and converged SCF state are reused artifacts. Plan only the newly requested "
                    "calculations; do not plan another relaxation or SCF. Preserve the parent "
                    f"scientific context exactly: {json.dumps(parent_context_data)}"
                )
            if plan_feedback:
                planner_query += (
                    "\n\nThe previous workflow plan failed deterministic validation. "
                    "Revise the plan while preserving the user's scientific request:\n"
                    + plan_feedback
                )
            proposed = self.agent.plan(query=planner_query)
            if not proposed:
                plan_feedback = "No valid subproblem blocks were returned."
                continue
            proposed = normalize_plan(query, proposed)
            if parent_state is not None:
                proposed = [
                    step for step in proposed
                    if step.get("tool") not in {"pw_relax", "pw_vc_relax", "pw_scf"}
                ]
                for new_id, step in enumerate(proposed, start=1):
                    step["id"] = new_id
            last_proposed = proposed
            plan_issues = validate_plan(query, proposed)
            blocking_plan_issues = [issue for issue in plan_issues if issue.blocking]
            if not blocking_plan_issues:
                subproblems = proposed
                break
            plan_feedback = "\n".join(issue.format() for issue in blocking_plan_issues)
            print(f"[plan] validation rejected attempt {plan_attempt}:\n{plan_feedback}")
        if not subproblems:
            if not self._uses_default_approval or not last_proposed:
                raise RuntimeError(f"Generated workflow plan failed validation after 3 attempts:\n{plan_feedback}")
            # Show the rejected draft and exact findings so the user can revise
            # it instead of being dropped back to the top-level prompt.
            subproblems = last_proposed

        assessment_banner = ""
        if scientific_assessment:
            assessment_banner = (
                "Scientific assessment (review before approval)\n"
                "----------------------------------------------\n"
                + json.dumps(scientific_assessment, indent=2, ensure_ascii=False)
                + "\n\n"
            )
        reused_banner = ""
        if parent_state is not None:
            reused_banner = (
                "Reused completed artifacts\n"
                "--------------------------\n"
                f"Parent workflow: {parent_state.run_dir}\n"
                + "\n".join(
                    f"REUSED: step {step.id} {step.tool} — attempt {step.attempts}"
                    for step in parent_state.steps
                    if step.status == "completed" and step.tool in {"pw_relax", "pw_vc_relax", "pw_scf"}
                )
                + "\n\n"
            )
        plan = assessment_banner + reused_banner + _plan_text(subproblems)
        print("\n" + plan)
        (self.agent.work_dir / "workflow_plan.txt").write_text(plan, encoding="utf-8")
        (self.agent.work_dir / "workflow_plan.json").write_text(
            json.dumps(subproblems, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        # The scientific strategy must be reviewable before expensive input
        # generation. Otherwise a generation error can bypass the user's only
        # opportunity to correct an over-expanded or scientifically wrong plan.
        if self._uses_default_approval:
            plan_decision = _approve_inputs_popup(
                plan,
                [],
                workflow_validator=lambda: [
                    issue.format() for issue in validate_plan(query, subproblems)
                    if issue.blocking
                ],
                review_stage="plan",
            )
            plan_action, plan_revision = _approval_result(plan_decision)
            if plan_action == "revise":
                (self.agent.work_dir / "plan_revision.json").write_text(
                    json.dumps({
                        "query": query,
                        "comment": plan_revision,
                        "superseded_run_dir": str(self.agent.work_dir),
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                revised_query = (
                    f"{query}\n\nUSER-REQUESTED SCIENTIFIC/PLAN REVISION (mandatory):\n"
                    f"{plan_revision}\nRegenerate the scientific assessment and minimal execution "
                    "plan before producing any input files."
                )
                return self.run(
                    revised_query,
                    run_id=run_id,
                    category=category,
                    task_type=task_type,
                    material_name=material_name,
                    reuse_run_dir=reuse_run_dir,
                )
            if plan_action != "approve":
                return {
                    "status": "cancelled_before_input_generation",
                    "run_dir": str(self.agent.work_dir),
                    "plan": subproblems,
                    "input_paths": [],
                }

        packages: List[Dict[str, Any]] = []
        input_paths: List[str] = []
        prior_parameter_memory: List[str] = []
        relaxation_seen = parent_state is not None and any(
            step.status == "completed" and step.tool in {"pw_relax", "pw_vc_relax"}
            for step in parent_state.steps
        )
        workflow_generation_context = (
            "Generate this input now as part of one pre-approved workflow. Use the exact shared "
            "Quantum ESPRESSO prefix 'tritondft_workflow' and outdir './'. Keep filenames consistent "
            "between dependent phonon/post-processing steps: the uniform q-grid ph.x uses "
            "fildyn='tritondft_workflow.dyn', q2r.x reads it and writes flfrc='tritondft_workflow.fc', "
            "and matdyn.x reads that flfrc. A separate Gamma Raman ph.x uses "
            "fildyn='tritondft_workflow.dynG', and its dynmat.x reads that exact dynG file. "
            "Choose this step's numerical and physical parameters now; do not wait for "
            "an earlier calculation to run. The complete workflow plan is:\n"
            f"Remote Quantum ESPRESSO version/capability hint: {getattr(self.agent, 'remote_qe_version', '') or 'unknown; the remote parser probe must confirm version-sensitive syntax'}.\n"
            + plan
        )
        if scientific_assessment:
            workflow_generation_context += (
                "\nThe pre-plan scientific assessment below is authoritative for stage-specific "
                "scope and global invariants:\n"
                + json.dumps(scientific_assessment, indent=2)
            )
        if parent_state is not None:
            workflow_generation_context += (
                "\nThis child workflow must consume the completed parent SCF state. Reuse and do "
                "not reinterpret this immutable parent context:\n"
                + json.dumps(parent_context_data, indent=2)
            )
        placeholder_memory = (
            "The workflow inputs are being generated before execution. For any pw.x step after "
            "vc-relax, choose all numerical and physical parameters now, independently of the "
            "relaxation result. Use the same material and pseudopotentials. The final relaxed "
            "CELL_PARAMETERS and ATOMIC_POSITIONS will be inserted automatically before execution."
        )
        for idx, step in enumerate(subproblems, start=1):
            print(f"[cluster-agent] generating input {idx}/{len(subproblems)}: {step.get('problem')}")
            package = self.agent.solve_sub_problem(
                step,
                problem_id=idx,
                query=query,
                total_memory=(
                    workflow_generation_context
                    + ("\nPreviously selected workflow parameter JSON (preserve shared settings):\n" + "\n".join(prior_parameter_memory) if prior_parameter_memory else "")
                    + ("\n" + placeholder_memory if relaxation_seen else "")
                ),
                material_info=material_info,
            )
            if package.get("status") != "cluster_input":
                raise RuntimeError(f"Expected cluster_input result, got: {package}")

            for path in package.get("input_paths", []):
                _set_workflow_prefix(path)
            if relaxation_seen and package.get("exec_name") == "pw.x":
                for path in package.get("input_paths", []):
                    if not _add_relaxed_structure_placeholder(path):
                        raise RuntimeError(
                            f"Could not add the relaxed-structure placeholder to {path}."
                        )

            packages.append(package)
            input_paths.extend(package.get("input_paths", []))
            if package.get("params_json"):
                prior_parameter_memory.append(
                    f"Step {idx} ({step.get('tool')}): {package['params_json']}"
                )
            if step.get("tool") == "pw_vc_relax":
                relaxation_seen = True

        # NSCF occupations and dos.x bz_sum form one deterministic integration
        # contract. Reconcile them before validation and before user approval.
        harmonize_dos_integration(subproblems, packages)

        validation_report_path = self.agent.work_dir / "validation_report.txt"
        generation_issues = validate_generated_workflow(query, subproblems, packages)
        validation_report_path.write_text(
            ("\n".join(issue.format() for issue in generation_issues) or "Validation passed.") + "\n",
            encoding="utf-8",
        )

        input_paths = _isolate_branch_packages(
            Path(self.agent.work_dir), subproblems, packages
        )
        plan = _plan_text(subproblems, packages)
        (self.agent.work_dir / "workflow_plan.txt").write_text(plan, encoding="utf-8")
        generation_issues = validate_generated_workflow(query, subproblems, packages)

        def workflow_validation_messages() -> List[str]:
            issues = validate_generated_workflow(query, subproblems, packages)
            validation_report_path.write_text(
                ("\n".join(issue.format() for issue in issues) or "Validation passed.") + "\n",
                encoding="utf-8",
            )
            return [issue.format() for issue in issues if issue.blocking]

        manifest = {
            "query": query,
            "plan": subproblems,
            "input_files": [str(Path(path).name) for path in input_paths],
            "parameter_proposals": [package.get("params_json", "") for package in packages],
            "validation": [issue.format() for issue in generation_issues],
            "status": "awaiting_approval",
        }
        manifest_path = self.agent.work_dir / "approval_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        raw_decision = (
            self.approval_callback(plan, input_paths, workflow_validation_messages)
            if self._uses_default_approval
            else self.approval_callback(plan, input_paths)
        )
        approval_action, revision_comment = _approval_result(raw_decision)
        if approval_action == "revise":
            manifest["status"] = "revision_requested"
            manifest["revision_comment"] = revision_comment
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            revisions_path = self.agent.work_dir / "revision_history.json"
            revisions_path.write_text(
                json.dumps([{
                    "query": query,
                    "comment": revision_comment,
                    "superseded_run_dir": str(self.agent.work_dir),
                }], indent=2) + "\n",
                encoding="utf-8",
            )
            revised_query = (
                f"{query}\n\nUSER-REQUESTED PLAN REVISION (mandatory):\n{revision_comment}\n"
                "Regenerate the complete workflow plan and all inputs. Explicitly incorporate this "
                "revision and do not submit the superseded draft."
            )
            print(f"[approval] Revision requested: {revision_comment}")
            return self.run(
                revised_query,
                run_id=run_id,
                category=category,
                task_type=task_type,
                material_name=material_name,
                reuse_run_dir=reuse_run_dir,
            )
        if approval_action != "approve":
            manifest["status"] = "cancelled"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return {
                "status": "cancelled",
                "run_dir": str(self.agent.work_dir),
                "plan": subproblems,
                "input_paths": input_paths,
            }

        # Approval allows expert edits, so validate the edited files and their
        # cross-step relationships again before any SSH connection is opened.
        edited_issues = validate_generated_workflow(query, subproblems, packages)
        blocking_edited_issues = [issue for issue in edited_issues if issue.blocking]
        if blocking_edited_issues:
            report = "\n".join(issue.format() for issue in blocking_edited_issues)
            raise RuntimeError(f"Approved inputs failed validation after editing:\n{report}")

        manifest["status"] = "approved"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        approved_dir = self.agent.work_dir / "approved_inputs"
        approved_dir.mkdir(parents=True, exist_ok=True)
        for path in input_paths:
            shutil.copy2(path, approved_dir / Path(path).name)
        packaged_work_dirs = set()
        for package in packages:
            work_dir = str(package["work_dir"])
            if work_dir in packaged_work_dirs:
                continue
            branch_inputs = [
                path for candidate in packages
                if str(candidate["work_dir"]) == work_dir
                for path in candidate.get("input_paths", [])
            ]
            branch_pseudo_dirs = {
                str(candidate.get("pseudo_dir") or self.agent.pseudo_dir)
                for candidate in packages
                if str(candidate["work_dir"]) == work_dir
            }
            if len(branch_pseudo_dirs) != 1:
                raise RuntimeError(
                    f"Branch {Path(work_dir).name} mixes incompatible pseudopotential libraries. "
                    "The planner must create separate scalar-relativistic and SOC branches."
                )
            package_pseudos_for_remote(
                branch_inputs,
                pseudo_dir=next(iter(branch_pseudo_dirs)),
                work_dir=work_dir,
            )
            packaged_work_dirs.add(work_dir)
        generated_context = create_workflow_context(
            query, self.agent.pseudo_dir, packages
        )
        child_context = generated_context
        parent_context = None
        if parent_state is not None:
            parent_context_path = Path(parent_state.run_dir) / "workflow_context.json"
            if not parent_context_path.is_file():
                raise RuntimeError("The parent workflow predates immutable context tracking and cannot be reused safely.")
            parent_context = json.loads(parent_context_path.read_text(encoding="utf-8"))
            child_context = WorkflowContext(
                query=query,
                pseudo_library=parent_context.get("pseudo_library", ""),
                pseudopotentials=parent_context.get("pseudopotentials", []),
                created_at=generated_context.created_at,
                parent_run_dir=parent_state.run_dir,
            )
        child_context.write_once(self.agent.work_dir)
        if parent_state is not None:
            parent_hashes = {
                item["name"]: item["sha256"] for item in parent_context.get("pseudopotentials", [])
            }
            child_hashes = {
                item["name"]: item["sha256"] for item in generated_context.pseudopotentials
            }
            parent_library = str(Path(parent_context.get("pseudo_library", "")).expanduser().resolve())
            selected_library = str(Path(self.agent.pseudo_dir).expanduser().resolve())
            if parent_library != selected_library:
                raise RuntimeError(
                    "Extension blocked: the requested XC/pseudopotential library differs from "
                    "the completed parent workflow."
                )
            if child_hashes and parent_hashes != child_hashes:
                raise RuntimeError(
                    "Extension blocked: the new inputs do not use the exact pseudopotentials "
                    "recorded by the completed parent workflow."
                )
        workflow_state = create_checkpoint(
            query, self.agent.work_dir, subproblems, packages
        )
        if parent_state is not None:
            workflow_state.parent_run_dir = parent_state.run_dir
            for checkpoint in workflow_state.steps:
                if not checkpoint.depends_on:
                    checkpoint.seed_remote_dir = parent_scf.remote_dir
                    checkpoint.reused_from_run = parent_state.run_dir
            workflow_state.save()
        if self._uses_default_approval:
            _launch_workflow_monitor(workflow_state.run_dir)
        # Fresh and resumed workflows deliberately share the same executor.
        # Each submission is recorded in a new immutable attempt directory.
        return self.resume(workflow_state.run_dir)

        # Legacy inline executor retained temporarily below for checkpoint-file
        # compatibility; it is unreachable and will be removed after migration.
        self.transport.ensure_connection()

        total_memory = ""
        results: List[Dict[str, Any]] = []
        conclusions: List[str] = []
        relaxed_structure = ""

        for idx, (step, package) in enumerate(zip(subproblems, packages), start=1):
            checkpoint = workflow_state.step(int(step["id"]))
            if checkpoint.status == "completed":
                print(f"[cluster-agent] step {idx} already completed; reusing checkpointed results.")
                continue
            if not workflow_state.dependencies_completed(checkpoint.id):
                checkpoint.set_status("blocked", "One or more dependencies are incomplete.")
                workflow_state.status = "awaiting_user"
                workflow_state.save()
                raise RuntimeError(
                    f"Step {checkpoint.id} is blocked by incomplete dependencies: "
                    f"{checkpoint.depends_on}"
                )
            print(f"\n[cluster-agent] step {idx}/{len(subproblems)}: {step.get('problem')}\n")
            for path in package.get("input_paths", []):
                if "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN" in Path(path).read_text(
                    encoding="utf-8"
                ):
                    if not relaxed_structure:
                        raise RuntimeError(
                            f"{Path(path).name} needs the relaxed structure, but no completed "
                            "relaxation output is available."
                        )
                    _insert_relaxed_structure(path, relaxed_structure)
                    if step.get("tool") == "pw_bands":
                        try:
                            labels = materialize_relaxed_band_path(path, relaxed_structure)
                        except Exception as exc:
                            print(
                                "[cluster-agent] relaxed-cell symmetry path generation failed; "
                                f"retaining the validated approved path ({exc})."
                            )
                        else:
                            (self.agent.work_dir / "band_path_labels.json").write_text(
                                json.dumps(labels, indent=2) + "\n", encoding="utf-8"
                            )
                    materialized_dir = self.agent.work_dir / "materialized_inputs"
                    materialized_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, materialized_dir / Path(path).name)
                    materialized_issues = validate_qe_input(
                        path,
                        tool=str(step.get("tool") or ""),
                        query=str(step.get("problem") or ""),
                    )
                    blocking_materialized = [issue for issue in materialized_issues if issue.blocking]
                    if blocking_materialized:
                        report = "\n".join(issue.format() for issue in blocking_materialized)
                        raise RuntimeError(
                            "A relaxation-dependent input failed validation after final structure "
                            f"materialization:\n{report}"
                        )

            local_run_dir = Path(package["work_dir"]).resolve()
            remote_dir = self.transport.remote_dir_for(local_run_dir)
            checkpoint.remote_dir = remote_dir
            for parent_id in checkpoint.depends_on:
                parent = workflow_state.step(parent_id)
                if parent.branch != checkpoint.branch and parent.remote_dir:
                    self.transport.clone_remote_directory(parent.remote_dir, remote_dir)
                    break
            checkpoint.attempts += 1
            checkpoint.set_status("running")
            workflow_state.status = "running"
            workflow_state.save()
            self.transport.archive_failure_markers(remote_dir, checkpoint.attempts)

            package["slurm_paths"] = self.agent.slurm_launcher.package(
                exec_name=package["exec_name"],
                qe_prefix=self.agent.remote_qe_bin_prefix,
                input_paths=package.get("input_paths", []),
                work_dir=str(local_run_dir),
                parallel_exec=self.agent.parallel_exec,
                parallel_np=self.agent.parallel_np,
                output_paths=package.get("output_paths", []),
                hardware_description=self.agent.hardware_description,
                probe_output_paths=None,
            )
            self.transport.upload_step_files(
                local_run_dir,
                remote_dir,
                [*package.get("input_paths", []), *package.get("slurm_paths", [])],
            )

            jobs: List[ClusterJob] = []
            for script_path in package.get("slurm_paths", []):
                script_name = Path(script_path).name
                job = self.transport.submit(remote_dir, script_name)
                print(f"[cluster-agent] submitted {script_name}: {job.submit_output}")
                jobs.append(job)
                checkpoint.job_ids.append(job.job_id)
                checkpoint.set_status("submitted")
                workflow_state.save()

            for job in jobs:
                self.transport.wait_for_job(job)

            self.transport.fetch_files(
                remote_dir,
                local_run_dir,
                [
                    *(Path(path).name for path in package.get("output_paths", [])),
                    "CRASH", "qe.err", "qe.out",
                ],
            )
            output_issues = []
            for output_path in package.get("output_paths", []):
                output_issues.extend(
                    validate_qe_output(output_path, exec_name=package.get("exec_name", ""))
                )
            blocking_output_issues = [issue for issue in output_issues if issue.blocking]
            if blocking_output_issues:
                report = "\n".join(issue.format() for issue in blocking_output_issues)
                raise RuntimeError(
                    f"Remote step {idx} failed deterministic output validation:\n{report}"
                )
            if step.get("tool") == "pw_vc_relax":
                output_paths = package.get("output_paths", [])
                if not output_paths:
                    raise RuntimeError("Relaxation package did not define an output path.")
                relaxed_structure = _extract_relaxed_structure(output_paths[0])
                (local_run_dir / "relaxed_structure.in").write_text(
                    relaxed_structure, encoding="utf-8"
                )
            parsed = self._parse_remote_step(query, step, package)
            results.append(parsed)

            judge = parsed.get("result_judge", "")
            prose = self.agent._judge_prose(judge)
            if prose:
                conclusions.append(f"Step {idx}: {prose}" if len(subproblems) > 1 else prose)
            total_memory += (
                f" Subproblem {idx}:\n System Results:\n {parsed.get('result_json', '')}\n"
                f" Conclusion of Subproblem {idx}: {judge}\n\n"
            )

            judge_json = parsed.get("judge_json") or {}
            if judge_json.get("status") != "done":
                checkpoint.set_status("awaiting_user", json.dumps(judge_json))
                workflow_state.status = "awaiting_user"
                workflow_state.save()
                raise RuntimeError(
                    f"Remote step {idx} did not pass result judging: {json.dumps(judge_json)}"
                )
            checkpoint.input_hashes = {
                path: file_sha256(path)
                for path in checkpoint.input_paths if Path(path).is_file()
            }
            checkpoint.set_status("completed")
            workflow_state.refresh_readiness()
            workflow_state.save()

        workflow_state.status = "completed"
        workflow_state.save()
        download_scope = self.download_callback("completed", workflow_state.run_dir)
        self._download_workflow_artifacts(workflow_state, download_scope)
        analysis = "\n\n".join(conclusions)
        plot_paths = generate_electronic_plots(self.agent.work_dir, query)
        plot_names = {Path(path).name for path in plot_paths}
        missing_plots = []
        if re.search(r"band(?: structure)? plot|plot.*band", query, re.I) and "band_structure.png" not in plot_names:
            missing_plots.append("band_structure.png")
        if re.search(r"total dos plot|plot.*total dos", query, re.I) and "total_dos.png" not in plot_names:
            missing_plots.append("total_dos.png")
        if re.search(r"projected dos plot|pdos plot|plot.*projected dos", query, re.I) and "projected_dos.png" not in plot_names:
            missing_plots.append("projected_dos.png")
        if missing_plots and download_scope in {"results", "all"}:
            raise RuntimeError(
                "The numerical workflow finished, but requested plot artifacts were not produced: "
                + ", ".join(missing_plots)
            )
        magnetic_moments = extract_magnetic_moments(self.agent.work_dir)
        self.agent._write_analysis(analysis, query)
        return {
            "status": "success",
            "run_dir": str(self.agent.work_dir),
            "steps": results,
            "analysis": analysis,
            "plot_paths": plot_paths,
            "magnetic_moments": magnetic_moments,
            "download_scope": download_scope,
        }

    def resume(self, run_dir: str) -> Dict[str, Any]:
        """Resume an approved workflow without regenerating completed steps."""
        workflow_state = WorkflowCheckpoint.load(run_dir)
        self.agent.work_dir = Path(workflow_state.run_dir)
        problems = workflow_state.verify_completed_inputs()
        if problems:
            raise RuntimeError(
                "Cannot safely resume because completed checkpoint artifacts changed:\n"
                + "\n".join(problems)
            )
        changed_steps: List[int] = []
        for checkpoint in workflow_state.steps:
            if checkpoint.status == "completed":
                continue
            for path, expected in list(checkpoint.input_hashes.items()):
                if Path(path).is_file() and file_sha256(path) != expected:
                    changed_steps.append(checkpoint.id)
                    checkpoint.input_hashes[path] = file_sha256(path)
                    break
        for step_id in changed_steps:
            workflow_state.invalidate_descendants(step_id)
        for checkpoint in workflow_state.steps:
            if checkpoint.status in {"failed", "awaiting_user", "running", "submitted"}:
                checkpoint.set_status("ready" if workflow_state.dependencies_completed(checkpoint.id) else "blocked")
        workflow_state.status = "running"
        workflow_state.refresh_readiness()
        workflow_state.save()

        self.transport.ensure_connection()
        remote_problems: List[str] = []
        for completed in workflow_state.steps:
            if completed.status == "completed" and completed.remote_dir:
                remote_problems.extend(self.transport.remote_files_ok(
                    completed.remote_dir,
                    [Path(path).name for path in completed.output_paths],
                ))
        if remote_problems:
            raise RuntimeError(
                "Cannot safely resume because completed remote artifacts changed:\n"
                + "\n".join(remote_problems)
            )
        if workflow_state.parent_run_dir:
            parent_state = WorkflowCheckpoint.load(workflow_state.parent_run_dir)
            parent_scf = next(
                (step for step in reversed(parent_state.steps) if step.tool == "pw_scf" and step.status == "completed"),
                None,
            )
            if parent_scf is None or not parent_scf.remote_dir:
                raise RuntimeError("The imported parent SCF checkpoint is no longer reusable.")
            parent_remote_problems = self.transport.remote_files_ok(
                parent_scf.remote_dir,
                [Path(path).name for path in parent_scf.output_paths],
            )
            if parent_remote_problems:
                raise RuntimeError(
                    "Cannot extend because the parent SCF remote artifacts are unavailable:\n"
                    + "\n".join(parent_remote_problems)
                )
            if hasattr(self.transport, "remote_qe_state_ok"):
                state_problems = self.transport.remote_qe_state_ok(parent_scf.remote_dir)
                if state_problems:
                    raise RuntimeError(
                        "Cannot extend because the parent QE save state is unavailable:\n"
                        + "\n".join(state_problems)
                    )

        query = workflow_state.query
        steps = workflow_state.plan
        packages = workflow_state.packages
        issues = validate_generated_workflow(query, steps, packages)
        blocking = [issue for issue in issues if issue.blocking]
        if blocking:
            raise RuntimeError(
                "Edited/resumed inputs failed validation:\n"
                + "\n".join(issue.format() for issue in blocking)
            )

        relaxed_structure = ""
        completed_relax = next(
            (
                checkpoint for checkpoint in workflow_state.steps
                if checkpoint.status == "completed"
                and checkpoint.tool in {"pw_relax", "pw_vc_relax"}
                and checkpoint.attempt_history
            ),
            None,
        )
        if completed_relax is not None:
            relaxed_path = Path(completed_relax.attempt_history[-1].local_dir) / "relaxed_structure.in"
            if relaxed_path.is_file():
                relaxed_structure = relaxed_path.read_text(encoding="utf-8")
        if not relaxed_structure:
            imported_relaxed = Path(workflow_state.run_dir) / "imported_relaxed_structure.in"
            if imported_relaxed.is_file():
                relaxed_structure = imported_relaxed.read_text(encoding="utf-8")

        results: List[Dict[str, Any]] = []
        conclusions: List[str] = []
        for index, (step, package) in enumerate(zip(steps, packages), start=1):
            checkpoint = workflow_state.step(int(step["id"]))
            if checkpoint.status == "completed":
                print(f"[resume] step {index}/{len(steps)} completed; reusing it.")
                continue
            if not workflow_state.dependencies_completed(checkpoint.id):
                checkpoint.set_status("blocked", "Waiting for an incomplete dependency.")
                workflow_state.save()
                continue
            parsed, new_relaxed = self._execute_resumed_step(
                query=query,
                index=index,
                total_steps=len(steps),
                step=step,
                package=package,
                checkpoint=checkpoint,
                workflow_state=workflow_state,
                relaxed_structure=relaxed_structure,
            )
            if new_relaxed:
                relaxed_structure = new_relaxed
            results.append(parsed)
            prose = self.agent._judge_prose(parsed.get("result_judge", ""))
            if prose:
                conclusions.append(f"Step {index}: {prose}")

        if any(step.status != "completed" for step in workflow_state.steps):
            workflow_state.status = "awaiting_user"
            workflow_state.save()
            return {
                "status": "awaiting_user",
                "run_dir": workflow_state.run_dir,
                "steps": results,
            }

        workflow_state.status = "completed"
        workflow_state.save()
        download_scope = self.download_callback("completed", workflow_state.run_dir)
        self._download_workflow_artifacts(workflow_state, download_scope)
        analysis = "\n\n".join(conclusions)
        plot_paths = generate_electronic_plots(self.agent.work_dir, query)
        magnetic_moments = extract_magnetic_moments(self.agent.work_dir)
        if analysis:
            self.agent._write_analysis(analysis, query)
        return {
            "status": "success",
            "run_dir": workflow_state.run_dir,
            "steps": results,
            "analysis": analysis,
            "plot_paths": plot_paths,
            "magnetic_moments": magnetic_moments,
            "download_scope": download_scope,
        }

    def _execute_resumed_step(
        self,
        *,
        query: str,
        index: int,
        total_steps: int,
        step: Dict[str, Any],
        package: Dict[str, Any],
        checkpoint,
        workflow_state: WorkflowCheckpoint,
        relaxed_structure: str,
    ) -> tuple[Dict[str, Any], str]:
        """Execute one immutable attempt and persist any failure for later resume."""
        attempt_number = checkpoint.attempts + 1
        task_name = _task_file_stem(step)
        attempt_dir = (
            Path(workflow_state.run_dir) / "attempts" /
            f"{checkpoint.id:02d}-{task_name}" / f"attempt_{attempt_number:03d}"
        )
        attempt_dir.mkdir(parents=True, exist_ok=False)
        attempt_package = dict(package)
        attempt_inputs: List[str] = []
        for original in package.get("input_paths", []):
            destination = attempt_dir / Path(original).name
            shutil.copy2(original, destination)
            attempt_inputs.append(str(destination))
        attempt_outputs = [str(attempt_dir / Path(path).name) for path in package.get("output_paths", [])]
        attempt_package.update(
            work_dir=str(attempt_dir),
            input_paths=attempt_inputs,
            output_paths=attempt_outputs,
            slurm_paths=[],
        )
        if (
            checkpoint.tool == "pw_phonon_gamma"
            and checkpoint.last_error.startswith("Clean attempt requested")
        ):
            for path in attempt_inputs:
                _force_ph_fresh_start(path)
        pseudo_source = Path(package["work_dir"]) / "pseudos"
        if pseudo_source.is_dir():
            shutil.copytree(pseudo_source, attempt_dir / "pseudos", dirs_exist_ok=True)
        if hasattr(self.transport, "remote_attempt_dir"):
            remote_dir = self.transport.remote_attempt_dir(
                Path(workflow_state.run_dir), checkpoint.id, task_name, attempt_number
            )
        else:
            remote_dir = self.transport.remote_dir_for(attempt_dir)
        attempt = AttemptCheckpoint(
            number=attempt_number,
            local_dir=str(attempt_dir),
            remote_dir=remote_dir,
            input_paths=attempt_inputs,
            output_paths=attempt_outputs,
        )
        checkpoint.attempt_history.append(attempt)
        checkpoint.attempts = attempt_number
        checkpoint.remote_dir = remote_dir
        checkpoint.output_paths = attempt_outputs
        workflow_state.save()
        try:
            print(f"\n[resume] step {index}/{total_steps}: {step.get('problem')}\n")
            for path in attempt_package.get("input_paths", []):
                text = Path(path).read_text(encoding="utf-8")
                if "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN" in text:
                    if not relaxed_structure:
                        raise RuntimeError("Relaxed structure is not available for this dependent step.")
                    _insert_relaxed_structure(path, relaxed_structure)
                    if step.get("tool") == "pw_bands":
                        try:
                            labels = materialize_relaxed_band_path(path, relaxed_structure)
                        except Exception as exc:
                            print(f"[resume] retaining the approved band path: {exc}")
                        else:
                            labels_path = Path(workflow_state.run_dir) / "band_path_labels.json"
                            labels_path.write_text(
                                json.dumps(labels, indent=2) + "\n", encoding="utf-8"
                            )

            local_run_dir = attempt_dir
            for parent_id in checkpoint.depends_on:
                parent = workflow_state.step(parent_id)
                if parent.remote_dir and hasattr(self.transport, "clone_remote_directory"):
                    self.transport.clone_remote_directory(parent.remote_dir, remote_dir)
                    break
            else:
                if checkpoint.seed_remote_dir and hasattr(self.transport, "clone_remote_directory"):
                    self.transport.clone_remote_directory(checkpoint.seed_remote_dir, remote_dir)
            checkpoint.job_ids = []
            checkpoint.set_status("running")
            attempt.status = "running"
            workflow_state.save()

            attempt_package["slurm_paths"] = self.agent.slurm_launcher.package(
                exec_name=attempt_package["exec_name"],
                qe_prefix=self.agent.remote_qe_bin_prefix,
                input_paths=attempt_package.get("input_paths", []),
                work_dir=str(local_run_dir),
                parallel_exec=self.agent.parallel_exec,
                parallel_np=self.agent.parallel_np,
                output_paths=attempt_package.get("output_paths", []),
                hardware_description=self.agent.hardware_description,
                probe_output_paths=None,
            )
            workflow_state.save()
            self.transport.upload_step_files(
                local_run_dir,
                remote_dir,
                [*attempt_package.get("input_paths", []), *attempt_package.get("slurm_paths", [])],
            )
            for script in attempt_package.get("slurm_paths", []):
                job = self.transport.submit(remote_dir, Path(script).name)
                checkpoint.job_ids.append(job.job_id)
                attempt.job_ids.append(job.job_id)
                checkpoint.set_status("submitted")
                attempt.status = "submitted"
                workflow_state.save()
                self.transport.wait_for_job(job)
            self.transport.fetch_files(
                remote_dir,
                local_run_dir,
                [
                    *(Path(path).name for path in attempt_package.get("output_paths", [])),
                    "CRASH", "qe.err", "qe.out",
                ],
            )

            output_issues = [
                issue
                for output in attempt_package.get("output_paths", [])
                for issue in validate_qe_output(output, exec_name=attempt_package.get("exec_name", ""))
                if issue.blocking
            ]
            if output_issues:
                raise RuntimeError("\n".join(issue.format() for issue in output_issues))

            new_relaxed = ""
            if step.get("tool") == "pw_vc_relax":
                new_relaxed = _extract_relaxed_structure(attempt_package["output_paths"][0])
                (local_run_dir / "relaxed_structure.in").write_text(new_relaxed, encoding="utf-8")
            parsed = self._parse_remote_step(query, step, attempt_package)
            judge_json = parsed.get("judge_json") or {}
            if judge_json.get("status") != "done":
                raise RuntimeError(f"Result validation failed: {json.dumps(judge_json)}")
            checkpoint.input_hashes = {
                path: file_sha256(path)
                for path in checkpoint.input_paths if Path(path).is_file()
            }
            checkpoint.set_status("completed")
            attempt.status = "completed"
            attempt.completed_at = checkpoint.completed_at
            workflow_state.refresh_readiness()
            workflow_state.save()
            return parsed, new_relaxed
        except Exception as exc:
            checkpoint.set_status("awaiting_user", str(exc))
            attempt.status = "failed"
            attempt.error = str(exc)
            workflow_state.status = "awaiting_user"
            workflow_state.save()
            raise

    def _parse_remote_step(
        self,
        query: str,
        subproblem: Dict[str, Any],
        package: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_paths = package.get("input_paths", [])
        subproblem_id = package.get("subproblem_id")
        work_dir = package.get("work_dir")
        input_list, output_list = get_qe_result(
            work_dir=work_dir,
            input_paths=input_paths,
            verbose=self.agent.verbose,
            subproblem_id=subproblem_id,
            output_paths=package.get("output_paths", []),
        )
        output_list = preprocess_output_list(output_list, verbose=self.agent.verbose)

        total_result_json = ""
        parse_requirement = get_parse_requirement(package.get("parse_requirement_key"))
        for input_file, output_file in zip(input_list, output_list):
            messages = get_prompt(
                prompt_type="result_parse",
                input_json=package.get("params_json", "{}"),
                input_file=input_file,
                output_text=output_file,
                fn=package.get("exec_name", ""),
                parse_requirement=parse_requirement,
            )
            result_out = self.agent.generator(
                messages[0]["content"],
                max_new_tokens=self.agent.max_new_tokens,
                return_full_text=False,
            )
            total_result_json += result_out[0]["generated_text"]

        judge_messages = get_prompt(
            prompt_type="result_judge",
            query=query,
            subproblem=subproblem["problem"],
            param_json=package.get("params_json", "{}"),
            result_json=total_result_json,
        )
        judge_out = self.agent.generator(
            judge_messages[0]["content"],
            max_new_tokens=self.agent.max_new_tokens,
            return_full_text=False,
        )
        judge_text = judge_out[0]["generated_text"]
        judge_json = extract_json_brutal(judge_text)

        if self.agent.output_log:
            output_to_log_file(
                self.agent.work_dir_root,
                self.agent.output_log_file,
                f"[Remote step parsed]\n{judge_text}\n",
            )

        return {
            "status": "success" if judge_json.get("status") == "done" else "notdone",
            "result_json": total_result_json,
            "result_judge": judge_text,
            "judge_json": judge_json,
            "package": package,
        }


def _prompt_remote_root(current: str = "") -> str:
    current = (current or "").strip()
    while True:
        if current:
            entered = input(f"Remote cluster parent directory [{current}]: ").strip()
            remote_root = entered or current
        else:
            remote_root = input(
                "Remote cluster parent directory, e.g. /scratch/$USER/qe_jobs: "
            ).strip()

        if not remote_root:
            print("Please enter a remote directory path where you have write permission.")
            continue
        if "/your_user" in remote_root or remote_root.endswith("/your_user"):
            print(
                "That path still contains the placeholder 'your_user'. "
                "Use your real cluster scratch/project path."
            )
            current = ""
            continue
        return remote_root.rstrip("/")


def _prompt_dft_code(default: str = "") -> str:
    default = (default or os.environ.get("CLUSTER_AGENT_DFT_CODE", "")).strip().lower()
    if default in {"qe", "quantum espresso", "quantum-espresso", "quantumespresso"}:
        default = "qe"
    elif default in {"vasp", "v"}:
        default = "vasp"
    else:
        default = "qe"

    while True:
        entered = input(
            "\nWhich code do you want to use for this simulation?\n"
            "  a) Quantum ESPRESSO\n"
            "  b) VASP\n"
            f"Select a or b [{ 'a' if default == 'qe' else 'b' }]: "
        ).strip().lower()
        choice = entered or ("a" if default == "qe" else "b")
        if choice in {"a", "qe", "quantum espresso", "quantum-espresso", "quantumespresso"}:
            return "qe"
        if choice in {"b", "v", "vasp"}:
            return "vasp"
        print("Please choose 'a' for Quantum ESPRESSO or 'b' for VASP.")


def _prompt_vasp_potcar_root(current: str = "") -> str:
    current = (current or os.environ.get("VASP_POTCAR_ROOT", "") or os.environ.get("VASP_PP_PATH", "")).strip()
    while True:
        if current:
            entered = input(f"Licensed VASP POTCAR root on local machine or cluster [{current}]: ").strip()
            root = entered or current
        else:
            root = input(
                "Licensed VASP POTCAR root on local machine or cluster, e.g. /home/$USER/VASP_PP: "
            ).strip()
        if not root:
            print("VASP requires a licensed POTCAR tree. Please enter its local or cluster path.")
            continue
        return root


def interactive_main() -> None:
    env_parser = argparse.ArgumentParser(add_help=False)
    env_parser.add_argument(
        "--env-file",
        default=os.environ.get("CLUSTER_AGENT_ENV_FILE", ".env.cluster"),
        help="Local env file with cluster-agent defaults and API keys",
    )
    env_args, remaining_args = env_parser.parse_known_args()
    _load_env_file(".env")
    _load_env_file(env_args.env_file, override=True)

    if "-h" not in remaining_args and "--help" not in remaining_args:
        print(WELCOME_BANNER, flush=True)
        if _env_missing_cluster_setup(env_args.env_file):
            _run_super_user_setup(env_args.env_file)
            _load_env_file(env_args.env_file, override=True)

        _ensure_env_defaults(env_args.env_file)
        _load_env_file(env_args.env_file, override=True)

        if _env_missing_api_keys(env_args.env_file):
            print(
                f"\nPlease modify {env_args.env_file} with your OPENAI_API_KEY "
                "and MP_API_KEY before running TritonDFT."
            )
            input("Press Enter after you have saved the file...")
            _load_env_file(env_args.env_file, override=True)

    parser = argparse.ArgumentParser(description="Interactive local-to-Slurm DFT cluster agent")
    parser.add_argument("--env-file", default=env_args.env_file, help="Local env file with cluster-agent defaults and API keys")
    parser.add_argument(
        "--ssh-target",
        default=os.environ.get("CLUSTER_AGENT_SSH_TARGET", ""),
        help="SSH alias or user@host for the cluster",
    )
    parser.add_argument(
        "--remote-root",
        default=os.environ.get("CLUSTER_AGENT_REMOTE_ROOT", ""),
        help="Remote parent directory for run folders",
    )
    parser.add_argument("--model", default=os.environ.get("CLUSTER_AGENT_MODEL", "gpt-4o"))
    parser.add_argument("--backend", default=os.environ.get("CLUSTER_AGENT_BACKEND", "openai"))
    parser.add_argument("--work-dir", default=os.environ.get("CLUSTER_AGENT_WORK_DIR", "tmp"))
    parser.add_argument(
        "--resume",
        default="",
        help="Resume an existing run directory containing workflow_state.json",
    )
    parser.add_argument(
        "--extend",
        default="",
        help="Create a child workflow that reuses a completed relaxation/SCF workflow",
    )
    parser.add_argument(
        "--fresh-start-step",
        type=int,
        default=0,
        help="With --resume, schedule a clean immutable attempt seeded from its completed parent",
    )
    parser.add_argument(
        "--parallel-np",
        type=int,
        default=int(os.environ.get("CLUSTER_AGENT_PARALLEL_NP", "0")),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--poll-seconds", type=int, default=int(os.environ.get("CLUSTER_AGENT_POLL_SECONDS", "30")))
    parser.add_argument(
        "--remote-qe-bin-dir",
        default=os.environ.get("CLUSTER_AGENT_REMOTE_QE_BIN_DIR", ""),
        help="Remote cluster directory containing QE executables; leave empty when module load exposes pw.x on PATH",
    )
    parser.add_argument(
        "--remote-qe-version",
        default=os.environ.get("CLUSTER_AGENT_QE_VERSION", ""),
        help="Remote QE version hint used for version-sensitive input such as Hubbard syntax",
    )
    parser.add_argument(
        "--qe-slurm-template",
        default=(
            os.environ.get("TRITONDFT_QE_SLURM_TEMPLATE")
            or os.environ.get("TRITONDFT_SLURM_TEMPLATE", "")
        ),
        help="QE-specific Slurm template file",
    )
    parser.add_argument(
        "--vasp-slurm-template",
        default=os.environ.get("TRITONDFT_VASP_SLURM_TEMPLATE", ""),
        help="VASP-specific Slurm template file",
    )
    parser.add_argument(
        "--dft-code",
        choices=["qe", "vasp"],
        default=os.environ.get("CLUSTER_AGENT_DFT_CODE", ""),
        help="DFT code to use: qe or vasp. If omitted, TritonDFT asks interactively.",
    )
    parser.add_argument(
        "--remote-vasp-command",
        default=os.environ.get("CLUSTER_AGENT_REMOTE_VASP_COMMAND", ""),
        help="Remote VASP command override; blank lets TritonDFT infer vasp_std/vasp_gam/vasp_ncl from the request",
    )
    parser.add_argument(
        "--vasp-potcar-root",
        default=os.environ.get("CLUSTER_AGENT_VASP_POTCAR_ROOT", ""),
        help="Licensed VASP POTCAR root on the local machine or cluster used to assemble POTCAR files",
    )
    parser.add_argument(
        "--vasp-functional",
        default=os.environ.get("CLUSTER_AGENT_VASP_FUNCTIONAL", ""),
        help="VASP POTCAR functional override, e.g. PBE, PBEsol, LDA, PW91; blank lets TritonDFT infer it",
    )
    parser.add_argument(
        "--no-query-info",
        action="store_true",
        default=os.environ.get("CLUSTER_AGENT_NO_QUERY_INFO", "").lower() in {"1", "true", "yes", "on"},
    )
    parser.add_argument("--no-master", action="store_true", help="Do not open SSH ControlMaster connection")
    args = parser.parse_args(remaining_args)

    if not args.ssh_target:
        args.ssh_target = input("SSH target alias or user@host: ").strip()
    if not args.ssh_target:
        raise ValueError("SSH target is required. Set CLUSTER_AGENT_SSH_TARGET or pass --ssh-target.")

    dft_code = args.dft_code or _prompt_dft_code()
    remote_root = _prompt_remote_root(args.remote_root)

    transport = SSHClusterTransport(
        ssh_target=args.ssh_target,
        remote_root=remote_root,
        poll_seconds=args.poll_seconds,
        keep_master=not args.no_master,
        verbose=True,
    )
    if dft_code == "vasp":
        potcar_root = _prompt_vasp_potcar_root(args.vasp_potcar_root)
        vasp_agent = VASPAgent(
            model=args.model,
            backend=args.backend,
            verbose=True,
            work_dir=args.work_dir,
            need_query_info=not args.no_query_info,
            output_log=True,
            output_log_file="remote_cluster_agent.log",
            potcar_root=potcar_root,
            vasp_command=args.remote_vasp_command,
            functional=args.vasp_functional,
            slurm_template_path=args.vasp_slurm_template,
        )
        agent = RemoteClusterVASPAgent(
            vasp_agent=vasp_agent,
            transport=transport,
            parallel_np=args.parallel_np or 1,
            vasp_command=args.remote_vasp_command,
            slurm_template_path=args.vasp_slurm_template,
        )
    else:
        if args.qe_slurm_template:
            os.environ["TRITONDFT_SLURM_TEMPLATE"] = args.qe_slurm_template
        dft_agent = DFTAgent(
            model=args.model,
            backend=args.backend,
            verbose=True,
            work_dir=args.work_dir,
            run_mode="cluster_package",
            need_query_info=not args.no_query_info,
            evaluation_mode=False,
            output_log=True,
            output_log_file="remote_cluster_agent.log",
            parallel_np=args.parallel_np,
            auto_confirm=True,
        )
        dft_agent.remote_qe_bin_prefix = args.remote_qe_bin_dir
        dft_agent.remote_qe_version = args.remote_qe_version
        agent = RemoteClusterDFTAgent(dft_agent, transport)

    print("you are all set to run TritonDFT")
    print(f"\nRemote cluster DFT agent is ready ({'VASP' if dft_code == 'vasp' else 'Quantum ESPRESSO'}). Type 'exit' or 'quit' to stop.\n")
    try:
        if args.resume:
            if dft_code != "qe":
                raise ValueError("Phase 1 --resume currently supports Quantum ESPRESSO workflows.")
            try:
                if args.fresh_start_step:
                    agent.fresh_start_step(args.resume, args.fresh_start_step)
                resumed_state = WorkflowCheckpoint.load(args.resume)
                if any(step.status == "awaiting_user" for step in resumed_state.steps):
                    result = agent.recovery_console(
                        args.resume,
                        next(
                            (step.last_error for step in resumed_state.steps if step.status == "awaiting_user"),
                            "Workflow is awaiting user guidance.",
                        ),
                    )
                else:
                    result = agent.resume(args.resume)
                print(json.dumps(result, indent=2))
            except Exception as exc:
                try:
                    state = WorkflowCheckpoint.load(args.resume)
                    state.mark_unfinished_failure(str(exc))
                except Exception:
                    pass
                print(f"[cluster-agent] resume paused: {exc}")
                try:
                    recovered = agent.recovery_console(args.resume, str(exc))
                    if recovered:
                        print(json.dumps(recovered, indent=2))
                except Exception as recovery_exc:
                    print(f"[cluster-agent] recovery paused: {recovery_exc}")
        if args.extend:
            if dft_code != "qe":
                raise ValueError("--extend currently supports Quantum ESPRESSO workflows.")
            extension_query = input("New calculation to append to the completed workflow: ").strip()
            if extension_query:
                result = agent.run(extension_query, reuse_run_dir=args.extend)
                print(json.dumps(result, indent=2))
        while True:
            query = input("DFT request> ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue
            try:
                reuse_run_dir = ""
                if query.lower() == "extend" or query.lower().startswith("extend "):
                    if dft_code != "qe":
                        print("Workflow extension currently supports Quantum ESPRESSO only.")
                        continue
                    reuse_run_dir = query[6:].strip()
                    if not reuse_run_dir:
                        reuse_run_dir = input("Completed workflow directory to extend: ").strip()
                    if not reuse_run_dir:
                        continue
                    query = input("New calculation to append: ").strip()
                    if not query:
                        continue
                transport.set_remote_root(_prompt_remote_root(transport.remote_root))
                result = (
                    agent.run(query, reuse_run_dir=reuse_run_dir)
                    if dft_code == "qe"
                    else agent.run(query)
                )
                print(json.dumps(result, indent=2))
            except KeyboardInterrupt:
                print("\n[cluster-agent] interrupted. You can type another request or exit.")
            except Exception as exc:
                saved_run_dir = ""
                try:
                    state_path = Path(agent.agent.work_dir) / "workflow_state.json"
                    if state_path.is_file():
                        state = WorkflowCheckpoint.load(agent.agent.work_dir)
                        state.mark_unfinished_failure(str(exc))
                        saved_run_dir = state.run_dir
                        print(
                            f"[cluster-agent] checkpoint saved. Resume with: "
                            f"bash scripts/run_cluster_agent.sh --env-file {args.env_file} "
                            f"--resume {state.run_dir}"
                        )
                except Exception:
                    pass
                print(f"[cluster-agent] failed: {exc}")
                if saved_run_dir and dft_code == "qe":
                    try:
                        recovered = agent.recovery_console(saved_run_dir, str(exc))
                        if recovered:
                            print(json.dumps(recovered, indent=2))
                    except Exception as recovery_exc:
                        print(f"[cluster-agent] recovery paused: {recovery_exc}")
    finally:
        close = input("Close persistent SSH connection? [y/N] ").strip().lower()
        if close == "y":
            transport.close_connection()


if __name__ == "__main__":
    interactive_main()
