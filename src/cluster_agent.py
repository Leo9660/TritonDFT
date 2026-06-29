import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from DFTAgent import DFTAgent
from prompt import get_prompt
from prompt.tool_requirements import get_parse_requirement
from tool import get_spec
from utils import (
    extract_json_brutal,
    get_qe_result,
    output_to_log_file,
    package_pseudos_for_remote,
    preprocess_output_list,
)


DEFAULT_USER_SLURM_TEMPLATE = "~/.tritondft/example_slurm_job_file.txt"


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
        "CLUSTER_AGENT_REMOTE_QE_BIN_DIR",
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
    return not data.get("CLUSTER_AGENT_SSH_TARGET") or not data.get("CLUSTER_AGENT_REMOTE_ROOT")


def _env_missing_api_keys(path: str) -> bool:
    data = _read_env_file(path)
    return not data.get("OPENAI_API_KEY") or not data.get("MP_API_KEY")


def _ensure_env_defaults(path: str) -> None:
    data = _read_env_file(path)
    defaults: Dict[str, str] = {}
    template_value = data.get("TRITONDFT_SLURM_TEMPLATE", "").strip()
    if not template_value or template_value == "example_slurm_job_file.txt":
        defaults["TRITONDFT_SLURM_TEMPLATE"] = DEFAULT_USER_SLURM_TEMPLATE
    if defaults:
        _write_env_file(path, defaults)
    _ensure_user_slurm_template(_read_env_file(path).get("TRITONDFT_SLURM_TEMPLATE", ""))


def _ensure_user_slurm_template(template_path: str) -> None:
    """Create a per-user Slurm example template when the default path is used."""
    if not template_path:
        return
    destination = Path(template_path).expanduser()
    if destination.exists():
        return
    if destination != Path(DEFAULT_USER_SLURM_TEMPLATE).expanduser():
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
    print(f"[cluster-agent] Created user Slurm template: {destination}")


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


def _default_alias_for(hostname: str) -> str:
    parts = [p for p in hostname.split(".") if p]
    if len(parts) > 1 and parts[0].lower() in {"login", "logon", "ssh"}:
        return re.sub(r"[^A-Za-z0-9_-]+", "-", parts[1]).strip("-") or "cluster"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", parts[0]).strip("-") if parts else "cluster"


def _ensure_ssh_config_host(username: str, hostname: str) -> str:
    existing = _find_matching_ssh_host(username, hostname)
    if existing:
        print(f"Found existing SSH config entry: Host {existing}")
        return existing

    alias = _default_alias_for(hostname)
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


def _run_super_user_setup(env_file: str) -> None:
    print("To begin, you need to provide below information\n")

    print("Please provide your cluster address:")
    hostname = input("> ").strip()
    print("and it will collect the address\n")

    print("Please provide your user id:")
    username = input("> ").strip()
    print("and it will collect the id\n")

    print("password:")
    print("  TritonDFT does not store your password. SSH will ask for your password/OTP when the connection opens.")

    print("\nplease provide the working directory path in your cluster:")
    remote_root = input("> ").strip().rstrip("/")
    print("and it will collect the path\n")

    if not username or not hostname or not remote_root:
        raise ValueError("User name, cluster address, and cluster working directory are required.")

    ssh_target = _ensure_ssh_config_host(username=username, hostname=hostname)
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
                DEFAULT_USER_SLURM_TEMPLATE,
            ),
            "CLUSTER_AGENT_REMOTE_QE_BIN_DIR": os.environ.get("CLUSTER_AGENT_REMOTE_QE_BIN_DIR", ""),
            "CLUSTER_AGENT_NO_QUERY_INFO": os.environ.get("CLUSTER_AGENT_NO_QUERY_INFO", "true"),
        },
    )
    print(f"\nSetup complete. Cluster settings were written to {env_file}.")


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
        return f"{self.remote_root}/{local_run_dir.name}"

    def upload_directory(self, local_dir: Path, remote_dir: str) -> None:
        remote_q = shlex.quote(remote_dir)
        _run_interactive(
            f"ssh {shlex.quote(self.ssh_target)} 'mkdir -p {remote_q}'",
            verbose=self.verbose,
        )
        _run_interactive(
            "rsync -az --progress -e ssh "
            f"{shlex.quote(str(local_dir) + '/')} "
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

    @staticmethod
    def _extract_job_id(sbatch_output: str) -> str:
        match = re.search(r"\b(\d+)\b", sbatch_output or "")
        return match.group(1) if match else ""


RELAXED_STRUCTURE_PLACEHOLDER = """! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN
! Relaxed CELL_PARAMETERS and ATOMIC_POSITIONS from the vc-relax step will be inserted here before execution.
! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_END"""


def _plan_text(subproblems: List[Dict[str, Any]]) -> str:
    lines = ["TritonDFT execution plan", "========================"]
    for idx, step in enumerate(subproblems, start=1):
        tool_name = step.get("tool") or "unknown"
        why = (step.get("why") or "").strip()
        if not why:
            try:
                why = get_spec(tool_name).description
            except Exception:
                why = "This step contributes to the requested workflow result."
        lines.extend(
            [
                "",
                f"{idx}. {step.get('problem') or tool_name}",
                f"   Tool: {tool_name}",
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

    insert_at = min(start for start, _ in spans)
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
    marker = RELAXED_STRUCTURE_PLACEHOLDER + "\n"
    text = text[:insert_at] + marker + text[insert_at:]
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
    text = Path(path).read_text(encoding="utf-8")
    errors: List[str] = []
    if re.search(r"(?mi)^\s*&system\b", text):
        required_patterns = {
            "&control namelist": r"(?mi)^\s*&control\b",
            "&electrons namelist": r"(?mi)^\s*&electrons\b",
            "ATOMIC_SPECIES card": r"(?mi)^\s*ATOMIC_SPECIES\b",
            "K_POINTS card": r"(?mi)^\s*K_POINTS\b",
        }
        for label, pattern in required_patterns.items():
            if not re.search(pattern, text):
                errors.append(f"missing {label}")
        has_positions = re.search(r"(?mi)^\s*ATOMIC_POSITIONS\b", text)
        has_placeholder = "TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN" in text
        if not has_positions and not has_placeholder:
            errors.append("missing ATOMIC_POSITIONS or relaxed-structure placeholder")
        control = re.search(r"(?mis)^\s*&control\b(.*?)^\s*/\s*$", text)
        if control and not re.search(r"(?mi)^\s*calculation\s*=", control.group(1)):
            errors.append("&control is missing calculation")
    elif re.search(r"(?mi)^\s*&inputph\b", text):
        for key in ("prefix", "outdir", "fildyn", "tr2_ph"):
            if not re.search(rf"(?mi)^\s*{re.escape(key)}\s*=", text):
                errors.append(f"&inputph is missing {key}")
    return errors


def _approve_inputs_popup(plan: str, input_paths: List[str]) -> bool:
    print("\n[approval] All inputs are ready.")
    print("[approval] Opening the TritonDFT approval window; execution is paused until you choose Approve & Run or Cancel.")
    print("[approval] If the window is behind your IDE, use Cmd-Tab to select Python/TritonDFT.")
    try:
        import tkinter as tk
        from tkinter import messagebox, ttk

        root = tk.Tk()
        root.title("TritonDFT plan and input approval")
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

        decision = {"approved": False}

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
            report = (
                "\n\n".join(messages)
                if messages
                else "Basic structural validation passed for every generated input."
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
            decision["approved"] = True
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

        controls = ttk.Frame(root)
        controls.pack(fill="x", padx=10, pady=(0, 10))
        ttk.Label(
            controls,
            text="Review or edit any tab. Approve & Run saves the edits and unlocks cluster submission.",
        ).pack(side="left")
        ttk.Button(controls, text="Cancel", command=cancel).pack(side="right", padx=(8, 0))
        ttk.Button(controls, text="Approve & Run", command=approve).pack(side="right")
        ttk.Button(controls, text="Validate", command=refresh_validation).pack(side="right", padx=(0, 8))
        root.protocol("WM_DELETE_WINDOW", cancel)
        refresh_validation()
        root.mainloop()
        return decision["approved"]
    except Exception as exc:
        print(f"[approval] GUI unavailable ({exc}). Falling back to terminal approval.")
        print("\n" + plan)
        print("Generated input files:")
        for path in input_paths:
            print(f"  - {path}")
        while True:
            answer = input(
                "Type 'approve' to run, 'edit' after editing the files above, or 'cancel': "
            ).strip().lower()
            if answer in {"approve", "yes", "y"}:
                return True
            if answer in {"cancel", "no", "n"}:
                return False
            if answer == "edit":
                print("Edit the files in your editor, then return here to approve or cancel.")


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
    ):
        self.agent = dft_agent
        self.transport = transport
        self.approval_callback = approval_callback or _approve_inputs_popup
        self.agent.run_mode = "cluster_input"

    def run(
        self,
        query: str,
        *,
        run_id: int = 0,
        category: str = "unknown",
        task_type: str = "",
        material_name: str = "",
    ) -> Dict[str, Any]:
        self.agent._prepare_run_directory(
            query=query,
            material_name=material_name,
            task_type=task_type,
            run_id=run_id,
            category=category,
        )

        material_info: Dict[str, Any] = {}
        if self.agent.need_query_info:
            material_info = self.agent.info_query(query)

        subproblems = self.agent.plan(query=query)
        if not subproblems:
            raise RuntimeError("No valid plan was generated.")

        plan = _plan_text(subproblems)
        print("\n" + plan)
        (self.agent.work_dir / "workflow_plan.txt").write_text(plan, encoding="utf-8")
        (self.agent.work_dir / "workflow_plan.json").write_text(
            json.dumps(subproblems, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        packages: List[Dict[str, Any]] = []
        input_paths: List[str] = []
        relaxation_seen = False
        workflow_generation_context = (
            "Generate this input now as part of one pre-approved workflow. Use the exact shared "
            "Quantum ESPRESSO prefix 'tritondft_workflow' and outdir './'. Keep filenames consistent "
            "between dependent phonon/post-processing steps: ph.x fildyn='tritondft_workflow.dyn', "
            "q2r.x reads that fildyn and writes flfrc='tritondft_workflow.fc', and matdyn.x reads "
            "that flfrc. Choose this step's numerical and physical parameters now; do not wait for "
            "an earlier calculation to run. The complete workflow plan is:\n"
            + plan
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
            if step.get("tool") == "pw_vc_relax":
                relaxation_seen = True

        manifest = {
            "query": query,
            "plan": subproblems,
            "input_files": [str(Path(path).name) for path in input_paths],
            "status": "awaiting_approval",
        }
        manifest_path = self.agent.work_dir / "approval_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        if not self.approval_callback(plan, input_paths):
            manifest["status"] = "cancelled"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return {
                "status": "cancelled",
                "run_dir": str(self.agent.work_dir),
                "plan": subproblems,
                "input_paths": input_paths,
            }

        manifest["status"] = "approved"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        approved_dir = self.agent.work_dir / "approved_inputs"
        approved_dir.mkdir(parents=True, exist_ok=True)
        for path in input_paths:
            shutil.copy2(path, approved_dir / Path(path).name)
        package_pseudos_for_remote(
            input_paths,
            pseudo_dir=self.agent.pseudo_dir,
            work_dir=str(self.agent.work_dir),
        )
        self.transport.ensure_connection()

        total_memory = ""
        results: List[Dict[str, Any]] = []
        conclusions: List[str] = []
        relaxed_structure = ""

        for idx, (step, package) in enumerate(zip(subproblems, packages), start=1):
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

            local_run_dir = Path(package["work_dir"]).resolve()
            remote_dir = self.transport.remote_dir_for(local_run_dir)

            probe_output_paths: List[str] = []
            if package.get("exec_name") in {"pw.x", "ph.x"}:
                print("[cluster-agent] running a short remote probe before choosing Slurm parallelism.")
                _, probe_output_paths, probe_script_paths = self.agent.slurm_launcher.package_probe(
                    exec_name=package["exec_name"],
                    qe_prefix=self.agent.remote_qe_bin_prefix,
                    input_paths=package.get("input_paths", []),
                    work_dir=str(local_run_dir),
                    parallel_np=self.agent.parallel_np,
                )
                self.transport.upload_directory(local_run_dir, remote_dir)
                probe_jobs: List[ClusterJob] = []
                for script_path in probe_script_paths:
                    probe_job = self.transport.submit(remote_dir, Path(script_path).name)
                    print(
                        f"[cluster-agent] submitted probe {Path(script_path).name}: "
                        f"{probe_job.submit_output}"
                    )
                    probe_jobs.append(probe_job)
                for probe_job in probe_jobs:
                    self.transport.wait_for_job(probe_job)
                self.transport.fetch_directory(remote_dir, local_run_dir)

            package["slurm_paths"] = self.agent.slurm_launcher.package(
                exec_name=package["exec_name"],
                qe_prefix=self.agent.remote_qe_bin_prefix,
                input_paths=package.get("input_paths", []),
                work_dir=str(local_run_dir),
                parallel_exec=self.agent.parallel_exec,
                parallel_np=self.agent.parallel_np,
                output_paths=package.get("output_paths", []),
                hardware_description=self.agent.hardware_description,
                probe_output_paths=probe_output_paths,
            )
            self.transport.upload_directory(local_run_dir, remote_dir)

            jobs: List[ClusterJob] = []
            for script_path in package.get("slurm_paths", []):
                script_name = Path(script_path).name
                job = self.transport.submit(remote_dir, script_name)
                print(f"[cluster-agent] submitted {script_name}: {job.submit_output}")
                jobs.append(job)

            for job in jobs:
                self.transport.wait_for_job(job)

            self.transport.fetch_directory(remote_dir, local_run_dir)
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
                raise RuntimeError(
                    f"Remote step {idx} did not pass result judging: {json.dumps(judge_json)}"
                )

        analysis = "\n\n".join(conclusions)
        self.agent._write_analysis(analysis, query)
        return {
            "status": "success",
            "run_dir": str(self.agent.work_dir),
            "steps": results,
            "analysis": analysis,
        }

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

    remote_root = _prompt_remote_root(args.remote_root)

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
    transport = SSHClusterTransport(
        ssh_target=args.ssh_target,
        remote_root=remote_root,
        poll_seconds=args.poll_seconds,
        keep_master=not args.no_master,
        verbose=True,
    )
    agent = RemoteClusterDFTAgent(dft_agent, transport)

    print("you are all set to run TritonDFT")
    print("\nRemote cluster DFT agent is ready. Type 'exit' or 'quit' to stop.\n")
    try:
        while True:
            query = input("DFT request> ").strip()
            if query.lower() in {"exit", "quit", "q"}:
                break
            if not query:
                continue
            try:
                transport.set_remote_root(_prompt_remote_root(transport.remote_root))
                result = agent.run(query)
                print(json.dumps(result, indent=2))
            except KeyboardInterrupt:
                print("\n[cluster-agent] interrupted. You can type another request or exit.")
            except Exception as exc:
                print(f"[cluster-agent] failed: {exc}")
    finally:
        close = input("Close persistent SSH connection? [y/N] ").strip().lower()
        if close == "y":
            transport.close_connection()


if __name__ == "__main__":
    interactive_main()
