from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


STEP_STATES = {
    "pending", "ready", "submitted", "running", "completed", "failed",
    "repairing", "awaiting_user", "blocked", "cancelled",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class AttemptCheckpoint:
    number: int
    status: str = "created"
    local_dir: str = ""
    remote_dir: str = ""
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)
    job_ids: List[str] = field(default_factory=list)
    error: str = ""
    created_at: str = field(default_factory=_now)
    completed_at: str = ""


@dataclass
class StepCheckpoint:
    id: int
    tool: str
    problem: str
    branch: str
    depends_on: List[int] = field(default_factory=list)
    status: str = "pending"
    job_ids: List[str] = field(default_factory=list)
    input_paths: List[str] = field(default_factory=list)
    output_paths: List[str] = field(default_factory=list)
    input_hashes: Dict[str, str] = field(default_factory=dict)
    remote_dir: str = ""
    attempts: int = 0
    last_error: str = ""
    started_at: str = ""
    completed_at: str = ""
    attempt_history: List[AttemptCheckpoint] = field(default_factory=list)
    seed_remote_dir: str = ""
    reused_from_run: str = ""

    def set_status(self, status: str, error: str = "") -> None:
        if status not in STEP_STATES:
            raise ValueError(f"Unsupported workflow step status: {status}")
        self.status = status
        if status == "running" and not self.started_at:
            self.started_at = _now()
        if status == "completed":
            self.completed_at = _now()
            self.last_error = ""
        elif error:
            self.last_error = error


@dataclass
class WorkflowCheckpoint:
    version: int
    query: str
    run_dir: str
    status: str
    plan: List[Dict[str, Any]]
    packages: List[Dict[str, Any]]
    steps: List[StepCheckpoint]
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    parent_run_dir: str = ""

    @property
    def path(self) -> Path:
        return Path(self.run_dir) / "workflow_state.json"

    def save(self) -> None:
        self.updated_at = _now()
        payload = json.dumps(asdict(self), indent=2) + "\n"
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(self.path)

    def step(self, step_id: int) -> StepCheckpoint:
        for checkpoint in self.steps:
            if checkpoint.id == step_id:
                return checkpoint
        raise KeyError(f"Unknown workflow step id {step_id}")

    def dependencies_completed(self, step_id: int) -> bool:
        checkpoint = self.step(step_id)
        return all(self.step(parent).status == "completed" for parent in checkpoint.depends_on)

    def refresh_readiness(self) -> None:
        for checkpoint in self.steps:
            if checkpoint.status in {"completed", "running", "submitted", "failed", "awaiting_user", "cancelled"}:
                continue
            checkpoint.status = "ready" if self.dependencies_completed(checkpoint.id) else "blocked"

    def invalidate_descendants(self, changed_step_id: int) -> List[int]:
        invalidated: List[int] = []
        frontier = {changed_step_id}
        while frontier:
            parents = set(frontier)
            frontier = set()
            for checkpoint in self.steps:
                if checkpoint.id == changed_step_id or checkpoint.id in invalidated:
                    continue
                if parents.intersection(checkpoint.depends_on):
                    checkpoint.status = "pending"
                    checkpoint.job_ids = []
                    checkpoint.last_error = "Upstream input/result changed."
                    checkpoint.completed_at = ""
                    invalidated.append(checkpoint.id)
                    frontier.add(checkpoint.id)
        self.refresh_readiness()
        self.save()
        return invalidated

    def verify_completed_inputs(self, *, require_local_outputs: bool = False) -> List[str]:
        problems: List[str] = []
        for checkpoint in self.steps:
            if checkpoint.status != "completed":
                continue
            for path, expected in checkpoint.input_hashes.items():
                if not Path(path).is_file():
                    problems.append(f"Completed step {checkpoint.id} input is missing: {path}")
                elif file_sha256(path) != expected:
                    problems.append(f"Completed step {checkpoint.id} input changed: {path}")
            if require_local_outputs:
                for path in checkpoint.output_paths:
                    if not Path(path).is_file() or Path(path).stat().st_size == 0:
                        problems.append(f"Completed step {checkpoint.id} output is missing/empty: {path}")
        return problems

    def mark_unfinished_failure(self, error: str) -> None:
        for checkpoint in self.steps:
            if checkpoint.status in {"running", "submitted", "ready"}:
                checkpoint.set_status("awaiting_user", error)
                break
        self.status = "awaiting_user"
        self.save()

    @classmethod
    def load(cls, run_dir: str | Path) -> "WorkflowCheckpoint":
        path = Path(run_dir).expanduser().resolve() / "workflow_state.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded_steps = []
        for step in data["steps"]:
            step["attempt_history"] = [
                AttemptCheckpoint(**attempt)
                for attempt in step.get("attempt_history", [])
            ]
            loaded_steps.append(StepCheckpoint(**step))
        data["steps"] = loaded_steps
        return cls(**data)


def infer_dependencies(steps: Sequence[Dict[str, Any]]) -> List[List[int]]:
    """Infer artifact dependencies from the normalized subproblem sequence."""
    dependencies: List[List[int]] = []
    last_relax: Optional[int] = None
    last_scf: Optional[int] = None
    last_bands: Optional[int] = None
    last_nscf: Optional[int] = None
    last_grid_ph: Optional[int] = None
    last_gamma_ph: Optional[int] = None
    last_q2r: Optional[int] = None
    for step in steps:
        step_id = int(step["id"])
        tool = str(step.get("tool") or "")
        problem = str(step.get("problem") or "").lower()
        parents: List[int] = []
        if tool in {"pw_relax", "pw_vc_relax"}:
            last_relax = step_id
        elif tool == "pw_scf":
            if last_relax is not None:
                parents = [last_relax]
            last_scf = step_id
        elif tool == "pw_bands":
            if last_scf is not None:
                parents = [last_scf]
            last_bands = step_id
        elif tool == "bands_post":
            if last_bands is not None:
                parents = [last_bands]
        elif tool == "pw_nscf":
            if last_scf is not None:
                parents = [last_scf]
            last_nscf = step_id
        elif tool in {"dos_post", "projwfc_post"}:
            if last_nscf is not None:
                parents = [last_nscf]
        elif tool == "pw_phonon_gamma":
            if last_scf is not None:
                parents = [last_scf]
            if "dispersion" in problem or "q-grid" in problem or "q grid" in problem or "q-point mesh" in problem:
                last_grid_ph = step_id
            else:
                last_gamma_ph = step_id
        elif tool == "q2r_post":
            if last_grid_ph is not None:
                parents = [last_grid_ph]
            last_q2r = step_id
        elif tool == "matdyn_post":
            if last_q2r is not None:
                parents = [last_q2r]
        elif tool == "dynmat_post":
            if last_gamma_ph is not None:
                parents = [last_gamma_ph]
        dependencies.append(parents)
    return dependencies


def infer_branches(steps: Sequence[Dict[str, Any]]) -> List[str]:
    branches: List[str] = []
    for step in steps:
        tool = str(step.get("tool") or "")
        problem = str(step.get("problem") or "").lower()
        is_soc_refinement = bool(
            ("soc" in problem or "spin-orbit" in problem or "spin orbit" in problem)
            and ("without soc" not in problem and "scalar" not in problem)
        )
        if is_soc_refinement and tool in {"pw_scf", "pw_bands", "bands_post", "pw_nscf", "dos_post", "projwfc_post"}:
            branch = "soc_refinement"
        elif tool in {"pw_relax", "pw_vc_relax", "pw_scf"}:
            branch = "core"
        elif tool in {"pw_bands", "bands_post"}:
            branch = "bands"
        elif tool in {"pw_nscf", "dos_post", "projwfc_post"}:
            branch = "dos_pdos"
        elif tool in {"q2r_post", "matdyn_post"}:
            branch = "phonon_dispersion"
        elif tool == "dynmat_post":
            branch = "gamma_raman"
        elif tool == "pw_phonon_gamma":
            is_grid = "dispersion" in problem or "q-grid" in problem or "q grid" in problem or "q-point mesh" in problem
            branch = "phonon_dispersion" if is_grid else "gamma_raman"
        else:
            branch = f"step_{step.get('id')}"
        branches.append(branch)
    return branches


def create_checkpoint(
    query: str,
    run_dir: str | Path,
    plan: List[Dict[str, Any]],
    packages: List[Dict[str, Any]],
) -> WorkflowCheckpoint:
    dependencies = infer_dependencies(plan)
    branches = infer_branches(plan)
    checkpoints: List[StepCheckpoint] = []
    for step, package, parents, branch in zip(plan, packages, dependencies, branches):
        inputs = list(package.get("input_paths", []))
        checkpoints.append(StepCheckpoint(
            id=int(step["id"]),
            tool=str(step.get("tool") or ""),
            problem=str(step.get("problem") or ""),
            branch=branch,
            depends_on=parents,
            input_paths=inputs,
            output_paths=list(package.get("output_paths", [])),
            input_hashes={path: file_sha256(path) for path in inputs if Path(path).is_file()},
        ))
    state = WorkflowCheckpoint(
        version=1,
        query=query,
        run_dir=str(Path(run_dir).resolve()),
        status="approved",
        plan=plan,
        packages=packages,
        steps=checkpoints,
    )
    state.refresh_readiness()
    state.save()
    return state
