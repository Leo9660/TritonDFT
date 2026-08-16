from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkflowContext:
    query: str
    pseudo_library: str
    pseudopotentials: List[Dict[str, str]]
    code: str = "quantum_espresso"
    created_at: str = ""
    parent_run_dir: str = ""

    def write_once(self, run_dir: str | Path) -> Path:
        path = Path(run_dir) / "workflow_context.json"
        payload = json.dumps(asdict(self), indent=2) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise RuntimeError("The immutable workflow context already exists and differs.")
        path.write_text(payload, encoding="utf-8")
        return path


def create_workflow_context(
    query: str,
    pseudo_library: str,
    packages: Sequence[Dict[str, Any]],
) -> WorkflowContext:
    files: Dict[str, Dict[str, str]] = {}
    for package in packages:
        pseudo_dir = Path(package["work_dir"]) / "pseudos"
        if not pseudo_dir.is_dir():
            continue
        for path in sorted(pseudo_dir.iterdir()):
            if path.is_file():
                files[path.name] = {"name": path.name, "sha256": _sha256(path)}
    return WorkflowContext(
        query=query,
        pseudo_library=str(Path(pseudo_library).expanduser().resolve()),
        pseudopotentials=list(files.values()),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
