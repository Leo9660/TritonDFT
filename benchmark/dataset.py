from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _format_parameter_block(parameters: Dict[str, Any]) -> str:
    """Return a human readable block describing calculation parameters."""
    if not parameters:
        return ""
    lines = ["Additional parameters:"]
    for key, value in parameters.items():
        pretty_key = key.replace("_", " ").capitalize()
        lines.append(f"- {pretty_key}: {value}")
    return "\n".join(lines)


@dataclass
class DataItem:
    """Structured benchmark record returned to downstream evaluation."""

    id: str
    task: str
    prompt: str
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "id": self.id,
            "task": self.task,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


def _render_material_summary(info: Dict[str, Any]) -> str:
    name = info.get("name") or info.get("formula") or "the material"
    formula = info.get("formula")
    space_group = info.get("space_group")
    structure = info.get("structure")
    cell_type = info.get("cell_type", "primitive cell")
    lattice = info.get("lattice_constants_angstrom", {})

    summary_parts = [f"material = {name}"]
    if formula and formula not in name:
        summary_parts[-1] += f" ({formula})"
    if space_group:
        summary_parts.append(f"space group {space_group}")
    if structure:
        summary_parts.append(f"structure = {structure}")
    summary = summary_parts[0]
    if len(summary_parts) > 1:
        summary += " with " + " and ".join(summary_parts[1:])

    summary += f" using the {cell_type}"
    if lattice:
        lat_parts = [f"{axis} = {value} Å" for axis, value in lattice.items()]
        summary += f" (lattice constants: {', '.join(lat_parts)})"
    return summary + "."


def _material_query_context(info: Dict[str, Any]) -> str:
    """Build the exact material sentence used in templated prompts."""
    name = info.get("name") or info.get("formula") or "the material"
    space_group = info.get("space_group", "unknown space group")
    structure = info.get("structure", "unspecified structure")
    cell_type = info.get("cell_type", "primitive cell")
    lattice = info.get("lattice_constants_angstrom", {})
    if lattice:
        lattice_text = ", ".join(f"{axis} = {value} Å" for axis, value in lattice.items())
    else:
        lattice_text = "not specified"
    return (
        f"material = {name} with space group {space_group} and structure = {structure} "
        f"using the {cell_type}, lattice constant(s) = {lattice_text}"
    )


_PSEUDOPOTENTIAL_GUIDANCE = {
    "LDA": "Use norm-conserving pseudopotentials generated within the LDA.",
    "PBE": "Use PAW or ultrasoft pseudopotentials compatible with PBE.",
    "PBE_SOL": "Use pseudopotentials generated with the PBEsol functional and keep them consistent across all species.",
}


def _pseudopotential_text(family: Optional[str]) -> str:
    if not family:
        return ""
    key = family.strip().upper()
    if key not in _PSEUDOPOTENTIAL_GUIDANCE:
        raise ValueError(
            f"Unsupported pseudopotential family '{family}'. Choose from {list(_PSEUDOPOTENTIAL_GUIDANCE)}."
        )
    return _PSEUDOPOTENTIAL_GUIDANCE[key]


@dataclass
class MaterialRecord:
    """Single material instance stored on disk."""

    id: str
    info: Dict[str, Any]
    parameters: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any]
    source_path: Path

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], *, source_path: Path) -> "MaterialRecord":
        return cls(
            id=payload.get("id") or payload.get("info", {}).get("name") or "material",
            info=payload.get("info", {}) or {},
            parameters=payload.get("parameters", {}) or {},
            ground_truth=payload.get("ground_truth", {}) or {},
            metadata=payload.get("metadata", {}) or {},
            source_path=source_path,
        )

    def summary(self) -> str:
        return _render_material_summary(self.info)

    def query_context(self) -> str:
        return _material_query_context(self.info)


@dataclass
class TaskTemplate:
    """Metadata that controls how prompts are constructed for a task."""

    name: str
    description: str
    prompt_template: str
    required_ground_truth: List[str]
    default_parameters: Dict[str, Any]
    source_path: Path

    @classmethod
    def from_json(cls, path: Path) -> "TaskTemplate":
        payload = json.loads(path.read_text(encoding="utf-8"))
        required_ground_truth = payload.get("required_ground_truth", []) or []
        return cls(
            name=payload["name"],
            description=payload.get("description", ""),
            prompt_template=payload["prompt_template"],
            required_ground_truth=required_ground_truth,
            default_parameters=payload.get("default_parameters", {}) or {},
            source_path=path,
        )


class BenchmarkDataset:
    """
    Discovers benchmark tasks and builds templated prompts.

    Directory layout:

    benchmark/
      questions/
        vc_relax.json
      materials/
        simple_materials.json
    """

    def __init__(self, data_root: Optional[Path] = None):
        package_root = Path(__file__).resolve().parent
        self.base_dir = Path(data_root or package_root)
        self.questions_dir = self.base_dir / "questions"
        self.materials_dir = self.base_dir / "materials"
        if not self.questions_dir.exists():
            raise FileNotFoundError(f"Questions directory missing: {self.questions_dir}")
        if not self.materials_dir.exists():
            raise FileNotFoundError(f"Materials directory missing: {self.materials_dir}")
        self._templates = self._load_templates()
        self._materials = self._load_materials()

    def _load_templates(self) -> Dict[str, TaskTemplate]:
        templates: Dict[str, TaskTemplate] = {}
        for manifest in sorted(self.questions_dir.glob("*.json")):
            template = TaskTemplate.from_json(manifest)
            templates[template.name] = template
        if not templates:
            raise RuntimeError(f"No task templates found under {self.questions_dir}")
        return templates

    def _load_materials(self) -> Dict[str, MaterialRecord]:
        materials: Dict[str, MaterialRecord] = {}
        for path in sorted(self.materials_dir.rglob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "materials" in payload:
                entries = payload["materials"]
            elif isinstance(payload, list):
                entries = payload
            elif isinstance(payload, dict):
                entries = [payload]
            else:
                raise ValueError(f"Unsupported material format in {path}")
            for entry in entries:
                record = MaterialRecord.from_payload(entry, source_path=path)
                if record.id in materials:
                    raise ValueError(f"Duplicated material id '{record.id}' at {path}")
                materials[record.id] = record
        if not materials:
            raise RuntimeError(f"No materials found under {self.materials_dir}")
        return materials

    @property
    def tasks(self) -> List[str]:
        return list(self._templates.keys())

    def collect(
        self,
        task_name: Optional[str] = None,
        parameter_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        pseudopotential_family: Optional[str] = "LDA",
    ) -> List[DataItem]:
        """
        Build benchmark prompts.

        Args:
            task_name: Limit collection to a single task.
            parameter_overrides: Optional mapping from task name to parameter dict
                that should override both template defaults and material parameters.
            pseudopotential_family: Select pseudopotential guidance text (LDA, PBE, PBE_SOL).
        """
        task_names = [task_name] if task_name else self.tasks
        items: List[DataItem] = []
        overrides = parameter_overrides or {}
        pseudo_text = _pseudopotential_text(pseudopotential_family)
        pseudo_tag = (
            pseudopotential_family.strip().upper() if pseudopotential_family else None
        )

        for name in task_names:
            if name not in self._templates:
                raise KeyError(f"Unknown task '{name}'. Available: {self.tasks}")
            template = self._templates[name]
            merged_override = overrides.get(name, {})
            for material in self._materials.values():
                parameters = {
                    **template.default_parameters,
                    **material.parameters,
                    **merged_override,
                }
                prompt = template.prompt_template.format(
                    material_summary=material.summary(),
                    material_context=material.query_context(),
                    parameter_text=_format_parameter_block(parameters),
                    pseudopotential_text=pseudo_text,
                ).strip()
                ground_truth = dict(material.ground_truth)
                for field in template.required_ground_truth:
                    ground_truth.setdefault(field, None)
                metadata = {
                    "description": template.description,
                    "parameters": parameters,
                    "material_info": material.info,
                    "pseudopotential_family": pseudo_tag,
                }
                items.append(
                    DataItem(
                        id=f"{name}:{material.id}",
                        task=name,
                        prompt=prompt,
                        ground_truth=ground_truth,
                        metadata=metadata,
                    )
                )
        return items


__all__ = [
    "BenchmarkDataset",
    "DataItem",
    "MaterialRecord",
    "TaskTemplate",
]
