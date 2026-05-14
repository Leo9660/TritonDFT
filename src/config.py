"""
Minimal pseudopotential configuration loader.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to load config.yaml; install it via `pip install pyyaml`.") from exc

DEFAULT_PSEUDO_DIRS = {
    "LDA": "PseudoDojo/SR_v0.4.1/LDA_standard",
    "PBE": "PseudoDojo/SR_v0.4.1/PBE_standard",
    "PBESOL": "PseudoDojo/SR_v0.4.1/PBEsol_standard",
    "PBE_FR": "PseudoDojo/SR_v0.4.1/PBE_fr",
    "PBESOL_FR": "PseudoDojo/SR_v0.4.1/PBEsol_fr",
}
DEFAULT_QE_BIN_DIR = "QuantumE/bin"


@dataclass(frozen=True)
class PseudoPaths:
    LDA: str
    PBE: str
    PBESOL: str
    PBE_FR: str
    PBESOL_FR: str

    @classmethod
    def from_dict(cls, data: dict) -> "PseudoPaths":
        if not data:
            data = {}
        return cls(
            LDA=data.get("LDA") or data.get("lda") or DEFAULT_PSEUDO_DIRS["LDA"],
            PBE=data.get("PBE") or data.get("pbe") or DEFAULT_PSEUDO_DIRS["PBE"],
            PBESOL=data.get("PBESOL") or data.get("pbesol") or DEFAULT_PSEUDO_DIRS["PBESOL"],
            PBE_FR=data.get("PBE_FR") or data.get("pbe_fr") or DEFAULT_PSEUDO_DIRS["PBE_FR"],
            PBESOL_FR=data.get("PBESOL_FR") or data.get("pbesol_fr") or DEFAULT_PSEUDO_DIRS["PBESOL_FR"],
        )

    def as_dict(self) -> dict:
        return {
            "lda": self.LDA,
            "pbe": self.PBE,
            "pbesol": self.PBESOL,
            "pbe_fr": self.PBE_FR,
            "pbesol_fr": self.PBESOL_FR,
        }


@dataclass(frozen=True)
class Config:
    pseudo: PseudoPaths
    qe_bin_dir: str
    path: Path

    @classmethod
    def load(cls, config_name: Optional[str] = None) -> "Config":
        repo_root = Path(__file__).resolve().parent.parent
        config_root = repo_root / "config"
        if config_name:
            path = Path(config_name)
            if not path.is_absolute():
                path = config_root / config_name
        else:
            path = config_root / "config.yaml"

        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
            pseudo_section = data.get("pseudo", {})
            qe_bin_dir = data.get("qe_bin_dir")
        else:
            pseudo_section = {}
            qe_bin_dir = None

        # Resolve every pseudo dir against repo_root so pw.x finds them
        # regardless of how deeply the run's cwd is nested (e.g. when
        # work_dir is /workspace/tmp/<date>/<run>/ — 3 levels deep).
        pseudo = PseudoPaths.from_dict(pseudo_section)
        pseudo = PseudoPaths(
            LDA=str((repo_root / pseudo.LDA).resolve()) if not Path(pseudo.LDA).is_absolute() else pseudo.LDA,
            PBE=str((repo_root / pseudo.PBE).resolve()) if not Path(pseudo.PBE).is_absolute() else pseudo.PBE,
            PBESOL=str((repo_root / pseudo.PBESOL).resolve()) if not Path(pseudo.PBESOL).is_absolute() else pseudo.PBESOL,
            PBE_FR=str((repo_root / pseudo.PBE_FR).resolve()) if not Path(pseudo.PBE_FR).is_absolute() else pseudo.PBE_FR,
            PBESOL_FR=str((repo_root / pseudo.PBESOL_FR).resolve()) if not Path(pseudo.PBESOL_FR).is_absolute() else pseudo.PBESOL_FR,
        )

        final_qe_bin = qe_bin_dir or str((repo_root / DEFAULT_QE_BIN_DIR).resolve())
        return cls(
            pseudo=pseudo,
            qe_bin_dir=final_qe_bin,
            path=path,
        )
