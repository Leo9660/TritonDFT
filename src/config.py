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

        def _resolve(p: str) -> str:
            return str((repo_root / p).resolve()) if not Path(p).is_absolute() else p

        resolved = {
            "LDA": _resolve(pseudo.LDA),
            "PBE": _resolve(pseudo.PBE),
            "PBESOL": _resolve(pseudo.PBESOL),
            "PBE_FR": _resolve(pseudo.PBE_FR),
            "PBESOL_FR": _resolve(pseudo.PBESOL_FR),
        }

        # Fallback for missing/empty pseudo dirs.  PseudoDojo FR sets ship
        # separately from SR; if the FR directory doesn't exist (or is
        # empty) we transparently re-point the FR slot to its non-FR sibling
        # so a spin-orbit request degrades gracefully to scalar-relativistic
        # instead of crashing with "file ... not found" inside pw.x.  Same
        # idea: if any slot is missing, fall back to PBE_standard as a
        # last-resort.
        def _has_upfs(d: str) -> bool:
            p = Path(d)
            return p.is_dir() and any(p.glob("*.upf")) or any(p.glob("*.UPF"))

        pbe_fallback = resolved["PBE"] if _has_upfs(resolved["PBE"]) else None
        if pbe_fallback:
            if not _has_upfs(resolved["PBE_FR"]):
                resolved["PBE_FR"] = pbe_fallback
            if not _has_upfs(resolved["PBESOL_FR"]):
                resolved["PBESOL_FR"] = resolved["PBESOL"] if _has_upfs(resolved["PBESOL"]) else pbe_fallback
            if not _has_upfs(resolved["LDA"]):
                resolved["LDA"] = pbe_fallback
            if not _has_upfs(resolved["PBESOL"]):
                resolved["PBESOL"] = pbe_fallback

        pseudo = PseudoPaths(**resolved)

        final_qe_bin = qe_bin_dir or str((repo_root / DEFAULT_QE_BIN_DIR).resolve())
        return cls(
            pseudo=pseudo,
            qe_bin_dir=final_qe_bin,
            path=path,
        )
