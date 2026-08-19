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
    "PBE_FR": "PseudoDojo/FR_v0.4/PBE_standard",
    "PBESOL_FR": "PseudoDojo/FR_v0.4/PBEsol_standard",
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
    remote_qe_bin_dir: str
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
            remote_qe_bin_dir = data.get("remote_qe_bin_dir")
        else:
            pseudo_section = {}
            qe_bin_dir = None
            remote_qe_bin_dir = None

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

        # Never cross-fallback between XC families or relativistic levels.
        # A missing same-family library must fail clearly instead of silently
        # producing a scientifically inconsistent calculation.
        def _has_upfs(d: str) -> bool:
            p = Path(d)
            return p.is_dir() and any(p.glob("*.upf")) or any(p.glob("*.UPF"))

        pseudo = PseudoPaths(**resolved)

        final_qe_bin = qe_bin_dir or str((repo_root / DEFAULT_QE_BIN_DIR).resolve())
        return cls(
            pseudo=pseudo,
            qe_bin_dir=final_qe_bin,
            remote_qe_bin_dir=remote_qe_bin_dir or "",
            path=path,
        )

# ── Pseudopotential library selection ────────────────────────────────────────
# PseudoDojo ships a regular tree:
#     <root>/SR_v0.4.1/<XC>_<accuracy>      scalar-relativistic
#     <root>/FR_v0.4/<XC>_<accuracy>        fully relativistic (for SOC)
# so the three axes a user picks (functional, relativistic treatment, accuracy)
# map onto a path without asking them to configure ten directories. The root is
# derived from whatever they already configured for PBE.
XC_CHOICES = ("LDA", "PBE", "PBEsol")
REL_CHOICES = ("SR", "FR")
ACCURACY_CHOICES = ("standard", "stringent")
# PseudoDojo publishes no fully-relativistic LDA library.
UNAVAILABLE = {("LDA", "FR")}
DEFAULT_PSEUDO_CHOICE = ("PBE", "SR", "standard")

_REL_DIRS = {"SR": "SR_v0.4.1", "FR": "FR_v0.4"}


def resolve_pseudo_dir(pseudo_paths, xc: str, rel: str, accuracy: str):
    """Map (functional, relativistic treatment, accuracy) to a library path.

    Returns (path, error). `error` is set when the combination does not exist,
    in which case the caller should fall back rather than run with the wrong
    pseudopotentials.
    """
    import os

    xc = (xc or "PBE").strip()
    rel = (rel or "SR").strip().upper()
    accuracy = (accuracy or "standard").strip().lower()

    canonical = {c.lower(): c for c in XC_CHOICES}
    xc = canonical.get(xc.lower(), xc)

    if xc not in XC_CHOICES:
        return None, f"unknown functional {xc!r}"
    if rel not in REL_CHOICES:
        return None, f"unknown relativistic treatment {rel!r}"
    if accuracy not in ACCURACY_CHOICES:
        return None, f"unknown accuracy {accuracy!r}"
    if (xc, rel) in UNAVAILABLE:
        return None, f"PseudoDojo has no fully-relativistic {xc} library"

    # <root>/<SR_v0.4.1|FR_v0.4>/<XC>_<accuracy>, root taken from the configured
    # PBE entry so a site-specific install location still works.
    base = getattr(pseudo_paths, "PBE", "") or ""
    root = os.path.dirname(os.path.dirname(base))
    if not root:
        return None, "pseudopotential root is not configured"
    path = os.path.join(root, _REL_DIRS[rel], f"{xc}_{accuracy}")
    if not os.path.isdir(path):
        return None, f"library not present on disk: {path}"
    return path, None
