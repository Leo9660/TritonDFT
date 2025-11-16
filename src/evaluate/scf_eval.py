import re
from pathlib import Path
from typing import Dict, Any, Optional


def run_scf_metrics_input(input_path: Path) -> Dict[str, Any]:
    """
    Placeholder for SCF input metrics extraction.
    Currently empty; will be implemented later.
    """
    return {}


def run_scf_metrics(input_path: Path, output_path: Path) -> Dict[str, Optional[float]]:
    """
    Parse a QE SCF/relax output file and capture the final total energy line.
    Quantum ESPRESSO prints the converged energy with a leading '!'.

    Returns a dict containing the extracted total energy in Ry if found.
    """
    text = Path(output_path).read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"!\s+total energy\s*=\s*([-+]?\d+\.?\d*(?:[Ee][-+]?\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    if not matches:
        return {"total_energy_ry": None}
    final_energy = float(matches[-1])
    return {"total_energy_ry": final_energy}


__all__ = ["run_scf_metrics_input", "run_scf_metrics_output"]
