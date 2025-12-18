from typing import Optional

from config import PseudoPaths

pw_requirement_template = """
    ### Minimal Rules (hard constraints)
    [Sections]
    - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
    - Always set verbosity='low' in &control.

    [Lattice specification]
    - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
    * e.g., fcc → ibrav=2, bcc → ibrav=3, sc → ibrav=1.
    - For ibrav=0: set celldm(1)=<alat in Bohr> in &system and provide CELL_PARAMETERS (alat) with dimensionless 3×3 coefficients.

    [Energy cutoffs]
    - Always infer cutoffs from pseudopotential quality, not use defaults.
    - Typical ranges (plane-wave basis):
        * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
        * GBRV ultrasoft: ecutwfc ≈ 40-60 Ry.
    - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

    [Atomic positions]
    - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
    - Each atomic line MUST be: <species> <x> <y> <z>
    - Ensure nat equals the number of position lines.
    - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

    [Pseudopotentials]
    - You MUST set pseudo_dir according to the following instructions, or the script will be invalid.
    - {pseudo_dir_instructions}
    - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
    - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

    [Filenames / reuse]
    - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
    - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
    - Always set in &control: outdir='./' and wfcdir='./'.

    [Convergence thresholds]
    - Put conv_thr ONLY in &electrons.
    - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

    [K_POINTS offsets]
    - Syntax: K_POINTS automatic  nk1 nk2 nk3  k1 k2 k3  (six integers).
    - Offset parity only: even→0 (Γ-centered), odd→1 (half-grid shift). e.g., 2≡0, 3≡1.
    - Recommended defaults:
        * Insulators/semiconductors (occupations='fixed'): use 0 0 0.
        * Metals (occupations='smearing'): use 1 1 1.
    - Symmetry-aware tweaks:
        * High-symmetry cubic (fcc/bcc/sc): 0 0 0 (fewer irreducible k-points, faster).
        * Low-symmetry/metallic systems: 1 1 1 (reduces Γ singularities).
        * 2D slabs with nk3=1 (vacuum along c): use 0 0 0 (do NOT shift along the non-sampled direction).
    - Keep offsets consistent across axes unless you have a clear reason (e.g., layered anisotropy).

    [Header formatting & discipline]
    - Section headers must be single-line with qualifiers (e.g., "ATOMIC_POSITIONS (crystal)", "CELL_PARAMETERS (alat)", "K_POINTS automatic").
    - ecutwfc/ecutrho/occupations MUST be inside &system.
    - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
    - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
    - No explanations, comments, or extra markdown in outputs.
"""

scf_parse_requirement = """
    - total energy (final !-marked energy in Ry)
    - Fermi level (in eV)
    - band gap (in eV, 0.0 if metallic)
    - lattice constant (in Å, for cubic systems only)
"""

vc_relax_parse_requirement = """
    - final structure of all atoms (final CELL_PARAMETERS and ATOMIC_POSITIONS in the output)
"""


def _format_pseudo_dir_instructions(pseudo_paths: "PseudoPaths") -> str:
    return (
        f"If functional=LDA, set pseudo_dir={pseudo_paths.LDA}. "
        f"If functional=PBE, set pseudo_dir={pseudo_paths.PBE}. "
        f"If functional=PBEsol, set pseudo_dir={pseudo_paths.PBESOL}."
    )


def get_pw_requirement(pseudo_paths: "PseudoPaths") -> str:
    instructions = _format_pseudo_dir_instructions(pseudo_paths)
    return pw_requirement_template.format(pseudo_dir_instructions=instructions).strip()


_PARSE_REQUIREMENTS = {
    "scf": scf_parse_requirement,
    "vc-relax": vc_relax_parse_requirement,
}


def get_parse_requirement(key: Optional[str]) -> str:
    if not key:
        return ""
    return _PARSE_REQUIREMENTS.get(key, "")
