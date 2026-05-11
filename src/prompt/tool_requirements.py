from typing import Optional

from config import PseudoPaths

# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc → ibrav=2, bcc → ibrav=3, sc → ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3×3 lattice vectors in Å.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3×3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ≈ 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss (broadening) in Ry (e.g., degauss=0.01).
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You don't need to set up the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

#     [K_POINTS offsets — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Offset parity only:
#         * even → 0  (Γ-centered)
#         * odd  → 1  (half-grid shift)
#     - Recommended defaults:
#         * Insulators / semiconductors (occupations='fixed'):
#             use 0 0 0.
#         * Metals (occupations='smearing'):
#             use 1 1 1.
#     - Symmetry-aware tweaks:
#         * High-symmetry cubic (fcc / bcc / sc):
#             0 0 0 (fewer irreducible k-points, faster).
#         * Low-symmetry or metallic systems:
#             1 1 1 (reduces Γ singularities).
#         * 2D slabs with nk3 = 1 (vacuum along c):
#             use 0 0 0 (do NOT shift along the non-sampled direction).
#     - Keep offsets consistent across axes unless there is a clear physical reason (e.g., strong anisotropy).

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.

#     [Material-specific considerations]
#     - Based on the materials, you should choose whether to use Van der Waals correction or not.
#     - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., "ATOMIC_POSITIONS (crystal)", "CELL_PARAMETERS (alat)", "K_POINTS automatic").
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

# Active QE requirement prompt iteration 1.
# Iteration result summary:
# - Test case: Si, primitive diamond, LDA, vc-relax, strict 1 meV/atom target
# - Successful QE run folder:
#   /workspace/TritonDFT/2026-04-14_si_highacc_iter2/2026-04-14/Si_vc-relax_165044_8bdf96e2
# - First attempt in this run: 8x8x8 (unsuccessful)
# - Final successful QE input used: ecutwfc = 100 Ry, K_POINTS automatic / 10 10 10 1 1 1
# - Conclusion: strengthening the QE-side requirement guidance moved the Si 1 meV/atom
#   run from the old 6x6x6 behavior to a successful 10x10x10 result.
# Medium-accuracy check:
# - Test case: Si, primitive diamond, LDA, vc-relax, strict 10 meV/atom target
# - Current rerun folder: /workspace/TritonDFT/2026-04-14_si_mediumacc_iter1_rerun/2026-04-14/Si_vc-relax_170352_20831e02
# - Current chosen QE input: ecutwfc = 80 Ry, K_POINTS automatic / 12 12 12 1 1 1
# - Interpretation: the strengthened requirement no longer underestimates, but now overshoots
#   the scientist reference 8x8x8 for the 10 meV/atom Si case.
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc → ibrav=2, bcc → ibrav=3, sc → ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3×3 lattice vectors in Å.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3×3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ≈ 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss (broadening) in Ry (e.g., degauss=0.01).
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You don't need to set up the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - nk1 nk2 nk3 MUST be chosen from an accuracy-first perspective, not a cost-first perspective.
#     - For strict targets such as 1 meV/atom, you MUST treat underestimating k-point density as unacceptable.
#     - If multiple meshes seem plausible, you MUST choose the denser one.
#     - A slightly over-conservative mesh is preferred over a too-coarse mesh.
#     - Do NOT choose a coarser mesh just to reduce irreducible k-points or runtime.
#     - Do NOT use "high-symmetry cubic" as a reason to reduce nk1 nk2 nk3.
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser input mesh.

#     - Practical mesh policy:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a dense Monkhorst-Pack mesh appropriate for high-accuracy total energies.
#         * Metals (occupations='smearing'):
#             choose an even denser mesh than for insulators if needed for comparable accuracy.
#         * For small primitive cells of simple semiconductors or insulators, especially cubic systems,
#           assume that high-accuracy targets generally require denser meshes than generic default choices.
#         * For a strict 1 meV/atom-style target, do NOT stop at a merely reasonable or low-cost mesh.

#     - Offset policy:
#         * Offset parity only: even → 0, odd → 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.

#     [Material-specific considerations]
#     - Based on the materials, you should choose whether to use Van der Waals correction or not.
#     - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., "ATOMIC_POSITIONS (crystal)", "CELL_PARAMETERS (alat)", "K_POINTS automatic").
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

# Previous active QE requirement prompt iteration 2 preserved for comparison.
# Result summary:
# - Si 1 meV/atom remained correct at 10x10x10.
# - Si 10 meV/atom improved from 12x12x12 to 9x9x9, but still slightly overshot 8x8x8.
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.
#
#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc → ibrav=2, bcc → ibrav=3, sc → ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3×3 lattice vectors in Å.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3×3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.
#
#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ≈ 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.
#
#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).
#
#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss (broadening) in Ry (e.g., degauss=0.01).
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.
#
#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You don't need to set up the k path in nscf, but set it when calculation = 'bands'.
#
#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.
#
#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.
#
#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.
#
#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).
#
#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser or denser input mesh by itself.
#
#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly described as very strict / high accuracy:
#             - treat underestimating k-point density as unacceptable;
#             - if two meshes are plausible, choose the denser one;
#             - a slight upward bias is preferred.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - choose the smallest mesh that is still likely to satisfy the target;
#             - do NOT automatically jump to the densest plausible mesh;
#             - prefer a modest safety margin, not a large one.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - prefer a reasonable low-cost mesh that is still physically sensible;
#             - avoid unnecessary over-conservative density.
#
#     - Material-class policy:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed.
#         * For small primitive cells of simple semiconductors or insulators, especially cubic systems:
#             - 1 meV/atom targets usually require clearly denser meshes than generic defaults;
#             - 10 meV/atom targets should be stricter than loose defaults, but should not be over-tightened without reason.
#
#     - Anti-overshoot rule:
#         * For medium or loose targets, do NOT choose a substantially denser mesh unless the material or task clearly requires it.
#         * Prefer the smallest mesh that is likely to meet the stated target, not the largest mesh that could possibly work.
#
#     - Offset policy:
#         * Offset parity only: even → 0, odd → 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.
#
#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.
#
#     [Material-specific considerations]
#     - Based on the materials, you should choose whether to use Van der Waals correction or not.
#     - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.
#
#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., "ATOMIC_POSITIONS (crystal)", "CELL_PARAMETERS (alat)", "K_POINTS automatic").
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

# Active QE requirement prompt iteration 3.
# Iteration 3 result summary:
# - Si 10 meV/atom medium test folder: /workspace/TritonDFT/2026-04-15_si_mediumacc_iter3/2026-04-15/Si_vc-relax_034824_5bac3e06
# - Generated K_POINTS automatic / 12 12 12 1 1 1, overshooting the 8x8x8 medium target.
# Weakness analysis of iteration 2:
# - It fixed the major underestimation problem and preserved the 1 meV/atom Si case at 10x10x10.
# - But its medium-accuracy guidance is still too qualitative, so the model keeps a noticeable
#   upward safety margin and can still overshoot 10 meV/atom cases (e.g. 9x9x9 instead of 8x8x8).
# - The next prompt therefore keeps the strict 1 meV/atom rule, but makes the 10 meV/atom rule
#   more explicitly "minimal mesh that satisfies the target" for simple small semiconductors.
#
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc → ibrav=2, bcc → ibrav=3, sc → ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3×3 lattice vectors in Å.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3×3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ≈ 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss (broadening) in Ry (e.g., degauss=0.01).
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You don't need to set up the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser or denser input mesh by itself.

#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly described as very strict / high accuracy:
#             - treat underestimating k-point density as unacceptable;
#             - if two meshes are plausible, choose the denser one;
#             - a small upward bias is preferred.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - choose the smallest mesh that is still likely to satisfy the target;
#             - do NOT automatically add an extra upward safety step if the current mesh is already plausible;
#             - for simple small semiconductors or insulators, prefer a near-minimal mesh rather than a conservative one.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - prefer a reasonable low-cost mesh that is still physically sensible;
#             - avoid unnecessary over-conservative density.

#     - Material-class calibration:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed.
#         * For small primitive cells of simple cubic-like semiconductors or insulators:
#             - 1 meV/atom targets generally justify a clearly dense mesh;
#             - 10 meV/atom targets should usually be only one step tighter than a loose/default choice, not many steps tighter.

#     - Anti-overshoot rule:
#         * For medium or loose targets, prefer the smallest mesh that is likely to meet the stated target.
#         * Do NOT choose a substantially denser mesh unless the material, metallicity, low symmetry, or another explicit physical requirement clearly justifies it.
#         * If two medium-accuracy meshes are both plausible, prefer the lower one unless there is a concrete reason to round up.

#     - Offset policy:
#         * Offset parity only: even → 0, odd → 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.

#     [Material-specific considerations]
#     - Based on the materials, you should choose whether to use Van der Waals correction or not.
#     - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., "ATOMIC_POSITIONS (crystal)", "CELL_PARAMETERS (alat)", "K_POINTS automatic").
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

# Iteration 4
# Iteration 4 result summary:
# - Si 10 meV/atom medium test folder: /workspace/TritonDFT/2026-04-15_si_mediumacc_iter4/2026-04-15/Si_vc-relax_035341_1baf932b
# - Generated K_POINTS automatic / 12 12 12 1 1 1, still overshooting the 8x8x8 medium target.
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss in Ry.
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed':
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You do not need to set the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf.
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>'.
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control.

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Symmetry may reduce the irreducible set internally, but that is NOT a reason by itself to choose a denser or coarser input mesh.

#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly very strict / high accuracy:
#             - treat underestimating k-point density as unacceptable;
#             - if two meshes are plausible, choose the denser one;
#             - a small upward bias is preferred.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - choose the MINIMUM mesh that is still likely to satisfy the target;
#             - once a mesh is sufficient, STOP and do NOT increase it further;
#             - unnecessary extra k-point density is a mistake for medium targets.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - prefer a reasonable low-cost mesh that is still physically sensible.

#     - Material-class calibration:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed.
#         * For small primitive cells of simple semiconductors or insulators:
#             - 1 meV/atom targets justify a clearly dense mesh;
#             - 10 meV/atom targets should remain close to the first sufficient mesh, not a conservative mesh;
#             - do NOT apply the 1 meV/atom mindset to a 10 meV/atom case.

#     - Medium-target stopping rule:
#         * For medium targets, prefer the LOWEST mesh that still appears sufficient.
#         * If two medium-accuracy meshes are both plausible, you MUST choose the lower one.
#         * Only round upward when there is a concrete physical reason such as metallicity, low symmetry, strong anisotropy, or another explicit difficulty.
#         * Do NOT round upward just for safety margin.
#         * Do NOT round upward merely because a denser mesh might also work.

#     - Offset policy:
#         * Offset parity only: even -> 0, odd -> 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b'.
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or weights.

#     [Material-specific considerations]
#     - Based on the material, choose whether to use Van der Waals correction.
#     - Based on material properties, choose whether to use magnetic moments, spin polarization, or spin-orbit coupling.
#     - Use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers.
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

#Iteration 5
# Iteration 5 result summary:
# - Si 10 meV/atom medium test folder: /workspace/TritonDFT/2026-04-15_si_mediumacc_iter5/2026-04-15/Si_vc-relax_041915_7ba858f5
# - Generated K_POINTS automatic / 10 10 10 1 1 1, improved from 12x12x12 but still above the 8x8x8 medium target.
#Result 10x10x10
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss in Ry.
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You do not need to set the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf.
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>'.
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control.

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser or denser input mesh by itself.

#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly described as very strict / high accuracy:
#             - treat underestimating k-point density as unacceptable;
#             - if two meshes are plausible, choose the denser one;
#             - a small upward bias is preferred.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - choose a mesh that is accurate but not aggressively over-conservative;
#             - prefer the smallest clearly safe mesh;
#             - do NOT apply the 1 meV/atom mindset to a 10 meV/atom case.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - prefer a reasonable low-cost mesh that is still physically sensible.

#     - Material-class calibration:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed.
#         * For small primitive cells of simple semiconductors or insulators:
#             - 1 meV/atom targets justify a clearly dense mesh;
#             - 10 meV/atom targets should usually be only moderately denser than a loose/default choice;
#             - for 10 meV/atom, avoid jumping more than one density step above a reasonable baseline unless there is a clear physical reason.

#     - Anti-overshoot rule:
#         * For medium targets, do NOT choose a substantially denser mesh unless the material, metallicity, low symmetry, or another explicit physical requirement clearly justifies it.
#         * If two medium-accuracy meshes are both plausible, prefer the lower one unless there is a concrete reason to round up.

#     - Offset policy:
#         * Offset parity only: even -> 0, odd -> 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b'.
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or weights.

#     [Material-specific considerations]
#     - Based on the material, choose whether to use Van der Waals correction.
#     - Based on material properties, choose whether to use magnetic moments, spin polarization, or spin-orbit coupling.
#     - Use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers.
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

#Iteration 6
# Iteration 6 result summary:
# - Si 10 meV/atom medium test folder: /workspace/TritonDFT/2026-04-15_si_mediumacc_iter6/2026-04-15/Si_vc-relax_042544_ff280a50
# - Generated K_POINTS automatic / 8 8 8 1 1 1, matching the 8x8x8 medium target.
# - Si 1 meV/atom high test folder: /workspace/TritonDFT/2026-04-15_si_highacc_iter6/2026-04-15/Si_vc-relax_042716_701527d0
# - Generated K_POINTS automatic / 12 12 12 1 1 1, above the 10x10x10 high target; QE input also failed formatting.
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss in Ry.
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You do not need to set the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf.
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>'.
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control.

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser or denser input mesh by itself.

#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly described as very strict / high accuracy:
#             - treat underestimating k-point density as unacceptable;
#             - if two meshes are plausible, choose the denser one;
#             - a small upward bias is preferred.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - choose a mesh that is accurate but not aggressively over-conservative;
#             - prefer a moderate mesh, not a dense mesh;
#             - do NOT apply the 1 meV/atom mindset to a 10 meV/atom case.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - prefer a reasonable low-cost mesh that is still physically sensible.

#     - Material-class calibration:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed.
#         * For small primitive cells of simple semiconductors or insulators:
#             - 1 meV/atom targets justify a clearly dense mesh;
#             - 10 meV/atom targets should stay near the medium regime and should not drift toward the high-accuracy regime;
#             - for 10 meV/atom, prefer about one step above a loose/default mesh, not two or more steps above it, unless there is a clear physical reason.

#     - Anti-overshoot rule:
#         * For medium targets, do NOT choose a substantially denser mesh unless the material, metallicity, low symmetry, anisotropy, or another explicit physical requirement clearly justifies it.
#         * If two medium-accuracy meshes are both plausible, prefer the lower one unless there is a concrete physical reason to round up.
#         * For simple fixed-occupation semiconductors, unnecessary extra density counts against the choice.

#     - Offset policy:
#         * Offset parity only: even -> 0, odd -> 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b'.
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or weights.

#     [Material-specific considerations]
#     - Based on the material, choose whether to use Van der Waals correction.
#     - Based on material properties, choose whether to use magnetic moments, spin polarization, or spin-orbit coupling.
#     - Use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers.
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

#Iteration 7
# Iteration 7 result summary:
# - Si 1 meV/atom high test folder: /workspace/TritonDFT/2026-04-15_si_highacc_iter7/2026-04-15/Si_vc-relax_043342_fd304e6b
# - Generated K_POINTS automatic / 8 8 8 1 1 1, below the 10x10x10 high target and therefore incorrect.
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.
#
#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.
#
#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.
#
#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).
#
#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss in Ry.
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.
#
#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You do not need to set the k path in nscf, but set it when calculation = 'bands'.
#
#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.
#
#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.
#
#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.
#
#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).
#
#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - nk1 nk2 nk3 MUST be chosen from an accuracy-first perspective, not a cost-first perspective.
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser input mesh.
#
#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly described as very strict / high accuracy:
#             - treat underestimating k-point density as unacceptable;
#             - if multiple meshes seem plausible, you MUST choose the denser one;
#             - a slightly over-conservative mesh is preferred over a too-coarse mesh.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - choose a mesh that is accurate but not aggressively over-conservative;
#             - prefer a moderate mesh, not a dense mesh;
#             - do NOT apply the 1 meV/atom mindset to a 10 meV/atom case.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - prefer a reasonable low-cost mesh that is still physically sensible.
#
#     - Practical mesh policy:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose an even denser mesh than for insulators if needed for comparable accuracy.
#         * For small primitive cells of simple semiconductors or insulators, especially cubic systems:
#             - 1 meV/atom targets generally require denser meshes than generic default choices;
#             - 10 meV/atom targets should stay in the medium regime and should not drift toward the high-accuracy regime;
#             - for 10 meV/atom, prefer about one step above a reasonable loose/default mesh, not two or more steps above it, unless there is a clear physical reason.
#
#     - Anti-overshoot rule:
#         * For medium targets, do NOT choose a substantially denser mesh unless the material, metallicity, low symmetry, anisotropy, or another explicit physical requirement clearly justifies it.
#         * If two medium-accuracy meshes are both plausible, prefer the lower one unless there is a concrete reason to round up.
#         * For simple fixed-occupation semiconductors, unnecessary extra density counts against the choice.
#
#     - Offset policy:
#         * Offset parity only: even -> 0, odd -> 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.
#
#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.
#
#     [Material-specific considerations]
#     - Based on the materials, you should choose whether to use Van der Waals correction or not.
#     - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.
#
#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., 'ATOMIC_POSITIONS (crystal)', 'CELL_PARAMETERS (alat)', 'K_POINTS automatic').
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - No explanations, comments, or extra markdown in outputs.
# """

# Iteration 8 result summary (Gemini 2.5 Flash, Si diamond primitive, LDA vc-relax):
# - 1 meV/atom high-accuracy test:
#   /workspace/TritonDFT/2026-04-16_si_highacc_iter8_gemini/2026-04-16/Si_vc-relax_035737_766e1eaa
#   Generated K_POINTS automatic / 10 10 10 1 1 1, matching the scientist high-accuracy target.
#   QE execution stopped because Gemini placed pseudo_dir in &system, but the k-point choice was correct.
# - 10 meV/atom medium-accuracy test:
#   /workspace/TritonDFT/2026-04-16_si_mediumacc_iter8_gemini/2026-04-16/Si_vc-relax_040032_f288bada
#   Generated K_POINTS automatic / 8 8 8 1 1 1, matching the scientist medium-accuracy target.
# - 20 meV/atom low-accuracy test:
#   /workspace/TritonDFT/2026-04-16_si_lowacc_iter8_gemini/2026-04-16/Si_vc-relax_040242_1ea5dd26
#   Generated K_POINTS automatic / 6 6 6 1 1 1, matching the scientist low-accuracy target.
# - Conclusion: iteration 8 successfully enforces the target-floor / positive-side behavior
#   for Si across high, medium, and low accuracy tiers when run with Gemini 2.5 Flash.
#Iteration 8
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss in Ry.
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You do not need to set the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Choose nk1 nk2 nk3 from the requested energy-accuracy tier first, then adjust only if material physics clearly requires it.
#     - The selected mesh MUST NOT be below the accuracy-tier lower bound.
#     - Prefer the closest mesh from the positive side: choose the smallest mesh that is greater than or equal to the target-tier mesh.
#     - Do NOT choose a denser mesh than the target-tier mesh unless there is a concrete physical reason such as metallicity, low symmetry, strong anisotropy, or another explicit difficulty.
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser or denser input mesh by itself.

#     - Accuracy-tier floor policy:
#         * If the requested target is around 1 meV/atom or explicitly very strict / high accuracy:
#             - never choose below a high-accuracy mesh;
#             - for small primitive fixed-occupation semiconductors such as diamond Si, the target-tier mesh is 10x10x10;
#             - if uncertain, choose 10x10x10 or the next denser physically justified mesh, never 8x8x8 or lower.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - never choose below a medium-accuracy mesh;
#             - for small primitive fixed-occupation semiconductors such as diamond Si, the target-tier mesh is 8x8x8;
#             - prefer 8x8x8 over 9x9x9, 10x10x10, or 12x12x12 unless a concrete physical reason requires extra density.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - never choose below a loose but physically sensible mesh;
#             - for small primitive fixed-occupation semiconductors such as diamond Si, the target-tier mesh is 6x6x6;
#             - avoid unnecessary over-conservative density.

#     - Material-class calibration:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed.
#         * For small primitive cells of simple semiconductors or insulators, especially cubic systems:
#             - high accuracy should approach the high-accuracy target from above, not below;
#             - medium accuracy should approach the medium target from above, not drift into high-accuracy density;
#             - loose accuracy should remain low cost while still meeting the loose target.

#     - Offset policy:
#         * Offset parity only: even -> 0, odd -> 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.

#     [Material-specific considerations]
#     - Based on the materials, choose whether to use Van der Waals correction or not.
#     - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., 'ATOMIC_POSITIONS (crystal)', 'CELL_PARAMETERS (alat)', 'K_POINTS automatic').
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - Do NOT put commas after namelist assignments in QE input files.
#     - No explanations, comments, or extra markdown in outputs.
# """

#Iteration 9
#Iteration 10
#
# pw_requirement_template = """
#     ### Minimal Rules (hard constraints)
#     [Sections]
#     - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
#     - Always set verbosity='low' in &control.

#     [Lattice specification]
#     - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
#     * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
#     - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
#     (A) Recommended: use explicit angstrom lattice vectors:
#         - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
#         - Do NOT set celldm(1).
#     (B) Alternative: use alat-scaled lattice vectors:
#         - Set celldm(1)=<alat in Bohr> in &system.
#         - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
#         - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

#     [Energy cutoffs]
#     - Always infer cutoffs from pseudopotential quality, not use defaults.
#     - Typical ranges (plane-wave basis):
#         * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
#         * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
#     - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

#     [Calculation Logic]
#     - IF calculation = 'vc-relax' OR 'vc-md':
#         1. You MUST include the &CELL namelist.
#         2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

#     [Occupations & Smearing]
#     - IF occupations = 'smearing':
#         1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
#         2. You MUST specify degauss in Ry.
#         3. Do NOT use the keyword 'sigma'.
#     - IF occupations = 'fixed' (insulators):
#         1. Do NOT specify smearing or degauss.

#     [band calculations]
#     - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
#     - You do not need to set the k path in nscf, but set it when calculation = 'bands'.

#     [Atomic positions]
#     - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
#     - Each atomic line MUST be: <species> <x> <y> <z>
#     - Ensure nat equals the number of position lines.
#     - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

#     [Pseudopotentials]
#     - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
#     - {pseudo_dir_instructions}
#     - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
#     - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

#     [Filenames / reuse]
#     - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
#     - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
#     - Always set in &control: outdir='./' and wfcdir='./'.

#     [Convergence thresholds]
#     - Put conv_thr ONLY in &electrons.
#     - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

#     [K_POINTS selection — VC-relax / SCF / NSCF only]
#     - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
#     - Syntax:
#         K_POINTS automatic
#         nk1 nk2 nk3 k1 k2 k3
#     - Choose nk1 nk2 nk3 from the requested energy-accuracy tier and the material class.
#     - Two equal hard gates apply:
#         * Accuracy gate: the selected mesh MUST NOT be below the expected expert/reference mesh for that material and accuracy tier.
#         * Cost gate: the selected mesh MUST NOT be denser than the smallest mesh that is still safely greater than or equal to the expected expert/reference mesh, unless a concrete material-specific reason requires extra density.
#     - Choose the closest mesh from the positive side: use the smallest mesh that is greater than or equal to the expected expert/reference mesh.
#     - If a material-specific reference or benchmark table value is available in the user query, benchmark metadata, memory, or surrounding context, use that value as the lower bound.
#     - If no explicit reference is available, infer the lower bound from physical principles: primitive-cell size, reciprocal-cell volume, metallicity, band gap, symmetry, anisotropy, and requested energy accuracy.
#     - Do NOT hard-code a single material's mesh as a general answer.
#     - Do NOT choose a denser mesh than the inferred lower bound unless there is a concrete physical reason such as metallicity, low symmetry, strong anisotropy, very small gap, or another explicit difficulty.
#     - Be strict about not inflating the inferred lower bound: larger primitive cells, multi-atom insulating cells, and smaller Brillouin zones usually need less raw nk density than one- or two-atom primitive metals/semiconductors at the same accuracy target.
#     - Do NOT copy a dense mesh scale from small primitive metals/semiconductors into larger insulating cells unless reciprocal-space resolution, metallicity, anisotropy, or another concrete physical reason requires it.
#     - For strict/high-accuracy targets, do not accidentally choose a medium- or loose-accuracy mesh. A lower raw nk mesh is allowed only when the larger cell or smaller Brillouin zone clearly preserves high-accuracy reciprocal-space resolution.
#     - Verify that the candidate mesh is at or above the inferred floor for the specific material, cell, and accuracy tier before accepting it.
#     - Once the accuracy gate is clearly passed, high computational cost is also invalid: do NOT choose a denser mesh than the smallest valid positive-side mesh unless there is a concrete material-specific reason.
#     - Do not keep a dense k-point mesh merely because it is safer after the floor is satisfied.
#     - Symmetry may reduce the irreducible set internally, but that is NOT a justification for choosing a coarser or denser input mesh by itself.

#     - Accuracy-tier policy:
#         * If the requested target is around 1 meV/atom or explicitly very strict / high accuracy:
#             - never choose below the material-specific high-accuracy reference or inferred high-accuracy floor;
#             - if uncertain between two adjacent meshes, choose the smaller mesh only if it is still at or above the inferred floor;
#             - otherwise choose the next denser mesh.
#             - if the lower mesh might actually correspond to medium accuracy, reject it and choose the next denser mesh.
#             - the model must not declare the lower mesh acceptable without concrete material/cell/accuracy-tier reasoning.
#         * If the requested target is around 10 meV/atom or a medium-accuracy target:
#             - never choose below the material-specific medium-accuracy reference or inferred medium-accuracy floor;
#             - avoid drifting into high-accuracy density when the medium floor is already satisfied;
#             - prefer the smallest mesh at or above the medium floor.
#         * If the requested target is around 20 meV/atom or a loose-accuracy target:
#             - never choose below the material-specific loose-accuracy reference or inferred loose-accuracy floor;
#             - prefer the smallest mesh at or above the loose floor.

#     - Material-class calibration:
#         * Insulators / semiconductors (occupations='fixed'):
#             choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier and material-specific reciprocal lattice scale.
#             For larger multi-atom insulating cells, a lower raw nk1 nk2 nk3 mesh can satisfy the same reciprocal-space resolution; do not over-densify solely because the target is strict.
#         * Metals (occupations='smearing'):
#             choose a denser mesh than for comparable insulators when needed for Fermi-surface convergence.
#         * Small primitive cells usually need denser raw nk1 nk2 nk3 than large conventional cells for the same reciprocal-space resolution.
#         * Do not assume all diamond-structure semiconductors share the same k-point floor; use the specific material and accuracy tier.

#     - Offset policy:
#         * Offset parity only: even -> 0, odd -> 1.
#         * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
#         * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
#         * Keep offsets consistent across axes unless there is a clear physical reason not to.
#         * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

#     [K_POINTS for BANDS calculations]
#     - Applies ONLY when calculation = 'bands'.
#     - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
#     - Format:
#         K_POINTS crystal_b
#         N
#         kx ky kz npts
#         ...
#     - N is the number of high-symmetry path nodes.
#     - Each line defines a path node:
#         * (kx, ky, kz) are crystal coordinates.
#         * npts is the number of interpolated k-points to the NEXT node.
#     - The final node SHOULD have npts = 1.
#     - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
#     - Do NOT enumerate every k-point explicitly.

#     [Material-specific considerations]
#     - Before writing the input, actively decide whether each advanced physical parameter is needed: vdw_corr, spin polarization, Hubbard U, noncolin, lspinorb, and smearing.
#     - Do NOT include advanced parameters by default, but do NOT omit them when the material class strongly indicates they are physically required.
#     - If an advanced parameter is present in the parameter JSON, preserve it in the QE input unless it is syntactically invalid for QE.

#     [Van der Waals / dispersion]
#     - Use vdw_corr when the material is likely layered, weakly bonded between structural units, molecular, van der Waals bonded, or contains stacked quintuple/septuple layers.
#     - This often includes layered chalcogenides/tellurides, topological insulators, graphite-like systems, molecular crystals, and other materials whose interlayer binding is not well described by plain LDA/GGA.
#     - If vdw_corr is needed, put it in &system.
#     - If vdw_corr is present in the parameter JSON, it MUST appear in &system.
#     - Prefer vdw_corr = 'dft-d3' unless the user specifies a different QE-supported dispersion correction.
#     - Use only QE-supported vdw_corr strings. Do NOT output unsupported variants such as 'dft-d3.abc'.
#     - If the material is a dense 3D covalent, ionic, or metallic solid with no layered or weakly-bound character, omit vdw_corr.

#     [Spin / magnetism]
#     - Use spin polarization for magnetic materials, transition-metal magnets, open-shell atoms/ions, or materials described as ferromagnetic, antiferromagnetic, or ferrimagnetic.
#     - Do not use spin polarization for nonmagnetic closed-shell semiconductors or insulators unless requested.

#     [Hubbard U]
#     - Use Hubbard U only for localized correlated d/f electron systems where DFT+U is physically expected.
#     - Do not add Hubbard U for ordinary sp semiconductors, simple metals, or materials without localized correlated electrons.

#     [Noncollinear and spin-orbit coupling]
#     - Use noncolin=.true. and lspinorb=.true. for heavy-element topological materials, strong-SOC compounds, materials where band inversion/topology is central, or when the user asks for SOC.
#     - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.
#     - If lspinorb=.true. or noncolin=.true. is used for SOC, use fully relativistic pseudopotentials and a fully relativistic pseudo_dir. Do NOT use scalar-relativistic pseudopotentials for SOC calculations.
#     - Do not enable SOC for light-element ordinary systems unless requested.

#     [Smearing]
#     - Use smearing for metals and semimetals.
#     - Do not use smearing for clear insulators or semiconductors unless needed for convergence.
#     - For metallic or topological semimetal cases, prefer Marzari-Vanderbilt smearing with a reasonable degauss.

#     [Header formatting & discipline]
#     - Section headers must be single-line with qualifiers (e.g., 'ATOMIC_POSITIONS (crystal)', 'CELL_PARAMETERS (alat)', 'K_POINTS automatic').
#     - ecutwfc/ecutrho/occupations MUST be inside &system.
#     - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
#     - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
#     - Do NOT put commas after namelist assignments in QE input files.
#     - No explanations, comments, or extra markdown in outputs.
# """

pw_requirement_template = """
    ### Minimal Rules (hard constraints)
    [Sections]
    - Required section order (when applicable): &control, &system, &electrons, &ions (if ions move), &cell (if cell moves), ATOMIC_SPECIES, ATOMIC_POSITIONS (crystal), [CELL_PARAMETERS (alat) if ibrav=0], K_POINTS automatic or an explicit list.
    - Always set verbosity='low' in &control.

    [Lattice specification]
    - For known Bravais lattices (ibrav > 0): set lattice constant(s) in Bohr (a[Bohr] = a[Å]/0.529177).
    * e.g., fcc -> ibrav=2, bcc -> ibrav=3, sc -> ibrav=1.
    - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
    (A) Recommended: use explicit angstrom lattice vectors:
        - Put CELL_PARAMETERS (angstrom) and provide 3x3 lattice vectors in A.
        - Do NOT set celldm(1).
    (B) Alternative: use alat-scaled lattice vectors:
        - Set celldm(1)=<alat in Bohr> in &system.
        - Put CELL_PARAMETERS (alat) with dimensionless 3x3 coefficients.
        - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

    [Energy cutoffs]
    - Always infer cutoffs from pseudopotential quality, not use defaults.
    - Typical ranges (plane-wave basis):
        * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
        * GBRV ultrasoft: ecutwfc ~= 40-60 Ry.
    - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

    [Calculation Logic]
    - IF calculation = 'vc-relax' OR 'vc-md':
        1. You MUST include the &CELL namelist.
        2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

    [Occupations & Smearing]
    - IF occupations = 'smearing':
        1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
        2. You MUST specify degauss in Ry.
        3. Do NOT use the keyword 'sigma'.
    - IF occupations = 'fixed' (insulators/semiconductors):
        1. Do NOT specify smearing or degauss.

    [band calculations]
    - When calculation = 'nscf', you MUST set nbnd as the number of Kohn-Sham states + 10.
    - You do not need to set the k path in nscf, but set it when calculation = 'bands'.

    [Atomic positions]
    - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
    - Each atomic line MUST be: <species> <x> <y> <z>
    - Ensure nat equals the number of position lines.
    - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

    [Pseudopotentials]
    - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
    - {pseudo_dir_instructions}
    - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
    - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.
    - pseudo_dir MUST be inside &control, never inside &system.

    [Filenames / reuse]
    - For brand-new runs: set &control prefix='system_<number>' (system_0, system_1, ...).
    - For follow-up runs: reuse the exact same prefix/outdir/wfcdir as the source SCF.
    - Always set in &control: outdir='./' and wfcdir='./'.

    [Convergence thresholds]
    - Put conv_thr ONLY in &electrons.
    - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

    [K_POINTS selection — VC-relax / SCF / NSCF only]
    - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
    - Syntax:
        K_POINTS automatic
        nk1 nk2 nk3 k1 k2 k3
    - Choose nk1 nk2 nk3 from the requested energy-accuracy tier and the specific material/cell.
    - The selected mesh MUST NOT be below the inferred floor required by that material and accuracy tier.
    - Underestimation is the first failure condition. If a candidate mesh is below the inferred floor, reject it immediately, even if it is cheaper.
    - Only after the no-underestimate condition is satisfied, lower cost is a hard requirement: choose the smallest mesh still greater than or equal to the inferred floor.
    - Treat the k-point choice as an ordered candidate search from lower cost to higher cost; stop only at the first mesh that satisfies the inferred floor.
    - Do NOT add extra density merely for safety if the floor is already satisfied.
    - Do NOT round upward twice. The requested accuracy tier determines the floor; once the floor is met, do not add another conservative step.
    - Infer the lower bound from physical principles: primitive vs conventional cell, reciprocal-cell size, material class, metallicity or band gap, symmetry, anisotropy, and requested energy accuracy.
    - If the current candidate is below the inferred floor, increase the mesh; do not stop at a below-floor intermediate value.
    - Do NOT hard-code one material's mesh as a general answer for another material.
    - Do NOT choose a denser mesh than the inferred floor unless there is a concrete physical reason such as metallicity, low symmetry, strong anisotropy, very small gap, or another explicit difficulty.
    - Symmetry may reduce the irreducible set internally, but that is NOT a reason to choose a coarser or denser input mesh by itself.

    - Accuracy-tier policy:
        * Around 1 meV/atom or explicitly high accuracy:
            - never choose below the material-specific high-accuracy floor;
            - choose the smallest mesh at or above that floor;
            - a medium-accuracy or loose-accuracy mesh is invalid for this tier, even if it is cheaper;
            - do not make the mesh denser solely because the word "strict" appears.
        * Around 10 meV/atom or medium accuracy:
            - never choose below the material-specific medium-accuracy floor;
            - choose the smallest mesh at or above that floor;
            - avoid drifting into high-accuracy density when the medium floor is already satisfied.
        * Around 20 meV/atom or loose accuracy:
            - never choose below the material-specific loose-accuracy floor;
            - choose the smallest mesh at or above that floor;
            - avoid drifting into medium- or high-accuracy density when the loose floor is already satisfied.

    - Material-class calibration:
        * Insulators / semiconductors (occupations='fixed'):
            choose a Monkhorst-Pack mesh appropriate to the requested accuracy tier and reciprocal-space resolution.
        * Metals (occupations='smearing'):
            choose a denser mesh than for comparable insulators when needed for Fermi-surface convergence.
        * Do not assume all materials in the same prototype share the same floor; use the specific material, cell size, and accuracy tier.
        * Small primitive cells usually need denser raw nk1 nk2 nk3 than large conventional cells for the same reciprocal-space resolution.
        * Once that reciprocal-space resolution requirement is met, selecting a denser mesh is wrong unless a concrete physical issue is named.

    - Offset policy:
        * Offset parity only: even -> 0, odd -> 1.
        * If the user explicitly requests a half-shifted Monkhorst-Pack grid, honor that request.
        * If the user explicitly requests an automatic half-shifted grid, prefer 1 1 1 whenever valid.
        * Keep offsets consistent across axes unless there is a clear physical reason not to.
        * For nk3 = 1 slab calculations, do NOT shift the non-sampled direction.

    [K_POINTS for BANDS calculations]
    - Applies ONLY when calculation = 'bands'.
    - K_POINTS MUST use 'crystal_b' (NOT 'automatic').
    - Format:
        K_POINTS crystal_b
        N
        kx ky kz npts
        ...
    - N is the number of high-symmetry path nodes.
    - Each line defines a path node:
        * (kx, ky, kz) are crystal coordinates.
        * npts is the number of interpolated k-points to the NEXT node.
    - The final node SHOULD have npts = 1.
    - Do NOT use Monkhorst-Pack grids, offsets, or k-point weights.
    - Do NOT enumerate every k-point explicitly.

    [Material-specific considerations]
    - Based on the materials, choose whether to use Van der Waals correction or not.
    - Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling.
    - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.

    [Header formatting & discipline]
    - Section headers must be single-line with qualifiers (e.g., 'ATOMIC_POSITIONS (crystal)', 'CELL_PARAMETERS (alat)', 'K_POINTS automatic').
    - ecutwfc/ecutrho/occupations MUST be inside &system.
    - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
    - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
    - Do NOT put commas after namelist assignments in QE input files.
    - QE namelist values MUST be evaluated literals. Do NOT write arithmetic expressions such as "10.26 / 0.529177"; compute the number first and write only the numeric value.
    - No explanations, comments, or extra markdown in outputs.
"""



bandsx_requirement_template = """
    ### Minimal Rules (hard constraints)
    [Purpose]
    - bands.x is post-processing only. It reads eigenvalues produced by pw.x (calculation='bands') and writes band-structure data for plotting.

    [Prerequisite]
    - A completed pw.x bands run MUST exist with the same prefix/outdir that bands.x will read.

    [Input file shape]
    - The input MUST contain exactly one namelist: &BANDS ... /
    - No extra sections, no comments, no markdown.

    [Paths / reuse]
    - Always set in &BANDS: prefix='<same as pw.x>', outdir='./'
    - prefix and outdir MUST match the source pw.x run exactly.

    [Output]
    - Always set filband='<prefix>.band' (or a fixed name 'bands.dat').
    - If lsym is supported: set lsym=.false. (do not apply symmetry operations to a k-path).
"""

dosx_requirement_template = """
    [Purpose]
    - dos.x is post-processing only. It reads data produced by pw.x (usually calculation='nscf') to calculate the Density of States.

    [Prerequisite]
    - A completed pw.x run (nscf recommended) MUST exist with the same prefix/outdir that dos.x will read.

    [Input file shape]
    - The input MUST contain exactly one namelist: &DOS ... /
    - Only output one script block with &DOS section inside the <scripts> ... </scripts> tags. You should NOT only include &projwfc or any other namelists.
    - No extra sections, no comments, no markdown.

    [Paths / reuse]
    - Always set in &DOS: prefix='<same as pw.x>', outdir='./'
    - prefix and outdir MUST match the source pw.x run exactly.

    [Output]
    - Always set fildos='<prefix>.dos' (or a fixed name 'dos.dat').
    - bz_sum='smearing' is recommended for metallic systems or general usage.
    - Do NOT include &projwfc or any other namelists. Only &DOS.
"""

ph_requirement_template = """
    ### Minimal Rules (hard constraints)
    [Purpose]
    - ph.x performs Density Functional Perturbation Theory (DFPT) calculations.
    - It computes phonon dynamical matrices and related linear-response properties.
    - This template is generic and does NOT assume Γ-only, dispersion, or Raman unless explicitly requested.

    [Prerequisite]
    - A completed pw.x SCF run MUST exist.
    - prefix and outdir MUST match the source pw.x SCF run exactly.

    [Input file shape]
    - The input MUST contain exactly one namelist: &inputph ... /
    - No extra sections, no comments, no markdown.

    [Paths / reuse]
    - Always set in &inputph: prefix='<same as pw.x>', outdir='./'
    - prefix/outdir MUST be reused exactly from the SCF run.

    [q-point specification]
    - To avoid format errors, prefer using the "Automatic Grid" mode (ldisp = .true.):
        * ldisp = .true.
        * nq1 = 1, nq2 = 1, nq3 = 1  (For Γ-point only)
        * nq1, nq2, nq3 > 1          (For Dispersion / Grid)
    - If you use ldisp = .false., you MUST append the q-point coordinates (e.g., '0.0 0.0 0.0') after the namelist.

    [Convergence]
    - You MUST set tr2_ph (DFPT self-consistency threshold).
    - tr2_ph should be stricter than the SCF electronic convergence.

    [Outputs]
    - You MUST set fildyn to specify the dynamical-matrix output filename.
    - If a uniform q-mesh is used (ldisp = .true.), multiple dynamical matrices will be written.

    [Optional controls]
    - asr may be set to enforce the acoustic sum rule. NOTICE: asr is a LOGICAL flag (.true. or .false.) in ph.x.
    - recover may be enabled to allow restart of interrupted runs.

    [Forbidden features unless explicitly requested]
    - Do NOT set raman=.true. unless Raman intensities are explicitly required.
    - Do NOT set elph=.true. unless electron-phonon coupling is explicitly required.
"""

evx_requirement_template = """
    ### Minimal Rules (hard constraints)
    [Purpose]
    - ev.x is post-processing only.
    - It fits the equation of state (E–V curve) using total energies computed by pw.x.
    - Typical use: extract equilibrium volume, bulk modulus, and its pressure derivative.

    [Prerequisite]
    - A set of completed pw.x SCF calculations at different cell volumes MUST exist.
    - All SCF runs MUST:
        * Use the same prefix.
        * Use the same pseudopotentials, cutoffs, k-point mesh, and occupations.
        * Differ ONLY in lattice volume (uniform scaling).

    [Input file shape]
    - The input MUST contain exactly one namelist: &EV ... /
    - No extra sections, no comments, no markdown.

    [Energy–volume data]
    - You MUST provide:
        * A list of volumes and corresponding total energies, OR
        * A filename containing volume–energy pairs.
    - Energies MUST be in Ry.
    - Volumes MUST be in (Bohr)^3.

    [Paths / reuse]
    - If reading from file:
        * The file MUST be accessible from the working directory.
    - prefix/outdir are NOT used by ev.x unless explicitly required by the input format.

    [Equation of state]
    - You MUST specify the equation-of-state fitting form.
    - Common choices include:
        * Birch–Murnaghan
        * Murnaghan
        * Vinet

    [Output]
    - ev.x outputs:
        * Equilibrium volume.
        * Bulk modulus.
        * Pressure derivative of the bulk modulus.
    - The fitted parameters MUST be extracted from stdout.

    [Forbidden usage]
    - Do NOT run ev.x without a consistent volume scan.
    - Do NOT mix energies from different pseudopotentials, k-meshes, or cutoffs.
"""

# Legacy requirement text captured from the previous `tool/QE_setup.py`
_legacy_pw_requirement_comment = """
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
    - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
    - {PseudoDojo_dir}
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

vc_relax_parse_requirement = """
    - final structure of all atoms (final CELL_PARAMETERS and ATOMIC_POSITIONS in the output)
    - k point mesh used (should not change between vc-relax and scf)
"""

scf_parse_requirement = """
    - scf total energy (in Ry)
    - the number of Kohn-Sham states for band structure calculation
"""

nscf_parse_requirement = """
    - nscf total energy (in Ry)
"""

pw_bands_parse_requirement = """
    - Confirm that the pw.x bands calculation finished successfully ("JOB DONE").
    - Confirm that eigenvalues were produced along the specified k-path.
    - If band gap, VBM, or CBM cannot be determined, set them to null.
"""

bandsx_parse_requirement = """
    - Confirm that bands.x finished successfully and produced the band data file.
    - Extract the band output filename if present, otherwise set to null.
"""

dosx_parse_requirement = """
    - Confirm that dos.x finished successfully (look for "JOB DONE").
    - Extract the DOS output filename if present, otherwise set to null.
"""

dosx_parse_requirement = """
    - Confirm that dos.x finished successfully (look for "JOB DONE").
    - Extract filename if present, otherwise set to null
"""

evx_parse_requirement = """
    - Confirm that ev.x finished successfully.
    - Extract the equilibrium volume.
    - Extract the bulk modulus.
    - Extract the pressure derivative of the bulk modulus if present.
"""

def _format_pseudo_dir_instructions(pseudo_paths: "PseudoPaths") -> str:
    return (
        f"- In &control, you MUST set pseudo_dir according to the functional.\n"
        f"  * Default (no spin-orbit coupling):\n"
        f"    - If functional=LDA, set pseudo_dir={pseudo_paths.LDA}.\n"
        f"    - If functional=PBE, set pseudo_dir={pseudo_paths.PBE}.\n"
        f"    - If functional=PBEsol, set pseudo_dir={pseudo_paths.PBESOL}.\n"
        f"  * With spin-orbit coupling:\n"
        f"    - If functional=PBE, set pseudo_dir={pseudo_paths.PBE_FR}.\n"
        f"    - If functional=PBEsol, set pseudo_dir={pseudo_paths.PBESOL_FR}."
    )

def get_pw_requirement(pseudo_paths: "PseudoPaths") -> str:
    instructions = _format_pseudo_dir_instructions(pseudo_paths)
    return pw_requirement_template.format(pseudo_dir_instructions=instructions).strip()

def get_bandsx_requirement() -> str:
    return bandsx_requirement_template.strip()

def get_dosx_requirement() -> str:
    return dosx_requirement_template.strip()

def get_ph_requirement() -> str:
    return ph_requirement_template.strip()

def get_evx_requirement() -> str:
    return evx_requirement_template.strip()

_PARSE_REQUIREMENTS = {
    "scf": scf_parse_requirement,
    "vc-relax": vc_relax_parse_requirement,
    "nscf": nscf_parse_requirement,
    "pw_bands": pw_bands_parse_requirement,
    "bandsx": bandsx_parse_requirement,
    "dosx": dosx_parse_requirement,
    "evx": evx_parse_requirement,
}


def get_parse_requirement(key: Optional[str]) -> str:
    if not key:
        return ""
    return _PARSE_REQUIREMENTS.get(key, "")
