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
    - For ibrav=0, you MUST choose exactly ONE of the following lattice encodings:
    (A) Recommended: use explicit angstrom lattice vectors:
        - Put CELL_PARAMETERS (angstrom) and provide 3×3 lattice vectors in Å.
        - Do NOT set celldm(1).
    (B) Alternative: use alat-scaled lattice vectors:
        - Set celldm(1)=<alat in Bohr> in &system.
        - Put CELL_PARAMETERS (alat) with dimensionless 3×3 coefficients.
        - If CELL_PARAMETERS (alat) is used but celldm(1) is missing, it is INVALID.

    [Energy cutoffs]
    - Always infer cutoffs from pseudopotential quality, not use defaults.
    - Typical ranges (plane-wave basis):
        * PseudoDojo / SSSP standard: ecutwfc = 60-120 Ry.
        * GBRV ultrasoft: ecutwfc ≈ 40-60 Ry.
    - If pseudopotential family unknown, start from ecutwfc = 80 Ry.

    [Calculation Logic]
    - IF calculation = 'vc-relax' OR 'vc-md':
        1. You MUST include the &CELL namelist.
        2. Inside &CELL, set cell_dynamics (e.g., 'bfgs') and press (target pressure).

    [Occupations & Smearing]
    - IF occupations = 'smearing':
        1. You MUST specify smearing type (e.g., smearing='gaussian', 'mv', 'mp').
        2. You MUST specify degauss (broadening) in Ry (e.g., degauss=0.01).
        3. Do NOT use the keyword 'sigma'.
    - IF occupations = 'fixed' (insulators):
        1. Do NOT specify smearing or degauss.

    [band calculations]
    - nbnd (CRITICAL): when calculation = 'nscf' OR 'bands', you MUST include
      enough empty states. Use occupied Kohn-Sham states + 16 by default (preserve
      a larger user value). Occupied states are ceil(nelec/2) for collinear
      calculations, but ceil(nelec) for noncollinear/SOC calculations. Without it pw.x computes only
      the occupied bands, so there is NO conduction band and neither a band gap
      nor a band structure can be obtained.
    - This pipeline gives the two run types separate jobs. Quantum ESPRESSO itself
      would accept a k-path in an nscf run (bands.x can post-process either), but
      here we keep the roles distinct so each step has one purpose and the band gap
      always comes from a proper Brillouin-zone sampling:
        * calculation = 'nscf'  -> UNIFORM grid. Use "K_POINTS automatic", denser
          than the SCF grid. This is the step that yields the band gap. Do not put
          a high-symmetry path here.
        * calculation = 'bands' -> HIGH-SYMMETRY PATH. Use "K_POINTS crystal_b"
          (see the BANDS section below), never 'automatic'. This is the step
          bands.x post-processes into a dispersion plot.
    - For a band GAP on a semiconductor/insulator, set occupations='fixed' so pw.x
      prints "highest occupied, lowest unoccupied level". With occupations='smearing'
      it prints only a Fermi energy and the gap cannot be read off.

    [Atomic positions]
    - Header MUST be exactly: ATOMIC_POSITIONS (crystal)
    - Each atomic line MUST be: <species> <x> <y> <z>
    - Ensure nat equals the number of position lines.
    - Ensure ntyp equals the number of unique species labels. ATOMIC_SPECIES must contain exactly those species, one line per species.

    [Coordinate precision — this decides how fast the run is]
    - Copy the given coordinates EXACTLY. Write at least 10 decimal places, in
      both ATOMIC_POSITIONS and CELL_PARAMETERS.
    - Never round to 4 decimals. Writing 0.6667 where the value is 2/3 is an
      error of 3.3e-5, which exceeds Quantum ESPRESSO's symmetry tolerance
      (accep = 1e-5). QE then fails to recognise the space group: measured on
      hexagonal MoS2, this cut the detected symmetry operations from 24 to 8,
      inflated the irreducible k-points from ~40 to 129, and made the relaxation
      roughly three times slower for an identical calculation.
    - Coordinates that are exact fractions must be written out: 1/3 ->
      0.3333333333, 2/3 -> 0.6666666667, 1/6 -> 0.1666666667.
    - Padding a rounded value with zeros does NOT satisfy this. 0.6667000000 has
      ten decimals but is still the 3.3e-5 error above; the digits must be the
      real ones (0.6666666667), not a short value stretched to length.
    - This costs nothing and is never a scientific choice: the numbers are given
      to you, so transcribe them at full precision.

    [Pseudopotentials]
    - You must set pseudo_dir according to the following rules, or your run will be completely invalid:
    - {pseudo_dir_instructions}
    - pseudo filenames in ATOMIC_SPECIES MUST be <element_lowercase>.upf (e.g., na.upf).
    - The number of ATOMIC_SPECIES entries MUST match ntyp exactly.

    [Filenames / reuse]
    - Set &control prefix='qerun'. Every step of this workflow MUST use this exact
      same prefix — nscf/bands read the SCF charge density from <outdir>/<prefix>.save,
      and bands.x reads the bands run's wavefunctions from the same place. Do NOT
      invent a per-step prefix such as 'system_0' / 'system_1'.
    - Always set in &control: outdir='./' and wfcdir='./'.

    [Convergence thresholds]
    - Put conv_thr ONLY in &electrons.
    - Put etot_conv_thr and forc_conv_thr ONLY in &control (ionic/cell minimization energy threshold).

    [K_POINTS offsets — VC-relax / SCF / NSCF only]
    - Applies ONLY when calculation = 'vc-relax', 'scf' or 'nscf'.
    - Syntax:
        K_POINTS automatic
        nk1 nk2 nk3 k1 k2 k3
    - Offset parity only:
        * even → 0  (Γ-centered)
        * odd  → 1  (half-grid shift)
    - Recommended defaults:
        * Insulators / semiconductors (occupations='fixed'):
            use 0 0 0.
        * Metals (occupations='smearing'):
            use 1 1 1.
    - Symmetry-aware tweaks:
        * High-symmetry cubic (fcc / bcc / sc):
            0 0 0 (fewer irreducible k-points, faster).
        * Low-symmetry or metallic systems:
            1 1 1 (reduces Γ singularities).
        * 2D slabs with nk3 = 1 (vacuum along c):
            use 0 0 0 (do NOT shift along the non-sampled direction).
    - Keep offsets consistent across axes unless there is a clear physical reason (e.g., strong anisotropy).

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
    - Dynamically decide whether vdW, magnetism, spin-orbit coupling, or DFT+U is relevant from the requested phase, dimensionality, material, and observable. Do not use fixed material-specific defaults.
    - When one of these choices is relevant, make its scope explicit and preserve it exactly within dependent steps of that branch. Do not force an electronic-only SOC refinement into vc-relax or the scalar-relativistic baseline. A SOC branch reuses geometry but requires its own fully relativistic SCF/save state.
    - A vdW choice is a complete contract: preserve vdw_corr, dftd3_version, damping/version semantics, dftd3_threebody, and any explicitly supplied D3/TS/XDM parameters exactly across relaxation and all electronic branches.
    - A bulk request must remain three-dimensionally periodic: do not add slab vacuum, dipole corrections, or isolated-boundary settings.
    - For AFM/ferrimagnetic order, remember that opposite initial moments on atoms of the same element may require distinct QE species labels that reference the same pseudopotential.
    - SOC requires compatible fully relativistic pseudopotentials and consistent noncollinear/SOC settings throughout the dependent workflow.
    - Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling.

    [DFT+U / Hubbard corrections — version-sensitive]
    - Hubbard input syntax is version-sensitive. Use the remote QE version supplied in workflow memory; never assume the local package version equals the cluster version.
    - QE 7.1 and newer use a HUBBARD card placed AFTER the namelists, format:
        HUBBARD (ortho-atomic)
        U <species>-<manifold> <value_in_eV>
      Example for Fe 3d with U = 4.5 eV:
        HUBBARD (ortho-atomic)
        U Fe-3d 4.5
    - The projector in parentheses is the Hubbard projector type; prefer
      'ortho-atomic' (use 'atomic' only if orthogonalization is problematic).
    - The manifold is element-symbol + orbital (e.g. Fe-3d, Mn-3d, Ni-3d, Ce-4f, V-3d).
    - QE 6.7 and older use the legacy lda_plus_u/Hubbard_U(i) &SYSTEM syntax and do not accept the new HUBBARD card.
    - If the remote version is unknown and DFT+U is required, explicitly record that version compatibility must be confirmed by the remote parser probe. Never mix old and new syntax in one input.
    - starting_magnetization, nspin, and occupations stay in &SYSTEM for all supported versions.

    [Numeric values — literals only (CRITICAL)]
    - NEVER write an arithmetic expression as an input value. Fortran namelist
      parsing treats '/' as the namelist TERMINATOR, so e.g.
      celldm(1)=2.87/0.529177 silently ENDS the &system namelist early and
      corrupts the entire input (later keywords and cards get ignored).
    - Every numeric value MUST be a single pre-computed literal — no '/', '*',
      '+', '-' operators, and no inline unit conversions.
    - Lattice parameter: either give celldm(1) in BOHR as a plain number
      (compute the Angstrom→Bohr conversion yourself: 1 Angstrom = 1.8897259886 Bohr),
      OR specify the cell in ANGSTROM using A=<value> (and B=, C=, cosAB=, ... as
      needed) in &system. Do NOT divide an Angstrom value by 0.529 inline.

    [Header formatting & discipline]
    - Section headers must be single-line with qualifiers (e.g., "ATOMIC_POSITIONS (crystal)", "CELL_PARAMETERS (alat)", "K_POINTS automatic").
    - ecutwfc/ecutrho/occupations MUST be inside &system.
    - conv_thr (and related electron-iteration knobs) MUST be inside &electrons.
    - Do NOT place NSCF/SCF-specific knobs outside their proper sections.
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
    - Always set in &BANDS: prefix='qerun', outdir='./'
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
    - Emin, Emax, and DeltaE are in eV. Unless the user explicitly requests a
      narrower energy interval, use a conservative full-DOS window of
      Emin=-15.0 eV, Emax=10.0 eV, and DeltaE=0.01 eV. Never convert these
      three values to Ry and never generate a total window narrower than 5 eV.
    - bz_sum MUST match the dense NSCF integration family exactly:
        * occupations='tetrahedra'     -> bz_sum='tetrahedra'
        * occupations='tetrahedra_lin' -> bz_sum='tetrahedra_lin'
        * occupations='tetrahedra_opt' -> bz_sum='tetrahedra_opt'
        * occupations='smearing' or 'fixed' -> bz_sum='smearing'
    - Do not set ngauss or degauss when bz_sum is a tetrahedron method.
    - Do NOT include &projwfc or any other namelists. Only &DOS.
"""

projwfc_requirement_template = """
    ### Minimal Rules (hard constraints)
    [Purpose]
    - projwfc.x reads dense uniform-grid NSCF wavefunctions and produces atomic-orbital projections for PDOS aggregation.
    [Input]
    - Output exactly one &PROJWFC namelist and no other namelists or prose.
    - Set prefix and outdir to exactly the dense DOS NSCF values.
    - Set filpdos='<prefix>.pdos' so downstream code can discover QE's per-atom/per-wavefunction files.
    - Set DeltaE in eV at a resolution appropriate for the requested plot.
    [Spin/orbitals]
    - projwfc.x writes all available atomic-wavefunction projections. Downstream deterministic aggregation separates the user-requested species and orbital channels.
    - Preserve the NSCF spin, SOC, Hubbard, pseudopotential, and magnetic-species setup through the shared prefix/outdir state.
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
    - You MUST set tr2_ph (DFPT self-consistency threshold). Use 1.0d-12, the
      Quantum ESPRESSO default, unless the query explicitly asks for higher
      accuracy.
    - Do NOT try to make tr2_ph "stricter than conv_thr": they measure different
      things (tr2_ph is a threshold on the perturbed charge density, conv_thr on
      the total energy), so there is no ordering to respect. Chasing conv_thr
      pushes tr2_ph to values like 1.0d-14, which triples the run time for no
      gain in the phonon frequencies.

    [Cost budget — phonons are the most expensive step by far]
    - A DFPT run costs roughly 40x an SCF *per irreducible q-point*, and jobs here
      run under a wall-clock limit. Measured on 2-atom silicon: a 4x4x4 q-mesh did
      not finish in 30 minutes, while a 2x2x2 mesh completed in 23 seconds and
      still reproduced the 520 cm-1 optical mode.
    - Default to nq1=nq2=nq3=2 for a dispersion. q2r.x + matdyn.x interpolate the
      force constants onto a fine q-path afterwards, so a small mesh still yields
      a smooth dispersion curve. Only go denser if the query explicitly asks for a
      converged/high-accuracy phonon calculation.
    - For a Gamma-only calculation (stability check, Raman/IR mode listing) use
      nq1=nq2=nq3=1 — do not run a full mesh.

    [Outputs]
    - You MUST set fildyn to specify the dynamical-matrix output filename.
    - If a uniform q-mesh is used (ldisp = .true.), multiple dynamical matrices will be written.

    [Optional controls]
    - asr may be set to enforce the acoustic sum rule. NOTICE: asr is a LOGICAL flag (.true. or .false.) in ph.x.
    - recover may be enabled to allow restart of interrupted runs.

    [Raman and dielectric response]
    - The valid Quantum ESPRESSO keyword is lraman, never raman.
    - If this subproblem explicitly requests Raman coefficients, it MUST be a separate Gamma calculation with lraman=.true., ldisp=.false., trans=.true., and the Gamma q point after the namelist.
    - A uniform ldisp q-grid for phonon dispersion does not replace the separate Gamma Raman response calculation.
    - Use epsil=.true. for the appropriate nonmetal Gamma dielectric/Born-charge response when IR properties are requested; do not enable it blindly for metals.

    [Forbidden features unless explicitly requested]
    - Do NOT set lraman=.true. unless Raman coefficients are explicitly required by this subproblem.
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

def get_projwfc_requirement() -> str:
    return projwfc_requirement_template.strip()

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
