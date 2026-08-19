"""Deterministic Quantum ESPRESSO input-syntax checklist.

The schemas below are transcribed from the official QE 7.5 input manuals for
the executables supported by TritonDFT.  This module deliberately checks only
syntax: namelist ownership, known variable names, scalar lexical types, and
documented enumerations.  Scientific and producer/consumer checks remain in
``validation.workflow``.

Official references:
  https://www.quantum-espresso.org/Doc/INPUT_PW.html
  https://www.quantum-espresso.org/Doc/INPUT_BANDS.html
  https://www.quantum-espresso.org/Doc/INPUT_DOS.html
  https://www.quantum-espresso.org/Doc/INPUT_PH.html
  https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html
  https://www.quantum-espresso.org/Doc/INPUT_PP.html
  https://www.quantum-espresso.org/Doc/INPUT_Q2R.html
  https://www.quantum-espresso.org/Doc/INPUT_MATDYN.html
  https://www.quantum-espresso.org/Doc/INPUT_DYNMAT.html
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple


QE_SYNTAX_REFERENCE_VERSION = "7.5"


def _words(value: str) -> FrozenSet[str]:
    return frozenset(value.lower().split())


# Array variables are stored by their base name: amass(1), starting_magnetization(2),
# and celldm(3) are consequently validated without enumerating indices.
PW_NAMELISTS: Mapping[str, FrozenSet[str]] = {
    "control": _words("""
        calculation title verbosity restart_mode wf_collect nstep iprint tstress
        tprnfor dt outdir wfcdir prefix lkpoint_dir max_seconds etot_conv_thr
        forc_conv_thr disk_io pseudo_dir tefield dipfield lelfield nberrycyc
        lorbm lberry gdir nppstr gate twochem lfcp trism
    """),
    "system": _words("""
        ibrav celldm a b c cosab cosac cosbc nat ntyp nbnd nbnd_cond
        tot_charge starting_charge tot_magnetization starting_magnetization
        ecutwfc ecutrho ecutfock nr1 nr2 nr3 nr1s nr2s nr3s nosym nosym_evc
        noinv no_t_rev force_symmorphic use_all_frac occupations
        one_atom_occupations starting_spin_angle degauss_cond nelec_cond
        degauss smearing nspin sic_gamma pol_type sic_energy sci_vb sci_cb
        noncolin ecfixed qcutz q2sigma input_dft ace exx_fraction
        screening_parameter exxdiv_treatment x_gamma_extrapolation ecutvcut
        nqx1 nqx2 nqx3 localization_thr hubbard_occ hubbard_beta
        starting_ns_eigenvalue dmft dmft_prefix ensemble_energies edir emaxpos
        eopreg eamp angle1 angle2 lforcet constrained_magnetization
        fixed_magnetization lambda report lspinorb assume_isolated esm_bc esm_w
        esm_efield esm_nfit lgcscf gcscf_mu gcscf_conv_thr gcscf_beta
        vdw_corr london london_s6 london_c6 london_rvdw london_rcut
        dftd3_version dftd3_threebody ts_vdw_econv_thr ts_vdw_isolated xdm
        xdm_a1 xdm_a2 space_group uniqueb origin_choice rhombohedral zgate
        relaxz block block_1 block_2 block_height nextffield lda_plus_u
        lda_plus_u_kind hubbard_u hubbard_j0 hubbard_alpha hubbard_j
        starting_ns_eigenvalue
    """),
    "electrons": _words("""
        electron_maxstep exx_maxstep scf_must_converge conv_thr adaptive_thr
        conv_thr_init conv_thr_multi mixing_mode mixing_beta mixing_ndim
        mixing_fixed_ns diagonalization diago_thr_init diago_cg_maxiter
        diago_david_ndim diago_rmm_ndim diago_rmm_conv diago_gs_nblock
        diago_full_acc efield efield_cart efield_phase startingpot startingwfc
        tqr real_space
    """),
    "ions": _words("""
        ion_positions ion_velocities ion_dynamics pot_extrapolation
        wfc_extrapolation remove_rigid_rot ion_temperature tempw fnosep nhpcl
        nhptyp nhgrp fnhscl ndega tolp delta_t nraise refold_pos upscale
        bfgs_ndim tgdiis_step trust_radius_max trust_radius_min trust_radius_ini
        w_1 w_2 fire_alpha_init fire_falpha fire_nmin fire_f_inc fire_f_dec
        fire_dtmax
    """),
    "cell": _words("cell_dynamics press wmass cell_factor press_conv_thr cell_dofree"),
}

OTHER_SCHEMAS: Mapping[str, Mapping[str, FrozenSet[str]]] = {
    "bands.x": {"bands": _words("prefix outdir filband spin_component lsigma lp filp lsym no_overlap plot_2d firstk lastk")},
    "dos.x": {"dos": _words("prefix outdir bz_sum ngauss degauss emin emax deltae fildos")},
    "projwfc.x": {"projwfc": _words("prefix outdir ngauss degauss emin emax deltae lsym diag_basis pawproj filpdos filproj lwrite_overlaps lbinary_data kresolveddos tdosinboxes n_proj_boxes irmin irmax plotboxes")},
    "ph.x": {"inputph": _words("""
        amass outdir prefix niter_ph tr2_ph alpha_mix nmix_ph verbosity reduce_io
        max_seconds dftd3_hess fildyn fildrho fildvscf epsil lrpa lnoloc trans
        lraman lmultipole eth_rps eth_ns dek recover low_directory_check only_init
        qplot q2d q_in_band_form electron_phonon el_ph_nsigma el_ph_sigma ahc_dir
        ahc_nbnd ahc_nbndskip skip_upper lshift_q zeu zue elop fpol ldisp nogg
        asr ldiag lqdir search_sym nq1 nq2 nq3 nk1 nk2 nk3 k1 k2 k3
        diagonalization read_dns_bare ldvscf_interpolate wpot_dir do_long_range
        do_charge_neutral start_irr last_irr nat_todo modenum start_q last_q
        dvscf_star drho_star
    """)},
    "pp.x": {
        "inputpp": _words("title prefix outdir filplot plot_num spin_component emin emax delta_e degauss_ldos use_gauss_ldos sample_bias kpoint kband lsign nc n0"),
        "plot": _words("nfile filepp weight iflag output_format fileout interpolation e1 e2 e3 x0 nx ny nz radius"),
    },
    "q2r.x": {"input": _words("fildyn flfrc zasr loto_2d write_lr")},
    "matdyn.x": {"input": _words("flfrc asr huang dos nk1 nk2 nk3 deltae ndos degauss fldos flfrq flvec fleig fldyn at l1 l2 l3 ntyp amass readtau fltau la2f q_in_band_form q_in_cryst_coord eigen_similarity fd na_ifc nosym loto_2d loto_disable read_lr write_frc")},
    "dynmat.x": {"input": _words("fildyn q amass asr remove_interaction_blocks axis lperm lplasma filout fileig filmol filxsf loto_2d el_ph_nsig el_ph_sigma")},
}

REQUIRED_NAMELISTS: Mapping[str, Tuple[str, ...]] = {
    "pw.x": ("control", "system", "electrons"),
    "bands.x": ("bands",),
    "dos.x": ("dos",),
    "projwfc.x": ("projwfc",),
    "ph.x": ("inputph",),
    "pp.x": ("inputpp",),
    "q2r.x": ("input",),
    "matdyn.x": ("input",),
    "dynmat.x": ("input",),
}

ENUMS: Mapping[Tuple[str, str, str], FrozenSet[str]] = {
    ("pw.x", "control", "calculation"): _words("scf nscf bands relax md vc-relax vc-md"),
    ("pw.x", "control", "verbosity"): _words("low high debug medium default minimal"),
    ("pw.x", "control", "restart_mode"): _words("from_scratch restart"),
    ("pw.x", "system", "occupations"): _words("smearing tetrahedra tetrahedra_lin tetrahedra_opt fixed from_input"),
    ("pw.x", "system", "smearing"): frozenset({"gaussian", "gauss", "mp", "methfessel-paxton", "mv", "marzari-vanderbilt", "cold", "fd", "fermi-dirac"}),
    ("pw.x", "cell", "cell_dynamics"): _words("none sd damp-pr bfgs pr w"),
    ("pw.x", "cell", "cell_dofree"): frozenset({"all", "ibrav", "a", "b", "c", "fixa", "fixb", "fixc", "x", "y", "z", "xy", "xz", "yz", "xyz", "shape", "volume", "2dxy", "2dshape", "epitaxial_ab", "epitaxial_ac", "epitaxial_bc"}),
    ("pw.x", "system", "dftd3_version"): _words("2 3 4 5 6"),
    ("dos.x", "dos", "bz_sum"): _words("smearing tetrahedra tetrahedra_lin tetrahedra_opt"),
    ("q2r.x", "input", "zasr"): _words("no simple crystal one-dim zero-dim all"),
    ("matdyn.x", "input", "asr"): _words("no simple crystal one-dim zero-dim all"),
    ("dynmat.x", "input", "asr"): _words("no simple crystal one-dim zero-dim all"),
}

INTEGER_KEYS = _words("""
    nstep iprint gdir nppstr ibrav nat ntyp nbnd nbnd_cond nr1 nr2 nr3 nr1s
    nr2s nr3s nqx1 nqx2 nqx3 nspin dftd3_version origin_choice electron_maxstep
    exx_maxstep mixing_ndim diago_cg_maxiter diago_david_ndim diago_rmm_ndim
    diago_gs_nblock niter_ph nmix_ph nq1 nq2 nq3 nk1 nk2 nk3 k1 k2 k3
    start_irr last_irr nat_todo modenum start_q last_q ngauss firstk lastk
    spin_component plot_num kpoint kband nc n0 nfile iflag output_format nx ny nz
    ndos ntyp l1 l2 l3 n_proj_boxes irmin irmax axis el_ph_nsig
""")

LOGICAL_KEYS = _words("""
    tstress tprnfor tefield dipfield lelfield lorbm lberry gate twochem lfcp trism
    nosym nosym_evc noinv no_t_rev force_symmorphic use_all_frac
    one_atom_occupations starting_spin_angle noncolin x_gamma_extrapolation dmft
    ensemble_energies lforcet lspinorb lgcscf london dftd3_threebody
    ts_vdw_isolated xdm uniqueb rhombohedral zgate relaxz scf_must_converge
    adaptive_thr mixing_fixed_ns diago_rmm_conv diago_full_acc tqr real_space
    remove_rigid_rot refold_pos lsigma lp lsym no_overlap plot_2d reduce_io
    dftd3_hess epsil lrpa lnoloc trans lraman lmultipole recover
    low_directory_check only_init qplot q2d q_in_band_form lshift_q zeu zue elop
    fpol ldisp nogg ldiag lqdir search_sym read_dns_bare ldvscf_interpolate
    do_long_range do_charge_neutral dvscf_star drho_star diag_basis pawproj
    lwrite_overlaps lbinary_data kresolveddos tdosinboxes plotboxes use_gauss_ldos
    lsign loto_2d write_lr huang dos readtau la2f q_in_cryst_coord fd na_ifc
    loto_disable read_lr write_frc remove_interaction_blocks lperm lplasma
""")


@dataclass(frozen=True)
class SyntaxFinding:
    code: str
    message: str


_NAMELIST_RE = re.compile(r"(?mis)^\s*&([a-z][a-z0-9_]*)\b(.*?)^\s*/\s*$")
_ASSIGNMENT_RE = re.compile(
    r"(?mi)^\s*([a-z][a-z0-9_]*(?:\s*\([^=\n]*\))?)\s*=\s*"
    r"('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|[^,!\n/]+)"
)


def _base_key(raw: str) -> str:
    return re.sub(r"\s*\(.*\)\s*$", "", raw.strip().lower())


def _bare_value(raw: str) -> str:
    return raw.strip().strip("'\"").strip().lower()


def _schema(exec_name: str) -> Mapping[str, FrozenSet[str]]:
    return PW_NAMELISTS if exec_name == "pw.x" else OTHER_SCHEMAS.get(exec_name, {})


def validate_qe_syntax(path: str, exec_name: str) -> List[SyntaxFinding]:
    """Validate one QE input against the official-document checklist."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    findings: List[SyntaxFinding] = []
    schema = _schema(exec_name)
    if not schema:
        return findings

    matches = list(_NAMELIST_RE.finditer(text))
    present = [match.group(1).lower() for match in matches]
    for required in REQUIRED_NAMELISTS.get(exec_name, ()):
        if required not in present:
            findings.append(SyntaxFinding("QE_NAMELIST_REQUIRED", f"{exec_name} requires &{required.upper()}."))
    for name in present:
        if name not in schema:
            findings.append(SyntaxFinding("QE_NAMELIST_UNKNOWN", f"&{name.upper()} is not documented for {exec_name}."))
    if len(present) != len(set(present)):
        findings.append(SyntaxFinding("QE_NAMELIST_DUPLICATE", "A namelist appears more than once."))
    documented_order = list(schema)
    recognized = [name for name in present if name in schema]
    if recognized != sorted(recognized, key=documented_order.index):
        findings.append(SyntaxFinding(
            "QE_NAMELIST_ORDER_INVALID",
            f"{exec_name} namelists must follow the documented order: "
            + ", ".join(f"&{name.upper()}" for name in documented_order) + ".",
        ))

    # A leading ampersand without a matching slash is otherwise easy to miss.
    starts = re.findall(r"(?mi)^\s*&([a-z][a-z0-9_]*)\b", text)
    if len(starts) != len(matches):
        findings.append(SyntaxFinding("QE_NAMELIST_UNTERMINATED", "At least one namelist is missing its terminating '/'."))

    for match in matches:
        name = match.group(1).lower()
        allowed = schema.get(name)
        if allowed is None:
            continue
        body = match.group(2)
        for assignment in _ASSIGNMENT_RE.finditer(body):
            raw_key, raw_value = assignment.groups()
            key = _base_key(raw_key)
            if key not in allowed:
                owners = [candidate for candidate, keys in schema.items() if key in keys]
                if owners:
                    findings.append(SyntaxFinding(
                        "QE_KEYWORD_WRONG_NAMELIST",
                        f"{key} is in &{name.upper()}, but the official {exec_name} input schema places it in "
                        + "/".join(f"&{owner.upper()}" for owner in owners) + ".",
                    ))
                else:
                    findings.append(SyntaxFinding("QE_KEYWORD_UNKNOWN", f"{key} is not documented for {exec_name} &{name.upper()}."))
                continue
            value = _bare_value(raw_value)
            choices = ENUMS.get((exec_name, name, key))
            enum_valid = value in choices if choices is not None else True
            if key == "cell_dofree" and "+" in value and choices is not None:
                enum_valid = all(part in choices for part in value.split("+"))
            if choices is not None and not enum_valid:
                findings.append(SyntaxFinding(
                    "QE_ENUM_INVALID",
                    f"{key}={raw_value.strip()} is invalid; allowed values are: {', '.join(sorted(choices))}.",
                ))
            if key in INTEGER_KEYS and not re.fullmatch(r"[+-]?\d+", value):
                findings.append(SyntaxFinding("QE_INTEGER_INVALID", f"{key} requires an integer, found {raw_value.strip()}."))
            if key in LOGICAL_KEYS and value not in {".true.", ".false.", "true", "false", "t", "f"}:
                findings.append(SyntaxFinding("QE_LOGICAL_INVALID", f"{key} requires a Fortran logical value, found {raw_value.strip()}."))

    return findings


def remove_undocumented_namelist_keywords(path: str, exec_name: str) -> List[str]:
    """Remove generated assignments that are absent from an executable's schema.

    This intentionally handles only unknown keys. A key belonging to another
    namelist is not moved automatically because relocation can change meaning.
    """
    schema = _schema(exec_name)
    if not schema:
        return []
    source = Path(path)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    current = ""
    repaired: List[str] = []
    output: List[str] = []
    for line in lines:
        start = re.match(r"^\s*&([a-z][a-z0-9_]*)\b", line, re.I)
        if start:
            current = start.group(1).lower()
            output.append(line)
            continue
        if current and re.match(r"^\s*/\s*(?:!.*)?$", line):
            current = ""
            output.append(line)
            continue
        assignment = re.match(r"^\s*([a-z][a-z0-9_]*(?:\s*\([^=\n]*\))?)\s*=", line, re.I)
        allowed = schema.get(current)
        if assignment and allowed is not None:
            key = _base_key(assignment.group(1))
            if key not in allowed and not any(key in keys for keys in schema.values()):
                repaired.append(f"Removed undocumented {exec_name} &{current.upper()} keyword: {key}")
                continue
        output.append(line)
    if repaired:
        source.write_text("\n".join(output).rstrip() + "\n", encoding="utf-8")
    return repaired


def force_from_scratch(path: str, exec_name: str) -> List[str]:
    """Rewrite restart_mode='restart' to 'from_scratch' on a newly generated step.

    Deterministic repair rather than a rejection, because there is no judgement
    involved: a step the agent has just written has no checkpoint of its own to
    resume from, so 'from_scratch' is the only correct value, always.

    The model reaches for 'restart' by confusing two different things — reusing
    a previous SCF's charge density (which is what calculation='nscf' plus a
    shared prefix/outdir does) with resuming an interrupted run of THIS step
    (which is what restart_mode governs). Left in place, QE looks for wavefunction
    and XML checkpoints that either do not exist or belong to the SCF step, and
    the failure is sometimes a crash and sometimes silently wrong numbers.

    Returns a list of human-readable repairs, empty if nothing was changed.
    """
    if os.path.basename(exec_name or "") not in ("pw.x", "ph.x"):
        return []
    try:
        text = Path(path).read_text(errors="ignore")
    except OSError:
        return []

    pattern = re.compile(
        r"(?mi)^(\s*restart_mode\s*=\s*)(['\"])\s*restart\s*\2(.*)$"
    )
    if not pattern.search(text):
        return []
    fixed = pattern.sub(lambda m: f"{m.group(1)}'from_scratch'{m.group(3)}", text)
    try:
        Path(path).write_text(fixed, encoding="utf-8")
    except OSError:
        return []
    return [f"{os.path.basename(path)}: restart_mode 'restart' -> 'from_scratch'"]
