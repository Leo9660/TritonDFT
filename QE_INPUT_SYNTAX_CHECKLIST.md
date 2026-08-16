# Quantum ESPRESSO input syntax checklist

This checklist is enforced by `src/validation/qe_syntax.py` before TritonDFT
opens an SSH connection or submits a job. It is intentionally separate from
scientific validation: passing this checklist means that the input follows the
documented QE grammar, not that the physical model is appropriate.

Reference schema version: Quantum ESPRESSO 7.5. The remote QE parser probe
remains authoritative for older cluster installations and version-sensitive
features such as Hubbard syntax.

## Checks applied to every supported executable

- The required namelist exists and is terminated by `/`.
- No undocumented or duplicate namelist is present.
- Namelists occur in the order required by the official manual.
- Every parsed keyword belongs to the namelist in which it appears.
- Unknown keywords are rejected rather than silently passed to QE.
- Indexed variables are checked by base name, for example `amass(1)` and
  `starting_magnetization(2)`.
- Documented integer and logical values have valid Fortran lexical forms.
- Closed enumerations are checked, including `calculation`, `restart_mode`,
  `occupations`, `smearing`, `cell_dynamics`, `cell_dofree`,
  `dftd3_version`, `bz_sum`, `asr`, and `zasr`.
- Existing executable-specific card and row checks in `workflow.py` run after
  this checklist.

## Official manuals covered

- [`pw.x` (`INPUT_PW`)](https://www.quantum-espresso.org/Doc/INPUT_PW.html):
  `&CONTROL`, `&SYSTEM`, `&ELECTRONS`, `&IONS`, and `&CELL`, plus the existing
  structure and k-point card checks.
- [`bands.x` (`INPUT_BANDS`)](https://www.quantum-espresso.org/Doc/INPUT_BANDS.html):
  `&BANDS`.
- [`dos.x` (`INPUT_DOS`)](https://www.quantum-espresso.org/Doc/INPUT_DOS.html):
  `&DOS`.
- [`projwfc.x` (`INPUT_PROJWFC`)](https://www.quantum-espresso.org/Doc/INPUT_PROJWFC.html):
  `&PROJWFC`.
- [`ph.x` (`INPUT_PH`)](https://www.quantum-espresso.org/Doc/INPUT_PH.html):
  `&INPUTPH`; q-point layout is additionally checked by the workflow validator
  and the remote parser probe.
- [`pp.x` (`INPUT_PP`)](https://www.quantum-espresso.org/Doc/INPUT_PP.html):
  `&INPUTPP` and optional `&PLOT`.
- [`q2r.x` (`INPUT_Q2R`)](https://www.quantum-espresso.org/Doc/INPUT_Q2R.html):
  `&INPUT`.
- [`matdyn.x` (`INPUT_MATDYN`)](https://www.quantum-espresso.org/Doc/INPUT_MATDYN.html):
  `&INPUT`.
- [`dynmat.x` (`INPUT_DYNMAT`)](https://www.quantum-espresso.org/Doc/INPUT_DYNMAT.html):
  `&INPUT`.

## Failure policy

Every finding from this checklist is `FATAL`. It is shown in the Validation tab
and blocks approval, SSH connection, and submission. Typical messages are:

```text
FATAL QE_KEYWORD_WRONG_NAMELIST [bands.in]: vdw_corr is in &CONTROL, but ... &SYSTEM.
FATAL QE_ENUM_INVALID [vc-relax.in]: cell_dofree='hexagonal' is invalid ...
FATAL QE_KEYWORD_UNKNOWN [scf.in]: <keyword> is not documented ...
```

## Scope boundary

The checklist covers every QE executable currently exposed by TritonDFT. When a
new executable is added to `tool_map.py`, its official input manual and schema
must be added here before the tool can be considered syntax-validated. Runtime
capability probes are still required because the Expanse installation may be
older than the 7.5 reference documentation.
