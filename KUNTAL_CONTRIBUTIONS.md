# Kuntal's TritonDFT development contributions

This document summarizes the functionality developed on the `kuntal-version`
branch relative to `Leo9660/TritonDFT:main`. It is intended to provide a durable
record of the motivation, design work, implementation, testing, and user-facing
changes contributed while extending TritonDFT for practical local-to-HPC DFT
workflows.

The summary is organized by capability rather than commit order. Before opening
a merge request, regenerate the comparison against the current upstream branch
and move any still-uncommitted work into reviewed commits.

## 1. Remote-cluster super-user interface

- Added an interactive `DFT request>` interface for researchers who generate
  calculations locally and execute them on remote Slurm clusters.
- Added first-run configuration for API keys, SSH targets, remote work roots,
  DFT backends, pseudopotential locations, and Slurm templates.
- Added SSH alias discovery and configuration while preserving existing user
  SSH settings.
- Added persistent SSH ControlMaster support, connection testing, remote
  directory management, `rsync` upload/download, `sbatch` submission, and
  `squeue` monitoring.
- Added a `tritondft-cluster` entry point and launcher script.
- Removed the requirement for Quantum ESPRESSO to be installed on the local
  Mac when calculations are executed remotely.
- Added detailed cluster installation and troubleshooting documentation.

## 2. Cluster-neutral Slurm generation

- Added user-owned QE and VASP Slurm templates under `~/.tritondft`.
- Preserved site-specific partitions, accounts, module commands, and launchers
  instead of replacing the entire template.
- Added support for both `srun`-based and `mpirun`-based clusters.
- Prevented an `srun` template from being silently changed back to `mpirun`.
- Synchronized `--nodes`, `--tasks-per-node`, `--ntasks`, MPI ranks, and QE
  parallel flags.
- Added user-specified maximum nodes and cores per node to the planning review,
  then clamped generated resource requests to that allocation.
- Added deterministic fallback resource selection when LLM resource planning
  is unavailable or malformed.

## 3. Approval-first planning and editable inputs

- Separated scientific assessment, workflow planning, input generation, and
  execution approval.
- Added a planning/approval window with tabs for:
  - complete plan;
  - dependency graph;
  - generated input files;
  - cross-step validation;
  - available cluster resources;
  - plain-language revision requests.
- Ensured that no cluster job is submitted before the user approves the plan
  and generated inputs.
- Added plan revision without discarding the complete request.
- Added editable input review when rerunning an existing step with new
  parameters.
- Replaced anonymous names such as `input_1_1.in` with task-oriented names such
  as `vc-relax.in`, `scf.in`, `bands.in`, and `dos.in`.

## 4. Dependency-graph execution and parallel branches

- Represented workflows as a directed acyclic graph instead of a fixed linear
  list.
- Added deterministic dependency and branch inference.
- Kept required sequences such as `vc-relax -> scf -> nscf` ordered.
- Allowed independent descendants to be submitted as parallel branches after
  their common dependency completes, including bands, DOS/PDOS, Gamma Raman,
  and phonon-dispersion workflows where scientifically appropriate.
- Added a readable workflow graph that exposes sequential and parallel stages
  before approval.
- Added branch isolation so scalar-relativistic and fully relativistic/SOC
  calculations use separate directories and incompatible `.save` data cannot
  contaminate another branch.
- Added explicit producer-to-consumer artifact staging:
  - Gamma `ph.x` dynamical matrix to `dynmat.x`;
  - q-grid dynamical-matrix family to `q2r.x`;
  - real-space force constants from `q2r.x` to `matdyn.x`.
- Added pre-submission failure when a required parent artifact is unavailable.

## 5. Persistent checkpoints, recovery, and reruns

- Added persistent workflow, step, dependency, attempt, job, input-hash, and
  remote-directory state.
- Added immutable attempt directories so repairs do not overwrite the evidence
  from a failed calculation.
- Added recovery states including queued, running, completed, blocked,
  awaiting-user, and failed.
- Added commands to discover, open, resume, retry, or cleanly restart a selected
  step in an existing workflow.
- Added direct forms such as `resume latest` and
  `resume <number> --fresh-start-step <id>`.
- Reused completed upstream calculations instead of restarting the entire
  workflow.
- Added clean phonon restarts that disable incompatible `recover=.true.` state
  when q-grid settings change.
- Added an interactive recovery path where the user can discuss an error, ask
  the agent for a repair, edit the input manually, retry, download results, or
  stop.
- Deferred result download until completion/failure and user confirmation,
  rather than copying the whole remote directory after every step.

## 6. Quantum ESPRESSO input syntax validation

- Added an executable-specific syntax checklist based on the official Quantum
  ESPRESSO input documentation for `pw.x`, `bands.x`, `dos.x`, `ph.x`,
  `dynmat.x`, `q2r.x`, `matdyn.x`, and `projwfc.x`.
- Added validation for required namelists, documented keywords, assignment
  syntax, cards, k-point rows, numerical ranges, and Fortran namelist closure.
- Added deterministic removal of hallucinated/undocumented namelist keywords.
- Added normalization of final namelist commas for compatibility with older or
  site-patched QE builds.
- Added output validation for `CRASH`, QE error banners, missing `JOB DONE`, and
  incomplete outputs before marking a step complete.
- Added optional QE-version configuration for version-sensitive syntax such as
  DFT+U.

## 7. Scientific cross-step validation and normalization

- Added checks that related `pw.x` stages retain compatible structures,
  pseudopotentials, exchange-correlation families, cutoffs, k-point policies,
  spin settings, and workflow prefixes.
- Added scalar-relativistic versus fully relativistic pseudopotential checks.
- Added pseudopotential metadata inspection for XC and relativistic type.
- Added isolated scalar and SOC electronic workflows where required.
- Added vdW-model inheritance and validation, including D3 version and
  three-body settings.
- Added DOS NSCF/`dos.x` integration consistency, including the recurring
  tetrahedra-versus-smearing mismatch.
- Added conservative DOS energy-window normalization while respecting an
  explicitly requested narrow window.
- Added electron-count-aware `nbnd` selection:
  - collinear calculations use occupied bands plus a configurable safety
    margin;
  - noncollinear/SOC calculations account for spinor band counting before
    adding the safety margin.
- Added checks for missing band paths, invalid high-symmetry rows, missing DOS
  or PDOS producers, inconsistent phonon/Raman workflows, and mismatched
  dynamical-matrix filenames.
- Added canonical artifact names shared by producers and postprocessors.

## 8. Structure handling and symmetry-aware band paths

- Improved Materials Project structure coercion and primitive-cell handling.
- Added explicit relaxed-structure placeholders to every dependent input.
- Added extraction of final `CELL_PARAMETERS` and `ATOMIC_POSITIONS` from
  relaxation output and insertion into downstream inputs immediately before
  submission.
- Added symmetry-aware high-symmetry band-path materialization using the final
  accepted structure while retaining an approved fallback path if symmetry
  analysis fails.
- Added stage-aware structure selection so questions about the SCF structure
  use the structure actually materialized in `scf.in`, rather than the original
  relaxation input.
- Added CIF export and local VESTA integration, including discovery of
  `VESTA.app` inside a user-selected directory and a link to the official VESTA
  download when it is unavailable.

## 9. Deterministic structural-analysis tool

- Added tool routing for structural questions rather than relying only on LLM
  inference.
- Added calculations for:
  - lattice constants and lattice angles;
  - crystal system, space group, and point group;
  - species-pair bond lengths with periodic boundary conditions;
  - coordination environments;
  - species-resolved bond angles;
  - nearest-neighbor distances.
- Added provenance showing which concrete structure file was analyzed.
- Used `pymatgen` for generic periodic-structure analysis across materials.

## 10. Workflow dashboard and user interaction

- Added a persistent workflow monitor with execution status, branch, attempt,
  and Slurm job information.
- Added validation and error details without closing the dashboard when a job
  pauses for user input.
- Added a Structure tab with structure inspection, CIF export, VESTA launch,
  viewer location, download link, and folder access.
- Added an Input files tab with subtabs for the concrete inputs used by every
  step and repaired attempt.
- Added editable Band structure and DOS plot panels with configurable axis
  limits and energy-reference controls.
- Added automatic reopening of completed workflows from the `DFT request>`
  interface.
- Added an Ask Results tab that restricts retrieval to files inside the selected
  workflow.

## 11. Evidence-grounded result questions

- Added local search over calculated output and supporting workflow files.
- Required answers to include exact file paths, line numbers, and complete
  supporting lines.
- Added confidence and derivation sections.
- Added verified safe arithmetic so quantities not printed directly can be
  derived from cited values.
- Routed structural questions to the deterministic structural-analysis tool.
- Added an explicit consent boundary before relevant excerpts are sent to a
  configured external LLM.

## 12. Results and plotting

- Added automatic band, total-DOS, and projected-DOS data discovery.
- Added consistent electronic energy references derived from calculation
  outputs rather than arbitrary plot limits.
- Added spin-aware total DOS handling and orbital grouping for projected DOS.
- Added magnetic-moment extraction for magnetic workflows.
- Added plot-setting persistence so users can adjust energy and intensity
  windows without rerunning DFT.

## 13. VASP cluster workflow

- Added a VASP planning, input-generation, validation, approval, packaging, and
  remote-execution path.
- Added POSCAR/INCAR/KPOINTS/POTCAR handling and pseudopotential selection.
- Added VASP-specific Slurm configuration and remote POTCAR assembly support.
- Kept VASP configuration separate from the Quantum ESPRESSO workflow.

## 14. Tests and regression coverage

- Added dedicated test modules for workflow validation, checkpoint state, and
  cluster approval/execution behavior.
- Added regression tests for Slurm launcher preservation, resource clamping,
  pseudopotential separation, relaxed-structure propagation, workflow graph
  generation, resume commands, phonon recovery, QE syntax, DOS integration,
  DOS windows, SOC-aware `nbnd`, artifact filenames, and remote artifact
  transfer.
- At the time this document was created, the focused suites passed:
  - `test/test_cluster_approval.py`: 32 tests;
  - `test/test_workflow_validation.py`: 41 tests;
  - `test/test_workflow_state.py`: 5 tests.

## 15. Known follow-up work

- Dashboard result tabs are not yet fully generated from the workflow plan.
  Band structure and DOS are currently always registered; Raman, phonon
  dispersion, PDOS, and magnetic-summary tabs should appear dynamically only
  when the corresponding calculation exists.
- A Raman results panel still needs deterministic parsing and plotting of
  `dynmat.x` frequencies, Raman activities/tensors, and the generated
  `.modes`/output files.
- Broader end-to-end tests on multiple clusters and Quantum ESPRESSO versions
  should supplement the current unit/regression tests.

## Merge-request preparation

Before submitting this branch:

1. Fetch and compare against the latest `upstream/main`.
2. Review and commit the current working-tree changes in logical groups.
3. Run the complete project test suite in addition to the focused tests above.
4. Perform at least one clean end-to-end QE workflow and one resume-after-error
   workflow on a supported cluster.
5. Update the test counts and remove any item from "Known follow-up work" that
   is completed before submission.
6. Use this document as the basis of the merge-request description, but keep
   the MR summary shorter and link here for the full contribution record.
