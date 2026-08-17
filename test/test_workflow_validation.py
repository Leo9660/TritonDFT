import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from validation import (
    inherit_shared_pw_settings,
    inherit_vdw_model,
    harmonize_dos_integration,
    harmonize_dos_window,
    harmonize_nbnd,
    normalize_plan,
    validate_generated_workflow,
    validate_plan,
    validate_qe_input,
    validate_qe_output,
    validate_qe_syntax,
    remove_undocumented_namelist_keywords,
)
from cluster_agent import _add_relaxed_structure_placeholder, _workflow_repair_indices
from results.electronic_reference import electronic_reference, electronic_references
from results.evidence_qa import evaluate_calculation, parse_evidence_answer, search_workflow_evidence, verify_evidence
from workflow_monitor import _calculation_input_files, _input_tab_label, _relaxed_structure_to_cif, _resolve_vesta_location
from tool.structural_analysis import call_structural_analysis_tool


def pw_input(calculation="scf", *, kpoints="K_POINTS automatic\n4 4 4 0 0 0", ecutrho=320, extra=""):
    return f"""&control
 calculation='{calculation}',
 prefix='wf',
 outdir='./',
/
&system
 ibrav=2,
 celldm(1)=10.2,
 nat=2,
 ntyp=1,
 input_dft='PBE',
 ecutwfc=80,
 ecutrho={ecutrho},
 {extra}
/
&electrons
 conv_thr=1.0d-8,
/
ATOMIC_SPECIES
Si 28.0855 si.upf
ATOMIC_POSITIONS (crystal)
Si 0 0 0
Si .25 .25 .25
{kpoints}
"""


class WorkflowValidationTests(unittest.TestCase):
    def test_unknown_dynmat_keyword_is_removed_before_llm_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dynmat.in"
            path.write_text(
                "&INPUT\n prefix='wf',\n fildyn='wf.dynG',\n asr='crystal',\n/\n",
                encoding="utf-8",
            )
            repairs = remove_undocumented_namelist_keywords(str(path), "dynmat.x")
            self.assertEqual(len(repairs), 1)
            self.assertNotRegex(path.read_text(), r"(?mi)^\s*prefix\s*=")
            codes = {finding.code for finding in validate_qe_syntax(str(path), "dynmat.x")}
            self.assertNotIn("QE_KEYWORD_UNKNOWN", codes)

    def test_wrong_namelist_keyword_is_not_silently_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pw.in"
            path.write_text(
                "&CONTROL\n calculation='scf',\n ecutwfc=80,\n/\n"
                "&SYSTEM\n ibrav=1,\n nat=1,\n ntyp=1,\n ecutwfc=80,\n/\n"
                "&ELECTRONS\n conv_thr=1d-8,\n/\n",
                encoding="utf-8",
            )
            repairs = remove_undocumented_namelist_keywords(str(path), "pw.x")
            self.assertEqual(repairs, [])
            self.assertRegex(path.read_text(), r"(?mi)^\s*ecutwfc\s*=\s*80")

    def test_nbnd_default_is_occupied_plus_margin_and_soc_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pseudos = root / "pseudos"
            pseudos.mkdir()
            (pseudos / "mo.upf").write_text('<PP_HEADER z_valence="14.0"/>', encoding="utf-8")
            (pseudos / "s.upf").write_text('<PP_HEADER z_valence="6.0"/>', encoding="utf-8")
            common = """&control
 calculation='{calculation}',
 pseudo_dir='{pseudo}',
/
&system
 ibrav=0, nat=6, ntyp=2, nbnd=40,
 {spin}
/
ATOMIC_SPECIES
Mo 95.95 mo.upf
S 32.06 s.upf
ATOMIC_POSITIONS (crystal)
Mo 0 0 0
Mo 0 0 .5
S 0 0 .1
S 0 0 .2
S 0 0 .6
S 0 0 .7
K_POINTS automatic
2 2 2 0 0 0
"""
            collinear = root / "nscf.in"
            soc = root / "bands.in"
            collinear.write_text(common.format(calculation="nscf", pseudo=pseudos, spin=""), encoding="utf-8")
            soc.write_text(common.format(calculation="bands", pseudo=pseudos, spin="noncolin=.true.,\n lspinorb=.true.,"), encoding="utf-8")
            packages = [{"exec_name": "pw.x", "input_paths": [str(collinear)]},
                        {"exec_name": "pw.x", "input_paths": [str(soc)]}]
            harmonize_nbnd(packages)
            self.assertRegex(collinear.read_text(), r"nbnd\s*=\s*42")
            self.assertRegex(soc.read_text(), r"nbnd\s*=\s*68")

    def test_input_file_tabs_prefer_concrete_execution_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = root / "approved_inputs"
            attempt1 = root / "attempts" / "01-vc-relax" / "attempt_001"
            attempt2 = root / "attempts" / "02-scf" / "attempt_002"
            for directory in (approved, attempt1, attempt2):
                directory.mkdir(parents=True)
            (approved / "scf.in").write_text("approved", encoding="utf-8")
            (attempt1 / "vc-relax.in").write_text("relax", encoding="utf-8")
            scf = attempt2 / "scf.in"
            scf.write_text("scf", encoding="utf-8")
            paths = _calculation_input_files(root)
            self.assertEqual(paths, [attempt1 / "vc-relax.in", scf])
            self.assertEqual(_input_tab_label(root, scf), "02-scf/attempt_002/scf.in")

    def test_structural_tool_calculates_generic_periodic_species_pair_bonds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            branch = root / "branches" / "core"
            branch.mkdir(parents=True)
            (branch / "relaxed_structure.in").write_text(
                "CELL_PARAMETERS (angstrom)\n"
                "3.0 0 0\n0 3.0 0\n0 0 10.0\n"
                "ATOMIC_POSITIONS (crystal)\n"
                "Mo 0 0 0.5\nS 0.5 0 0.5\n",
                encoding="utf-8",
            )
            result = call_structural_analysis_tool(root, "What is the Mo-S bond length?")
            self.assertIsNotNone(result)
            self.assertEqual(result["call"]["task"], "species_pair_bonds")
            self.assertAlmostEqual(result["result"]["shortest_distance_angstrom"], 1.5)
            self.assertTrue(result["result"]["first_shell_bonds"])
            self.assertTrue(result["evidence"])

    def test_structural_tool_returns_generic_lattice_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relaxed_structure.in").write_text(
                "CELL_PARAMETERS (angstrom)\n"
                "3 0 0\n0 4 0\n0 0 5\n"
                "ATOMIC_POSITIONS (crystal)\nSi 0 0 0\n",
                encoding="utf-8",
            )
            result = call_structural_analysis_tool(root, "What are alpha beta gamma and the cell volume?")
            self.assertEqual(result["call"]["task"], "lattice")
            self.assertEqual(result["result"]["volume_angstrom3"], 60.0)
            self.assertEqual(result["result"]["gamma_degree"], 90.0)

    def test_structural_tool_calculates_periodic_species_triplet_angles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relaxed_structure.in").write_text(
                "CELL_PARAMETERS (angstrom)\n"
                "10 0 0\n0 10 0\n0 0 10\n"
                "ATOMIC_POSITIONS (crystal)\n"
                "Mo 0.5 0.5 0.5\nS 0.6 0.5 0.5\nS 0.5 0.6 0.5\n",
                encoding="utf-8",
            )
            result = call_structural_analysis_tool(root, "What are the S-Mo-S bond angles?")
            self.assertEqual(result["call"]["task"], "species_triplet_angles")
            angles = result["result"]["distinct_angles"]
            self.assertTrue(any(abs(item["angle_degree"] - 90.0) < 1e-8 for item in angles))

    def test_structural_tool_uses_executed_scf_geometry_for_scf_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relax = root / "attempts" / "01-vc-relax" / "attempt_001"
            scf = root / "attempts" / "02-scf" / "attempt_001"
            relax.mkdir(parents=True)
            scf.mkdir(parents=True)
            base = "CELL_PARAMETERS (angstrom)\n3 0 0\n0 3 0\n0 0 {c}\nATOMIC_POSITIONS (crystal)\nSi 0 0 0\n"
            (relax / "relaxed_structure.in").write_text(base.format(c=4), encoding="utf-8")
            (scf / "scf.in").write_text(base.format(c=5), encoding="utf-8")
            result = call_structural_analysis_tool(root, "What is the symmetry of the structure used in the SCF calculation?")
            self.assertEqual(Path(result["structure_file"]), (scf / "scf.in").resolve())

    def test_structural_tool_compares_executed_initial_and_scf_structures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relax = root / "attempts" / "01-vc-relax" / "attempt_001"
            scf = root / "attempts" / "02-scf" / "attempt_001"
            relax.mkdir(parents=True)
            scf.mkdir(parents=True)
            cubic = "CELL_PARAMETERS (angstrom)\n3 0 0\n0 3 0\n0 0 3\nATOMIC_POSITIONS (crystal)\nSi 0 0 0\n"
            tetragonal = cubic.replace("0 0 3", "0 0 4")
            (relax / "vc-relax.in").write_text(cubic, encoding="utf-8")
            (scf / "scf.in").write_text(tetragonal, encoding="utf-8")
            result = call_structural_analysis_tool(root, "Did the symmetry change after relaxation?")
            self.assertEqual(result["call"]["task"], "symmetry_comparison")
            self.assertTrue(result["result"]["changed"])
            self.assertEqual(Path(result["result"]["final_structure_file"]), (scf / "scf.in").resolve())

    def test_result_evidence_is_scoped_ranked_and_line_addressable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            branch = root / "branches" / "soc_refinement"
            branch.mkdir(parents=True)
            output = branch / "nscf.out"
            output.write_text(
                "calculation starts\n"
                "highest occupied, lowest unoccupied level (ev): 4.1000 5.2500\n"
                "JOB DONE.\n",
                encoding="utf-8",
            )
            save = branch / "tritondft.save"
            save.mkdir()
            (save / "data-file-schema.xml").write_text("band gap 999 eV", encoding="utf-8")
            evidence = search_workflow_evidence(root, "What is the band gap?")
            self.assertTrue(evidence)
            self.assertEqual(evidence[0].relative_path, "branches/soc_refinement/nscf.out")
            self.assertEqual(evidence[0].line_number, 2)
            self.assertTrue(verify_evidence(evidence[0]))
            self.assertFalse(any(".save" in item.relative_path for item in evidence))

    def test_result_answer_accepts_only_known_evidence_ids(self):
        parsed = parse_evidence_answer(
            '{"answer":"The gap is 1.15 eV.","evidence_ids":["E1","E99"],'
            '"confidence":"high","derivation":"5.25 - 4.10"}',
            {"E1"},
        )
        self.assertEqual(parsed["evidence_ids"], ["E1"])

    def test_result_math_is_evaluated_without_arbitrary_python(self):
        self.assertAlmostEqual(evaluate_calculation("0.5 * sqrt(12.3**2)"), 6.15)
        self.assertAlmostEqual(evaluate_calculation("degrees(acos(0.5))"), 60.0)
        with self.assertRaises(ValueError):
            evaluate_calculation("__import__('os').system('echo unsafe')")

    def test_structural_evidence_prefers_relaxed_geometry_over_starting_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            approved = root / "approved_inputs"
            final = root / "branches" / "core"
            approved.mkdir(parents=True)
            final.mkdir(parents=True)
            starting = "CELL_PARAMETERS (angstrom)\n3 0 0\n0 3 0\n0 0 10\nATOMIC_POSITIONS (crystal)\nMo 0 0 0.25\nMo 0 0 0.75\n"
            relaxed = starting.replace("0 0 10", "0 0 12.3")
            (approved / "vc-relax.in").write_text(starting, encoding="utf-8")
            (final / "relaxed_structure.in").write_text(relaxed, encoding="utf-8")
            evidence = search_workflow_evidence(root, "c axis distance between Mo atoms")
            self.assertTrue(evidence)
            self.assertEqual(evidence[0].path.name, "relaxed_structure.in")

    def test_vesta_parent_directory_resolves_to_macos_app_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "VESTA"
            application = parent / "VESTA.app"
            application.mkdir(parents=True)
            with patch("workflow_monitor.sys.platform", "darwin"):
                self.assertEqual(_resolve_vesta_location(parent), str(application.resolve()))

    def test_relaxed_qe_structure_can_be_exported_for_vesta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "relaxed_structure.in").write_text(
                "CELL_PARAMETERS (angstrom)\n"
                "3.16 0.0 0.0\n1.58 2.73664 0.0\n0.0 0.0 12.30\n"
                "ATOMIC_POSITIONS (crystal)\n"
                "Mo 0.333333 0.666667 0.25\nS 0.333333 0.666667 0.621\n"
            )
            cif = _relaxed_structure_to_cif(root)
            self.assertTrue(cif.is_file())
            text = cif.read_text(encoding="utf-8")
            self.assertIn("_cell_length_a", text)
            self.assertIn("Mo", text)

    def test_electronic_reference_prefers_soc_nscf_over_scalar_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scalar = root / "branches" / "core"
            soc = root / "branches" / "soc_refinement"
            scalar.mkdir(parents=True)
            soc.mkdir(parents=True)
            (scalar / "scf.out").write_text("highest occupied level (ev): 20.0\n")
            (soc / "scf.out").write_text("the Fermi energy is 5.5 ev\n")
            (soc / "nscf.out").write_text(
                "highest occupied, lowest unoccupied level (ev): 4.0 5.0\n"
            )
            refs = electronic_references(root)
            self.assertEqual(refs["vbm"], 4.0)
            self.assertEqual(refs["cbm"], 5.0)
            self.assertEqual(refs["midgap"], 4.5)
            self.assertEqual(electronic_reference(root, "vbm"), 4.0)

    def test_official_schema_rejects_misplaced_vdw_keywords(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bands.in"
            text = pw_input("bands", kpoints="K_POINTS crystal_b\n2\n0 0 0 20\n0.5 0 0 1")
            text = text.replace("prefix='wf',", "prefix='wf',\n vdw_corr='DFT-D3',\n dftd3_version=4,")
            path.write_text(text, encoding="utf-8")
            codes = {finding.code for finding in validate_qe_syntax(str(path), "pw.x")}
            self.assertIn("QE_KEYWORD_WRONG_NAMELIST", codes)

    def test_official_schema_rejects_invalid_cell_dofree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vc-relax.in"
            text = pw_input("vc-relax") + "&cell\n cell_dofree='hexagonal',\n/\n"
            path.write_text(text, encoding="utf-8")
            findings = validate_qe_syntax(str(path), "pw.x")
            self.assertTrue(any(f.code == "QE_ENUM_INVALID" and "cell_dofree" in f.message for f in findings))

    def test_official_schema_accepts_documented_d3bj_and_cell_dofree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vc-relax.in"
            text = pw_input("vc-relax", extra="vdw_corr='DFT-D3',\n dftd3_version=4,")
            text += "&cell\n cell_dynamics='bfgs',\n cell_dofree='all',\n/\n"
            path.write_text(text, encoding="utf-8")
            self.assertFalse(validate_qe_syntax(str(path), "pw.x"))

    def test_new_scf_step_rejects_restart_mode_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scf.in"
            path.write_text(
                pw_input().replace("prefix='wf',", "prefix='wf',\n restart_mode='restart',"),
                encoding="utf-8",
            )
            codes = {issue.code for issue in validate_qe_input(str(path), tool="pw_scf")}
            self.assertIn("NEW_STEP_RESTART_MODE_INVALID", codes)

    def test_lda_input_rejects_pbe_upf_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            (pseudo_dir / "si.upf").write_text('<UPF><PP_HEADER functional="PBE"/></UPF>\n')
            path = root / "scf.in"
            text = pw_input().replace("input_dft='PBE'", "input_dft='LDA'")
            text = text.replace("prefix='wf',", "prefix='wf',\n pseudo_dir='./pseudos',")
            path.write_text(text, encoding="utf-8")
            codes = {issue.code for issue in validate_qe_input(str(path), tool="pw_scf")}
            self.assertIn("PSEUDO_XC_MISMATCH", codes)

    def test_soc_requires_fully_relativistic_upf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pseudo_dir = root / "pseudos"
            pseudo_dir.mkdir()
            (pseudo_dir / "si.upf").write_text('<UPF><PP_HEADER functional="PBE" relativistic="scalar"/></UPF>\n')
            path = root / "scf.in"
            text = pw_input().replace(
                "input_dft='PBE'",
                "input_dft='PBE',\n noncolin=.true.,\n lspinorb=.true.",
            ).replace("prefix='wf',", "prefix='wf',\n pseudo_dir='./pseudos',")
            path.write_text(text, encoding="utf-8")
            codes = {issue.code for issue in validate_qe_input(str(path), tool="pw_scf")}
            self.assertIn("SOC_PSEUDO_NOT_FULLY_RELATIVISTIC", codes)
            (pseudo_dir / "si.upf").write_text(
                '<UPF><PP_HEADER functional="PBE"/>fully-relativistic pseudopotential</UPF>\n'
            )
            codes = {issue.code for issue in validate_qe_input(str(path), tool="pw_scf")}
            self.assertNotIn("SOC_PSEUDO_NOT_FULLY_RELATIVISTIC", codes)

    def test_plan_rejects_optional_duplicate_and_unrequested_convergence_jobs(self):
        plan = [
            {"id": 8, "tool": "pw_scf", "problem": "Converge cutoff and k-point sampling", "input": "structure", "why": "test convergence"},
            {"id": 2, "tool": "pw_scf", "problem": "SOC SCF", "input": "relaxed geometry", "why": "SOC bands"},
            {"id": 9, "tool": "pw_bands", "problem": "Optionally calculate scalar bands", "input": "SCF", "why": "optional comparison"},
        ]
        issues = validate_plan("relax and calculate SOC bands", plan)
        codes = {issue.code for issue in issues}
        self.assertIn("UNREQUESTED_CONVERGENCE_STEP", codes)
        self.assertIn("OPTIONAL_STEP_IN_EXECUTABLE_PLAN", codes)
        self.assertIn("DUPLICATE_SCF_WORKFLOWS", codes)

    def test_production_parameters_and_scalar_soc_scf_branches_are_valid(self):
        plan = [
            {
                "id": 1,
                "tool": "pw_vc_relax",
                "problem": "Relax the bulk 2H-MoS2 cell and atomic positions",
                "input": "scalar-relativistic pseudopotentials, converged cutoffs, and Gamma-centered k-point mesh",
                "why": "Produce the equilibrium scalar-relativistic geometry",
            },
            {
                "id": 2,
                "tool": "pw_scf",
                "problem": "Validate the relaxed structure with a scalar-relativistic SCF calculation",
                "input": "relaxed structure and scalar-relativistic pseudopotentials",
                "why": "Provide the structural baseline",
            },
            {
                "id": 3,
                "tool": "pw_scf",
                "problem": "Perform a fully relativistic SOC SCF calculation",
                "input": "fixed relaxed geometry and fully relativistic pseudopotentials",
                "why": "Generate the SOC save state for bands and DOS",
            },
        ]
        codes = {issue.code for issue in validate_plan("relax bulk MoS2 and calculate bands and DOS", plan)}
        self.assertNotIn("UNREQUESTED_CONVERGENCE_STEP", codes)
        self.assertNotIn("DUPLICATE_SCF_WORKFLOWS", codes)

    def test_normalized_plan_ids_follow_execution_order(self):
        plan = [
            {"id": 7, "tool": "pw_scf", "problem": "SCF"},
            {"id": 2, "tool": "pw_vc_relax", "problem": "relax"},
        ]
        normalized = normalize_plan("relax then SCF", plan)
        self.assertEqual([step["tool"] for step in normalized], ["pw_vc_relax", "pw_scf"])
        self.assertEqual([step["id"] for step in normalized], [1, 2])

    def test_cross_step_validation_rejects_dftd3_version_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relax = root / "vc-relax.in"
            scf = root / "scf.in"
            relax.write_text(
                pw_input("vc-relax", extra="vdw_corr='dft-d3',\n dftd3_version=2,"),
                encoding="utf-8",
            )
            scf.write_text(
                pw_input("scf", extra="vdw_corr='DFT-D3',\n dftd3_version=4,\n noncolin=.true.,\n lspinorb=.true.,"),
                encoding="utf-8",
            )
            steps = [
                {"id": 1, "tool": "pw_vc_relax", "problem": "relax"},
                {"id": 2, "tool": "pw_scf", "problem": "SOC SCF"},
            ]
            packages = [
                {"exec_name": "pw.x", "input_paths": [str(relax)], "pseudo_dir": "/scalar"},
                {"exec_name": "pw.x", "input_paths": [str(scf)], "pseudo_dir": "/fully-relativistic"},
            ]
            codes = {
                issue.code for issue in validate_generated_workflow("bulk bands with SOC", steps, packages)
            }
            self.assertIn("VDW_MODEL_MISMATCH", codes)

    def test_unphysical_gamma_acoustic_modes_block_phonon_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon-grid.out"
            path.write_text(
                "q = ( 0.000000000 0.000000000 0.000000000 )\n"
                "freq ( 1) = -13.8 [THz] = -460.8 [cm-1]\n"
                "freq ( 2) = -13.8 [THz] = -460.8 [cm-1]\n"
                "freq ( 3) = -13.8 [THz] = -460.8 [cm-1]\n"
                "JOB DONE.\n",
                encoding="utf-8",
            )
            codes = {issue.code for issue in validate_qe_output(str(path), exec_name="ph.x")}
            self.assertIn("GAMMA_ACOUSTIC_MODES_INVALID", codes)

    def test_final_namelist_assignment_does_not_require_trailing_comma(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon-grid.in"
            path.write_text(
                "&inputph\n prefix='wf',\n outdir='./',\n fildyn='wf.dyn',\n"
                " tr2_ph=1.0d-14,\n recover=.false.\n/\n",
                encoding="utf-8",
            )
            codes = {issue.code for issue in validate_qe_input(str(path), tool="pw_phonon_gamma")}
            self.assertNotIn("NAMELIST_FINAL_COMMA_MISSING", codes)

    def test_band_nscf_plan_is_normalized_to_bands_mode(self):
        plan = [{"id": 1, "problem": "Eigenvalues along the high-symmetry k-point path", "tool": "pw_nscf", "input": "SCF"}]
        self.assertEqual(normalize_plan("band structure", plan)[0]["tool"], "pw_bands")

    def test_phonons_are_ordered_before_mutating_nscf_branches(self):
        plan = [
            {"id": 1, "problem": "SCF", "tool": "pw_scf", "input": "structure"},
            {"id": 2, "problem": "DOS states", "tool": "pw_nscf", "input": "SCF"},
            {"id": 3, "problem": "Gamma phonons", "tool": "pw_phonon_gamma", "input": "SCF"},
            {"id": 4, "problem": "DOS", "tool": "dos_post", "input": "NSCF"},
        ]
        tools = [step["tool"] for step in normalize_plan("phonons and DOS", plan)]
        self.assertLess(tools.index("pw_phonon_gamma"), tools.index("pw_nscf"))

    def test_raman_and_dispersion_require_two_ph_steps(self):
        plan = [
            {"id": 1, "problem": "phonons on uniform q mesh", "tool": "pw_phonon_gamma", "input": "SCF"},
            {"id": 2, "problem": "force constants", "tool": "q2r_post", "input": "dynamical matrices"},
            {"id": 3, "problem": "phonon dispersion", "tool": "matdyn_post", "input": "force constants"},
            {"id": 4, "problem": "Gamma modes", "tool": "dynmat_post", "input": "Gamma matrix"},
        ]
        codes = {i.code for i in validate_plan("phonon dispersion and Raman", plan)}
        self.assertIn("RAMAN_DISPERSION_PH_SPLIT_REQUIRED", codes)

    def test_placeholder_cannot_corrupt_band_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bands.in"
            path.write_text(pw_input("bands", kpoints="K_POINTS crystal_b\n2\n0.5 0.5 0.5 20\n0 0 0 1"))
            self.assertTrue(_add_relaxed_structure_placeholder(str(path)))
            text = path.read_text()
            self.assertIn("\n0.5 0.5 0.5 20\n", text)
            self.assertNotIn("0.! TRITONDFT", text)
            self.assertFalse([i for i in validate_qe_input(str(path), tool="pw_bands") if i.blocking])

    def test_raman_ph_requires_lraman(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ph.in"
            path.write_text("&inputph\n prefix='wf',\n outdir='./',\n fildyn='wf.dynG',\n tr2_ph=1d-14,\n/\n0 0 0\n")
            codes = {i.code for i in validate_qe_input(str(path), tool="pw_phonon_gamma", query="Gamma Raman coefficients")}
            self.assertIn("RAMAN_NOT_IN_THIS_PH", codes)

    def test_shared_settings_are_inherited_not_reselected(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "relax.in"
            second = Path(tmp) / "scf.in"
            first.write_text(pw_input("vc-relax", extra="nspin=2,\n starting_magnetization(1)=0.5,"))
            second.write_text(pw_input("scf", ecutrho=640))
            packages = [
                {"exec_name": "pw.x", "input_paths": [str(first)]},
                {"exec_name": "pw.x", "input_paths": [str(second)]},
            ]
            inherit_shared_pw_settings(packages)
            text = second.read_text()
            self.assertRegex(text, r"ecutrho\s*=\s*320")
            self.assertRegex(text, r"nspin\s*=\s*2")
            self.assertIn("starting_magnetization(1)=0.5", text)

    def test_vdw_model_is_inherited_without_overwriting_soc_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relax = root / "vc-relax.in"
            soc = root / "scf-soc.in"
            bands = root / "bands.in"
            relax.write_text(pw_input(
                "vc-relax",
                extra="vdw_corr='grimme-d3',\n dftd3_version=4,\n dftd3_threebody=.true.,",
            ))
            soc.write_text(pw_input(
                "scf",
                extra="noncolin=.true.,\n lspinorb=.true.,",
            ).replace("si.upf", "si-fr.upf"))
            bands.write_text(pw_input(
                "bands",
                kpoints="K_POINTS crystal_b\n2\n0 0 0 20\n0.5 0 0 1",
                extra="noncolin=.true.,\n lspinorb=.true.,",
            ).replace("si.upf", "si-fr.upf"))
            packages = [
                {"exec_name": "pw.x", "input_paths": [str(relax)]},
                {"exec_name": "pw.x", "input_paths": [str(soc)]},
                {"exec_name": "pw.x", "input_paths": [str(bands)]},
            ]
            inherit_vdw_model(packages)
            for path in (soc, bands):
                text = path.read_text()
                self.assertRegex(text, r"(?mi)^\s*vdw_corr\s*=\s*'grimme-d3'")
                self.assertRegex(text, r"(?mi)^\s*dftd3_version\s*=\s*4")
                self.assertRegex(text, r"(?mi)^\s*dftd3_threebody\s*=\s*\.true\.")
                self.assertIn("lspinorb=.true.", text)
                self.assertIn("si-fr.upf", text)

    def test_spin_resolved_request_rejects_nonmagnetic_setup(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "scf.in"
            source.write_text(pw_input("scf"))
            steps = [{"id": 1, "problem": "SCF", "tool": "pw_scf", "input": "structure"}]
            packages = [{"exec_name": "pw.x", "input_paths": [str(source)], "params_json": "{}"}]
            codes = {i.code for i in validate_generated_workflow("spin-resolved projected DOS and local magnetic moments", steps, packages)}
            self.assertIn("SPIN_SETUP_MISSING", codes)
            self.assertIn("INITIAL_MOMENTS_MISSING", codes)

    def test_dos_tetrahedron_smearing_mismatch_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            nscf = Path(tmp) / "nscf.in"
            dos = Path(tmp) / "dos.in"
            nscf.write_text(pw_input("nscf", extra="occupations='tetrahedra',"))
            dos.write_text("&DOS\n prefix='wf',\n outdir='./',\n fildos='wf.dos',\n bz_sum='smearing',\n/\n")
            steps = [
                {"id": 1, "problem": "dense DOS states", "tool": "pw_nscf", "input": "SCF"},
                {"id": 2, "problem": "total DOS", "tool": "dos_post", "input": "NSCF"},
            ]
            packages = [
                {"exec_name": "pw.x", "input_paths": [str(nscf)], "params_json": "{}"},
                {"exec_name": "dos.x", "input_paths": [str(dos)], "params_json": "{}"},
            ]
            codes = {i.code for i in validate_generated_workflow("total DOS", steps, packages)}
            self.assertIn("DOS_INTEGRATION_MISMATCH", codes)

    def test_dos_integration_is_harmonized_before_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            nscf = Path(tmp) / "nscf.in"
            dos = Path(tmp) / "dos.in"
            nscf.write_text(pw_input("nscf", extra="occupations='tetrahedra_opt',"))
            dos.write_text(
                "&DOS\n prefix='wf',\n outdir='./',\n fildos='wf.dos',\n"
                " bz_sum='smearing',\n ngauss=0,\n degauss=0.01,\n/\n"
            )
            steps = [
                {"id": 1, "problem": "dense DOS states", "tool": "pw_nscf", "input": "SCF"},
                {"id": 2, "problem": "total DOS", "tool": "dos_post", "input": "NSCF"},
            ]
            packages = [
                {"exec_name": "pw.x", "input_paths": [str(nscf)], "params_json": "{}"},
                {"exec_name": "dos.x", "input_paths": [str(dos)], "params_json": "{}"},
            ]
            harmonize_dos_integration(steps, packages)
            text = dos.read_text()
            self.assertIn("bz_sum='tetrahedra_opt'", text)
            self.assertNotRegex(text, r"(?mi)^\s*(?:ngauss|degauss)\s*=")
            codes = {i.code for i in validate_generated_workflow("total DOS", steps, packages)}
            self.assertNotIn("DOS_INTEGRATION_MISMATCH", codes)

    def test_narrow_generated_dos_window_is_replaced_before_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            dos = Path(tmp) / "dos.in"
            dos.write_text(
                "&DOS\n prefix='wf',\n outdir='./',\n fildos='wf.dos',\n"
                " Emin=-0.734986,\n Emax=0.734986,\n DeltaE=0.000734986,\n"
                " bz_sum='smearing',\n/\n"
            )
            steps = [{"id": 1, "problem": "total DOS", "tool": "dos_post", "input": "NSCF"}]
            packages = [{"exec_name": "dos.x", "input_paths": [str(dos)], "params_json": "{}"}]
            harmonize_dos_window(steps, packages, "calculate the total DOS")
            text = dos.read_text()
            self.assertRegex(text, r"(?mi)^\s*Emin\s*=\s*-15\.0,")
            self.assertRegex(text, r"(?mi)^\s*Emax\s*=\s*10\.0,")
            self.assertRegex(text, r"(?mi)^\s*DeltaE\s*=\s*0\.01,")
            codes = {i.code for i in validate_qe_input(str(dos), tool="dos_post")}
            self.assertNotIn("DOS_ENERGY_WINDOW_TOO_NARROW", codes)

    def test_narrow_dos_window_is_detected_by_validator(self):
        with tempfile.TemporaryDirectory() as tmp:
            dos = Path(tmp) / "dos.in"
            dos.write_text(
                "&DOS\n prefix='wf',\n outdir='./',\n fildos='wf.dos',\n"
                " Emin=-0.734986,\n Emax=0.734986,\n DeltaE=0.01,\n/\n"
            )
            codes = {i.code for i in validate_qe_input(str(dos), tool="dos_post")}
            self.assertIn("DOS_ENERGY_WINDOW_TOO_NARROW", codes)

    def test_dos_mismatch_targets_only_dos_post_for_llm_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            nscf = Path(tmp) / "nscf.in"
            dos = Path(tmp) / "dos.in"
            nscf.write_text(pw_input("nscf", extra="occupations='tetrahedra',"))
            dos.write_text("&DOS\n prefix='wf',\n outdir='./',\n fildos='wf.dos',\n bz_sum='smearing',\n/\n")
            steps = [
                {"id": 1, "problem": "dense DOS states", "tool": "pw_nscf", "input": "SCF"},
                {"id": 2, "problem": "total DOS", "tool": "dos_post", "input": "NSCF"},
            ]
            packages = [
                {"exec_name": "pw.x", "input_paths": [str(nscf)], "params_json": "{}"},
                {"exec_name": "dos.x", "input_paths": [str(dos)], "params_json": "{}"},
            ]
            issues = [
                issue for issue in validate_generated_workflow("total DOS", steps, packages)
                if issue.blocking
            ]
            self.assertEqual(_workflow_repair_indices(issues, steps, packages), [1])


if __name__ == "__main__":
    unittest.main()
