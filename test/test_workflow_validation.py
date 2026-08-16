import tempfile
import unittest
from pathlib import Path

from validation import (
    inherit_shared_pw_settings,
    harmonize_dos_integration,
    normalize_plan,
    validate_generated_workflow,
    validate_plan,
    validate_qe_input,
    validate_qe_output,
    validate_qe_syntax,
)
from cluster_agent import _add_relaxed_structure_placeholder, _workflow_repair_indices


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
