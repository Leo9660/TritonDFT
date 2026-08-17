import tempfile
import unittest
import json
from pathlib import Path
import os
import sys
import types
from unittest.mock import patch

if "mp_api.client" not in sys.modules:
    mp_api_module = types.ModuleType("mp_api")
    mp_api_client_module = types.ModuleType("mp_api.client")
    mp_api_client_module.MPRester = object
    mp_api_module.client = mp_api_client_module
    sys.modules["mp_api"] = mp_api_module
    sys.modules["mp_api.client"] = mp_api_client_module

from cluster_agent import (
    ClusterJob,
    RemoteClusterDFTAgent,
    SSHClusterTransport,
    _add_relaxed_structure_placeholder,
    _approval_result,
    _env_missing_cluster_setup,
    _extract_relaxed_structure,
    _ensure_env_defaults,
    _input_validation_errors,
    _insert_relaxed_structure,
    _enforce_workflow_artifact_names,
    _force_ph_fresh_start,
    _normalize_namelist_final_commas,
    _plan_text,
    _pw_pseudo_dirs_for_work_dir,
    _may_clone_parent_qe_state,
    _workflow_graph_text,
    _discover_workflows,
    _resolve_workflow_to_open,
    _parse_resume_command,
    _required_parent_artifacts,
)
from execute_code.slurm import SlurmLauncher
from execute_code.slurm import _create_probe_script, _ensure_parameter, _enforce_safe_qe_parallel_flags
from execute_code.slurm_template import render_slurm_script
from DFTAgent import DFTAgent, _generate_nonempty_text
from workflow_state import WorkflowCheckpoint, create_checkpoint


PW_INPUT = """&control
 calculation = 'scf',
 prefix = 'generated_prefix',
/
&system
 ibrav = 2,
 celldm(1) = 10.2,
 nat = 2,
 ntyp = 1,
/
&electrons
 conv_thr = 1.0d-8,
/
ATOMIC_SPECIES
Si 28.0855 si.upf
ATOMIC_POSITIONS (crystal)
Si 0.0 0.0 0.0
Si 0.25 0.25 0.25
CELL_PARAMETERS (angstrom)
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
K_POINTS automatic
4 4 4 0 0 0
"""


RELAX_OUTPUT = """Begin final coordinates
CELL_PARAMETERS (angstrom)
5.40 0.00 0.00
0.00 5.40 0.00
0.00 0.00 5.40
ATOMIC_POSITIONS (crystal)
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
End final coordinates
convergence has been achieved
JOB DONE.
"""


class PlaceholderTests(unittest.TestCase):
    def test_resume_prompt_parses_fresh_start_step(self):
        self.assertEqual(_parse_resume_command("resume 1 --fresh-start-step 6"), ("1", 6))
        self.assertEqual(_parse_resume_command("resume latest"), ("latest", None))

    def test_merced_template_preserves_srun_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "merced.sh"
            template.write_text(
                "#!/bin/bash\n#SBATCH --partition=medium\n#SBATCH --nodes=1\n"
                "#SBATCH --tasks-per-node=16\nmodule load quantum-espresso\n"
                "srun --mpi=pmix -n 1 $exe -in $INPUT > $OUTPUT\n",
                encoding="utf-8",
            )
            rendered = render_slurm_script(
                exec_path="pw.x", input_path="scf.in", output_path="scf.out",
                command_line="export OMP_NUM_THREADS=1; mpirun --allow-run-as-root -np 16 $exe -nk 4 -in $INPUT > $OUTPUT",
                nodes=1, tasks_per_node=16, work_dir=".", time_limit="01:00:00",
                template_path=str(template),
            )
            self.assertIn("srun --mpi=pmix -n 16 $exe -nk 4 -in $INPUT > $OUTPUT", rendered)
            self.assertNotIn("mpirun", rendered)
            self.assertIn("#SBATCH --partition=medium", rendered)

    def test_template_ntasks_tracks_nodes_times_tasks_per_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "merced.sh"
            template.write_text(
                "#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --tasks-per-node=16\n"
                "#SBATCH --ntasks=16\nsrun -n 16 $exe -in $INPUT > $OUTPUT\n",
                encoding="utf-8",
            )
            rendered = render_slurm_script(
                exec_path="pw.x", input_path="scf.in", output_path="scf.out",
                command_line="mpirun -np 48 $exe -in $INPUT > $OUTPUT",
                nodes=2, tasks_per_node=24, work_dir=".", time_limit="01:00:00",
                template_path=str(template),
            )
            self.assertIn("#SBATCH --ntasks=48", rendered)
            self.assertIn("srun -n 48", rendered)

    def test_slurm_resource_plan_is_clamped_to_user_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "scf.in"
            source.write_text(PW_INPUT, encoding="utf-8")

            class Generator:
                def __call__(self, *_args, **_kwargs):
                    return [{"generated_text": (
                        "Analysis: large calculation\n"
                        "Command: mpirun -np 200 $exe -in $INPUT > $OUTPUT\n"
                        'Slurm: {"nodes": 4, "tasks_per_node": 128, "time_limit": "01:00:00"}'
                    )}]

            launcher = SlurmLauncher(generator=Generator(), max_new_tokens=100)
            launcher.set_resource_limits(2, 24)
            plan = launcher._generate_slurm_auto_parallel_plan(
                exec_name="pw.x", qe_prefix="", input_path=str(source), input_name="scf.in",
                output_name="scf.out", parallel_np=48, hardware_description=None,
            )
            self.assertEqual(plan["nodes"], 2)
            self.assertEqual(plan["tasks_per_node"], 24)
            self.assertEqual(plan["mpi_ranks"], 48)
            self.assertIn("-np 48", plan["command_line"])

    def test_ph_probe_inserts_fortran_namelist_parameter_with_comma(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "phonon.in"
            source.write_text(
                "&inputph\n"
                "  prefix='wf',\n"
                "  outdir='./',\n"
                "  tr2_ph=1.0d-14,\n"
                "  fildyn='wf.dyn',\n"
                "/\n",
                encoding="utf-8",
            )
            probe = Path(_create_probe_script(str(source), exec_name="ph.x"))
            text = probe.read_text(encoding="utf-8")
            self.assertIn("max_seconds = 120,", text)
            self.assertNotIn("max_seconds = 120\n", text)

    def test_probe_replacement_preserves_required_comma(self):
        original = "&control\n  max_seconds=30\n/\n"
        updated = _ensure_parameter(original, "max_seconds", "120")
        self.assertIn("max_seconds = 120,", updated)

    def test_missing_per_user_ssh_alias_triggers_cluster_setup(self):
        old_home = os.environ.get("HOME")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["HOME"] = tmp
                env_path = Path(tmp) / ".env.cluster"
                env_path.write_text(
                    "CLUSTER_AGENT_SSH_TARGET=expanse\n"
                    "CLUSTER_AGENT_REMOTE_ROOT=/scratch/user/tritondft\n",
                    encoding="utf-8",
                )

                self.assertTrue(_env_missing_cluster_setup(str(env_path)))

                ssh_dir = Path(tmp) / ".ssh"
                ssh_dir.mkdir()
                (ssh_dir / "config").write_text(
                    "Host expanse\n"
                    "  HostName login.expanse.sdsc.edu\n"
                    "  User user\n",
                    encoding="utf-8",
                )
                self.assertFalse(_env_missing_cluster_setup(str(env_path)))
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)

    def test_direct_ssh_target_does_not_require_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env.cluster"
            env_path.write_text(
                "CLUSTER_AGENT_SSH_TARGET=user@login.expanse.sdsc.edu\n"
                "CLUSTER_AGENT_REMOTE_ROOT=/scratch/user/tritondft\n",
                encoding="utf-8",
            )
            self.assertFalse(_env_missing_cluster_setup(str(env_path)))

    def test_default_slurm_template_is_user_local_and_created(self):
        old_home = os.environ.get("HOME")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["HOME"] = tmp
                env_path = Path(tmp) / ".env.cluster"
                env_path.write_text(
                    "OPENAI_API_KEY=x\n"
                    "MP_API_KEY=y\n"
                    "TRITONDFT_SLURM_TEMPLATE=example_slurm_job_file.txt\n",
                    encoding="utf-8",
                )

                _ensure_env_defaults(str(env_path))

                env_text = env_path.read_text(encoding="utf-8")
                self.assertIn(
                    "TRITONDFT_SLURM_TEMPLATE=~/.tritondft/example_qe_slurm_job_file.txt",
                    env_text,
                )
                user_template = Path(tmp) / ".tritondft" / "example_qe_slurm_job_file.txt"
                self.assertTrue(user_template.exists())
                self.assertIn("#SBATCH", user_template.read_text(encoding="utf-8"))
        finally:
            if old_home is not None:
                os.environ["HOME"] = old_home
            else:
                os.environ.pop("HOME", None)

    def test_empty_model_responses_retry_without_becoming_parser_failures(self):
        class EventuallyReturnsInput:
            def __init__(self):
                self.calls = 0
                self.budgets = []

            def __call__(self, _prompt, *, max_new_tokens, return_full_text):
                self.calls += 1
                self.budgets.append(max_new_tokens)
                if self.calls < 3:
                    return [{"generated_text": ""}]
                return [{"generated_text": "<scripts><script>&control\n/</script></scripts>"}]

        generator = EventuallyReturnsInput()
        text = _generate_nonempty_text(
            generator,
            "prompt",
            max_new_tokens=8192,
            attempts=3,
        )
        self.assertIn("&control", text)
        self.assertEqual(generator.calls, 3)
        self.assertEqual(generator.budgets, [8192, 8192, 8192])

    def test_placeholder_is_visible_and_materializes_final_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input_2_1.in"
            output_path = Path(tmp) / "output_1_1.out"
            input_path.write_text(PW_INPUT, encoding="utf-8")
            output_path.write_text(RELAX_OUTPUT, encoding="utf-8")

            self.assertTrue(_add_relaxed_structure_placeholder(str(input_path)))
            preview = input_path.read_text(encoding="utf-8")
            self.assertIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN", preview)
            self.assertIn("ibrav = 0", preview)
            self.assertNotIn("celldm(1)", preview)
            self.assertNotIn("Si 0.25 0.25 0.25", preview)

            structure = _extract_relaxed_structure(str(output_path))
            self.assertTrue(_insert_relaxed_structure(str(input_path), structure))
            materialized = input_path.read_text(encoding="utf-8")
            self.assertNotIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER", materialized)
            self.assertIn("CELL_PARAMETERS (angstrom)", materialized)
            self.assertIn("5.40 0.00 0.00", materialized)
            self.assertIn("ATOMIC_POSITIONS (crystal)", materialized)

    def test_plan_explains_why_each_step_exists(self):
        text = _plan_text(
            [
                {
                    "problem": "Relax the structure",
                    "tool": "pw_vc_relax",
                    "input": "Initial crystal structure",
                    "why": "Obtain the equilibrium geometry for all later calculations",
                },
                {
                    "problem": "Run SCF",
                    "tool": "pw_scf",
                    "input": "Relaxed structure",
                    "why": "Create the converged charge density",
                },
            ]
        )
        self.assertIn("Why: Obtain the equilibrium geometry", text)
        self.assertIn("No cluster job is submitted until you approve", text)

    def test_incomplete_pw_input_is_rejected_before_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input_2_1.in"
            path.write_text(
                "&system\n  ibrav=0,\n/\n&electrons\n/\nATOMIC_SPECIES\nSi 28 si.upf\n"
                "! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_BEGIN\n"
                "! relaxed structure\n"
                "! TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER_END\n",
                encoding="utf-8",
            )
            errors = _input_validation_errors(str(path))
            self.assertIn("pw.x input is missing &control.", errors)
            self.assertIn("pw.x input is missing K_POINTS.", errors)

    def test_packaged_slurm_scripts_do_not_overwrite_between_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "input_1_1.in"
            second = Path(tmp) / "input_2_1.in"
            first.write_text(PW_INPUT, encoding="utf-8")
            second.write_text(PW_INPUT, encoding="utf-8")
            launcher = SlurmLauncher(generator=None, max_new_tokens=1)
            launcher._generate_slurm_auto_parallel_plan = lambda **_kwargs: {
                "mpi_ranks": 1,
                "command_line": "mpirun -np 1 $exe -in $INPUT > $OUTPUT",
                "time_limit": "00:10:00",
                "nodes": 1,
                "tasks_per_node": 1,
            }
            launcher._generate_slurm_script = lambda **kwargs: (
                f"#!/bin/bash\nINPUT={kwargs['input_path']}\n"
            )

            first_scripts = launcher.package(
                "pw.x", "", [str(first)], tmp, False, 1
            )
            second_scripts = launcher.package(
                "pw.x", "", [str(second)], tmp, False, 1
            )

            self.assertNotEqual(first_scripts, second_scripts)
            self.assertTrue(Path(first_scripts[0]).exists())
            self.assertTrue(Path(second_scripts[0]).exists())

    def test_empty_auto_parallel_response_uses_fallback_plan(self):
        class EmptyGenerator:
            def __call__(self, *_args, **_kwargs):
                return [{"generated_text": ""}]

        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input_1_1.in"
            input_path.write_text(PW_INPUT.replace("'scf'", "'vc-relax'"), encoding="utf-8")
            launcher = SlurmLauncher(
                generator=EmptyGenerator(),
                max_new_tokens=100,
                verbose=False,
            )
            scripts = launcher.package(
                "pw.x",
                "",
                [str(input_path)],
                tmp,
                False,
                8,
                output_paths=[str(Path(tmp) / "output_1_1.out")],
            )

            script = Path(scripts[0]).read_text(encoding="utf-8")
            self.assertIn("mpirun --allow-run-as-root -np 8", script)
            self.assertIn("#SBATCH -t 04:00:00", script)


class _FakeAgent:
    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.need_query_info = False
        self.pseudo_dir = str(work_dir / "unused-pseudos")
        self.remote_qe_bin_prefix = ""
        self.parallel_np = 4
        self.parallel_exec = False
        self.hardware_description = None
        self.slurm_launcher = _FakeSlurmLauncher()
        self.run_mode = ""
        self.generated = []
        Path(self.pseudo_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.pseudo_dir) / "si.upf").write_text("fake pseudo\n", encoding="utf-8")

    def _prepare_run_directory(self, **_kwargs):
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def plan(self, query):
        return [
            {
                "id": 1,
                "problem": "Relax",
                "tool": "pw_vc_relax",
                "input": "Initial structure",
                "why": "Produce the equilibrium geometry",
            },
            {
                "id": 2,
                "problem": "SCF",
                "tool": "pw_scf",
                "input": "Relaxed structure",
                "why": "Produce the converged ground state",
            },
        ]

    def solve_sub_problem(self, step, problem_id, **_kwargs):
        input_path = self.work_dir / f"input_{problem_id}_1.in"
        output_path = self.work_dir / f"output_{problem_id}_1.out"
        mode = {"pw_vc_relax": "vc-relax", "pw_scf": "scf"}[step["tool"]]
        input_path.write_text(PW_INPUT.replace("'scf'", f"'{mode}'"), encoding="utf-8")
        self.generated.append(problem_id)
        return {
            "status": "cluster_input",
            "input_paths": [str(input_path)],
            "output_paths": [str(output_path)],
            "slurm_paths": [],
            "work_dir": str(self.work_dir),
            "exec_name": "pw.x",
            "subproblem_id": problem_id,
            "params_json": "{}",
        }

    @staticmethod
    def _judge_prose(judge):
        return "done"

    def _write_analysis(self, analysis, query):
        (self.work_dir / "analysis.json").write_text(analysis, encoding="utf-8")


class _FakeTransport:
    def __init__(self, state):
        self.state = state
        self.ensure_calls = 0
        self.submissions = []
        self.directory_fetches = 0

    def ensure_connection(self):
        self.ensure_calls += 1
        if not self.state["approved"]:
            raise AssertionError("SSH opened before approval")

    def remote_dir_for(self, local_run_dir):
        return f"/remote/{local_run_dir.name}"

    def upload_directory(self, local_dir, remote_dir):
        self.state["uploaded"].append(remote_dir)

    def upload_step_files(self, local_dir, remote_dir, paths):
        self.state["uploaded"].append(remote_dir)
        self.state.setdefault("uploaded_files", []).extend(Path(path).name for path in paths)

    def submit(self, remote_dir, script_name):
        self.submissions.append(script_name)
        return ClusterJob("1", script_name, remote_dir, "Submitted batch job 1")

    def archive_failure_markers(self, remote_dir, attempt):
        return None

    def wait_for_job(self, job):
        return None

    def fetch_directory(self, remote_dir, local_dir):
        self.directory_fetches += 1

    def fetch_files(self, remote_dir, local_dir, names):
        if "vc-relax.out" in names and not (local_dir / "vc-relax.out").exists():
            (local_dir / "vc-relax.out").write_text(RELAX_OUTPUT, encoding="utf-8")
        if "scf.out" in names:
            (local_dir / "scf.out").write_text("JOB DONE.\n", encoding="utf-8")
        return [str(local_dir / name) for name in names if (local_dir / name).is_file()]


class _FakeSlurmLauncher:
    def package_probe(self, *, input_paths, work_dir, **_kwargs):
        probe_inputs = []
        probe_outputs = []
        probe_scripts = []
        for input_path in input_paths:
            stem = Path(input_path).stem
            probe_input = Path(work_dir) / f"{stem}_probe.in"
            probe_output = Path(work_dir) / f"{stem}_probe.out"
            probe_script = Path(work_dir) / f"slurm_probe_{stem}.sh"
            probe_input.write_text(Path(input_path).read_text(encoding="utf-8"), encoding="utf-8")
            probe_script.write_text("#!/bin/bash\n", encoding="utf-8")
            probe_inputs.append(str(probe_input))
            probe_outputs.append(str(probe_output))
            probe_scripts.append(str(probe_script))
        return probe_inputs, probe_outputs, probe_scripts

    def package(self, *, input_paths, work_dir, probe_output_paths=None, **_kwargs):
        if probe_output_paths:
            assert all(Path(path).exists() for path in probe_output_paths)
        scripts = []
        for input_path in input_paths:
            script = Path(work_dir) / f"slurm_job_{Path(input_path).stem}.sh"
            script.write_text("#!/bin/bash\n", encoding="utf-8")
            scripts.append(str(script))
        return scripts


class _TestRemoteAgent(RemoteClusterDFTAgent):
    def _parse_remote_step(self, query, subproblem, package):
        return {
            "result_json": "{}",
            "result_judge": '{"status":"done","desc":"done"}',
            "judge_json": {"status": "done"},
        }


class ApprovalWorkflowTests(unittest.TestCase):
    def test_workflow_browser_discovers_and_resolves_latest_and_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "2026-08-16" / "older_run"
            newer = root / "2026-08-17" / "newer_run"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            (older / "workflow_state.json").write_text(json.dumps({
                "status": "completed", "updated_at": "2026-08-16T12:00:00+00:00",
                "query": "older calculation",
            }))
            (newer / "workflow_state.json").write_text(json.dumps({
                "status": "awaiting_user", "updated_at": "2026-08-17T12:00:00+00:00",
                "query": "newer calculation",
            }))
            workflows = _discover_workflows(root)
            self.assertEqual([item["name"] for item in workflows], ["newer_run", "older_run"])
            self.assertEqual(_resolve_workflow_to_open("latest", root), str(newer.resolve()))
            self.assertEqual(_resolve_workflow_to_open("2", root), str(older.resolve()))
            self.assertEqual(_resolve_workflow_to_open(str(older), root), str(older.resolve()))

    def test_scalar_save_state_is_not_cloned_into_soc_scf(self):
        parent = types.SimpleNamespace(id=1, tool="pw_vc_relax", branch="core")
        child = types.SimpleNamespace(id=2, tool="pw_scf", branch="soc_refinement")
        plan = [
            {"id": 1, "tool": "pw_vc_relax"},
            {"id": 2, "tool": "pw_scf"},
        ]
        packages = [
            {"exec_name": "pw.x", "pseudo_dir": "/pseudo/PBE_SR"},
            {"exec_name": "pw.x", "pseudo_dir": "/pseudo/PBE_FR"},
        ]
        self.assertFalse(_may_clone_parent_qe_state(parent, child, plan, packages))

    def test_soc_save_state_is_cloned_into_dependent_soc_bands(self):
        parent = types.SimpleNamespace(id=2, tool="pw_scf", branch="soc_refinement")
        child = types.SimpleNamespace(id=3, tool="pw_bands", branch="soc_refinement")
        plan = [{"id": 2, "tool": "pw_scf"}, {"id": 3, "tool": "pw_bands"}]
        packages = [
            {"exec_name": "pw.x", "pseudo_dir": "/pseudo/PBE_FR"},
            {"exec_name": "pw.x", "pseudo_dir": "/pseudo/PBE_FR"},
        ]
        self.assertTrue(_may_clone_parent_qe_state(parent, child, plan, packages))

    def test_postprocessors_do_not_create_false_pseudo_library_conflicts(self):
        packages = [
            {"work_dir": "/run/soc", "exec_name": "pw.x", "pseudo_dir": "/pseudo/PBE_FR"},
            {"work_dir": "/run/soc", "exec_name": "bands.x", "pseudo_dir": "/pseudo/PBE"},
            {"work_dir": "/run/soc", "exec_name": "dos.x", "pseudo_dir": "/pseudo/PBE"},
        ]
        self.assertEqual(
            _pw_pseudo_dirs_for_work_dir(packages, "/run/soc", "/fallback"),
            {"/pseudo/PBE_FR"},
        )

    def test_workflow_graph_shows_parallel_electronic_branches(self):
        steps = [
            {"id": 1, "tool": "pw_vc_relax", "problem": "Scalar-relativistic vc-relax"},
            {"id": 2, "tool": "pw_scf", "problem": "Fully relativistic SOC SCF"},
            {"id": 3, "tool": "pw_bands", "problem": "SOC bands"},
            {"id": 4, "tool": "bands_post", "problem": "Postprocess SOC bands"},
            {"id": 5, "tool": "pw_nscf", "problem": "SOC dense-grid NSCF"},
            {"id": 6, "tool": "dos_post", "problem": "SOC total DOS"},
        ]
        graph = _workflow_graph_text(steps)
        self.assertIn("[1] pw_vc_relax", graph)
        self.assertIn("├── ○ [3] pw_bands", graph)
        self.assertIn("└── ○ [5] pw_nscf", graph)
        self.assertIn("Parallel opportunity", graph)
        self.assertIn("step(s) 2", graph)

    def test_workflow_graph_can_render_execution_states(self):
        steps = [
            {"id": 1, "tool": "pw_scf", "problem": "SCF"},
            {"id": 2, "tool": "pw_nscf", "problem": "Dense NSCF"},
        ]
        graph = _workflow_graph_text(steps, {1: "completed", 2: "running"})
        self.assertIn("✓ [1]", graph)
        self.assertIn("● [2]", graph)

    def test_approval_decision_supports_revision_and_legacy_callbacks(self):
        self.assertEqual(_approval_result(True), ("approve", ""))
        self.assertEqual(_approval_result(False), ("cancel", ""))
        self.assertEqual(
            _approval_result({"action": "revise", "revision": " add vc-relax "}),
            ("revise", "add vc-relax"),
        )

    def test_repeated_failure_stays_in_recovery_console_until_new_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "scf.in"
            input_path.write_text(PW_INPUT, encoding="utf-8")
            plan = [{"id": 1, "problem": "SCF", "tool": "pw_scf", "input": "structure", "why": "ground state"}]
            packages = [{
                "input_paths": [str(input_path)],
                "output_paths": [str(root / "scf.out")],
                "work_dir": str(root),
                "exec_name": "pw.x",
            }]
            checkpoint = create_checkpoint("SCF", root, plan, packages)
            checkpoint.step(1).set_status("awaiting_user", "first failure")
            checkpoint.status = "awaiting_user"
            checkpoint.save()
            remote = _TestRemoteAgent(
                _FakeAgent(root),
                _FakeTransport({"approved": True, "uploaded": []}),
                approval_callback=lambda *_args: True,
            )
            with patch("builtins.input", side_effect=["retry", "new"]), patch.object(
                remote, "resume", side_effect=RuntimeError("second failure")
            ) as resume:
                result = remote.recovery_console(str(root), "first failure")
            self.assertEqual(result["status"], "new_request")
            self.assertEqual(resume.call_count, 1)

    def test_explicit_lda_selects_lda_pseudopotential_library(self):
        agent = object.__new__(DFTAgent)
        agent.pseudo_dirs = types.SimpleNamespace(
            LDA="/pseudo/LDA", PBE="/pseudo/PBE", PBESOL="/pseudo/PBEsol",
            PBE_FR="/pseudo/PBE-FR", PBESOL_FR="/pseudo/PBEsol-FR",
        )
        self.assertEqual(agent.select_pseudo_dir("Calculate Raman using LDA"), "/pseudo/LDA")
        self.assertEqual(agent.pseudo_dir, "/pseudo/LDA")

    def test_assessed_soc_scope_controls_pseudopotential_relativity(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = object.__new__(DFTAgent)
            agent.pseudo_dirs = types.SimpleNamespace(
                PBE="/pseudo/PBE", PBE_FR="/pseudo/PBE_FR",
                PBESOL="/pseudo/PBESOL", PBESOL_FR="/pseudo/PBESOL_FR",
                LDA="/pseudo/LDA",
            )
            optional = {"soc_policy": {"classification": "optional_refinement", "scope": "electronic_only"}}
            required = {"soc_policy": {"classification": "required", "scope": "electronic_only"}}
            self.assertEqual(agent.select_pseudo_dir("bulk MoS2 bands", optional), "/pseudo/PBE")
            self.assertEqual(agent.select_pseudo_dir("bulk MoS2 bands", required), "/pseudo/PBE_FR")
            self.assertEqual(
                agent.select_pseudo_dir("bulk MoS2 bands", required, target_scope="baseline"),
                "/pseudo/PBE",
            )

    def test_fresh_phonon_branch_forces_recover_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon-grid.in"
            path.write_text("&inputph\n nq1=4,\n nq2=4,\n nq3=4,\n recover=.true.,\n/\n")
            _force_ph_fresh_start(str(path))
            text = path.read_text(encoding="utf-8")
            self.assertIn("recover=.false.,", text)
            self.assertNotIn("recover=.true.", text)

    def test_phonon_command_forces_safe_diagonalization_group(self):
        command = _enforce_safe_qe_parallel_flags(
            "ph.x", "mpirun -np 32 $exe -nk 1 -in $INPUT > $OUTPUT"
        )
        self.assertIn("$exe -nd 1 -nk 1", command)

    def test_legacy_phonon_input_is_deterministically_migrated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonon-grid.in"
            path.write_text(
                "&inputph\n prefix='wf',\n outdir='./',\n fildyn='si.dyn',\n"
                " tr2_ph=1.0d-14,\n recover=.false.\n/\n",
                encoding="utf-8",
            )
            step = {"tool": "pw_phonon_gamma", "problem": "uniform phonon q-grid dispersion"}
            _enforce_workflow_artifact_names(step, str(path))
            _normalize_namelist_final_commas(str(path))
            text = path.read_text(encoding="utf-8")
            self.assertIn("fildyn='tritondft_workflow.dyn'", text)
            self.assertIn("recover=.false.,\n/", text)

    def test_branch_seed_copies_only_qe_state(self):
        transport = SSHClusterTransport("expanse", "/remote", verbose=False)
        with patch("cluster_agent._run_interactive") as runner:
            transport.clone_remote_directory("/remote/run/branches/core", "/remote/run/branches/bands")
        command = runner.call_args.args[0]
        self.assertIn("tritondft_workflow.save", command)
        self.assertIn("tritondft_workflow.xml", command)
        self.assertNotIn("cp -a /remote/run/branches/core/. ", command)

    def test_dynmat_declares_gamma_dynamical_matrix_as_parent_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raman-analysis.in"
            path.write_text(
                "&input\n fildyn='tritondft_workflow.dynG',\n asr='simple',\n/\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _required_parent_artifacts(
                    {"tool": "dynmat_post"}, [str(path)]
                ),
                ["tritondft_workflow.dynG"],
            )

    def test_remote_artifact_copy_checks_and_copies_dynmat_input(self):
        transport = SSHClusterTransport("expanse", "/remote", verbose=False)
        with patch("cluster_agent._run_interactive") as runner:
            transport.copy_remote_artifacts(
                "/remote/run/03-phonon", "/remote/run/04-dynmat",
                ["tritondft_workflow.dynG"],
            )
        command = runner.call_args.args[0]
        self.assertIn("Missing required workflow artifact", command)
        self.assertIn("tritondft_workflow.dynG", command)
        self.assertIn("cp -a", command)

    def test_all_inputs_are_generated_and_approved_before_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {"approved": False, "uploaded": []}
            agent = _FakeAgent(Path(tmp))
            transport = _FakeTransport(state)

            def approve(plan, input_paths):
                self.assertEqual(agent.generated, [1, 2])
                self.assertEqual(len(input_paths), 2)
                self.assertEqual([Path(path).name for path in input_paths], ["vc-relax.in", "scf.in"])
                self.assertIn("Step 1: Relax", plan)
                self.assertIn("Input file(s): vc-relax.in", plan)
                self.assertIn("Step 2: SCF", plan)
                self.assertIn("Input file(s): scf.in", plan)
                downstream = Path(input_paths[1]).read_text(encoding="utf-8")
                self.assertIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER", downstream)
                self.assertEqual(transport.ensure_calls, 0)
                self.assertFalse(list(Path(tmp).glob("slurm_job_*.sh")))
                state["approved"] = True
                return True

            downloads = []
            remote = _TestRemoteAgent(
                agent,
                transport,
                approval_callback=approve,
                download_callback=lambda event, run_dir: downloads.append((event, run_dir)) or "none",
            )
            result = remote.run("vc-relax then scf")

            self.assertEqual(result["status"], "success")
            self.assertEqual(transport.ensure_calls, 1)
            self.assertEqual(
                transport.submissions,
                ["slurm_job_vc-relax.sh", "slurm_job_scf.sh"],
            )
            self.assertFalse(list(Path(tmp).rglob("*_probe.in")))
            self.assertNotIn("CRASH", state.get("uploaded_files", []))
            self.assertNotIn("qe.out", state.get("uploaded_files", []))
            self.assertEqual(transport.directory_fetches, 0)
            self.assertEqual(downloads, [("completed", str(Path(tmp).resolve()))])
            context = json.loads((Path(tmp) / "workflow_context.json").read_text(encoding="utf-8"))
            self.assertEqual(context["pseudo_library"], str(Path(agent.pseudo_dir).resolve()))
            checkpoint = WorkflowCheckpoint.load(tmp)
            self.assertEqual([len(step.attempt_history) for step in checkpoint.steps], [1, 1])
            self.assertTrue(all("attempt_001" in step.remote_dir for step in checkpoint.steps))
            approved_scf = (Path(tmp) / "branches" / "core" / "scf.in").read_text(encoding="utf-8")
            self.assertIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER", approved_scf)
            materialized = next(
                (Path(tmp) / "attempts" / "02-scf").glob("attempt_*/scf.in")
            ).read_text(encoding="utf-8")
            self.assertIn("5.40 0.00 0.00", materialized)
            self.assertNotIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER", materialized)


if __name__ == "__main__":
    unittest.main()
