import tempfile
import unittest
from pathlib import Path
import os
import sys
import types

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
    _add_relaxed_structure_placeholder,
    _env_missing_cluster_setup,
    _extract_relaxed_structure,
    _ensure_env_defaults,
    _input_validation_errors,
    _insert_relaxed_structure,
    _plan_text,
)
from execute_code.slurm import SlurmLauncher
from DFTAgent import _generate_nonempty_text


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
"""


class PlaceholderTests(unittest.TestCase):
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
                    "TRITONDFT_SLURM_TEMPLATE=~/.tritondft/example_slurm_job_file.txt",
                    env_text,
                )
                user_template = Path(tmp) / ".tritondft" / "example_slurm_job_file.txt"
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
            self.assertIn("missing &control namelist", errors)
            self.assertIn("missing K_POINTS card", errors)

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
        input_path.write_text(
            PW_INPUT.replace("'scf'", f"'{step['tool'].replace('pw_', '')}'"),
            encoding="utf-8",
        )
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

    def ensure_connection(self):
        self.ensure_calls += 1
        if not self.state["approved"]:
            raise AssertionError("SSH opened before approval")

    def remote_dir_for(self, local_run_dir):
        return f"/remote/{local_run_dir.name}"

    def upload_directory(self, local_dir, remote_dir):
        self.state["uploaded"].append(remote_dir)

    def submit(self, remote_dir, script_name):
        self.submissions.append(script_name)
        return ClusterJob("1", script_name, remote_dir, "Submitted batch job 1")

    def wait_for_job(self, job):
        return None

    def fetch_directory(self, remote_dir, local_dir):
        for probe_input in local_dir.glob("*_probe.in"):
            probe_output = probe_input.with_suffix(".out")
            probe_output.write_text(
                "number of k points= 8\nnumber of Kohn-Sham states= 4\n",
                encoding="utf-8",
            )
        if not (local_dir / "output_1_1.out").exists():
            (local_dir / "output_1_1.out").write_text(RELAX_OUTPUT, encoding="utf-8")
        (local_dir / "output_2_1.out").write_text("JOB DONE.\n", encoding="utf-8")


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
    def test_all_inputs_are_generated_and_approved_before_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = {"approved": False, "uploaded": []}
            agent = _FakeAgent(Path(tmp))
            transport = _FakeTransport(state)

            def approve(plan, input_paths):
                self.assertEqual(agent.generated, [1, 2])
                self.assertEqual(len(input_paths), 2)
                downstream = Path(input_paths[1]).read_text(encoding="utf-8")
                self.assertIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER", downstream)
                self.assertEqual(transport.ensure_calls, 0)
                self.assertFalse(list(Path(tmp).glob("slurm_job_*.sh")))
                state["approved"] = True
                return True

            remote = _TestRemoteAgent(agent, transport, approval_callback=approve)
            result = remote.run("vc-relax then scf")

            self.assertEqual(result["status"], "success")
            self.assertEqual(transport.ensure_calls, 1)
            self.assertEqual(len(transport.submissions), 4)
            materialized = (Path(tmp) / "input_2_1.in").read_text(encoding="utf-8")
            self.assertIn("5.40 0.00 0.00", materialized)
            self.assertNotIn("TRITONDFT_RELAXED_STRUCTURE_PLACEHOLDER", materialized)


if __name__ == "__main__":
    unittest.main()
