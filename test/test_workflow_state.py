import json
import tempfile
import unittest
from pathlib import Path

from src.workflow_state import WorkflowCheckpoint, create_checkpoint, infer_branches, infer_dependencies


class WorkflowStateTests(unittest.TestCase):
    def setUp(self):
        self.plan = [
            {"id": 1, "tool": "pw_vc_relax", "problem": "relax"},
            {"id": 2, "tool": "pw_scf", "problem": "SCF"},
            {"id": 3, "tool": "pw_bands", "problem": "bands"},
            {"id": 4, "tool": "bands_post", "problem": "post-process bands"},
            {"id": 5, "tool": "pw_nscf", "problem": "DOS NSCF"},
            {"id": 6, "tool": "dos_post", "problem": "total DOS"},
            {"id": 7, "tool": "projwfc_post", "problem": "PDOS"},
            {"id": 8, "tool": "pw_phonon_gamma", "problem": "Gamma phonons and Raman"},
            {"id": 9, "tool": "dynmat_post", "problem": "Raman tensors"},
            {"id": 10, "tool": "pw_phonon_gamma", "problem": "phonon dispersion q-grid"},
            {"id": 11, "tool": "q2r_post", "problem": "force constants"},
            {"id": 12, "tool": "matdyn_post", "problem": "phonon dispersion"},
        ]

    def test_dependency_graph_exposes_independent_post_scf_branches(self):
        dependencies = infer_dependencies(self.plan)
        by_id = {step["id"]: parents for step, parents in zip(self.plan, dependencies)}
        self.assertEqual(by_id[2], [1])
        self.assertEqual(by_id[3], [2])
        self.assertEqual(by_id[5], [2])
        self.assertEqual(by_id[8], [2])
        self.assertEqual(by_id[10], [2])
        self.assertEqual(by_id[4], [3])
        self.assertEqual(by_id[6], [5])
        self.assertEqual(by_id[7], [5])
        self.assertEqual(by_id[9], [8])
        self.assertEqual(by_id[11], [10])
        self.assertEqual(by_id[12], [11])

    def test_branch_classification(self):
        branches = dict(zip((step["id"] for step in self.plan), infer_branches(self.plan)))
        self.assertEqual(branches[1], "core")
        self.assertEqual(branches[3], "bands")
        self.assertEqual(branches[5], "dos_pdos")
        self.assertEqual(branches[8], "gamma_raman")
        self.assertEqual(branches[10], "phonon_dispersion")

    def test_checkpoint_round_trip_and_targeted_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packages = []
            for step in self.plan:
                branch = infer_branches([step])[0]
                branch_dir = root / "branches" / branch
                branch_dir.mkdir(parents=True, exist_ok=True)
                input_path = branch_dir / f"input_{step['id']}_1.in"
                output_path = branch_dir / f"output_{step['id']}_1.out"
                input_path.write_text("input\n", encoding="utf-8")
                output_path.write_text("output\n", encoding="utf-8")
                packages.append({"input_paths": [str(input_path)], "output_paths": [str(output_path)]})

            state = create_checkpoint("test", root, self.plan, packages)
            for checkpoint in state.steps:
                checkpoint.set_status("completed")
            state.save()
            loaded = WorkflowCheckpoint.load(root)
            self.assertEqual(loaded.status, "approved")
            invalidated = loaded.invalidate_descendants(5)
            self.assertEqual(set(invalidated), {6, 7})
            self.assertEqual(loaded.step(4).status, "completed")
            self.assertEqual(loaded.step(8).status, "completed")
            self.assertEqual(loaded.step(6).status, "ready")
            self.assertEqual(loaded.step(7).status, "ready")
            json.loads((root / "workflow_state.json").read_text(encoding="utf-8"))

    def test_extension_provenance_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "phonon.in"
            input_path.write_text("input\n", encoding="utf-8")
            state = create_checkpoint(
                "Raman extension",
                root,
                [{"id": 1, "tool": "pw_phonon_gamma", "problem": "Gamma Raman"}],
                [{"input_paths": [str(input_path)], "output_paths": [str(root / "phonon.out")]}],
            )
            state.parent_run_dir = "/parent/run"
            state.step(1).seed_remote_dir = "/remote/parent/scf/attempt_001"
            state.step(1).reused_from_run = "/parent/run"
            state.save()
            loaded = WorkflowCheckpoint.load(root)
            self.assertEqual(loaded.parent_run_dir, "/parent/run")
            self.assertEqual(loaded.step(1).seed_remote_dir, "/remote/parent/scf/attempt_001")

    def test_completed_input_mutation_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.in"
            output_path = root / "output.out"
            input_path.write_text("before\n", encoding="utf-8")
            output_path.write_text("done\n", encoding="utf-8")
            state = create_checkpoint(
                "test", root, [{"id": 1, "tool": "pw_scf", "problem": "SCF"}],
                [{"input_paths": [str(input_path)], "output_paths": [str(output_path)]}],
            )
            state.step(1).set_status("completed")
            state.save()
            input_path.write_text("after\n", encoding="utf-8")
            self.assertIn("input changed", state.verify_completed_inputs()[0])


if __name__ == "__main__":
    unittest.main()
