import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure repository modules are importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmark import BenchmarkDataset, DataItem  # noqa: E402
from DFTAgent import DFTAgent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DFTAgent on prompts generated from the BenchmarkDataset."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of prompts to evaluate across all tasks (0 = no limit).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model identifier passed to DFTAgent.",
    )
    parser.add_argument(
        "--backend",
        default="openai",
        help="Generation backend used by DFTAgent (e.g., openai, hf, vllm).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for each model call inside DFTAgent.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the generator.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter.",
    )
    parser.add_argument(
        "--need-query-info",
        action="store_true",
        help="Enable Material Project info query inside the agent.",
    )
    parser.add_argument(
        "--evaluation-mode",
        action="store_true",
        help="Toggle evaluation mode flag on the agent.",
    )
    parser.add_argument(
        "--output-log",
        action="store_true",
        help="Save DFTAgent logs into work_dir/output_log_file.",
    )
    parser.add_argument(
        "--work-dir",
        default="tmp",
        help="Root directory where DFTAgent writes intermediate files.",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional OpenAI API key forwarded to DFTAgent.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="Optional OpenAI base URL forwarded to DFTAgent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print generated prompts without running the agent.",
    )
    parser.add_argument(
        "--pseudo",
        default="LDA",
        help="Pseudopotential family guidance passed to BenchmarkDataset (LDA, PBE, PBE_sol).",
    )
    parser.add_argument(
        "--category",
        default="metal",
        help="Material category passed to BenchmarkDataset.",
    )
    parser.add_argument(
        "--start-from-type",
        default="",
        help="Skip prompts until a task with this type is encountered (matches DataItem.task).",
    )
    parser.add_argument(
        "--start-from-id",
        default="",
        help="Skip prompts until this material ID is encountered (matches metadata.material_id or DataItem.id).",
    )
    parser.add_argument(
        "--task-type",
        default="all",
        choices=["band_gap", "vc_relax", "all", "dos"],
        help="Task type filter for benchmark questions.",
    )
    parser.add_argument(
        "--run-mode",
        default="mpirun",
        choices=["mpirun", "local", "slurm"],
        help="Execution mode for QE runs.",
    )
    parser.add_argument(
        "--parallel-np",
        type=int,
        default=1,
        help="MPI ranks for mpirun or Slurm runs.",
    )
    parser.add_argument(
        "--auto-parallel",
        action="store_true",
        help="Enable auto-parallel probing and command generation.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark logging to CSV.",
    )
    parser.add_argument(
        "--benchmark-file",
        default="benchmark.csv",
        help="CSV file for benchmark outputs.",
    )
    parser.add_argument(
        "--hardware-description",
        default=None,
        help="Optional hardware description passed to auto-parallel prompt.",
    )
    return parser.parse_args()


def collect_prompts(
    limit: int,
    pseudopotential_family: str,
    category: str,
    task_type: str,
) -> List[DataItem]:
    dataset = BenchmarkDataset(category=category, task_type=task_type)
    prompts = []
    for task in dataset.tasks:
        prompts.extend(
            dataset.collect(
                task_name=task,
                pseudopotential_family=pseudopotential_family,
            )
        )
        if limit > 0 and len(prompts) >= limit:
            break
        print(f"Prompt {len(prompts)} collected from task {task}, {prompts[-1]}.")
    return prompts if limit == 0 else prompts[:limit]


def build_agent(args: argparse.Namespace) -> DFTAgent:
    return DFTAgent(
        model=args.model,
        dft_tool="quantum espresso",
        verbose=True,
        backend=args.backend,
        work_dir=args.work_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        need_query_info=args.need_query_info,
        evaluation_mode=args.evaluation_mode,
        output_log=args.output_log,
        output_log_file="benchmark_agent.log",
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        run_mode=args.run_mode,
        parallel_np=args.parallel_np,
        auto_parallel=args.auto_parallel,
        benchmark=args.benchmark,
        benchmark_file=args.benchmark_file,
        hardware_description=args.hardware_description,
        auto_confirm=True
    )


def main() -> None:
    args = parse_args()
    prompts = collect_prompts(args.limit, args.pseudo, args.category, args.task_type)

    start_from_type = (args.start_from_type or "").strip()
    start_from_id = (args.start_from_id or "").strip()
    started = not (start_from_type or start_from_id)

    if not prompts:
        print("No prompts collected from BenchmarkDataset. Exiting.")
        return

    agent = None if args.dry_run else build_agent(args)

    for idx, item in enumerate(prompts, start=1):
        header = f"[{idx}/{len(prompts)}][{item.task}]"
        material_info = (
            item.metadata.get("material_info", {}) if isinstance(item.metadata, dict) else {}
        )
        material_name = material_info.get("formula") or material_info.get("name") or ""
        material_structure = material_info.get("structure", "")
        category = (
            item.metadata.get("category", "") if isinstance(item.metadata, dict) else ""
        ) or args.category
        material_id = ""
        if isinstance(item.metadata, dict):
            # Preserve the material_id exactly as provided in the source JSON.
            material_id = item.metadata.get("material_id", "")
        if not started:
            # type_match = (not start_from_type) or item.task == start_from_type
            # id_match = (not start_from_id) or int(start_from_id) in {material_id, item.id}
            # print(item.task, material_id, item.id, type_match, id_match)
            # started = type_match and id_match
            if start_from_type and start_from_id:
                started = f"{start_from_type}_{start_from_id}" == material_id
            else:
                started = True
            print(f"Skipping prompt {item.task} {material_id or item.id}...")
            if not started:
                continue
        print(f"{header} Prompt (pseudo={args.pseudo}):\n{item.prompt}\n")
        print(
            f"{header} TaskType={item.task} | Material={material_name or 'unknown'} | Category={category} | MaterialID={material_id} | Structure={material_structure or 'unknown'}"
        )
        if args.dry_run:
            continue

        try:
            result = agent.run(
                item.prompt,
                run_id=idx,
                category=category,
                task_type=args.task_type,
                material_name=material_name,
                work_dir=f"/data/icml_test_bed/{args.model}/{category}_{material_id}",
            )
            print(f"{header} Result: {result}\n")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{header} Failed with error: {exc}\n")


if __name__ == "__main__":
    main()
