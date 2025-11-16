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
        "--tasks",
        nargs="+",
        default="vc_relax",
        help="Subset of benchmark tasks to evaluate (defaults to every registered task).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of prompts to evaluate across all tasks.",
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
        default=2048,
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
    return parser.parse_args()


def collect_prompts(
    tasks: Optional[List[str]],
    limit: int,
    pseudopotential_family: str,
) -> List[DataItem]:
    dataset = BenchmarkDataset()
    target_tasks = tasks or dataset.tasks
    prompts = []
    for task in target_tasks:
        prompts.extend(
            dataset.collect(
                task_name=task,
                pseudopotential_family=pseudopotential_family,
            )
        )
        if len(prompts) >= limit:
            break
        print(f"Prompt {len(prompts)} collected from task {task}, {prompts[-1]}.")
    return prompts[:limit]


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
    )


def main() -> None:
    args = parse_args()
    prompts = collect_prompts(args.tasks, args.limit, args.pseudo)

    if not prompts:
        print("No prompts collected from BenchmarkDataset. Exiting.")
        return

    agent = None if args.dry_run else build_agent(args)

    for idx, item in enumerate(prompts, start=1):
        header = f"[{idx}/{len(prompts)}][{item.task}]"
        print(f"{header} Prompt (pseudo={args.pseudo}):\n{item.prompt}\n")
        if args.dry_run:
            continue

        try:
            result = agent.run(item.prompt)
            print(f"{header} Result: {result}\n")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{header} Failed with error: {exc}\n")


if __name__ == "__main__":
    main()
