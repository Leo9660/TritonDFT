import sys
import argparse
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from DFTAgent import DFTAgent


def main():
    parser = argparse.ArgumentParser(description="Run DFT Agent with a query")
    parser.add_argument("query", help="The DFT query to execute")
    parser.add_argument("--model", default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument("--backend", default="auto", help="Backend: auto, openai, claude, gemini, vllm, hf (default: auto)")
    parser.add_argument("--dft-tool", default="quantum espresso", help="DFT tool (default: quantum espresso)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (default: 0.9)")
    parser.add_argument("--openai-base-url", default=None, help="OpenAI-compatible base URL (triggers openai backend)")
    parser.add_argument("--work-dir", default="", help="Working directory root")
    parser.add_argument("--output-log-file", default="evaluation.log", help="Log file path")
    parser.add_argument("--no-script-only", action="store_true", help="Disable script_only mode")
    parser.add_argument("--no-evaluation", action="store_true", help="Disable evaluation mode")
    parser.add_argument("--no-query-info", action="store_true", help="Disable need_query_info")
    parser.add_argument("--parallel-exec", action="store_true", help="Enable parallel execution")
    parser.add_argument("--vllm-tp-size", type=int, default=4, help="vLLM tensor parallel size (default: 4)")
    parser.add_argument("--category", default="unknown", help="Category label (default: unknown)")
    parser.add_argument("--run-id", type=int, default=0, help="Run ID for tracking (default: 0)")
    args = parser.parse_args()

    agent_kwargs = dict(
        model=args.model,
        dft_tool=args.dft_tool,
        verbose=True,
        backend=args.backend,
        work_dir=args.work_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        need_query_info=not args.no_query_info,
        parallel_exec=args.parallel_exec,
        evaluation_mode=not args.no_evaluation,
        output_log=True,
        output_log_file=args.output_log_file,
        script_only=not args.no_script_only,
    )

    if args.openai_base_url:
        agent_kwargs["openai_base_url"] = args.openai_base_url
    if args.vllm_tp_size:
        agent_kwargs["vllm_tensor_parallel_size"] = args.vllm_tp_size

    agent = DFTAgent(**agent_kwargs)
    result = agent.run(
        args.query,
        run_id=args.run_id,
        category=args.category,
    )


if __name__ == "__main__":
    main()
