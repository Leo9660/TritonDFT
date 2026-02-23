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
    parser.add_argument("--backend", default="openai", help="Backend: openai or vllm (default: openai)")
    parser.add_argument("--dft-tool", default="quantum espresso", help="DFT tool (default: quantum espresso)")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Max new tokens (default: 4096)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (default: 0.9)")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="OpenAI base URL")
    parser.add_argument("--work-dir", default="", help="Working directory")
    parser.add_argument("--output-log-file", default="evaluation.log", help="Log file path")
    parser.add_argument("--no-script-only", action="store_true", help="Disable script_only mode")
    parser.add_argument("--no-evaluation", action="store_true", help="Disable evaluation mode")
    parser.add_argument("--no-query-info", action="store_true", help="Disable need_query_info")
    parser.add_argument("--parallel-exec", action="store_true", help="Enable parallel execution")
    parser.add_argument("--vllm-tp-size", type=int, default=4, help="vLLM tensor parallel size (default: 4)")
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

    if args.backend == "openai":
        agent_kwargs["openai_base_url"] = args.openai_base_url
    elif args.backend == "vllm":
        agent_kwargs["vllm_tensor_parallel_size"] = args.vllm_tp_size

    # # BaTiO3 new
    # query = "Perform a vc-relax calculation for tetragonal BaTiO3 (space group P4mm, #99) using the PBE functional with PAW pseudopotentials."\
    # " Use a 6x6x6 Monkhorst-Pack k-point grid and a plane-wave cutoff energy of 650 eV. "\
    # "Return the relaxed lattice parameters and atomic positions."

    # # Example query
    # query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7," \
    # " and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for "\
    # " ecutwfc and the k-point. Use a half-shifted. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = Al with space group Fm-3m and structure = FCC using the primitive cell, \
    # perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    # lattice constant = 4.05 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    # and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    # ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = Fe with space group Im-3m and structure = BCC using the primitive cell, \
    # perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    # lattice constant = 2.87 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    # and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    # ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = NaCl with space group Fm-3m and structure = rocksalt using the primitive cell, \
    # perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    # lattice constant = 5.64 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    # and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    # ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = GaAs with space group F-43m and structure = zinc blende using the primitive cell, \
    # perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    # lattice constant = 5.65 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    # and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    # ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = MgO with space group Fm-3m and structure = rocksalt using the primitive cell, \
    # perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    # lattice constant = 4.21 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    # and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    # ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = Graphene (C) with space group P6₃/mmc and structure = hexagonal using the primitive cell, \
    # perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    # lattice constant = 2.46 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    # and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    # ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    # query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, " \
    # "and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for " \
    # "ecutwfc and the k-point. Use a half-shifted grid. After the vc-relax finishes, perform a self-consistent field (scf) calculation " \
    # "on the relaxed structure with consistent settings (same system) to obtain the final total energy. " \
    # "Return the fully relaxed structure (atomic parameters) and the final scf total energy."

    # query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "Lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, " \
    # "and conv_thr = 1.0e-8. Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for " \
    # "ecutwfc and the k-point. After the vc-relax finishes, run a self-consistent field (scf) calculation " \
    # "on the relaxed structure with consistent settings, then perform a non-self-consistent field (nscf) calculation " \
    # "and compute the band gap from that electronic structure."

    # query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7," \
    # " and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for "\
    # " ecutwfc and the k-point. Use a half-shifted. Return the fully relaxed structure (atomic parameters)."

    agent = DFTAgent(**agent_kwargs)
    result = agent.run(args.query)


if __name__ == "__main__":
    main()
