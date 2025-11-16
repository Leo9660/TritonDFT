import sys
from pathlib import Path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from DFTAgent import DFTAgent


def main():
    # Initialize the agent
    # agent = DFTAgent(
    #     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    #     # model = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     # model = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    #     # model = "openai/gpt-oss-120b",
    #     dft_tool="quantum espresso",
    #     verbose=True,
    #     work_dir="",
    #     max_new_tokens=4096
    # )

    # agent = DFTAgent(
    #     model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    #     # model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    #     dft_tool="quantum espresso",
    #     verbose=True,
    #     backend="vllm",
    #     work_dir="",
    #     max_new_tokens=2048,
    #     vllm_tensor_parallel_size=4,
    #     temperature=0.0,
    #     top_p=0.9,
    # )

    agent = DFTAgent(
        model="gpt-4o",
        dft_tool="quantum espresso",
        verbose=True,
        backend="openai",  # Set to OpenAI backend
        work_dir="",
        max_new_tokens=4096,
        temperature=0.0,
        top_p=0.9,
        # openai_api_key="your-openai-api-key",  # Provide OpenAI API key
        openai_base_url="https://api.openai.com/v1",  # Optional, for custom base URL
        need_query_info=True,
        parallel_exec=False,
        evaluation_mode=True,
        output_log=True,
        output_log_file="evaluation.log"
    )

    # BaTiO3 new
    query = "Perform a vc-relax calculation for tetragonal BaTiO3 (space group P4mm, #99) using the PBE functional with PAW pseudopotentials."\
    " Use a 6x6x6 Monkhorst-Pack k-point grid and a plane-wave cutoff energy of 650 eV. "\
    "Return the relaxed lattice parameters and atomic positions."

    # Example query
    query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7," \
    " and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for "\
    " ecutwfc and the k-point. Use a half-shifted. Return the fully relaxed structure (atomic parameters)."

    query = "For material = Al with space group Fm-3m and structure = FCC using the primitive cell, \
    perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    lattice constant = 4.05 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    query = "For material = Fe with space group Im-3m and structure = BCC using the primitive cell, \
    perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    lattice constant = 2.87 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    query = "For material = NaCl with space group Fm-3m and structure = rocksalt using the primitive cell, \
    perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    lattice constant = 5.64 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    query = "For material = GaAs with space group F-43m and structure = zinc blende using the primitive cell, \
    perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    lattice constant = 5.65 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    query = "For material = MgO with space group Fm-3m and structure = rocksalt using the primitive cell, \
    perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    lattice constant = 4.21 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    query = "For material = Graphene (C) with space group P6₃/mmc and structure = hexagonal using the primitive cell, \
    perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. \
    lattice constant = 2.46 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, \
    and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for \
    ecutwfc and the k-point. Use a half-shifted grid. Return the fully relaxed structure (atomic parameters)."

    query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, " \
    "and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for " \
    "ecutwfc and the k-point. Use a half-shifted grid. After the vc-relax finishes, perform a self-consistent field (scf) calculation " \
    "on the relaxed structure with consistent settings (same system) to obtain the final total energy. " \
    "Return the fully relaxed structure (atomic parameters) and the final scf total energy."

    query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    "Lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, " \
    "and conv_thr = 1.0e-8. Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for " \
    "ecutwfc and the k-point. After the vc-relax finishes, run a self-consistent field (scf) calculation " \
    "on the relaxed structure with consistent settings, then perform a non-self-consistent field (nscf) calculation " \
    "and compute the band gap from that electronic structure."

    # query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7," \
    # " and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for "\
    # " ecutwfc and the k-point. Use a half-shifted. Return the fully relaxed structure (atomic parameters)."

    result = agent.run(query)

if __name__ == "__main__":
    main()