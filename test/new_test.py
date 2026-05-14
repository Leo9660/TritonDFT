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
# mpr.materials.search(formula="Si", spacegroup_symbol="Fd-3m", fields=["material_id", "initial_structures"])
# mpr.materials.search(formula=\"Si\", spacegroup_symbol=\"Fd-3m\", fields=[\"material_id\", \"initial_structures\"])
    agent = DFTAgent(
        model="gpt-5.2",
        dft_tool="quantum espresso",
        verbose=True,
        backend="openai",  # Set to OpenAI backend
        work_dir="",
        max_new_tokens=4096,
        top_p=0.9,
        # openai_api_key="your-openai-api-key",  # Provide OpenAI API key
        openai_base_url="https://api.openai.com/v1",  # Optional, for custom base URL
        need_query_info=True,
        parallel_exec=False,
        evaluation_mode=False,
        output_log=True,
        output_log_file="evaluation.log",
        run_mode="mpirun",
        # run_mode="slurm",
        parallel_np=64,
        # auto_parallel=True,
        benchmark=True,
        auto_confirm=True,
        # script_only=True,
    )

    # agent = DFTAgent(
    #     model="claude-opus-4-5",
    #     dft_tool="quantum espresso",
    #     verbose=True,
    #     backend="claude",  # Set to OpenAI backend
    #     work_dir="",
    #     max_new_tokens=4096,
    #     top_p=0.9,
    #     # openai_api_key="your-openai-api-key",  # Provide OpenAI API key
    #     openai_base_url="https://api.openai.com/v1",  # Optional, for custom base URL
    #     need_query_info=True,
    #     parallel_exec=False,
    #     evaluation_mode=True,
    #     output_log=True,
    #     output_log_file="evaluation.log",
    #     run_mode="mpirun",
    #     # run_mode="slurm",
    #     parallel_np=16,
    #     auto_parallel=True,
    #     benchmark=True,
    #     auto_confirm=True,
    # )

    # agent = DFTAgent(
    #     model="gemini-2.5-pro",
    #     dft_tool="quantum espresso",
    #     verbose=True,
    #     backend="gemini",  # Set to OpenAI backend
    #     work_dir="",
    #     max_new_tokens=8192,
    #     top_p=0.9,
    #     # openai_api_key="your-openai-api-key",  # Provide OpenAI API key
    #     openai_base_url="https://api.openai.com/v1",  # Optional, for custom base URL
    #     need_query_info=True,
    #     parallel_exec=False,
    #     evaluation_mode=True,
    #     output_log=True,
    #     output_log_file="evaluation.log",
    #     run_mode="mpirun",
    #     # run_mode="slurm",
    #     parallel_np=32,
    #     # auto_parallel=True,
    #     benchmark=True,
    #     auto_confirm=True,
    #     script_only=True,
    # )

    # BaTiO3 new
    query = "Perform a vc-relax calculation for tetragonal BaTiO3 (space group P4mm, #99) using the PBE functional with PAW pseudopotentials."\
    " Use a 6x6x6 Monkhorst-Pack k-point grid and a plane-wave cutoff energy of 650 eV. "\
    "Return the relaxed lattice parameters and atomic positions."

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
    "Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, " \
    "and conv_thr = 1.0e-8. Set the k point as 6. " \
    "Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for " \
    "ecutwfc and the k-point."

    query = "For material = Si with space group Fd-3m and structure = diamond cubic, " \
    " Use a full workflow consisting of: 1) vc-relax, 2) scf, 3) nscf, 4) pw.x calculation=\'bands\', 5) band post-processing, and 6) DOS post-processing. "\
    "Do not need to explicitly return band structure. Just finish the calculation."

    # query = "For material = Si with space group Fd-3m and structure = diamond cubic, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = PBE. " \
    # "Make a reasonable educated guess for ecutwfc and the k-point."

    query = "For material = Si with space group Fd-3m and structure = diamond cubic, " \
        "perform a variable-cell relaxation (vc-relax) calculation with " \
        "exchange-correlation functional = PBE and an energy convergence threshold of 1 meV. " \
        "Make a reasonable educated guess for ecutwfc and the k-point."

    query = "For formula=BaTiO3 structure=perovskite atoms_per_primitive_cell=5 space_group=P4mm, " \
    "use a full workflow consisting of: 1) vc-relax, 2) scf, 3) nscf, 4) pw.x calculation=\"bands\", " \
    "5) band post-processing, and 6) DOS post-processing. Require a strict energy convergence of 10 meV/atom. " \
    "Set use a gamma-centered grid grid, make a educated guess for etot_conv_thr, forc_conv_thr, conv_thr, ecutwfc, k-point sampling. " \
    "Use PBE pseudopotentials. Do not need to explicitly return band structure. Just finish the calculation."

    query = "For material = Si with space group Fd-3m and structure = diamond cubic, " \
        "perform a variable-cell relaxation (vc-relax) calculation with " \
        "exchange-correlation functional = PBE and an energy convergence threshold of 1 meV. " \
        "Then perform a phonon calculation at the Gamma point to check dynamical stability. "

    query = "For material = Si with space group Fd-3m and structure = diamond cubic, " \
        "perform a variable-cell relaxation (vc-relax) calculation with " \
        "exchange-correlation functional = PBE and an energy convergence threshold of 1 meV. " \
        "Then perform a phonon calculation with the full phonon dispersion."

    query = (
        "For material = MgO with space group Fm-3m and rock-salt crystal structure, "
        "perform a variable-cell relaxation (vc-relax) calculation using the PBE "
        "exchange-correlation functional with an energy convergence threshold of 1 meV. "
        "Then compute the full phonon dispersion. "
        "In addition, perform Brillouin-zone integration to obtain the phonon density "
        "of states and evaluate temperature-dependent phonon thermodynamic properties, "
        "including the Helmholtz free energy, entropy, and constant-volume heat capacity, "
        "with the q-point sampling density adaptively selected to ensure convergence "
        "of the thermodynamic quantities."
    )

    query = (
        "NaCl (Fm-3m, rock-salt): vc-relax with PBE, followed by full phonon calculation "
        "to obtain phonon dispersion, phonon DOS, and phonon thermodynamic quantities."
    )

    query = (
        "Sb (R-3m, topological semimetal): elastic energy–strain response under small uniaxial strain."
    )

    query = ("Perform a elastic energy–strain response  for Sb (R-3m) under fixed uniaxial strain. "\
    "Compute the total energy for strains $\epsilon_{zz}$ ranging from -2% to +2%. Use high precision settings (conv_thr=1.0d-12)"\
    " and include spin-orbit coupling.")

    query = "Compute the electronic band structure of Si (space group Fd-3m) using the PBE functional along a standard high-symmetry k-path."

    query = "I want to calculate Raman spectra for HfTe5 with space group cmcm. Select input parameters for structural relaxation accordingly. It is known for having a sensitive electronic structure often near a topological phase transition. Use this to generate the vcrcelax input file"

    query = (
        "For material = Bi2Se3 with space group R-3m and rhombohedral crystal structure, "
        "perform a variable-cell relaxation (vc-relax) calculation using the PBE "
        "exchange-correlation functional with an energy convergence threshold of 1 meV. "
        "Then compute the full phonon dispersion along high-symmetry k-paths. "
        "Make a reasonable educated guess for ecutwfc and k-point sampling."
    )

    # query = "For formula=LiNbO3 structure=trigonal atoms_per_primitive_cell=10 space_group=R3c, use a full workflow consisting of: 1) vc-relax, 2) scf, 3) nscf, 4) pw.x calculation=\"bands\", 5) band post-processing, and 6) DOS post-processing. Set use a gamma-centered grid grid, make a educated guess for etot_conv_thr, forc_conv_thr, conv_thr, ecutwfc, k-point sampling. Use PBE pseudopotentials. Do not need to explicitly return band structure. Just finish the calculation."

    # query = "For material = Li with structure = bcc using the primitive cell, " \
    # "Use a full workflow consisting of: 1) vc-relax, 2) scf, 3) nscf, 4) band structure calculation with exchange-correlation functional = PBE. " \
    # " Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for k and ecutwfc." \
    # " Based on the materials, you should choose whether to use Van der Waals correction or not." \
    # " Based on the material properties, choose whether to use magnetic moments, spin polarization, spin-orbit coupling." \
    # " Remember that we need to use fully relativistic pseudopotentials when including spin-orbit coupling."

    # " Try not to choose the parameter values less than the mentioned default in Quantum ESPRESSO documentation. "

    # query = "For material = BaTiO3 with space group = P4mm using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # " Set the k point as 6. " \
    # "Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for " \
    # "ecutwfc and the k-point."

    # query = "For material = PbTiO3 with space group = P4mm using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-5, " \
    # "and conv_thr = 1.0e-8. Set the k point as 6. " \
    # "Use an automatic half-shifted Monkhorst-Pack grid, and make a reasonable educated guess for " \
    # "ecutwfc and the k-point."

    # query = "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, " \
    # "perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. " \
    # "lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7," \
    # " and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for "\
    # " ecutwfc and the k-point. Use a half-shifted. Return the fully relaxed structure (atomic parameters)."

    result = agent.run(query)

if __name__ == "__main__":
    main()
