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
        parallel_exec=True,
        parallel_np=12
    )

    # Example query
    
    query = "Perform a vc-relax calculation for diamond cubic silicon with LDA, a 6x6x6 (1 1 1) Monkhorst-Pack k-mesh, and ecutwfc = 100 Ry, ecutrho = 400 Ry. Return the equilibrium lattice constant (Å)."
    query = "Using PBE and the provided hexagonal P-6m2 (space group 187) structure of Y (2 atoms per unit cell), " \
    "perform a vc-relax calculation to determine the equilibrium lattice constants (a, c, c/a) and atomic coordinates."
    query = "Using PBE and the provided O₂ dimer structure in a 10x10x10 Å cubic box, perform a vc-relax calculation "\
    "to optimize the O-O bond length and atomic positions. Report: total energy per atom, optimized O–O bond length, and "\
    " whether the system converges to a spin-polarized ground state. Use a 4x4x4 k-point grid and cutoffs ecutwfc≈80–100 Ry, "\
    "ecutrho=4xecutwfc."

    query = "Run QE vc-relax (LDA) for CH3NH3PbI3 in the tetragonal I4cm structure." \
    "Use ecutwfc = 80 Ry, ecutrho = 320 Ry, and a 5x4x5 k-point grid (1 1 1 shift)." \
    "Disable symmetry and keep ion_dynamics='bfgs', cell_dynamics='bfgs'." \
    "Report final relaxed lattice constants (a, c, c/a), fractional atomic positions, total energy, and stress."

    query = "Run QE vc-relax using LDA for CH3NH3PbI3 in the cubic Fd-3m structure. " \
    "Use ecutwfc = 100 Ry, ecutrho = 400 Ry, and a 6x6x6 k-point grid (1 1 1 shift). " \
    "Disable symmetry (nosym, noinv) and set ion_dynamics='bfgs', cell_dynamics='bfgs'. " \
    "Report final relaxed lattice constants (a, b, c, angles), fractional atomic positions, total energy, and stress."

    # Si three pseudo potentials
    query = "Perform a vc-relax calculation for diamond cubic silicon with LDA, a 6x6x6 (1 1 1) Monkhorst-Pack k-mesh, " \
    "and ecutwfc = 100 Ry, ecutrho = 400 Ry. Return the equilibrium lattice constant (Å)."

    query = "Perform a vc-relax calculation for diamond cubic silicon with PBE, a 6x6x6 (1 1 1) Monkhorst-Pack k-mesh, " \
    "and ecutwfc = 100 Ry, ecutrho = 400 Ry. Return the equilibrium lattice constant (Å)."

    query = "Perform a vc-relax calculation for diamond cubic silicon with PBEsol, a 6x6x6 (1 1 1) Monkhorst-Pack k-mesh, " \
    "and ecutwfc = 100 Ry, ecutrho = 400 Ry. Return the equilibrium lattice constant (Å)."

    # BaTiO3 new
    query = "Perform a vc-relax calculation for tetragonal BaTiO3 (space group P4mm, #99) using the PBE functional with PAW pseudopotentials." \
    " Use a 6x6x6 Monkhorst-Pack k-point grid and a plane-wave cutoff energy of 650 eV. "\
    "Ensure that the relaxation is performed without enforcing symmetry constraints so that the system can relax into the polar ground state. "\
    "Return the relaxed lattice parameters and atomic positions."

    # Ti
    query = "Using PBE and the provided hexagonal P-6m2 (space group 187) Ti structure (3 atoms per unit cell), " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameters (a, c, and c/a) and atomic positions." \
    "Use a Γ-centered 5x5x9 k-point mesh and a plane-wave cutoff of 520 eV."

    # Y
    query = "Using PBE and the provided hexagonal P-6m2 (space group 187) Y structure (2 atoms per unit cell), " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameters (a, c, and c/a) and atomic positions. " \
    "Use a Γ-centered 7x7x4 k-point mesh and a plane-wave cutoff of 500 eV."

    # O2
    query = "Using PBE and the provided O2 dimer structure in a cubic box (no symmetry, 2 atoms per unit cell), " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameters and atomic positions. " \
    "Use a Γ-centered 4x4x4 k-point mesh and a plane-wave cutoff of 600 eV."

    # CH3NH3PbI3
    query = "Using LDA and the provided cubic Fd-3m (space group 227) CH3NH3PbI3 perovskite structure, " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameter a and the relaxed atomic positions. " \
    "Use a 6x6x6 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 100 Ry."

    query = "Using LDA and the provided tetragonal I4/mcm CH3NH3PbI3 perovskite structure, " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameters (a, c, and c/a) and the relaxed atomic positions. " \
    "Use a 5x4x5 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 80 Ry."

    query = "Using LDA and the provided tetragonal I4cm CH3NH3PbI3 perovskite structure, " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameters (a, c, and c/a) and the relaxed atomic positions. " \
    "Use a 5x4x5 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 80 Ry."

    # CuAl2O4 successful
    query = "Using PBE and the provided cubic Fd-3m (space group 227) CuAl2O4 spinel structure (14 atoms per unit cell), " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameter a, the oxygen internal parameter u, " \
    "and the relaxed atomic positions. Use an 8x8x8 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 600 eV."

    # TbInO3 not found
    query = "Using PBE and the provided hexagonal P6_3cm (space group 185) TbInO3 structure, " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameters (a, c, and c/a) and the relaxed atomic positions. " \
    "Use a Γ-centered 4x4x4 k-point mesh and a plane-wave cutoff of 600 eV."

    # CH3NH3PbI3 no MP entry
    query = "Using LDA and the provided cubic Fd-3m (space group 227) CH3NH3PbI3 perovskite structure, " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameter a and the relaxed atomic positions. " \
    "Use a 6x6x6 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 100 Ry."

    # PbTiO3 new
    query = "Using LDA to Run QE vc-relax for PbTiO3 in the tetragonal P4mm perovskite structure. " \
    "Use ecutwfc = 120 Ry, ecutrho = 480 Ry, and a 10x10x10 k-point grid. "\
    "Ensure that the relaxation is performed without enforcing symmetry constraints so that the system can relax into the polar ground state. "\
    "Report the final relaxed values of lattice constants a and c."

    # BaTiO3 new
    query = "Perform a vc-relax calculation for tetragonal BaTiO3 (space group P4mm, #99) using the PBE functional with PAW pseudopotentials."\
    " Use a 6x6x6 Monkhorst-Pack k-point grid and a plane-wave cutoff energy of 650 eV. "\
    "Return the relaxed lattice parameters and atomic positions."

    # Y2V2O7 parallel
    query = "Using PBE and the provided cubic Fd-3m (space group 227) Y2V2O7 pyrochlore structure in the fcc primitive cell (22 atoms), " \
    "perform a spin-polarized vc-relax to obtain the equilibrium lattice parameter a, the oxygen internal parameter u, " \
    "and the relaxed atomic positions. Use a 6x6x6 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 600 eV."

    # Y2Ti2O7 too slow
    query = "Using PBE and the provided cubic Fd-3m (space group 227) Y2Ti2O7 pyrochlore structure, " \
    "perform a vc-relax calculation to obtain the equilibrium lattice parameter a, the oxygen internal parameter u, " \
    "and the relaxed atomic positions. Use a 4x4x4 Monkhorst-Pack k-point mesh and a plane-wave cutoff of 520 eV."

    # Run the agent
    result = agent.run(query)


if __name__ == "__main__":
    main()