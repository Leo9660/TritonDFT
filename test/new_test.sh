#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: bash new_test.sh <query> [options]"
    echo ""
    echo "Examples:"
    echo '  bash new_test.sh "Perform a vc-relax calculation for tetragonal BaTiO3 (space group P4mm, #99) using the PBE functional with PAW pseudopotentials. Use a 6x6x6 Monkhorst-Pack k-point grid and a plane-wave cutoff energy of 650 eV. Return the relaxed lattice parameters and atomic positions."'
    echo ""
    echo '  bash new_test.sh "For material = Si with space group Fd-3m and structure = diamond cubic using the primitive cell, perform a variable-cell relaxation (vc-relax) calculation with exchange-correlation functional = LDA. lattice constant = 5.43 Å. Set the convergence criteria as etot_conv_thr = 1.0e-8, forc_conv_thr = 1.0e-7, and conv_thr = 1.0e-8. Use an automatic Monkhorst-Pack grid, and make a reasonable educated guess for ecutwfc and the k-point. Use a half-shifted. Return the fully relaxed structure (atomic parameters)."'
    echo ""
    echo "Options (pass after the query):"
    echo "  --model MODEL              Model name (default: gpt-4o)"
    echo "  --backend BACKEND          Backend: openai or vllm (default: openai)"
    echo "  --dft-tool TOOL            DFT tool (default: quantum espresso)"
    echo "  --max-new-tokens N         Max new tokens (default: 4096)"
    echo "  --temperature T            Temperature (default: 0.0)"
    echo "  --top-p P                  Top-p (default: 0.9)"
    echo "  --openai-base-url URL      OpenAI base URL"
    echo "  --work-dir DIR             Working directory"
    echo "  --output-log-file FILE     Log file path (default: evaluation.log)"
    echo "  --no-script-only           Disable script_only mode"
    echo "  --no-evaluation            Disable evaluation mode"
    echo "  --no-query-info            Disable need_query_info"
    echo "  --parallel-exec            Enable parallel execution"
    echo "  --vllm-tp-size N           vLLM tensor parallel size (default: 4)"
    exit 1
fi

python "$SCRIPT_DIR/new_test.py" "$@"
