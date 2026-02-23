#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: bash new_test.sh <query> [options]"
    echo ""
    echo "Examples:"
    echo '  bash new_test.sh "Perform a vc-relax calculation for BaTiO3 ..." --material-name BaTiO3 --task-type vc-relax'
    echo ""
    echo '  bash new_test.sh "For material = Si ... perform a vc-relax ..." --material-name Si --task-type vc-relax'
    echo ""
    echo "Tracking options:"
    echo "  --category CAT             Category label (default: unknown)"
    echo "  --run-id N                 Run ID for tracking (default: 0)"
    echo ""
    echo "Model options:"
    echo "  --model MODEL              Model name (default: gpt-4o)"
    echo "  --backend BACKEND          Backend: openai or vllm (default: openai)"
    echo "  --max-new-tokens N         Max new tokens (default: 4096)"
    echo "  --temperature T            Temperature (default: 0.0)"
    echo "  --top-p P                  Top-p (default: 0.9)"
    echo "  --openai-base-url URL      OpenAI base URL"
    echo "  --vllm-tp-size N           vLLM tensor parallel size (default: 4)"
    echo ""
    echo "Runtime options:"
    echo "  --dft-tool TOOL            DFT tool (default: quantum espresso)"
    echo "  --work-dir DIR             Working directory root"
    echo "  --output-log-file FILE     Log file path (default: evaluation.log)"
    echo "  --no-script-only           Disable script_only mode"
    echo "  --no-evaluation            Disable evaluation mode"
    echo "  --no-query-info            Disable need_query_info"
    echo "  --parallel-exec            Enable parallel execution"
    exit 1
fi

python "$SCRIPT_DIR/new_test.py" "$@"
