#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:pkg_resources is deprecated as an API:UserWarning}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

"$PYTHON" src/cluster_agent.py "$@"
