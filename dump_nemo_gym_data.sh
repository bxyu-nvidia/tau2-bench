#!/usr/bin/env bash
set -euo pipefail

if [[ "${TAU2_SKIP_UV_SYNC:-0}" != "1" ]]; then
    uv venv --python 3.12 .venv --allow-existing
    source .venv/bin/activate
    uv sync --active --extra knowledge
fi

PYTHON_BIN="${TAU2_DUMP_PYTHON:-python}"
PYTHONPATH="$(pwd)/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" dump_nemo_gym_data.py "$@"
