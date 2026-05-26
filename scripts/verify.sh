#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "error: no Python interpreter found; install python/python3 or set PYTHON=/path/to/python" >&2
    exit 127
  fi
fi

"${PYTHON_BIN}" -m ruff check .
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m compileall -q \
  causal_discovery \
  causal_inference \
  llm \
  preprocess \
  postprocess \
  report \
  user \
  utils \
  web_demo \
  global_setting \
  data/simulator \
  test_latex_functionality.py \
  install_latex.py \
  tests
