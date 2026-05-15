#!/usr/bin/env bash
set -euo pipefail

python -m ruff check .
python -m pytest -q
python -m compileall -q \
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
