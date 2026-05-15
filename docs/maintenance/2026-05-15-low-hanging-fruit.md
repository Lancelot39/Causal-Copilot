# Causal-Copilot Low-Hanging Fruit Backlog

Date: 2026-05-15

This backlog ranks small, main-safe maintenance pushes by risk reduction per unit of work. The guiding rule is to prefer daily changes that make future verification clearer before touching heavier algorithm behavior.

## Current Evidence

- Local branch: `main`, aligned with `origin/main` at `29c598c`.
- Public GitHub repo: `Lancelot39/Causal-Copilot`, public, 181 stars, 44 forks, 0 open issues, 0 open PRs when checked.
- Public README lists Aryan Philip under other contributors.
- Local machine only has `/usr/bin/python3` Python 3.9.6; README asks for Python 3.10 and recommends Docker.
- `python3 -m compileall -q .` passed across the checkout.
- `ruff check` on owned code with critical rules passed after fixing two undefined-name issues.
- Raw `pytest --collect-only` in an unconfigured repo tried to collect vendored external tests and failed with 44 dependency/import errors.
- After scoping default tests, `pytest -q` runs 3 smoke tests and skips 2 opt-in integration modules.
- `requirements_cpu.txt` cannot resolve under Python 3.9 because `gradio==5.40.0` requires Python 3.10 or newer; this matches the README's Python 3.10 requirement.
- README referenced `LICENSE`, but no root `LICENSE` file existed before this maintenance pass.

## Ranked Daily Pushes

### 1. Verification hygiene

Why first: this makes every later maintainer push safer and easier to review.

Scope:
- Add root `pyproject.toml` with scoped `pytest` and `ruff` config.
- Exclude vendored `externals/`, assets, and datasets from default lint/test traversal.
- Keep the first lint gate narrow: syntax-level and undefined-name rules only.
- Add `requirements_dev.txt`, `scripts/verify.sh`, and a minimal GitHub Actions workflow for the deterministic smoke gate.

Verification:
- `ruff check .`
- `python3 -m compileall -q causal_discovery causal_inference llm preprocess postprocess report user utils web_demo global_setting data/simulator test_latex_functionality.py install_latex.py`
- `pytest --collect-only -q` in a Python 3.10 environment with project dependencies installed.

### 2. Runtime break fixes

Why second: these are direct NameError risks in owned code.

Scope:
- Import `DataSimulator` in `causal_discovery/wrappers/fci.py`.
- Promote `logger` import to module scope in `causal_inference/DRL/hte_program.py`.

Verification:
- `ruff check causal_discovery causal_inference --select=E9,F63,F7,F82`
- Targeted import smoke tests in the configured Python 3.10 environment.

### 3. Licensing cleanup

Why third: README promises MIT licensing through `LICENSE`; the missing file is a distribution-quality gap.

Scope:
- Add standard MIT `LICENSE`.
- Confirm the copyright holder text with core maintainers if they prefer a named institution or author list over "Causal-Copilot contributors".

Verification:
- `test -f LICENSE`
- README link review.

### 4. Developer bootstrap clarity

Why fourth: local setup is the current blocker to full verification.

Scope:
- Add `.python-version` or document Python 3.10 explicitly in one canonical place.
- Add a `Makefile` or `scripts/verify.sh` that runs the default checks.
- Decide whether local setup should be Docker-first only or also support a Python 3.10 venv.

Verification:
- Fresh clone runbook on macOS.
- Fresh clone runbook in Docker.

### 5. CI baseline

Why fifth: CI should start with cheap deterministic checks before slow scientific dependencies.

Scope:
- Add GitHub Actions for Python 3.10.
- Start with `compileall` and scoped `ruff`.
- Add dependency install and pytest only after setup is deterministic.

Verification:
- Green CI on the default branch.

### 6. Test classification

Why sixth: some current "tests" are integration/manual scripts that require LLMs, Ollama, LaTeX, or API keys.

Scope:
- Mark tests as `unit`, `integration`, `llm`, `latex`, or `manual`.
- Convert `llm/test_ollama.py` into a skipped integration test unless Ollama is reachable.
- Convert `causal_discovery/tests/selection_test.py` into a manual or LLM-marked test; it currently depends on OpenAI access and long-running simulation.
- Keep `llm/test_ollama.py` outside default `testpaths` because importing the `llm` package initializes provider dependencies before module-level skip logic can run.

Verification:
- `pytest -m "not integration and not llm and not latex"`.
- Separate documented commands for expensive/manual suites.

### 7. Dependency normalization

Why seventh: the requirements files are pinned but installation behavior is fragile and uses `--no-deps`.

Scope:
- Decide whether `requirements_cpu.txt` is a lockfile or an install input.
- Split runtime, dev, latex, and optional GPU dependencies.
- Add a lightweight dev requirements file for lint/test tooling.

Verification:
- Fresh Python 3.10 venv install.
- Fresh Docker build.

### 8. Submodule metadata cleanup

Why eighth: `.gitmodules` points to `causal-learn`, while the repo has a vendored `externals/causal-learn` tree tracked as normal files.

Scope:
- Either remove stale `.gitmodules`, or convert the external dependency into a real submodule under the path actually used by code.
- Prefer removing stale metadata if the project intentionally vendors patched external code.

Verification:
- `git submodule status --recursive` should be empty if no submodules are intended.
- Fresh clone should not show confusing submodule instructions.

### 9. Docker entrypoint polish

Why ninth: Dockerfiles build an environment but default to `/bin/bash`, while README implies users can run the app.

Scope:
- Decide whether containers should launch the web demo by default or remain interactive.
- If launching the app, set `CMD ["python", "web_demo/demo.py"]` or document the explicit command.
- Add a health check only after startup is deterministic.

Verification:
- `docker build -f Dockerfile.cpu -t causal-copilot-cpu .`
- `docker run --rm -p 7860:7860 causal-copilot-cpu`

## Suggested Sequence

Day 1:
- Push verification hygiene, direct NameError fixes, and `LICENSE`.

Day 2:
- Expand smoke coverage around importable core modules after Python 3.10 dependency installation is reproducible.

Day 3:
- Add dependency-install CI or Docker build verification after resolver behavior is stable.

Day 4:
- Mark/skip LLM, Ollama, and LaTeX integration tests cleanly.

Day 5:
- Normalize dependency files and prove a fresh Python 3.10 setup.

Day 6:
- Resolve stale `.gitmodules` behavior.

Day 7:
- Polish Docker entrypoint and runtime docs.
