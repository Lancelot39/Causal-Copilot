# Causal-Copilot-Abel

Autonomous causal analysis agent — LLM-driven pipeline for causal discovery + inference.
UCSD + Abel.ai. arxiv:2504.13263. MIT License.

## Quick Reference

```bash
# Setup (conda required)
bash setup_cpu.sh          # Install system deps + conda env
source activate copilot    # Activate environment

# Run CLI
python main.py --data-file data/dataset/Abalone/Abalone.csv --initial-query "What causes high rings?"

# Run Web Demo
python web_demo/demo.py    # Gradio on :7860

# Environment
cp .env.example .env       # Configure LLM_PROVIDER, API keys
```

## Architecture

```
main.py (CLI entry)              web_demo/demo.py (Gradio entry)
       \                              /
        → global_setting/ (GlobalState dataclass — single source of truth)
        → preprocess/     (data loading, imputation, stat tests, EDA)
        → causal_discovery/
            filter.py     (LLM selects top-K algorithms)
            rerank.py     (score + select best algorithm)
            hyperparameter_selector.py (LLM tunes hyperparams)
            program.py    (execute algorithm)
            wrappers/     (27 algorithm implementations)
        → postprocess/    (bootstrap + LLM + KCI graph refinement)
        → causal_inference/
            inference.py  (Analysis class — 1717 lines, main orchestrator)
            DML/ DRL/ IV/ MetaLearners/ Uplift/ (estimator modules)
        → report/         (LaTeX PDF generation, 11 LLM calls)
        → user/discuss.py (post-analysis Q&A)
        → llm/            (OpenAI/OpenRouter/Ollama abstraction)
```

## Key Conventions

### State Management
- `GlobalState` (dataclass in `global_setting/state.py`) is the single source of truth
- All pipeline functions take and return `global_state` — explicit mutation, no singletons
- Subclasses: UserData, Statistics, Logging, Algorithm, Results, Inference

### Algorithm Wrappers
- All inherit from `CausalDiscoveryAlgorithm` (base.py) — NOT enforced with @abstractmethod
- Must implement: `fit(data) -> (adj_matrix, info_dict, model)`, `name` property
- Register in `wrappers/__init__.py` + add to `__all__`
- Each wrapper has self-contained `test_algorithm()` method (should be extracted to tests/)

### LLM Integration
- `LLMClient` in `llm/llm_client.py` — unified interface for 3 providers
- Config via env vars: `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
- Temperature hardcoded to 0.0 throughout (deterministic)
- Pydantic models used for structured LLM output parsing

### File Paths
- All paths are relative to project root — code assumes CWD is project root
- Context files: `causal_discovery/context/{algos,hyperparameters,benchmarking}/`
- Output: `{data_dir}/output_report/`, `{data_dir}/output_graph/`

## Known Issues

### Critical
- `eval()` in runtime_estimator.py (lines 130, 159) — code injection risk
- 22 bare `except:` blocks across causal_inference/ — masks all errors
- Test code embedded in production wrappers (250+ lines in pc.py alone)

### Important
- `externals/` bundles 5 libraries via git submodule + sys.path injection
- No test suite (only 2 test files: test_ollama.py, test_latex_functionality.py)
- No CI/CD, no pyproject.toml, not pip-installable
- Hardcoded file paths break if CWD != project root
- Code duplication: _create_background_knowledge() in PC and CDNOD (100 lines each)

## Dependencies

Core: torch, scikit-learn, pandas, numpy, scipy, statsmodels, networkx
Causal: dowhy, econml, gcastle, tigramite, lingam, causalnex, causal-learn (bundled)
LLM: openai
Web: gradio, fastapi
Report: plumbum (LaTeX), pypdf2, pydantic
System: graphviz (apt), texlive-latex-* (apt), TinyTeX

## Adjacency Matrix Convention
- `mat[i,j] = 1` means edge j → i (column causes row)
- Values: 0=none, 1=directed, 2=undirected, 3=bidirected, 4-7=PAG edge types
