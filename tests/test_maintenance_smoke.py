from pathlib import Path


def test_readme_license_reference_resolves():
    root = Path(__file__).resolve().parents[1]

    assert "See `LICENSE`" in (root / "README.md").read_text(encoding="utf-8")
    assert (root / "LICENSE").is_file()


def test_fci_test_algorithm_imports_data_simulator():
    root = Path(__file__).resolve().parents[1]
    source = (root / "causal_discovery" / "wrappers" / "fci.py").read_text(encoding="utf-8")

    assert "from data.simulator.dummy import DataSimulator" in source
    assert "simulator = DataSimulator()" in source


def test_drl_hte_program_has_module_logger_import():
    root = Path(__file__).resolve().parents[1]
    source = (root / "causal_inference" / "DRL" / "hte_program.py").read_text(encoding="utf-8")

    assert "from utils.logger import logger" in source
    assert "logger.info" in source


def test_cpu_dockerfile_uses_cpu_only_runtime():
    root = Path(__file__).resolve().parents[1]
    source = (root / "Dockerfile.cpu").read_text(encoding="utf-8")
    first_instruction = next(line for line in source.splitlines() if line.strip())

    assert "python:3.10-slim" in first_instruction
    assert "cuda" not in first_instruction.lower()
    assert "https://download.pytorch.org/whl/cpu" in source
    assert "torch==2.2.2" in source


def test_dockerfiles_launch_web_demo_by_default():
    root = Path(__file__).resolve().parents[1]

    for dockerfile_name in ("Dockerfile.cpu", "Dockerfile.gpu"):
        source = (root / dockerfile_name).read_text(encoding="utf-8")

        assert "EXPOSE 7860" in source
        assert 'CMD ["python", "web_demo/demo.py"]' in source


def test_dockerignore_excludes_local_and_generated_files():
    root = Path(__file__).resolve().parents[1]
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8").splitlines()
    ignored_patterns = {line.strip() for line in dockerignore if line.strip() and not line.startswith("#")}

    expected_patterns = {
        ".git",
        ".env",
        ".venv",
        "__pycache__",
        "*.py[cod]",
        ".pytest_cache",
        ".ruff_cache",
        "demo_data",
        "gradio_output",
        "output",
        "test_output",
    }

    assert expected_patterns <= ignored_patterns
