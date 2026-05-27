import os
import subprocess
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


def test_demo_keeps_core_gradio_component_bindings():
    source = Path("web_demo/demo.py").read_text(encoding="utf-8")

    for name in ("msg", "file_upload", "download_btn", "reset_btn", "chatbot", "demo_btns"):
        assert f"{name} =" in source

    assert "msg.submit(" in source
    assert "file_upload.upload(" in source
    assert "reset_btn.click(" in source
    assert "demo_btn.click(" in source


def test_verify_script_falls_back_to_python3_when_python_is_absent(tmp_path):
    root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "python-args.log"
    fake_python3 = bin_dir / "python3"
    fake_python3.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$*" >> "$VERIFY_PYTHON_STUB_LOG"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python3.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["VERIFY_PYTHON_STUB_LOG"] = str(log_path)
    env.pop("PYTHON", None)

    result = subprocess.run(
        ["/bin/bash", "scripts/verify.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    assert invocations[0] == "-m ruff check ."
    assert invocations[1] == "-m pytest -q"
    assert invocations[2].startswith("-m compileall -q")


def test_verify_script_disables_external_pytest_plugins_by_default(tmp_path):
    root = Path(__file__).resolve().parents[1]
    log_path = tmp_path / "python-env.log"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/sh\n"
        'printf "%s|%s\\n" "$PYTEST_DISABLE_PLUGIN_AUTOLOAD" "$*" >> "$VERIFY_PYTHON_STUB_LOG"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PYTHON"] = str(fake_python)
    env["VERIFY_PYTHON_STUB_LOG"] = str(log_path)
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)

    result = subprocess.run(
        ["/bin/bash", "scripts/verify.sh"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    invocations = log_path.read_text(encoding="utf-8").splitlines()
    assert invocations
    assert all(line.startswith("1|") for line in invocations)
