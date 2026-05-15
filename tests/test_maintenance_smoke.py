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
