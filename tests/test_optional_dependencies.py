from pathlib import Path


def test_causalnex_imports_are_lazy():
    for source_path in (
        "causal_discovery/wrappers/dynotears.py",
        "causal_discovery/wrappers/utils/ts_utils.py",
    ):
        top_level_imports = [
            line
            for line in Path(source_path).read_text().splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]

        assert not any("causalnex" in line for line in top_level_imports)


def test_latexmk_imports_are_lazy():
    for source_path in (
        "report/report_generation.py",
        "report/inference_report_generation.py",
    ):
        top_level_imports = [
            line
            for line in Path(source_path).read_text().splitlines()
            if line.startswith("import ") or line.startswith("from ")
        ]

        assert not any("latexmk" in line for line in top_level_imports)
