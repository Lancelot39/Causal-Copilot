from pathlib import Path


def _read_pins(requirements_path):
    pins = {}
    for line in Path(requirements_path).read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        package, version = stripped.split("==", 1)
        pins[package.lower()] = version.strip()
    return pins


def _read_requirements(requirements_path):
    return {
        line.strip().split("==", 1)[0].lower()
        for line in Path(requirements_path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_gradio_runtime_requirement_pins_are_resolvable():
    for requirements_path in ("requirements_cpu.txt", "requirements_gpu.txt"):
        pins = _read_pins(requirements_path)

        assert pins["gradio"].startswith("5.")
        assert tuple(map(int, pins["safehttpx"].split("."))) >= (0, 1, 6)
        assert tuple(map(int, pins["ruff"].split("."))) >= (0, 9, 3)
        assert tuple(map(int, pins["pygam"].split("."))) >= (0, 12, 0)
        assert tuple(map(int, pins["econml"].split("."))) >= (0, 16, 0)
        assert "causalnex" not in _read_requirements(requirements_path)
