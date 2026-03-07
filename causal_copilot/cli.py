"""Causal-Copilot CLI entry point."""
from __future__ import annotations

import argparse
import sys


def cmd_doctor(args):
    """Check environment and report installed capabilities."""
    print("Causal-Copilot Doctor")
    print("=" * 40)

    from causal_copilot import __version__
    print(f"Version: {__version__}")

    # Check core deps
    core_deps = ["numpy", "pandas", "scipy", "sklearn", "networkx", "statsmodels"]
    print("\nCore dependencies:")
    for dep in core_deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, "__version__", "ok")
            print(f"  + {dep} {ver}")
        except ImportError:
            print(f"  x {dep} NOT INSTALLED")

    # Check optional extras
    extras = {
        "agent": ["openai", "pydantic"],
        "inference": ["dowhy", "econml"],
        "viz": ["matplotlib", "seaborn"],
        "gpu": ["torch"],
        "web": ["gradio"],
        "algorithms": ["lingam", "tigramite"],
    }
    print("\nOptional extras:")
    for group, deps in extras.items():
        installed = []
        missing = []
        for dep in deps:
            try:
                __import__(dep)
                installed.append(dep)
            except ImportError:
                missing.append(dep)
        status = "+" if not missing else "~" if installed else "x"
        detail = f"missing: {', '.join(missing)}" if missing else "all installed"
        print(f"  {status} [{group}] {detail}")

    print("\nPlatform:")
    import platform
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Arch: {platform.machine()}")


def cmd_version(args):
    from causal_copilot import __version__
    print(f"causal-copilot {__version__}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="causal-copilot",
        description="Autonomous causal analysis from tabular data.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Check environment and dependencies")
    sub.add_parser("version", help="Show version")

    # Placeholder for future commands
    # sub.add_parser("quickstart", help="Run demo analysis on bundled data")
    # sub.add_parser("analyze", help="Run causal analysis on a CSV file")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "version":
        cmd_version(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
