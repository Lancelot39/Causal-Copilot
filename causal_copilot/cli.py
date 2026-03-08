"""Causal-Copilot CLI entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def cmd_analyze(args):
    """Run causal discovery on a CSV file."""
    from causal_copilot import CausalCopilot

    try:
        copilot = CausalCopilot(planner=args.planner)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    result = copilot.analyze(
        args.data,
        algorithm=args.algorithm,
        timeout=args.timeout,
        seed=args.seed,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"Result written to {out_path}")
    else:
        print(result.summary)
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")
        if result.provenance:
            p = result.provenance
            print(f"\nProvenance: {p.algorithm} | seed={p.seed} | {p.runtime_seconds:.1f}s")

    sys.exit(0 if result.status == "ok" else 1)


def cmd_quickstart(args):
    """Run a demo analysis on bundled synthetic data."""
    import numpy as np
    import pandas as pd
    from causal_copilot import CausalCopilot

    print("Causal-Copilot Quickstart")
    print("=" * 40)
    print("Generating synthetic data: X → Y → Z (linear, Gaussian noise)\n")

    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(size=n) * 0.3
    z = 0.6 * y + rng.normal(size=n) * 0.4
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    copilot = CausalCopilot()
    result = copilot.analyze(df, seed=42)

    print(f"Status: {result.status}")
    print(f"Summary: {result.summary}")

    if result.adjacency_matrix is not None:
        print(f"\nAdjacency matrix (columns → rows):")
        cols = ["X", "Y", "Z"]
        header = "     " + "  ".join(f"{c:>4}" for c in cols)
        print(header)
        for i, row_label in enumerate(cols):
            vals = "  ".join(f"{int(result.adjacency_matrix[i, j]):>4}" for j in range(len(cols)))
            print(f"  {row_label:>2} {vals}")

    if result.warnings:
        print(f"\nWarnings: {result.warnings}")

    if result.provenance:
        p = result.provenance
        print(f"\nProvenance: {p.algorithm} | planner={p.planner} | seed={p.seed} | {p.runtime_seconds:.1f}s")

    print("\nAssumptions:")
    for a in result.assumptions:
        print(f"  - {a}")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nFull result written to {out_path}")

    sys.exit(0 if result.status == "ok" else 1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="causal-copilot",
        description="Autonomous causal analysis from tabular data.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="Check environment and dependencies")
    sub.add_parser("version", help="Show version")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Run causal analysis on a CSV file")
    p_analyze.add_argument("data", help="Path to CSV file")
    p_analyze.add_argument("--output", "-o", help="Output JSON file path")
    p_analyze.add_argument("--algorithm", "-a", help="Force a specific algorithm")
    p_analyze.add_argument("--planner", "-p", default="rule", help="Planner: rule (default)")
    p_analyze.add_argument("--timeout", "-t", type=int, default=300, help="Timeout in seconds (default: 300)")
    p_analyze.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")

    # quickstart
    p_quick = sub.add_parser("quickstart", help="Run demo analysis on synthetic data")
    p_quick.add_argument("--output", "-o", help="Output JSON file path")

    args = parser.parse_args(argv)

    if args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "version":
        cmd_version(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "quickstart":
        cmd_quickstart(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
