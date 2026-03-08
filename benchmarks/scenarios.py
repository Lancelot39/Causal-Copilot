"""Synthetic benchmark scenarios for causal discovery evaluation.

Each scenario returns (data, ground_truth_adj) with a fixed seed.
Ground truth adjacency: mat[i,j]=1 means j→i (column causes row).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    name: str
    description: str
    data: pd.DataFrame
    ground_truth: np.ndarray
    columns: list


def _make_scenario(name, desc, gen_fn, seed=42) -> Scenario:
    rng = np.random.default_rng(seed)
    data, gt, cols = gen_fn(rng)
    return Scenario(name=name, description=desc, data=data, ground_truth=gt, columns=cols)


def linear_chain(rng, n=500) -> tuple:
    """X → Y → Z, linear Gaussian."""
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(size=n) * 0.3
    z = 0.6 * y + rng.normal(size=n) * 0.4
    cols = ["X", "Y", "Z"]
    gt = np.array(
        [
            [0, 0, 0],  # X row: nothing causes X
            [1, 0, 0],  # Y row: X→Y
            [0, 1, 0],  # Z row: Y→Z
        ]
    )
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), gt, cols


def fork(rng, n=500) -> tuple:
    """X ← Z → Y (common cause / fork)."""
    z = rng.normal(size=n)
    x = 0.7 * z + rng.normal(size=n) * 0.3
    y = 0.5 * z + rng.normal(size=n) * 0.4
    cols = ["X", "Y", "Z"]
    gt = np.array(
        [
            [0, 0, 1],  # X row: Z→X
            [0, 0, 1],  # Y row: Z→Y
            [0, 0, 0],  # Z row: nothing causes Z
        ]
    )
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), gt, cols


def collider(rng, n=500) -> tuple:
    """X → Z ← Y (collider / v-structure)."""
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    z = 0.6 * x + 0.4 * y + rng.normal(size=n) * 0.3
    cols = ["X", "Y", "Z"]
    gt = np.array(
        [
            [0, 0, 0],  # X row
            [0, 0, 0],  # Y row
            [1, 1, 0],  # Z row: X→Z, Y→Z
        ]
    )
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), gt, cols


def diamond(rng, n=500) -> tuple:
    """A → B, A → C, B → D, C → D (diamond / 4 nodes)."""
    a = rng.normal(size=n)
    b = 0.7 * a + rng.normal(size=n) * 0.3
    c = 0.5 * a + rng.normal(size=n) * 0.4
    d = 0.4 * b + 0.3 * c + rng.normal(size=n) * 0.3
    cols = ["A", "B", "C", "D"]
    gt = np.array(
        [
            [0, 0, 0, 0],  # A row
            [1, 0, 0, 0],  # B row: A→B
            [1, 0, 0, 0],  # C row: A→C
            [0, 1, 1, 0],  # D row: B→D, C→D
        ]
    )
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d}), gt, cols


def sparse_10(rng, n=500) -> tuple:
    """10-node sparse DAG: 6 directed edges."""
    v = [rng.normal(size=n) for _ in range(10)]
    # X0→X1, X0→X3, X1→X2, X3→X4, X4→X5, X2→X9
    v[1] = 0.6 * v[0] + rng.normal(size=n) * 0.4
    v[2] = 0.5 * v[1] + rng.normal(size=n) * 0.4
    v[3] = 0.7 * v[0] + rng.normal(size=n) * 0.3
    v[4] = 0.5 * v[3] + rng.normal(size=n) * 0.4
    v[5] = 0.4 * v[4] + rng.normal(size=n) * 0.5
    v[9] = 0.3 * v[2] + rng.normal(size=n) * 0.5
    cols = [f"X{i}" for i in range(10)]
    gt = np.zeros((10, 10), dtype=int)
    edges = [(0, 1), (0, 3), (1, 2), (3, 4), (4, 5), (2, 9)]
    for src, tgt in edges:
        gt[tgt, src] = 1  # mat[i,j]=1 means j→i
    return pd.DataFrame({c: v[i] for i, c in enumerate(cols)}), gt, cols


ALL_SCENARIOS: dict[str, Scenario] = {
    "linear_chain": _make_scenario("linear_chain", "X → Y → Z (3 nodes, linear)", linear_chain),
    "fork": _make_scenario("fork", "X ← Z → Y (common cause)", fork),
    "collider": _make_scenario("collider", "X → Z ← Y (v-structure)", collider),
    "diamond": _make_scenario("diamond", "A → B,C → D (4 nodes)", diamond),
    "sparse_10": _make_scenario("sparse_10", "10-node sparse DAG (6 edges)", sparse_10),
}
