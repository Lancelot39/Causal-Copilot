"""Benchmark tests — validate scenarios and evaluation metrics.

Run with: pytest benchmarks/ -v
"""

import numpy as np
import pytest

from benchmarks.evaluate import evaluate_adjacency
from benchmarks.scenarios import ALL_SCENARIOS, Scenario

# ---------------------------------------------------------------------------
# Scenario smoke tests
# ---------------------------------------------------------------------------


class TestScenarios:
    @pytest.mark.parametrize("name", list(ALL_SCENARIOS.keys()))
    def test_scenario_structure(self, name):
        s = ALL_SCENARIOS[name]
        assert isinstance(s, Scenario)
        assert s.data.shape[0] >= 100
        assert s.data.shape[1] >= 2
        assert s.ground_truth.shape == (s.data.shape[1], s.data.shape[1])
        assert list(s.data.columns) == s.columns

    @pytest.mark.parametrize("name", list(ALL_SCENARIOS.keys()))
    def test_scenario_deterministic(self, name):
        """Same scenario generated twice must produce identical data."""
        s1 = ALL_SCENARIOS[name]
        # Re-import to regenerate
        from benchmarks.scenarios import _make_scenario, collider, diamond, fork, linear_chain, sparse_10

        gen_fns = {
            "linear_chain": linear_chain,
            "fork": fork,
            "collider": collider,
            "diamond": diamond,
            "sparse_10": sparse_10,
        }
        s2 = _make_scenario(name, s1.description, gen_fns[name])
        np.testing.assert_array_equal(s1.data.values, s2.data.values)
        np.testing.assert_array_equal(s1.ground_truth, s2.ground_truth)

    @pytest.mark.parametrize("name", list(ALL_SCENARIOS.keys()))
    def test_ground_truth_is_dag(self, name):
        """Ground truth adjacency must have zero diagonal (no self-loops)."""
        gt = ALL_SCENARIOS[name].ground_truth
        assert np.all(np.diag(gt) == 0)

    def test_has_5_scenarios(self):
        assert len(ALL_SCENARIOS) == 5


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_perfect_prediction(self):
        gt = np.array([[0, 1], [0, 0]])
        m = evaluate_adjacency(gt, gt)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.shd == 0

    def test_empty_prediction(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.zeros_like(gt)
        m = evaluate_adjacency(pred, gt)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.false_negatives == 1

    def test_all_edges_predicted(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.ones_like(gt)
        np.fill_diagonal(pred, 0)
        m = evaluate_adjacency(pred, gt)
        assert m.recall == 1.0
        assert m.false_positives == 1  # extra reverse edge

    def test_reversed_edge(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.array([[0, 0], [1, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1
        assert m.shd == 2

    def test_handles_edge_types(self):
        """Predicted values 2 (undirected) and 3 (bidirected) count as edges."""
        gt = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred = np.array([[0, 2, 0], [0, 0, 3], [0, 0, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.true_positives == 2
        assert m.precision == 1.0

    def test_shd_symmetric(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.array([[0, 0], [1, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.shd == 2  # 1 missing + 1 extra
