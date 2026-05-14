"""Tests for evaluation metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.telern.evaluate import dcg_at_k, ndcg_at_k, mrr_at_k, compute_metrics


class TestDCG:
    def test_empty(self):
        assert dcg_at_k(np.array([]), 3) == 0.0

    def test_all_zeros(self):
        assert dcg_at_k(np.zeros(10), 5) == 0.0

    def test_perfect_ranking(self):
        scores = np.array([2.0, 2.0, 2.0, 1.0, 1.0])
        dcg = dcg_at_k(scores, 3)
        expected = (3 / np.log2(2)) + (3 / np.log2(3)) + (3 / np.log2(4))
        assert abs(dcg - expected) < 1e-6

    def test_k_larger_than_array(self):
        scores = np.array([2.0, 1.0])
        dcg = dcg_at_k(scores, 5)
        assert dcg > 0

    def test_dcg_decreases_with_rank(self):
        scores_good = np.array([2.0, 1.0, 0.0])
        scores_bad = np.array([0.0, 1.0, 2.0])
        assert dcg_at_k(scores_good, 3) > dcg_at_k(scores_bad, 3)


class TestNDCG:
    def test_perfect(self):
        scores = np.array([2.0, 1.0, 0.0])
        assert abs(ndcg_at_k(scores, 3) - 1.0) < 1e-6

    def test_all_zeros(self):
        assert ndcg_at_k(np.zeros(10), 5) == 0.0

    def test_imperfect(self):
        scores = np.array([1.0, 2.0, 0.0])  # suboptimal order
        ndcg = ndcg_at_k(scores, 3)
        assert 0.0 < ndcg < 1.0

    def test_empty(self):
        assert ndcg_at_k(np.array([]), 3) == 0.0


class TestMRR:
    def test_first_position(self):
        scores = np.array([1.0, 0.0, 0.0])
        assert abs(mrr_at_k(scores, 3) - 1.0) < 1e-6

    def test_second_position(self):
        scores = np.array([0.0, 1.0, 0.0])
        assert abs(mrr_at_k(scores, 3) - 0.5) < 1e-6

    def test_third_position(self):
        scores = np.array([0.0, 0.0, 1.0])
        assert abs(mrr_at_k(scores, 3) - 1.0 / 3) < 1e-6

    def test_no_relevant(self):
        scores = np.array([0.0, 0.0, 0.0])
        assert mrr_at_k(scores, 3) == 0.0

    def test_k_limit(self):
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        assert mrr_at_k(scores, 2) == 0.0  # relevant at pos 3, k=2


class TestComputeMetrics:
    def test_all_metrics_present(self):
        metrics = compute_metrics(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        for k in [3, 5, 10]:
            assert f"ndcg@{k}" in metrics
            assert f"mrr@{k}" in metrics

    def test_metric_values_in_range(self):
        metrics = compute_metrics(np.array([1.0, 0.5, 0.0, 0.0, 0.2]))
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_custom_k(self):
        metrics = compute_metrics(np.array([1.0, 0.0]), k_values=[1, 2])
        assert "ndcg@1" in metrics
        assert "mrr@1" in metrics
        assert "mrr@2" in metrics
