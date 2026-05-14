"""Tests for TELEN configuration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.telern.config import TELENConfig, GraphConfig, HyperNetworkConfig, MetaTrainingConfig


class TestGraphConfig:
    def test_defaults(self):
        cfg = GraphConfig()
        assert cfg.hidden_dim == 768
        assert cfg.gnn_layers == 3
        assert cfg.gnn_dropout == 0.1
        assert cfg.use_cross_ref_edges is True
        assert cfg.use_agency_edges is True
        assert cfg.use_temporal_edges is True
        assert cfg.use_semantic_edges is True
        assert cfg.semantic_knn == 10
        assert cfg.max_concepts_per_article == 8
        assert cfg.min_tfidf_score == 0.05

    def test_custom(self):
        cfg = GraphConfig(hidden_dim=512, gnn_layers=2, use_agency_edges=False)
        assert cfg.hidden_dim == 512
        assert cfg.gnn_layers == 2
        assert cfg.use_agency_edges is False


class TestHyperNetworkConfig:
    def test_defaults(self):
        cfg = HyperNetworkConfig()
        assert cfg.adaptation_rank == 64
        assert cfg.hn_hidden_dim == 512
        assert cfg.hn_layers == 3
        assert cfg.dropout == 0.1
        assert cfg.output_shift is True
        assert cfg.output_bias is True
        assert cfg.output_variance is True
        assert cfg.min_variance == 0.01


class TestMetaTrainingConfig:
    def test_defaults(self):
        cfg = MetaTrainingConfig()
        assert cfg.meta_lr == 3e-4
        assert cfg.inner_lr == 5e-3
        assert cfg.meta_batch_size == 4
        assert cfg.n_query == 32
        assert cfg.n_negatives == 256
        assert cfg.meta_epochs == 50
        assert cfg.temperature == 0.05
        assert cfg.train_split_year == 2018
        assert cfg.val_split_year == 2020
        assert cfg.max_state_articles == 500
        assert cfg.kl_weight == 0.001
        assert cfg.n_mc_samples == 1


class TestTELENConfig:
    def test_defaults(self):
        cfg = TELENConfig()
        assert cfg.backbone == "vinai/phobert-base-v2"
        assert cfg.hidden_dim == 768
        assert cfg.max_seq_length == 480
        assert isinstance(cfg.graph, GraphConfig)
        assert isinstance(cfg.hypernetwork, HyperNetworkConfig)
        assert isinstance(cfg.meta, MetaTrainingConfig)
        assert cfg.seed == 42

    def test_custom_nested(self):
        cfg = TELENConfig(
            hidden_dim=512,
            graph=GraphConfig(hidden_dim=512, gnn_layers=2),
            hypernetwork=HyperNetworkConfig(adaptation_rank=32),
        )
        assert cfg.hidden_dim == 512
        assert cfg.graph.hidden_dim == 512
        assert cfg.graph.gnn_layers == 2
        assert cfg.hypernetwork.adaptation_rank == 32

    def test_output_dir_is_string(self):
        cfg = TELENConfig()
        assert isinstance(cfg.output_dir, str)
        assert len(cfg.output_dir) > 0
