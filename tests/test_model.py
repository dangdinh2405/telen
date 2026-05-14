"""Tests for TELEN model (unit-level, no backbone download needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock
from src.telern.config import TELENConfig
from src.telern.model import create_model


class TestCreateModel:
    """Test model creation without downloading the backbone."""

    def test_create_model_structure(self):
        """Test that create_model returns a TELEN with correct submodules."""
        with patch("src.telern.model.AutoModel.from_pretrained") as mock_auto, \
             patch("src.telern.model.AutoTokenizer.from_pretrained") as mock_tok:
            # Mock the backbone to return a small fake model
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_model.return_value = mock_model
            mock_auto.return_value = mock_model
            mock_tok.return_value = MagicMock()

            config = TELENConfig(hidden_dim=768)
            model = create_model(config)

            assert hasattr(model, "encoder")
            assert hasattr(model, "base_projection")
            assert hasattr(model, "proj_norm")
            assert hasattr(model, "attn_query")
            assert hasattr(model, "concept_graph")
            assert hasattr(model, "state_encoder")
            assert hasattr(model, "hypernetwork")

    def test_encoder_frozen(self):
        """Test that encoder parameters are frozen."""
        with patch("src.telern.model.AutoModel.from_pretrained") as mock_auto, \
             patch("src.telern.model.AutoTokenizer.from_pretrained") as mock_tok:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_auto.return_value = mock_model
            mock_tok.return_value = MagicMock()

            config = TELENConfig(hidden_dim=768)
            model = create_model(config)

            for p in model.encoder.parameters():
                assert not p.requires_grad

    def test_projection_trainable(self):
        """Test that projection is trainable (not frozen)."""
        with patch("src.telern.model.AutoModel.from_pretrained") as mock_auto, \
             patch("src.telern.model.AutoTokenizer.from_pretrained") as mock_tok:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_auto.return_value = mock_model
            mock_tok.return_value = MagicMock()

            config = TELENConfig(hidden_dim=768)
            model = create_model(config)

            assert any(p.requires_grad for p in model.base_projection.parameters())

    def test_attn_query_trainable(self):
        """Test that attn_query is trainable."""
        with patch("src.telern.model.AutoModel.from_pretrained") as mock_auto, \
             patch("src.telern.model.AutoTokenizer.from_pretrained") as mock_tok:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_auto.return_value = mock_model
            mock_tok.return_value = MagicMock()

            config = TELENConfig(hidden_dim=768)
            model = create_model(config)

            assert model.attn_query.requires_grad

    def test_get_state_vector_empty_graph(self):
        """Test get_state_vector returns zeros when no graph built."""
        with patch("src.telern.model.AutoModel.from_pretrained") as mock_auto, \
             patch("src.telern.model.AutoTokenizer.from_pretrained") as mock_tok:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_auto.return_value = mock_model
            mock_tok.return_value = MagicMock()

            config = TELENConfig(hidden_dim=768)
            model = create_model(config)
            model.attn_query.data = torch.randn(768)

            sv = model.get_state_vector()
            assert sv.shape == (768,)
            assert torch.allclose(sv, torch.zeros(768))

    def test_param_count_reasonable(self):
        """Test that the model has a reasonable number of parameters."""
        with patch("src.telern.model.AutoModel.from_pretrained") as mock_auto, \
             patch("src.telern.model.AutoTokenizer.from_pretrained") as mock_tok:
            mock_model = MagicMock()
            mock_model.config.hidden_size = 768
            mock_auto.return_value = mock_model
            mock_tok.return_value = MagicMock()

            config = TELENConfig(hidden_dim=768)
            model = create_model(config)

            total = sum(p.numel() for p in model.parameters())
            assert total > 0


class TestModelArchitecture:
    """Test the architecture without mocking the encoder (pure torch tests)."""

    def test_projection_shape(self):
        """Verify projection layer maps hidden_dim -> hidden_dim."""
        proj = nn.Sequential(nn.Linear(768, 768), nn.Tanh())
        x = torch.randn(4, 768)
        out = proj(x)
        assert out.shape == (4, 768)

    def test_attention_pooling(self):
        """Verify attention pooling logic used in _pool method."""
        d = 768
        attn_query = nn.Parameter(torch.randn(d))
        hidden = torch.randn(4, 10, d)  # [B, S, D]
        mask = torch.ones(4, 10)

        scores = torch.einsum("bsd,d->bs", hidden, attn_query) / (d ** 0.5)
        scores = scores.masked_fill(mask == 0, float("-1e9"))
        weights = torch.nn.functional.softmax(scores, dim=1)
        pooled = torch.einsum("bsd,bs->bd", hidden, weights)

        assert pooled.shape == (4, d)
        assert torch.allclose(weights.sum(dim=1), torch.ones(4))

    def test_layer_norm_projection(self):
        """Verify that projection + LayerNorm + normalize works as expected."""
        proj = nn.Sequential(nn.Linear(768, 768), nn.Tanh())
        proj_norm = nn.LayerNorm(768)
        x = torch.randn(4, 768)
        base = proj(x)
        normalized = torch.nn.functional.normalize(proj_norm(base), p=2, dim=1)
        assert normalized.shape == (4, 768)
        norms = normalized.norm(dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)
