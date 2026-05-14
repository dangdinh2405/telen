"""Tests for HyperNetwork and StateEncoder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from src.telern.config import TELENConfig, HyperNetworkConfig
from src.telern.hypernetwork import HyperNetwork, StateEncoder


class TestStateEncoder:
    @pytest.fixture
    def encoder(self):
        return StateEncoder(dim=64)

    def test_init(self, encoder):
        assert isinstance(encoder.state_proj, torch.nn.Sequential)

    def test_forward_equal_weights(self, encoder):
        x = torch.randn(10, 64)
        out = encoder(x)
        assert out.shape == (64,)
        assert out.dtype == torch.float32

    def test_forward_with_weights(self, encoder):
        x = torch.randn(10, 64)
        w = torch.rand(10)
        out = encoder(x, w)
        assert out.shape == (64,)

    def test_forward_single_node(self, encoder):
        x = torch.randn(1, 64)
        out = encoder(x)
        assert out.shape == (64,)

    def test_forward_deterministic_in_eval(self, encoder):
        encoder.eval()
        torch.manual_seed(42)
        x = torch.randn(5, 64)
        out1 = encoder(x)
        out2 = encoder(x)
        assert torch.allclose(out1, out2)


class TestHyperNetwork:
    @pytest.fixture
    def config(self):
        return TELENConfig(
            hidden_dim=64,
            hypernetwork=HyperNetworkConfig(
                adaptation_rank=8,
                hn_hidden_dim=128,
                hn_layers=3,
                dropout=0.0,
                output_shift=True,
                output_bias=True,
                output_variance=True,
            ),
        )

    @pytest.fixture
    def hn(self, config):
        return HyperNetwork(config)

    def test_init(self, hn):
        assert isinstance(hn.trunk, torch.nn.Sequential)
        assert hn.basis_u.shape == (8, 64)  # r x d
        assert hn.basis_v.shape == (8, 64)
        assert hn.basis_b.shape == (8, 64)
        assert hn.head_logvar is not None

    def test_forward_single_state(self, hn):
        s = torch.randn(64)
        out = hn(s)
        assert "shift_matrix" in out
        assert "bias" in out
        assert "log_variance" in out
        assert out["shift_matrix"].shape == (64, 64)
        assert out["bias"].shape == (64,)
        assert out["log_variance"].shape == (64,)

    def test_forward_batch(self, hn):
        s = torch.randn(4, 64)
        out = hn(s)
        assert out["shift_matrix"].shape == (4, 64, 64)
        assert out["bias"].shape == (4, 64)
        assert out["log_variance"].shape == (4, 64)

    def test_output_no_variance(self):
        cfg = TELENConfig(
            hidden_dim=64,
            hypernetwork=HyperNetworkConfig(output_variance=False),
        )
        hn = HyperNetwork(cfg)
        out = hn(torch.randn(64))
        assert "log_variance" in out
        # When head_logvar is None, falls back to constant -3.0
        assert torch.allclose(out["log_variance"], torch.full((64,), -3.0))

    def test_shift_matrix_is_valid(self, hn):
        s = torch.randn(64)
        out = hn(s)
        assert not torch.isnan(out["shift_matrix"]).any()
        assert not torch.isinf(out["shift_matrix"]).any()

    def test_deterministic(self, hn):
        torch.manual_seed(42)
        s = torch.randn(64)
        out1 = hn(s)
        torch.manual_seed(42)
        s2 = torch.randn(64)
        out2 = hn(s2)
        for k in out1:
            assert torch.allclose(out1[k], out2[k])

    def test_gradients_flow(self, hn):
        s = torch.randn(64)
        out = hn(s)
        loss = out["shift_matrix"].sum() + out["bias"].sum() + out["log_variance"].sum()
        loss.backward()
        for name, p in hn.named_parameters():
            assert p.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(p.grad).any(), f"{name} gradient is NaN"
