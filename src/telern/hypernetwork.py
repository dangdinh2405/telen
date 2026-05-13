"""
HyperNetwork for TELEN.

Core innovation: Instead of learning fixed projection weights, the HyperNetwork
GENERATES the projection function from the current legal corpus state.

When new laws arrive → state vector changes → HyperNetwork produces new weights
→ embedding space adapts WITHOUT retraining.

Additionally outputs variance for stochastic embeddings (uncertainty-aware retrieval).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    """
    Generates embedding projection parameters from a legal state vector.

    Given state vector s ∈ R^d, produces:
      - ΔW: low-rank projection shift (weighted sum of learned rank-1 bases)
      - Δb: bias shift (weighted sum of learned bias bases)
      - log_σ²: per-dimension log-variance for stochastic embedding

    Architecture: Instead of generating giant parameter matrices directly,
    we store a compact set of learned basis vectors and use the HyperNetwork
    to generate ONLY the combination weights. This is parameter-efficient
    and forces generalization.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hn = config.hypernetwork
        d = config.hidden_dim
        r = hn.adaptation_rank
        hidden = hn.hn_hidden_dim

        # Shared trunk: state → latent code
        self.trunk = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Dropout(hn.dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(hn.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # Modulator: latent → combination weights for all outputs
        self.modulator = nn.Linear(hidden, 2 * r + r + 1)  # A_weights + B_weights + bias_weights + var_context

        # === Learned basis vectors (stored, not generated) ===
        # For ΔW = Σ_i w^A_i * (u_i ⊗ v_i^T) where u_i, v_i ∈ R^d
        self.basis_u = nn.Parameter(torch.randn(r, d) * 0.01)  # [r, d]
        self.basis_v = nn.Parameter(torch.randn(r, d) * 0.01)  # [r, d]

        # For Δb = Σ_i w^b_i * b_i where b_i ∈ R^d
        self.basis_b = nn.Parameter(torch.randn(r, d) * 0.01)  # [r, d]

        # Variance head
        if hn.output_variance:
            self.head_logvar = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, d),
            )
        else:
            self.head_logvar = None

    def forward(self, state_vector: torch.Tensor) -> dict:
        """
        Args:
            state_vector: [d] or [B, d] summarizing current legal landscape

        Returns dict with keys:
            "shift_matrix": [d, d] or [B, d, d] rank-r projection shift
            "bias": [d] or [B, d] bias shift
            "log_variance": [d] or [B, d] log variance for stochastic embedding
        """
        squeeze = state_vector.dim() == 1
        if squeeze:
            state_vector = state_vector.unsqueeze(0)  # [1, d]

        B, d = state_vector.shape
        r = self.config.hypernetwork.adaptation_rank

        # Shared representation
        h = self.trunk(state_vector)  # [B, hidden]
        modulated = self.modulator(h)  # [B, 2r + r + 1]

        # Split modulation weights
        w_A = modulated[:, :r]          # [B, r]
        w_B = modulated[:, r:2*r]       # [B, r]
        w_bias = modulated[:, 2*r:3*r]  # [B, r]

        # Build shift matrix: ΔW = Σ_i w^A_i * (u_i ⊗ v_i^T)
        # Weighted combination of basis vectors
        u_combined = w_A @ self.basis_u  # [B, d]
        v_combined = w_B @ self.basis_v  # [B, d]
        shift = torch.bmm(
            u_combined.unsqueeze(2),     # [B, d, 1]
            v_combined.unsqueeze(1),     # [B, 1, d]
        )  # [B, d, d]
        # Low-rank: this is rank-1. For rank r, generate r outer products and sum.
        # Simple yet effective: use weighted sum of r rank-1 components
        shift = shift.squeeze(0) if B == 1 else shift  # [d, d] or [B, d, d]
        if B == 1:
            shift = shift.unsqueeze(0)

        # Actually let's do proper rank-r: sum over rank dimension
        # w_A: [B, r], basis_u: [r, d]
        # For each rank i: w_A[:, i:i+1] * (basis_u[i:i+1]^T @ basis_v[i:i+1])
        # = Σ_i (w_A[:, i] * basis_u[i]) ⊗ (w_B[:, i] * basis_v[i])
        u_weighted = (w_A.unsqueeze(2) * self.basis_u.unsqueeze(0))  # [B, r, d]
        v_weighted = (w_B.unsqueeze(2) * self.basis_v.unsqueeze(0))  # [B, r, d]
        shift_ranked = torch.einsum("brd,bre->brde", u_weighted, v_weighted)  # [B, r, d, d]
        shift = shift_ranked.sum(dim=1)  # [B, d, d]

        # Bias
        bias = (w_bias.unsqueeze(2) * self.basis_b.unsqueeze(0)).sum(dim=1)  # [B, d]

        result = {"shift_matrix": shift, "bias": bias}

        # Log variance
        if self.head_logvar is not None:
            logvar = self.head_logvar(h)
            logvar = torch.clamp(logvar, min=-5.0, max=2.0)
            result["log_variance"] = logvar
        else:
            result["log_variance"] = torch.full((B, d), -3.0, device=h.device)

        if squeeze:
            result = {k: v.squeeze(0) for k, v in result.items()}

        return result


class StateEncoder(nn.Module):
    """
    Encodes the legal concept graph into a compact state vector.

    This is separate from the HyperNetwork so the graph computation
    can be cached and only updated when the graph changes.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, node_embeddings: torch.Tensor, node_weights: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            node_embeddings: [N, d] refined node embeddings from GNN
            node_weights: [N] optional attention weights

        Returns:
            state_vector: [d] summarizing the legal landscape
        """
        if node_weights is None:
            # Equal weight if none provided
            node_weights = torch.ones(
                node_embeddings.shape[0], device=node_embeddings.device
            )
        node_weights = F.softmax(node_weights, dim=0)
        pooled = (node_embeddings * node_weights.unsqueeze(1)).sum(dim=0)
        return self.state_proj(pooled)
