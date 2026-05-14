"""
TELEN: Temporal Evolving Legal Embedding Network.

Bi-encoder backbone + Legal Concept Graph + HyperNetwork projection.
Embedding space adapts dynamically to the legal corpus state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pyvi import ViTokenizer

from .config import TELENConfig
from .hypernetwork import StateEncoder, HyperNetwork
from .concept_graph import build_concept_graph


def wseg(text):
    return ViTokenizer.tokenize(text.replace("_", " "))


class BiEncoder(nn.Module):
    """Vietnamese bi-encoder backbone with attention pooling."""

    def __init__(self, model_name="bkai-foundation-models/vietnamese-bi-encoder"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dim = self.model.config.hidden_size
        self.attn_query = nn.Parameter(torch.randn(self.dim))
        self.scale = self.dim ** 0.5

    def forward(self, texts, max_len=480):
        segmented = [wseg(t) for t in texts]
        enc = self.tokenizer(segmented, padding=True, truncation=True,
                             max_length=max_len, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.attn_query.device)
        mask = enc["attention_mask"].to(self.attn_query.device)
        hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
        scores = torch.einsum("bsd,d->bs", hidden, self.attn_query) / self.scale
        scores = scores.masked_fill(mask == 0, float("-1e9"))
        weights = F.softmax(scores, dim=1)
        return torch.einsum("bsd,bs->bd", hidden, weights)


class TELEN(nn.Module):
    """Temporal Evolving Legal Embedding Network."""

    def __init__(self, config: TELENConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim

        # Bi-encoder backbone (frozen)
        self.encoder = BiEncoder()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Projection
        self.base_projection = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.proj_norm = nn.LayerNorm(d)
        self.attn_query = nn.Parameter(torch.randn(d))

        # Graph
        self.concept_graph = None
        self.law_id_to_idx = None

        # HyperNetwork
        self.state_encoder = StateEncoder(d)
        self.hypernetwork = HyperNetwork(config)

    def _pool(self, hidden, mask):
        """Attention-weighted pooling (for pre-tokenized inputs)."""
        scores = torch.einsum("bsd,d->bs", hidden, self.attn_query) / (self.config.hidden_dim ** 0.5)
        scores = scores.masked_fill(mask == 0, float("-1e9"))
        weights = F.softmax(scores, dim=1)
        return torch.einsum("bsd,bs->bd", hidden, weights)

    def encode_text(self, texts):
        return self.encoder(texts, max_len=self.config.max_seq_length)

    def get_state_vector(self):
        if self.concept_graph is None or self.concept_graph.num_nodes == 0:
            return torch.zeros(self.config.hidden_dim, device=self.attn_query.device)
        refined = self.concept_graph.forward()
        return self.state_encoder(refined)

    def adapt_embedding(self, raw, state_vec):
        base = self.base_projection(raw)
        hn = self.hypernetwork(state_vec)
        shift = raw @ hn["shift_matrix"].T + hn["bias"]
        mean = F.normalize(self.proj_norm(base + shift), p=2, dim=1)
        result = {"mean": mean, "log_variance": hn.get("log_variance")}
        if self.config.hypernetwork.output_variance:
            noise = 0.1 * hn["log_variance"].exp().clamp(min=0.001, max=0.25).sqrt().clamp(max=0.5)
            result["sample"] = F.normalize(mean + torch.randn_like(mean) * noise, p=2, dim=1)
        else:
            result["sample"] = mean
        return result

    def forward(self, texts, use_stochastic=False):
        raw = self.encode_text(texts)
        state = self.get_state_vector()
        adapted = self.adapt_embedding(raw, state)
        return {
            "embeddings": adapted["sample"] if use_stochastic else adapted["mean"],
            "mean": adapted["mean"],
            "log_variance": adapted.get("log_variance"),
            "state_vector": state,
        }

    def build_graph(self, df):
        self.concept_graph, self.law_id_to_idx = build_concept_graph(
            df, lambda t: self.encode_text([t])[0].detach(), self.config,
        )
        self.concept_graph = self.concept_graph.to(self.attn_query.device)

    def add_law(self, law_id, articles):
        if self.concept_graph is None: return
        if articles:
            emb = self.encode_text(articles[:5]).mean(dim=0)
            new_idx = self.concept_graph.num_nodes
            self.concept_graph.add_nodes([law_id], emb.unsqueeze(0))
            existing = self.concept_graph.node_embeddings[:-1]
            if len(existing) > 0:
                sim = F.cosine_similarity(emb.unsqueeze(0), existing)
                _, top = sim.topk(k=min(10, len(existing)))
                self.concept_graph.add_edges("semantic",
                    [(new_idx, i.item(), sim[i].item()) for i in top])


def create_model(config: TELENConfig) -> TELEN:
    return TELEN(config)
