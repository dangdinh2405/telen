"""
Legal Concept Graph — evolving knowledge backbone of TELEN.

Nodes: law entities + key terms extracted via TF-IDF
Edges: agency, temporal, semantic, cross-reference, term-document
GNN: Multi-layer sparse graph convolution
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer


# ═══════════════════════════════════════════════
# GNN Layers
# ═══════════════════════════════════════════════
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, adj):
        deg = adj.sum(dim=1).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5)
        norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        x = norm_adj @ x
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm(x)
        return x


class GNNEncoder(nn.Module):
    def __init__(self, dim, n_layers=3, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([GCNLayer(dim, dim, dropout) for _ in range(n_layers)])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj) + x  # residual
        return x


# ═══════════════════════════════════════════════
# Legal Concept Graph
# ═══════════════════════════════════════════════
class LegalConceptGraph(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.graph.hidden_dim

        self.node_ids = []
        self.node_embeddings = None
        self.edges = {"cross_ref": [], "agency": [], "temporal": [], "semantic": []}
        self._adj_cached = None
        self._adj_dirty = True

        self.gnn = GNNEncoder(config.graph.hidden_dim, config.graph.gnn_layers, config.graph.gnn_dropout)

    @property
    def num_nodes(self):
        return len(self.node_ids)

    @property
    def device(self):
        return self.gnn.layers[0].linear.weight.device

    def add_nodes(self, node_ids, embeddings):
        if self.node_embeddings is None:
            self.node_embeddings = embeddings
        else:
            self.node_embeddings = torch.cat([self.node_embeddings, embeddings], dim=0)
        self.node_ids.extend(node_ids)
        self._adj_dirty = True

    def add_edges(self, edge_type, edges):
        self.edges[edge_type].extend(edges)
        self._adj_dirty = True

    def build_adjacency(self):
        if not self._adj_dirty and self._adj_cached is not None:
            return self._adj_cached
        N = self.num_nodes
        adj = torch.zeros(N, N, device=self.device)

        for edge_type, use in [("cross_ref", self.config.graph.use_cross_ref_edges),
                                ("agency", self.config.graph.use_agency_edges),
                                ("temporal", self.config.graph.use_temporal_edges),
                                ("semantic", self.config.graph.use_semantic_edges)]:
            if not use or not self.edges[edge_type]:
                continue
            valid = [(s, d, w) for s, d, w in self.edges[edge_type] if s < N and d < N]
            if not valid:
                continue
            src = torch.tensor([e[0] for e in valid], device=self.device, dtype=torch.long)
            dst = torch.tensor([e[1] for e in valid], device=self.device, dtype=torch.long)
            wgt = torch.tensor([e[2] for e in valid], device=self.device, dtype=torch.float)
            adj.index_put_((src, dst), wgt, accumulate=True)
            adj.index_put_((dst, src), wgt, accumulate=True)

        adj = adj + torch.eye(N, device=self.device)
        self._adj_cached = adj
        self._adj_dirty = False
        return adj

    def forward(self):
        dev = self.device
        if self.node_embeddings.device != dev:
            self.node_embeddings = self.node_embeddings.to(dev)
        adj = self.build_adjacency()
        return self.gnn(self.node_embeddings, adj)

    def to(self, device):
        super().to(device)
        if self.node_embeddings is not None:
            self.node_embeddings = self.node_embeddings.to(device)
        return self


# ═══════════════════════════════════════════════
# Cross-reference extraction
# ═══════════════════════════════════════════════
CROSS_REF_PATTERNS = [
    (re.compile(r"(?:theo|theo quy định tại|căn cứ vào|căn cứ)\s+Điều\s+(\d+)\s+(?:của\s+)?(Luật|Bộ luật|Nghị định|Thông tư|Pháp lệnh)\s+([^,.;]+)"), "citation"),
    (re.compile(r"(Luật|Bộ luật|Nghị định|Thông tư|Pháp lệnh|Quyết định)\s+(?:số\s+)?([\d]+/[\d]+/[\w-]+)"), "reference"),
    (re.compile(r"sửa đổi[,，]\s*bổ sung\s+(?:một số điều của\s+)?(Luật|Nghị định|Thông tư)\s+([^,.;]+)"), "amendment"),
    (re.compile(r"(?:thay thế|bãi bỏ)\s+(?:Điều\s+(\d+)\s+(?:của\s+)?)?(Luật|Nghị định|Thông tư)\s+([^,.;]+)"), "replacement"),
]


def extract_key_terms(df, max_terms=200):
    texts = [(f"{row['title']} {row['text'][:500]}").replace("_", " ")
             for _, row in df.iterrows()]
    vectorizer = TfidfVectorizer(max_features=max_terms, ngram_range=(1, 2),
                                 min_df=3, max_df=0.8, token_pattern=r'(?u)\b\w+\b')
    tfidf = vectorizer.fit_transform(texts)
    scores = tfidf.max(axis=0).toarray().flatten()
    return list(vectorizer.get_feature_names_out()[scores.argsort()[::-1][:max_terms]])


def _law_matches_ref(law_id, ref_text):
    law_lower = law_id.lower().replace("_", " ").replace("-", " ")
    ref_lower = ref_text.lower().replace("_", " ").replace("-", " ")
    parts = law_id.split("/")
    if len(parts) >= 3:
        if parts[2].replace("_", " ") in ref_lower: return True
        if len(parts) >= 2 and parts[1] in ref_lower: return True
    return False


def build_concept_graph(df, encode_fn, config):
    """Build enhanced concept graph from training data."""
    graph = LegalConceptGraph(config)
    law_groups = df.groupby("law_id")
    law_ids = sorted(law_groups.groups.keys())
    N_laws = len(law_ids)
    print(f"  Building graph: {N_laws} law nodes...")

    # Law embeddings
    embs = []
    for lid in law_ids:
        group = law_groups.get_group(lid)
        texts = [f"{t}: {txt[:300]}" for t, txt in zip(group["title"], group["text"])]
        embs.append(torch.stack([encode_fn(t) for t in texts[:5]]).mean(dim=0))
    law_embs = torch.stack(embs)
    graph.add_nodes(law_ids, law_embs)
    law_id_to_idx = {lid: i for i, lid in enumerate(law_ids)}

    # Key term nodes
    print("  Extracting key terms...")
    key_terms = extract_key_terms(df, max_terms=200)
    term_embs = torch.stack([encode_fn(t) for t in key_terms])
    graph.add_nodes([f"TERM:{t}" for t in key_terms], term_embs)
    print(f"    {len(key_terms)} key terms")

    # Agency edges
    agency_edges = []
    for _, group in df.groupby("law_type"):
        same = group["law_id"].unique()
        for i in range(len(same)):
            for j in range(i + 1, len(same)):
                if same[i] in law_id_to_idx and same[j] in law_id_to_idx:
                    agency_edges.append((law_id_to_idx[same[i]], law_id_to_idx[same[j]], 0.3))
    graph.add_edges("agency", agency_edges)
    print(f"    Agency edges: {len(agency_edges)}")

    # Temporal edges
    temporal_edges = []
    for _, group in df.groupby("law_type"):
        yl = group.groupby("year")["law_id"].unique()
        for y1, y2 in zip(sorted(yl.keys()), sorted(yl.keys())[1:]):
            for l1 in yl[y1]:
                for l2 in yl[y2]:
                    if l1 in law_id_to_idx and l2 in law_id_to_idx:
                        temporal_edges.append((law_id_to_idx[l1], law_id_to_idx[l2], 0.2))
    graph.add_edges("temporal", temporal_edges)
    print(f"    Temporal edges: {len(temporal_edges)}")

    # Semantic edges (chunked k-NN)
    semantic_k = min(config.graph.semantic_knn, N_laws - 1)
    semantic_edges = []
    if N_laws > 1:
        chunk = 64
        for i in range(0, N_laws, chunk):
            end = min(i + chunk, N_laws)
            sim = F.cosine_similarity(law_embs[i:end].unsqueeze(1), law_embs.unsqueeze(0), dim=2)
            for j in range(sim.shape[0]):
                sim[j, i + j] = float("-inf")
            vals, idx = sim.topk(k=semantic_k, dim=1)
            for j in range(sim.shape[0]):
                for kk in range(semantic_k):
                    semantic_edges.append((i + j, idx[j, kk].item(), vals[j, kk].item()))
    graph.add_edges("semantic", semantic_edges)
    print(f"    Semantic edges: {len(semantic_edges)}")

    # Cross-reference edges
    cross_ref_edges = []
    for _, row in df.iterrows():
        src = row["law_id"]
        if src not in law_id_to_idx: continue
        for pattern, etype in CROSS_REF_PATTERNS:
            for match in pattern.findall(row["text"]):
                match_str = " ".join(match).lower() if isinstance(match, tuple) else str(match).lower()
                for tgt in law_ids:
                    if tgt != src and _law_matches_ref(tgt, match_str):
                        cross_ref_edges.append((law_id_to_idx[src], law_id_to_idx[tgt], 0.5))
                        break
    graph.add_edges("cross_ref", cross_ref_edges)
    print(f"    Cross-ref edges: {len(cross_ref_edges)}")

    # Term-document edges
    term_doc_edges = []
    law_texts = [(f"{row['title']} {row['text'][:300]}").replace("_", " ")
                 for _, row in df.iterrows()]
    vec = TfidfVectorizer(vocabulary=key_terms if key_terms else None)
    try:
        tfidf = vec.fit_transform(law_texts)
        for ti, term in enumerate(key_terms):
            if ti < tfidf.shape[1]:
                col = tfidf[:, ti].toarray().flatten()
                for lp in col.argsort()[::-1][:10]:
                    if col[lp] > 0.1 and lp < N_laws:
                        term_doc_edges.append((N_laws + ti, lp, float(col[lp])))
    except ValueError:
        pass
    graph.add_edges("semantic", term_doc_edges)
    print(f"    Term-doc edges: {len(term_doc_edges)}")
    print(f"  Total: {graph.num_nodes} nodes ({N_laws} laws + {len(key_terms)} terms)")

    return graph, law_id_to_idx
