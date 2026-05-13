"""
Evaluate TELEN with full benchmarks.

Metrics: NDCG@3, NDCG@5, NDCG@10, MRR@3, MRR@5, MRR@10

Baselines:
  - BM25 (lexical retrieval)
  - Frozen PhoBERT (vinai/phobert-base-v2)
  - DEk21 (huyydangg/DEk21_hcmute_embedding)
  - TELEN (ours)

Usage:
    python eval.py
"""
import sys; sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings("ignore")
import random, numpy as np, torch, torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pyvi import ViTokenizer

from src.telern.config import TELENConfig
from src.telern.model import create_model
from src.telern.evaluate import (
    BM25Baseline, FrozenPhoBERT, prepare_test_data,
    build_test_queries, build_test_corpus, compute_metrics, evaluate_bm25,
)

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = TELENConfig()

def wseg(text):
    return ViTokenizer.tokenize(text.replace("_", " "))

def evaluate_model(name, encode_fn, queries, corpus, corpus_ids, corpus_law_ids):
    """Generic evaluation for any embedding model."""
    print(f"\n  [{name}] Encoding corpus ({len(corpus)} docs)...")
    c_embs = []
    for i in range(0, len(corpus), 64):
        batch = [d["text"] for d in corpus[i:i+64]]
        embs = encode_fn(batch)
        if isinstance(embs, np.ndarray): embs = torch.tensor(embs)
        c_embs.append(embs.cpu())
    c_embs = torch.cat(c_embs, dim=0)

    print(f"  [{name}] Evaluating {len(queries)} queries...")
    all_m = defaultdict(list)
    for q in tqdm(queries, desc=f"  {name}"):
        q_emb = encode_fn([q["query_text"]])
        if isinstance(q_emb, np.ndarray): q_emb = torch.tensor(q_emb)
        sim = F.cosine_similarity(q_emb.cpu(), c_embs).numpy()
        rel = np.array([1.0 if corpus_law_ids[j]==q["law_id"] else 0.0 for j in range(len(corpus))])
        si = sim.argsort()[::-1]; sr = rel[si]
        for j,cid in enumerate(corpus_ids):
            if cid==q["query_id"]:
                p=np.where(si==j)[0]; sr=np.delete(sr,p[0]) if len(p)>0 else None; break
        for k in [3,5,10]:
            for mn,mv in compute_metrics(sr[:k],[k]).items(): all_m[mn].append(mv)
    return {n: np.mean(v) for n,v in all_m.items()}

# ── Data ──
test_df = prepare_test_data(config)
queries = build_test_queries(test_df, max_queries=300)
corpus = build_test_corpus(test_df)
corpus_ids = [d["article_id"] for d in corpus]
corpus_law_ids = [d["law_id"] for d in corpus]
train_df = test_df[test_df["year"] <= config.meta.train_split_year]
print(f"Test: {len(queries)} queries, {len(corpus)} docs, {test_df['law_id'].nunique()} laws")

results = {}

# ── BM25 ──
print("\n[1/4] BM25")
results["BM25"] = evaluate_bm25(queries, corpus)

# ── PhoBERT ──
print("\n[2/4] Frozen PhoBERT")
phobert = FrozenPhoBERT()
results["PhoBERT"] = evaluate_model("PhoBERT", lambda texts: phobert.encode(texts, batch_size=64), queries, corpus, corpus_ids, corpus_law_ids)

# ── DEk21 ──
print("\n[3/4] DEk21 (SOTA)")
dek21 = SentenceTransformer("huyydangg/DEk21_hcmute_embedding", device=device)
results["DEk21"] = evaluate_model("DEk21", lambda texts: dek21.encode([wseg(t) for t in texts], batch_size=64, show_progress_bar=False, normalize_embeddings=True, convert_to_tensor=True), queries, corpus, corpus_ids, corpus_law_ids)

# ── TELEN ──
print("\n[4/4] TELEN (Ours)")
telen = create_model(config).to(device)
ckpt = torch.load(config.output_dir + "/telen_best.pt", map_location=device, weights_only=False)
telen.hypernetwork.load_state_dict(ckpt["hypernetwork"])
telen.state_encoder.load_state_dict(ckpt["state_encoder"])
telen.base_projection.load_state_dict(ckpt["base_projection"])
telen.attn_query.data.copy_(ckpt["attn_query"])
if len(train_df) > 0: telen.build_graph(train_df)

def telen_encode(texts):
    with torch.no_grad():
        return telen(texts, use_stochastic=False)["embeddings"].cpu()

results["TELEN"] = evaluate_model("TELEN", telen_encode, queries, corpus, corpus_ids, corpus_law_ids)

# ── Summary ──
print("\n" + "=" * 75)
print("BENCHMARK RESULTS")
print("=" * 75)
h = f"{'Method':<15}"
for m in [3,5,10]: h += f" {'NDCG@'+str(m):>10} {'MRR@'+str(m):>10}"
print(h); print("-"*len(h))
for name in ["BM25", "PhoBERT", "DEk21", "TELEN"]:
    r = f"{name:<15}"
    for m in [3,5,10]: r += f" {results[name][f'ndcg@{m}']:>10.4f} {results[name][f'mrr@{m}']:>10.4f}"
    print(r)

print("\n--- Relative Improvement over Baselines ---")
for baseline in ["PhoBERT", "DEk21"]:
    print(f"  TELEN vs {baseline}:")
    for m in [3,5,10]:
        ni = (results["TELEN"][f"ndcg@{m}"] / max(results[baseline][f"ndcg@{m}"], 1e-6) - 1) * 100
        mi = (results["TELEN"][f"mrr@{m}"] / max(results[baseline][f"mrr@{m}"], 1e-6) - 1) * 100
        print(f"    NDCG@{m}: {ni:+.1f}%  MRR@{m}: {mi:+.1f}%")
print("Done!")
