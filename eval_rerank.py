"""
TELEN + Cross-encoder re-rank top-5 for MRR boost.
TELEN retrieves top-50, CE re-ranks top-5 to push relevant doc to rank 1.
"""
import sys; sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings("ignore")
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm; from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

from src.telern.config import TELENConfig
from src.telern.model import create_model
from src.telern.evaluate import prepare_test_data, build_test_queries, build_test_corpus, compute_metrics

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = TELENConfig()

# Data
test_df = prepare_test_data(config)
queries = build_test_queries(test_df, max_queries=300)
corpus = build_test_corpus(test_df)
corpus_ids = [d["article_id"] for d in corpus]
corpus_law_ids = [d["law_id"] for d in corpus]
train_df = test_df[test_df["year"] <= config.meta.train_split_year]
print(f"Test: {len(queries)} queries, {len(corpus)} docs")

# TELEN
print("Loading TELEN...")
telen = create_model(config).to(device)
ckpt = torch.load(config.output_dir + "/telen_best.pt", map_location=device, weights_only=False)
telen.hypernetwork.load_state_dict(ckpt["hypernetwork"])
telen.state_encoder.load_state_dict(ckpt["state_encoder"])
telen.base_projection.load_state_dict(ckpt["base_projection"])
telen.attn_query.data.copy_(ckpt["attn_query"])
if len(train_df)>0: telen.build_graph(train_df)

print("Encoding corpus...")
c_embs = []
for i in range(0,len(corpus),64):
    with torch.no_grad(): r=telen([d["text"] for d in corpus[i:i+64]],use_stochastic=False); c_embs.append(r["embeddings"].cpu())
c_embs = torch.cat(c_embs,dim=0)

# Cross-encoder
print("Loading cross-encoder...")
ce_encoder = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
ce_head = nn.Sequential(
    nn.Linear(ce_encoder.config.hidden_size, 512), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, 1),
).to(device)
ce_ckpt = torch.load("data/checkpoints/telen/cross_encoder_best.pt", map_location=device, weights_only=False)
ce_encoder.load_state_dict(ce_ckpt["encoder"])
ce_head.load_state_dict(ce_ckpt["head"])
ce_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
ce_encoder.eval(); ce_head.eval()

def ce_score(query, docs):
    scores = []
    for i in range(0, len(docs), 32):
        b_docs = docs[i:i+32]
        enc = ce_tokenizer([query]*len(b_docs), b_docs, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            out = ce_encoder(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            s = torch.sigmoid(ce_head(out.last_hidden_state[:,0,:])).squeeze(-1).cpu().numpy()
            if s.ndim==0: s=np.array([s])
            scores.append(s)
    return np.concatenate(scores)

# Evaluate: TELEN only vs TELEN + CE rerank top-5
rerank_k_vals = [5, 10, 20]
results = {}

# TELEN standalone
print("\n--- TELEN standalone ---")
v3_m = defaultdict(list)
for q in tqdm(queries, desc="TELEN"):
    with torch.no_grad(): qe=telen([q["query_text"]],use_stochastic=False)["embeddings"].cpu()
    sim=F.cosine_similarity(qe,c_embs).numpy()
    si=sim.argsort()[::-1]
    rel=np.array([1.0 if corpus_law_ids[j]==q["law_id"] else 0.0 for j in range(len(corpus))])[si]
    for j,cid in enumerate(corpus_ids):
        if cid==q["query_id"]: p=np.where(si==j)[0]; rel=np.delete(rel,p[0]) if len(p)>0 else None; break
    for k in [3,5,10]:
        for mn,mv in compute_metrics(rel[:k],[k]).items(): v3_m[mn].append(mv)
results["TELEN"] = {n:np.mean(v) for n,v in v3_m.items()}

# TELEN + CE rerank
for rerank_k in rerank_k_vals:
    print(f"\n--- TELEN + CE rerank top-{rerank_k} ---")
    all_m = defaultdict(list)
    for q in tqdm(queries, desc=f"CE-{rerank_k}"):
        with torch.no_grad(): qe=telen([q["query_text"]],use_stochastic=False)["embeddings"].cpu()
        sim=F.cosine_similarity(qe,c_embs).numpy()
        top_k=sim.argsort()[::-1][:rerank_k]
        top_docs=[corpus[idx]["text"] for idx in top_k]
        ce_s=ce_score(q["query_text"],top_docs)
        reranked=top_k[ce_s.argsort()[::-1]]
        remaining=sim.argsort()[::-1][~np.isin(sim.argsort()[::-1],top_k)]
        final=np.concatenate([reranked,remaining])
        rel=np.array([1.0 if corpus_law_ids[j]==q["law_id"] else 0.0 for j in final])
        for j,cid in enumerate(corpus_ids):
            if cid==q["query_id"]: p=np.where(final==j)[0]; rel=np.delete(rel,p[0]) if len(p)>0 else None; break
        for k in [3,5,10]:
            for mn,mv in compute_metrics(rel[:k],[k]).items(): all_m[mn].append(mv)
    results[f"TELEN+CE@{rerank_k}"] = {n:np.mean(v) for n,v in all_m.items()}

# Summary
print("\n"+"="*75)
print("MRR BOOST RESULTS")
print("="*75)
h=f"{'Method':<20}"
for m in [3,5,10]: h+=f" {'NDCG@'+str(m):>10} {'MRR@'+str(m):>10}"
print(h); print("-"*len(h))
for name, r in results.items():
    row=f"{name:<20}"
    for m in [3,5,10]: row+=f" {r[f'ndcg@{m}']:>10.4f} {r[f'mrr@{m}']:>10.4f}"
    print(row)

print("\n--- MRR Gain vs TELEN standalone ---")
base=results["TELEN"]
for name in [k for k in results if k!="TELEN"]:
    r=results[name]
    gains=[(m,(r[f"mrr@{m}"]/max(base[f"mrr@{m}"],1e-6)-1)*100) for m in [3,5,10]]
    print(f"  {name}: " + " | ".join(f"MRR@{m}: {g:+.1f}%" for m,g in gains))
print("Done!")
