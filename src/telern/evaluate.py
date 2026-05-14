"""
Evaluation for TELEN: NDCG@k and MRR@k.

Metrics:
  - NDCG@3, NDCG@5, NDCG@10
  - MRR@3, MRR@5, MRR@10

Baselines:
  - BM25 (lexical)
  - Frozen PhoBERT + mean pooling
  - TELEN (ours)

Evaluation setup:
  - Query = article title + first 100 chars of text
  - Relevant = other articles from the SAME law
  - Corpus = all articles from test years (held-out)
"""

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from .config import TELENConfig, DATA_DIR
from .model import TELEN, create_model as create_telen
from ..data import load_raw_data, extract_metadata, clean_data


# ═══════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════

def dcg_at_k(scores: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain at k."""
    scores = np.asarray(scores)[:k]
    if len(scores) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(scores) + 2))
    return np.sum((2.0**scores - 1) / discounts)


def ndcg_at_k(scores: np.ndarray, k: int) -> float:
    """Normalized DCG at k."""
    ideal = np.sort(scores)[::-1]
    dcg_val = dcg_at_k(scores, k)
    idcg_val = dcg_at_k(ideal, k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def mrr_at_k(scores: np.ndarray, k: int) -> float:
    """Mean Reciprocal Rank at k."""
    scores = np.asarray(scores)[:k]
    for rank, s in enumerate(scores, start=1):
        if s > 0:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    relevance_scores: np.ndarray, k_values: List[int] = [3, 5, 10]
) -> Dict[str, float]:
    """Compute NDCG@k and MRR@k from relevance scores."""
    metrics = {}
    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)
        metrics[f"mrr@{k}"] = mrr_at_k(relevance_scores, k)
    return metrics


# ═══════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════

def prepare_test_data(config: TELENConfig):
    """Prepare test data from held-out years."""
    print("Loading data...")
    df = load_raw_data(str(DATA_DIR / "train-00000-of-00001.parquet"))
    df = extract_metadata(df)
    df = clean_data(df, min_text_len=10)

    # Test split: articles from test years
    test_years = range(config.meta.val_split_year + 1, 2025)
    test_df = df[df["year"].isin(test_years)].reset_index(drop=True)

    print(f"  Test set: {len(test_df)} articles from {test_df['law_id'].nunique()} laws")
    return test_df


def build_test_queries(test_df: pd.DataFrame, max_queries: int = 500) -> List[Dict]:
    """Build query set from test articles."""
    # Group by law_id
    law_groups = test_df.groupby("law_id")

    queries = []
    for law_id, group in law_groups:
        articles = group.to_dict("records")
        if len(articles) < 3:  # Need at least 1 query + 2 relevant
            continue
        # Use each article as a potential query
        for article in articles[:2]:  # Max 2 queries per law
            queries.append({
                "query_id": article["id"],
                "query_text": f"{article['title']}: {article['text'][:500]}",
                "query_full": article["text"],
                "law_id": law_id,
            })

    if len(queries) > max_queries:
        queries = random.sample(queries, max_queries)

    print(f"  Queries: {len(queries)}")
    return queries


def build_test_corpus(test_df: pd.DataFrame) -> List[Dict]:
    """Build corpus of all test articles for retrieval."""
    corpus = []
    for _, row in test_df.iterrows():
        corpus.append({
            "article_id": row["id"],
            "text": f"{row['title']}: {row['text'][:500]}",
            "law_id": row["law_id"],
        })
    print(f"  Corpus: {len(corpus)} documents")
    return corpus


def evaluate_telen(
    model: TELEN,
    queries: List[Dict],
    corpus: List[Dict],
    batch_size: int = 64,
) -> Dict[str, float]:
    """
    Evaluate TELEN on retrieval metrics.

    For each query, rank all corpus documents by cosine similarity.
    Relevance = article is from the same law.
    """
    device = next(model.parameters()).device
    model.eval()

    # Encode corpus
    print("  Encoding corpus...")
    corpus_embeddings = []
    corpus_ids = [doc["article_id"] for doc in corpus]
    corpus_law_ids = [doc["law_id"] for doc in corpus]

    for i in tqdm(range(0, len(corpus), batch_size), desc="  Corpus"):
        batch = corpus[i:i + batch_size]
        texts = [doc["text"] for doc in batch]
        with torch.no_grad():
            result = model(texts, use_stochastic=False)
            corpus_embeddings.append(result["embeddings"].cpu())

    corpus_embeddings = torch.cat(corpus_embeddings, dim=0)  # [N_corpus, d]
    print(f"  Corpus embeddings: {corpus_embeddings.shape}")

    # Evaluate each query
    all_metrics = defaultdict(list)

    print("  Evaluating queries...")
    for query in tqdm(queries, desc="  Queries"):
        # Encode query
        with torch.no_grad():
            result = model([query["query_text"]], use_stochastic=False)
            query_emb = result["embeddings"].cpu()  # [1, d]

        # Cosine similarity with all corpus
        sim = F.cosine_similarity(
            query_emb, corpus_embeddings
        ).numpy()  # [N_corpus]

        # Build relevance scores (1.0 if same law, 0.0 otherwise)
        relevance = np.array([
            1.0 if corpus_law_ids[i] == query["law_id"] else 0.0
            for i in range(len(corpus))
        ])

        # Rank by similarity and compute metrics
        sorted_idx = sim.argsort()[::-1]
        sorted_relevance = relevance[sorted_idx]

        # Remove the query itself from results
        query_idx_in_corpus = None
        for i, cid in enumerate(corpus_ids):
            if cid == query["query_id"]:
                query_idx_in_corpus = i
                break

        if query_idx_in_corpus is not None:
            # Remove self-match
            mask = sorted_idx != query_idx_in_corpus
            sorted_relevance = sorted_relevance[mask]

        # Compute metrics
        for k in [3, 5, 10]:
            metrics = compute_metrics(sorted_relevance[:k], [k])
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)

    # Average over queries
    results = {name: np.mean(scores) for name, scores in all_metrics.items()}
    return results


# ═══════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════

class BM25Baseline:
    """Simple BM25 implementation using TF-IDF as approximation."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

    def fit(self, corpus: List[Dict]):
        self.corpus = corpus
        self.doc_texts = [doc["text"] for doc in corpus]
        self.doc_ids = [doc["article_id"] for doc in corpus]
        self.doc_law_ids = [doc["law_id"] for doc in corpus]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def search(self, query_text: str, k: int = 100) -> np.ndarray:
        query_vec = self.vectorizer.transform([query_text])
        scores = (self.tfidf_matrix @ query_vec.T).toarray().flatten()
        sorted_idx = scores.argsort()[::-1]
        return sorted_idx


def evaluate_bm25(queries: List[Dict], corpus: List[Dict]) -> Dict[str, float]:
    """Evaluate BM25 baseline."""
    print("  Building BM25 index...")
    bm25 = BM25Baseline()
    bm25.fit(corpus)

    all_metrics = defaultdict(list)

    print("  Evaluating queries...")
    for query in tqdm(queries, desc="  Queries"):
        sorted_idx = bm25.search(query["query_text"], k=100)

        # Remove self
        doc_ids = bm25.doc_ids
        query_idx = None
        for i, did in enumerate(doc_ids):
            if did == query["query_id"]:
                query_idx = i
                break

        relevance = np.array([
            1.0 if bm25.doc_law_ids[i] == query["law_id"] else 0.0
            for i in sorted_idx
        ])

        if query_idx is not None:
            pos = np.where(sorted_idx == query_idx)[0]
            if len(pos) > 0:
                relevance = np.delete(relevance, pos[0])

        for k in [3, 5, 10]:
            valid_rel = relevance[:k]
            metrics = compute_metrics(valid_rel, [k])
            for name, val in metrics.items():
                all_metrics[name].append(val)

    return {name: np.mean(scores) for name, scores in all_metrics.items()}


class FrozenPhoBERT:
    """Frozen PhoBERT with mean pooling baseline."""

    def __init__(self, model_name: str = "vinai/phobert-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                # Mean pooling
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = F.normalize(pooled, p=2, dim=1)
                embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)


def evaluate_frozen_phobert(
    queries: List[Dict], corpus: List[Dict]
) -> Dict[str, float]:
    """Evaluate frozen PhoBERT baseline."""
    print("  Loading frozen PhoBERT...")
    encoder = FrozenPhoBERT()

    print("  Encoding corpus...")
    corpus_texts = [doc["text"] for doc in corpus]
    corpus_embeddings = encoder.encode(corpus_texts)
    corpus_ids = [doc["article_id"] for doc in corpus]
    corpus_law_ids = [doc["law_id"] for doc in corpus]

    all_metrics = defaultdict(list)

    print("  Evaluating queries...")
    query_texts = [q["query_text"] for q in queries]
    query_embeddings = encoder.encode(query_texts)

    for i, query in enumerate(tqdm(queries, desc="  Queries")):
        query_emb = query_embeddings[i:i+1]
        sim = F.cosine_similarity(query_emb, corpus_embeddings).numpy()

        relevance = np.array([
            1.0 if corpus_law_ids[j] == query["law_id"] else 0.0
            for j in range(len(corpus))
        ])

        sorted_idx = sim.argsort()[::-1]
        sorted_relevance = relevance[sorted_idx]

        # Remove self
        for j, cid in enumerate(corpus_ids):
            if cid == query["query_id"]:
                mask = sorted_idx != j
                sorted_relevance = sorted_relevance[mask]
                break

        for k in [3, 5, 10]:
            metrics = compute_metrics(sorted_relevance[:k], [k])
            for name, val in metrics.items():
                all_metrics[name].append(val)

    return {name: np.mean(scores) for name, scores in all_metrics.items()}


# ═══════════════════════════════════════════════════════════
# Main evaluation entry point
# ═══════════════════════════════════════════════════════════

def run_full_evaluation(
    config: TELENConfig = None,
    checkpoint_path: str = None,
):
    """Run complete evaluation with all baselines and TELEN."""
    if config is None:
        config = TELENConfig()

    random.seed(config.seed)
    np.random.seed(config.seed)

    print("=" * 60)
    print("TELEN Evaluation")
    print("=" * 60)

    # Prepare test data
    test_df = prepare_test_data(config)
    queries = build_test_queries(test_df, max_queries=300)
    corpus = build_test_corpus(test_df)

    k_values = [3, 5, 10]
    results = {}

    # --- Baseline 1: BM25 ---
    print("\n" + "=" * 40)
    print("[1/3] BM25 Baseline")
    print("=" * 40)
    results["BM25"] = evaluate_bm25(queries, corpus)
    for m in k_values:
        print(f"  NDCG@{m}: {results['BM25'][f'ndcg@{m}']:.4f}  |  MRR@{m}: {results['BM25'][f'mrr@{m}']:.4f}")

    # --- Baseline 2: Frozen PhoBERT ---
    print("\n" + "=" * 40)
    print("[2/3] Frozen PhoBERT Baseline")
    print("=" * 40)
    results["PhoBERT"] = evaluate_frozen_phobert(queries, corpus)
    for m in k_values:
        print(f"  NDCG@{m}: {results['PhoBERT'][f'ndcg@{m}']:.4f}  |  MRR@{m}: {results['PhoBERT'][f'mrr@{m}']:.4f}")

    # --- TELEN ---
    print("\n" + "=" * 40)
    print("[3/3] TELEN (Ours)")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_telen(config)
    model = model.to(device)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.hypernetwork.load_state_dict(ckpt["hypernetwork"])
        model.state_encoder.load_state_dict(ckpt["state_encoder"])
        model.base_projection.load_state_dict(ckpt["base_projection"])
        model.attn_query.data.copy_(ckpt["attn_query"])
        # Rebuild graph
        model.build_graph(test_df[test_df["year"] <= config.meta.train_split_year])

    results["TELEN"] = evaluate_telen(model, queries, corpus)
    for m in k_values:
        print(f"  NDCG@{m}: {results['TELEN'][f'ndcg@{m}']:.4f}  |  MRR@{m}: {results['TELEN'][f'mrr@{m}']:.4f}")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    header = f"{'Method':<20}"
    for m in k_values:
        header += f" {'NDCG@'+str(m):>12} {'MRR@'+str(m):>12}"
    print(header)
    print("-" * len(header))

    for method in ["BM25", "PhoBERT", "TELEN"]:
        row = f"{method:<20}"
        for m in k_values:
            row += f" {results[method][f'ndcg@{m}']:>12.4f} {results[method][f'mrr@{m}']:>12.4f}"
        print(row)

    # Relative improvement
    print("\n--- Improvement over PhoBERT ---")
    for m in k_values:
        ndcg_imp = (results["TELEN"][f"ndcg@{m}"] / max(results["PhoBERT"][f"ndcg@{m}"], 1e-6) - 1) * 100
        mrr_imp = (results["TELEN"][f"mrr@{m}"] / max(results["PhoBERT"][f"mrr@{m}"], 1e-6) - 1) * 100
        print(f"  NDCG@{m}: {ndcg_imp:+.1f}%  |  MRR@{m}: {mrr_imp:+.1f}%")

    return results


if __name__ == "__main__":
    run_full_evaluation()
