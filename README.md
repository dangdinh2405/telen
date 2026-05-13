# TELEN: Temporal Evolving Legal Embedding Network

> **Vietnamese legal text embedding with meta-learning for continuous adaptation to new laws.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

TELEN introduces a **novel embedding architecture** designed specifically for Vietnamese legal text retrieval in RAG (Retrieval-Augmented Generation) systems. Unlike conventional static embedding models, TELEN generates embeddings that **adapt dynamically** to the current state of the legal corpus — enabling seamless integration of new laws without retraining.

### Key Innovations

1. **HyperNetwork-Driven Projection** — Instead of fixed projection weights, a HyperNetwork generates the embedding projection function from the current legal corpus state. When new laws are published, the embedding space adapts automatically.

2. **Legal Concept Graph (LCG)** — An evolving knowledge graph where nodes represent legal entities (laws, key terms) and edges encode cross-references, agency hierarchy, temporal sequences, and semantic similarity.

3. **State-Adaptive Embeddings** — Embeddings are not static vectors but are modulated by a learned "legal state vector" that summarizes the entire legal landscape at any point in time.

---

## Architecture

```
Legal Text
    ↓
Bi-Encoder (bkai-foundation-models/vietnamese-bi-encoder)
    ↓
Raw Representation [768-dim]
    ↓
┌─────────────────────────────────────┐
│  HyperNetwork(state_vector) → ΔW, Δb │  ← Generated, not learned!
│  Adapted Projection = Base + ΔW·x + Δb │
└─────────────────────────────────────┘
    ↓
Legal Concept Graph (GNN)
    ↓  state_vector
State Encoder ← current legal corpus
    ↓
L2-Normalized Embedding [768-dim]
```

## Benchmark Results

**Test set**: 1,406 Vietnamese legal articles from 2021 (held-out, unseen during training)

| Model | NDCG@3 | NDCG@5 | NDCG@10 | MRR@3 | MRR@5 | MRR@10 |
|---|---|---|---|---|---|---|
| **BM25** (bm25) | 0.5164 | 0.5628 | 0.5718 | 0.5016 | 0.5290 | 0.5354 |
| **PhoBERT-base-v2** (dense) | 0.4803 | 0.5305 | 0.5738 | 0.4503 | 0.4792 | 0.4961 |
| **DEk21** (dense) | 0.6651 | 0.6907 | 0.7286 | 0.6394 | 0.6553 | 0.6734 |
| **TELEN** (dense) | **0.8878** | **0.9097** | **0.9132** | **0.8686** | **0.8782** | **0.8782** |

### Relative Improvement

| Baseline | NDCG@3 | NDCG@10 | MRR@10 |
|---|---|---|---|
| vs PhoBERT (dense) | **+84.9%** | **+59.2%** | **+77.1%** |
| vs DEk21 (dense) | **+33.5%** | **+25.3%** | **+30.4%** |

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Inference

```python
from inference import TELENInference

# Load model
model = TELENInference()

# Encode legal texts
texts = [
    "Điều 1: Thông tư này quy định về quản lý thuế giá trị gia tăng...",
    "Điều 2: Đối tượng áp dụng là các tổ chức, cá nhân kinh doanh...",
]
embeddings = model.encode(texts)  # → [2, 768] normalized vectors

# Compute similarity
similarity = model.similarity(texts[0], texts[1])
print(f"Cosine similarity: {similarity:.4f}")

# Retrieve similar documents
results = model.retrieve(texts[0], corpus, top_k=10)
```

### Training

```bash
# Train TELEN from scratch
python train.py

# Train cross-encoder re-ranker (optional, for extra +2-3% gain)
python train_ce.py
```

### Evaluation

```bash
python eval.py
```

---

## Training Details

### Dataset
- **Source**: [another-symato/VMTEB-Zalo-legel-retrieval-wseg](https://huggingface.co/datasets/another-symato/VMTEB-Zalo-legel-retrieval-wseg) on HuggingFace
- **Content**: 61,425 Vietnamese legal articles (Thông tư, Nghị định, Luật, Pháp lệnh)
- **Period**: 1999–2021
- **Format**: Word-segmented Vietnamese text (underscore-separated compound words)

### Training Pipeline

| Stage | Description | Epochs | Trainable Params |
|---|---|---|---|
| 1. Contrastive Pretraining | Triplet + InfoNCE loss on same-law article pairs | 5 | ~1M (projection head) |
| 2. Meta-Training | HyperNetwork learns to adapt embedding space for future laws | 50 (early stop) | ~4M (HyperNetwork + State Encoder) |

### Hyperparameters

| Parameter | Value |
|---|---|
| Backbone | `bkai-foundation-models/vietnamese-bi-encoder` |
| Embedding dimension | 768 |
| Adaptation rank | 64 |
| GNN layers | 3 |
| Meta N-way, K-shot | 16-way, 5-shot |
| Negatives per query | 256 (50% hard + 50% random) |
| Temperature | 0.05 |
| Optimizer | AdamW + CosineAnnealingWarmRestarts |

### Hardware
- GPU: NVIDIA RTX 5070 Ti (16GB VRAM)
- Training time: ~8 hours (5 contrastive + 50 meta epochs)

---

## Continuous Adaptation

When a new law is published, TELEN adapts without retraining:

```python
# New law arrives
new_articles = [
    "Điều 1: Luật mới về trí tuệ nhân tạo...",
    "Điều 2: Các nguyên tắc áp dụng AI trong xét xử...",
]

# Update concept graph (milliseconds)
model.add_new_law("123/2025/l-ai", new_articles)

# Embedding space automatically adapts via HyperNetwork
# All subsequent query embeddings reflect the new legal landscape
embeddings = model.encode(["Điều 1: ..."])
```

---

## Project Structure

```
law-embedding/
├── dataset/
│   └── train-00000-of-00001.parquet   # Training data (61K legal articles)
├── src/
│   ├── data.py                    # Data loading utilities
│   └── telern/
│       ├── config.py              # Configuration
│       ├── model.py               # TELEN architecture
│       ├── concept_graph.py       # Legal Concept Graph + GNN
│       ├── hypernetwork.py        # HyperNetwork + StateEncoder
│       └── evaluate.py            # Evaluation metrics & baselines
├── data/checkpoints/telen/
│   └── telen_best.pt              # Pretrained model weights
├── train.py                       # Training script
├── train_ce.py                    # Cross-encoder training (optional)
├── eval.py                        # Evaluation script
├── inference.py                   # Inference API
├── requirements.txt
└── README.md
```

---

## Citation

```bibtex
@misc{telen2025,
  title={TELEN: Temporal Evolving Legal Embedding Network for Vietnamese Law},
  author={dangdinh},
  year={2026},
  publisher={Huggingface},
}
```

## License

MIT License — see [LICENSE](LICENSE) file for details.

## Acknowledgments

- `bkai-foundation-models/vietnamese-bi-encoder` — backbone bi-encoder
- `huyydangg/DEk21_hcmute_embedding` — baseline comparison (previous SOTA)
- `vinai/phobert-base-v2` — used in cross-encoder re-ranker
