"""
TELEN: Temporal Evolving Legal Embedding Network — Training Script.

Stages:
  1. Contrastive pretraining (5 epochs) — train projection head
  2. Meta-training (50 epochs) — train HyperNetwork + State Encoder

Usage:
    python train.py
"""
import sys, os, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from pyvi import ViTokenizer

sys.path.insert(0, ".")
from src.telern.config import TELENConfig, DATA_DIR
from src.telern.model import TELEN, create_model
from src.data import load_raw_data, extract_metadata, clean_data

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ═══════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════
def prepare_data(config):
    df = load_raw_data(str(DATA_DIR / "train-00000-of-00001.parquet"))
    df = extract_metadata(df); df = clean_data(df, min_text_len=10)
    articles_by_law = defaultdict(list)
    laws_by_year = defaultdict(list)
    for _, row in df.iterrows():
        articles_by_law[row["law_id"]].append({
            "id": row["id"], "title": row["title"], "text": row["text"],
            "law_type": row["law_type"], "year": row["year"],
        })
    for law_id in articles_by_law:
        laws_by_year[articles_by_law[law_id][0]["year"]].append(law_id)
    all_years = sorted(laws_by_year.keys())
    train_years = [y for y in all_years if y <= config.meta.train_split_year]
    val_years = [y for y in all_years if config.meta.train_split_year < y <= config.meta.val_split_year]
    test_years = [y for y in all_years if y > config.meta.val_split_year]
    return articles_by_law, laws_by_year, train_years, val_years, test_years, df

# ═══════════════════════════════════════════════════════════
# Contrastive Dataset
# ═══════════════════════════════════════════════════════════
class ContrastiveDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=480):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.law_groups = self.df.groupby("law_id")
        self.law_ids = list(self.law_groups.groups.keys())

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]; law_id = row["law_id"]
        wseg = lambda t: ViTokenizer.tokenize(t.replace("_", " "))
        anchor = wseg(f"{row['title']}: {row['text'][:400]}")
        group_idx = self.law_groups.groups[law_id]
        others = [i for i in group_idx if i != idx]
        pos_row = self.df.iloc[random.choice(others)] if others else row
        positive = wseg(f"{pos_row['title']}: {pos_row['text'][:400]}")
        neg_law = random.choice([l for l in self.law_ids if l != law_id])
        neg_row = self.df.iloc[random.choice(list(self.law_groups.groups[neg_law]))]
        negative = wseg(f"{neg_row['title']}: {neg_row['text'][:400]}")

        def tok(t): return self.tokenizer(t, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        return {f"{k}_{s}": tok(t)[k].squeeze(0)
                for t, s in [(anchor,"a"),(positive,"p"),(negative,"n")]
                for k in ["input_ids","attention_mask"]}

# ═══════════════════════════════════════════════════════════
# Stage 1: Contrastive Pretraining
# ═══════════════════════════════════════════════════════════
def contrastive_pretrain(model, df, config, epochs=5, batch_size=24, lr=3e-5):
    tokenizer = model.encoder.tokenizer
    dataset = ContrastiveDataset(df, tokenizer, config.max_seq_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainable = list(model.base_projection.parameters()) + [model.attn_query]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))

    print(f"  Contrastive pretraining: {epochs} epochs, {len(loader)} batches")
    model.train(); model.encoder.model.eval()

    for epoch in range(epochs):
        total = 0.0
        for batch in tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}"):
            a_ids=batch["input_ids_a"].to(device); a_mask=batch["attention_mask_a"].to(device)
            p_ids=batch["input_ids_p"].to(device); p_mask=batch["attention_mask_p"].to(device)
            n_ids=batch["input_ids_n"].to(device); n_mask=batch["attention_mask_n"].to(device)

            with torch.no_grad():
                ah=model._pool(model.encoder.model(input_ids=a_ids,attention_mask=a_mask).last_hidden_state,a_mask)
                ph=model._pool(model.encoder.model(input_ids=p_ids,attention_mask=p_mask).last_hidden_state,p_mask)
                nh=model._pool(model.encoder.model(input_ids=n_ids,attention_mask=n_mask).last_hidden_state,n_mask)

            ae=F.normalize(model.base_projection(ah),p=2,dim=1)
            pe=F.normalize(model.base_projection(ph),p=2,dim=1)
            ne=F.normalize(model.base_projection(nh),p=2,dim=1)

            trip=F.relu(0.3-(ae*pe).sum(1)+(ae*ne).sum(1)).mean()
            sim=ae@torch.cat([ae,pe,ne],dim=0).T/0.05
            infonce=F.cross_entropy(sim,torch.arange(len(a_ids),device=device)+len(a_ids))
            loss=trip+0.5*infonce

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0); opt.step(); sched.step()
            total+=loss.item()
        print(f"    Epoch {epoch+1} avg loss: {total/len(loader):.4f}")
    print("  Contrastive pretraining complete!")
    return model

# ═══════════════════════════════════════════════════════════
# Episode building
# ═══════════════════════════════════════════════════════════
def build_episode(articles_by_law, laws_by_year, state_years, query_year, config):
    mc = config.meta
    q_laws = laws_by_year.get(query_year, [])
    if len(q_laws) < 5: return None
    sampled = random.sample(q_laws, min(mc.n_query // 4, len(q_laws)))
    queries, positives, q_types = [], [], set()
    for lid in sampled:
        arts = articles_by_law[lid]
        if len(arts) < 2: continue
        qi, pi = random.sample(range(len(arts)), 2)
        queries.append(arts[qi]); positives.append(arts[pi])
        q_types.add(arts[qi]["law_type"])
    if len(queries) < 4: return None

    hard_neg, rand_neg = [], []
    for lid in q_laws:
        if lid in sampled: continue
        for a in articles_by_law[lid]:
            if a["law_type"] in q_types: hard_neg.append(a)
            else: rand_neg.append(a)

    nh = min(mc.n_negatives // 2, len(hard_neg))
    nr = min(mc.n_negatives - nh, len(rand_neg))
    negatives = (random.sample(hard_neg, nh) if nh > 0 else []) + (random.sample(rand_neg, nr) if nr > 0 else [])
    if len(negatives) < 4: return None
    return {"queries": queries, "positives": positives, "negatives": negatives}

# ═══════════════════════════════════════════════════════════
# Stage 2: Meta-Training
# ═══════════════════════════════════════════════════════════
def compute_loss(model, q_texts, p_texts, n_texts, state_vec, temp=0.05):
    n_q, n_p = len(q_texts), len(p_texts)
    if n_q == 0 or n_p == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    all_t = q_texts + p_texts + n_texts
    raw = model.encode_text(all_t)
    adapted = model.adapt_embedding(raw, state_vec)
    emb = adapted["mean"]
    qe = emb[:n_q]; pe = emb[n_q:n_q+n_p]; ne = emb[n_q+n_p:]

    if n_q == n_p:
        sim = torch.cat([(qe*pe).sum(1).unsqueeze(1)/temp, qe@ne.T/temp], dim=1)
        loss = F.cross_entropy(sim, torch.zeros(n_q, dtype=torch.long, device=device))
    else:
        loss = F.cross_entropy(qe @ torch.cat([pe, ne], dim=0).T / temp,
                               torch.arange(n_q, device=device).clamp(max=len(pe)-1))

    if model.config.hypernetwork.output_variance:
        lv = adapted.get("log_variance")
        if lv is not None: loss = loss + (lv.exp() - lv - 1).mean() * model.config.meta.kl_weight
    return loss

def validate(model, articles_by_law, laws_by_year, val_years, config):
    model.eval(); losses = []
    with torch.no_grad():
        for _ in range(30):
            qy = random.choice(val_years)
            if qy not in laws_by_year: continue
            sy = [y for y in sorted(laws_by_year.keys()) if y < qy]
            if len(sy) < 3: sy = [y for y in sorted(laws_by_year.keys()) if y <= qy]
            ep = build_episode(articles_by_law, laws_by_year, sy, qy, config)
            if ep is None: continue
            sv = model.get_state_vector()
            losses.append(compute_loss(model,
                [f"{q['title']}: {q['text'][:200]}" for q in ep["queries"]],
                [f"{p['title']}: {p['text'][:200]}" for p in ep["positives"]],
                [f"{n['title']}: {n['text'][:200]}" for n in ep["negatives"]],
                sv, config.meta.temperature).item())
    return sum(losses)/max(len(losses),1)

def meta_train(model, articles_by_law, laws_by_year, train_years, val_years, config, epochs=50, patience=15):
    trainable = (list(model.hypernetwork.parameters()) + list(model.state_encoder.parameters()) +
                 list(model.base_projection.parameters()) + [model.attn_query])
    opt = torch.optim.AdamW(trainable, lr=config.meta.meta_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    os.makedirs(config.output_dir, exist_ok=True)
    best_val, patience_ctr = float("inf"), 0

    for epoch in range(epochs):
        model.train(); total_loss = 0.0
        steps = config.meta.meta_batch_size * 100
        progress = tqdm(range(steps), desc=f"Meta Epoch {epoch+1}/{epochs}")
        for _ in progress:
            if len(train_years) < 3: break
            si = random.randint(2, len(train_years)-1)
            sy, qy = train_years[:si], train_years[si]
            if qy not in laws_by_year: continue
            ep = build_episode(articles_by_law, laws_by_year, sy, qy, config)
            if ep is None: continue
            sv = model.get_state_vector()
            loss = compute_loss(model,
                [f"{q['title']}: {q['text'][:200]}" for q in ep["queries"]],
                [f"{p['title']}: {p['text'][:200]}" for p in ep["positives"]],
                [f"{n['title']}: {n['text'][:200]}" for n in ep["negatives"]],
                sv, config.meta.temperature)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg = total_loss / max(steps, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg:.4f}")

        vl = validate(model, articles_by_law, laws_by_year, val_years, config)
        print(f"  Val loss: {vl:.4f}")
        sched.step(vl)

        if vl < best_val:
            best_val, patience_ctr = vl, 0
            torch.save({
                "hypernetwork": model.hypernetwork.state_dict(),
                "state_encoder": model.state_encoder.state_dict(),
                "base_projection": model.base_projection.state_dict(),
                "attn_query": model.attn_query,
                "epoch": epoch, "val_loss": vl,
            }, Path(config.output_dir) / "telen_best.pt")
            print(f"  Saved (val_loss={vl:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch+1}"); break
    print("Meta-training complete!")
    return model

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
def main():
    config = TELENConfig()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    articles_by_law, laws_by_year, train_years, val_years, test_years, df = prepare_data(config)
    print(f"  Train: {train_years[0]}-{train_years[-1]} ({len(train_years)}y)")
    print(f"  Val:   {val_years[0]}-{val_years[-1]} ({len(val_years)}y)")
    print(f"  Test:  {len(test_years)}y")

    # Model
    print("\nCreating TELEN...")
    model = create_model(config).to(device)
    print(f"  HyperNetwork: {sum(p.numel() for p in model.hypernetwork.parameters()):,} params")

    # Build graph
    print("\nBuilding concept graph...")
    train_df = df[df["year"].isin(train_years)]
    model.build_graph(train_df)
    print(f"  Graph: {model.concept_graph.num_nodes} nodes")

    # Stage 1
    print("\n" + "=" * 60)
    print("Stage 1: Contrastive Pretraining")
    print("=" * 60)
    model = contrastive_pretrain(model, train_df, config, epochs=5, batch_size=24, lr=3e-5)

    # Stage 2
    print("\n" + "=" * 60)
    print("Stage 2: Meta-Training")
    print("=" * 60)
    model = meta_train(model, articles_by_law, laws_by_year, train_years, val_years, config, epochs=50, patience=15)

    print(f"\nDone! Model saved to: {config.output_dir}/telen_best.pt")

if __name__ == "__main__":
    main()
