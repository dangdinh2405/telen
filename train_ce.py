"""
Train the cross-encoder re-ranker for legal text.

Usage:
    python train_ce.py

Trains a PhoBERT-based cross-encoder on legal article pairs
with margin ranking loss for re-ranking TELEN retrieval results.

Output: data/checkpoints/telen/cross_encoder_best.pt
"""
import sys; sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')
import warnings; warnings.filterwarnings("ignore")
import random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer

from src.telern.config import DATA_DIR
from src.data import load_raw_data, extract_metadata, clean_data

SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda")

# ── Data ──
print("Loading data...")
df = load_raw_data(str(DATA_DIR / "train-00000-of-00001.parquet"))
df = extract_metadata(df); df = clean_data(df, min_text_len=10)
train_df = df[df["year"] <= 2018]
print(f"  {len(train_df)} articles, {train_df['law_id'].nunique()} laws")

# ── Build pairs ──
print("Building pairs...")
law_groups = train_df.groupby("law_id")
law_ids = list(law_groups.groups.keys())
law_type_to_laws = defaultdict(list)
for lid in law_ids:
    lt = law_groups.get_group(lid).iloc[0]["law_type"]
    law_type_to_laws[lt].append(lid)

pairs = []
for law_id in tqdm(law_ids, desc="  Pairs"):
    group = law_groups.get_group(law_id)
    articles = group.to_dict("records")
    if len(articles) < 2: continue
    law_type = articles[0]["law_type"]
    same_type_laws = [l for l in law_type_to_laws.get(law_type, []) if l != law_id]

    for art in articles:
        q = f"{art['title']}: {art['text'][:400]}"
        pos = [a for a in articles if a["id"] != art["id"]]
        if pos:
            pairs.append((q, f"{random.choice(pos)['title']}: {random.choice(pos)['text'][:400]}", 1.0))
        if same_type_laws:
            neg_art = law_groups.get_group(random.choice(same_type_laws)).iloc[0]
            pairs.append((q, f"{neg_art['title']}: {neg_art['text'][:400]}", 0.0))
        diff = [l for l in law_ids if l != law_id and l not in same_type_laws]
        if diff:
            neg_art2 = law_groups.get_group(random.choice(diff)).iloc[0]
            pairs.append((q, f"{neg_art2['title']}: {neg_art2['text'][:400]}", 0.0))

n_pos = sum(1 for p in pairs if p[2] == 1.0)
if len(pairs) > 60000:
    pos_pairs = [p for p in pairs if p[2] == 1.0]
    neg_pairs = [p for p in pairs if p[2] == 0.0]
    pairs = random.sample(pos_pairs, min(30000, len(pos_pairs))) + random.sample(neg_pairs, min(30000, len(neg_pairs)))
print(f"  {len(pairs)} pairs ({sum(1 for p in pairs if p[2]==1.0)} pos)")

# ── Model ──
print("Loading PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
encoder = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
head = nn.Sequential(
    nn.Linear(encoder.config.hidden_size, 512), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
    nn.Linear(256, 1),
).to(device)
opt = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()), lr=1e-5, weight_decay=0.01)

# ── Train ──
B, epochs = 16, 10
steps_per_epoch = len(pairs) // B
print(f"\nTraining: {epochs} epochs, {steps_per_epoch} steps/epoch")
best_loss = float("inf")

for epoch in range(epochs):
    random.shuffle(pairs)
    epoch_loss = 0.0
    progress = tqdm(range(steps_per_epoch), desc=f"  Epoch {epoch+1}/{epochs}")
    for step in progress:
        start = (step * B) % max(len(pairs) - B, 1)
        batch = pairs[start:start + B]
        queries = [p[0] for p in batch]; docs = [p[1] for p in batch]
        labels = torch.tensor([p[2] for p in batch], dtype=torch.float, device=device)

        enc = tokenizer(queries, docs, padding=True, truncation=True, max_length=256, return_tensors="pt")
        input_ids = enc["input_ids"].to(device); attention_mask = enc["attention_mask"].to(device)
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        scores = head(out.last_hidden_state[:, 0, :]).squeeze(-1)

        pos_mask = labels == 1; neg_mask = labels == 0
        if pos_mask.any() and neg_mask.any():
            pos_scores = scores[pos_mask]; neg_scores = scores[neg_mask]
            loss = F.relu(0.3 - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0)).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(scores, labels)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0)
        opt.step()
        epoch_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = epoch_loss / steps_per_epoch
    print(f"    Epoch {epoch+1} avg loss: {avg_loss:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({"encoder": encoder.state_dict(), "head": head.state_dict()},
                   "data/checkpoints/telen/cross_encoder_best.pt")
        print(f"    Saved (loss={avg_loss:.4f})")

print("\nDone! Model saved to: data/checkpoints/telen/cross_encoder_best.pt")
