"""
TELEN Inference — encode legal texts to 768-dim embeddings.

Usage:
    from inference import TELENInference
    model = TELENInference()
    embeddings = model.encode(["Điều 1: Thông tư này quy định về..."])
    similarity = model.similarity(text1, text2)
"""
import sys; sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from pyvi import ViTokenizer

from src.telern.config import TELENConfig
from src.telern.model import create_model


class TELENInference:
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = TELENConfig()
        self.model = create_model(self.config).to(self.device)

        if checkpoint_path is None:
            checkpoint_path = self.config.output_dir + "/telen_best.pt"

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.hypernetwork.load_state_dict(ckpt["hypernetwork"])
        self.model.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.model.base_projection.load_state_dict(ckpt["base_projection"])
        self.model.attn_query.data.copy_(ckpt["attn_query"])
        self.model.eval()

        print(f"TELEN loaded on {self.device}")
        print(f"  HyperNetwork: {sum(p.numel() for p in self.model.hypernetwork.parameters()):,} params")
        print(f"  Ready for inference.")

    def build_graph(self, df):
        """Build concept graph from a DataFrame with [id, title, text, law_id, law_type, year] columns."""
        self.model.build_graph(df)

    def encode(self, texts: list, batch_size: int = 64) -> torch.Tensor:
        """Encode a list of legal texts to 768-dim normalized embeddings."""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            with torch.no_grad():
                result = self.model(batch, use_stochastic=False)
                embeddings.append(result["embeddings"].cpu())
        return torch.cat(embeddings, dim=0)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb = self.encode([text1, text2])
        return F.cosine_similarity(emb[0:1], emb[1:2]).item()

    def retrieve(self, query: str, corpus: list, top_k: int = 10) -> list:
        """Retrieve top-k most similar documents from a corpus."""
        query_emb = self.encode([query])
        corpus_embs = self.encode(corpus)
        sim = F.cosine_similarity(query_emb, corpus_embs).numpy()
        top_indices = sim.argsort()[::-1][:top_k]
        return [(int(i), float(sim[i])) for i in top_indices]


# ── Demo ──
if __name__ == "__main__":
    model = TELENInference()

    # Example queries
    q1 = "Điều 1: Thông tư này quy định về quản lý thuế giá trị gia tăng đối với hàng hóa nhập khẩu"
    q2 = "Điều 2: Đối tượng áp dụng là các tổ chức, cá nhân kinh doanh hàng hóa nhập khẩu"
    q3 = "Điều 1: Nghị định này quy định về xử phạt vi phạm hành chính trong lĩnh vực giao thông"

    print(f"\nSimilarity test:")
    print(f"  q1 vs q2 (same law): {model.similarity(q1, q2):.4f}")
    print(f"  q1 vs q3 (diff law):  {model.similarity(q1, q3):.4f}")
