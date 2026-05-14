"""TELEN configuration."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.parent  # repo root
DATA_DIR = ROOT / "dataset"
CHECKPOINT_DIR = ROOT / "data" / "checkpoints" / "telen"


@dataclass
class GraphConfig:
    """Legal Concept Graph configuration."""
    hidden_dim: int = 768
    gnn_layers: int = 3
    gnn_dropout: float = 0.1
    # Edge types
    use_cross_ref_edges: bool = True
    use_agency_edges: bool = True
    use_temporal_edges: bool = True
    use_semantic_edges: bool = True
    semantic_knn: int = 10
    # Concept extraction
    max_concepts_per_article: int = 8
    min_tfidf_score: float = 0.05


@dataclass
class HyperNetworkConfig:
    """HyperNetwork that generates projection weights from legal state."""
    adaptation_rank: int = 16  # Low-rank adaptation (reduced to prevent overfitting)
    hn_hidden_dim: int = 512
    hn_layers: int = 3
    dropout: float = 0.2
    # What the HyperNetwork outputs
    output_shift: bool = True       # ΔW for projection
    output_bias: bool = True        # Δb for projection
    output_variance: bool = True    # log σ² for stochastic embedding
    min_variance: float = 0.01      # minimum variance


@dataclass
class MetaTrainingConfig:
    """Meta-learning training configuration."""
    meta_lr: float = 1e-4
    inner_lr: float = 5e-3
    meta_batch_size: int = 4        # episodes per meta-update
    n_query: int = 32               # query articles per episode
    n_negatives: int = 256          # negative articles per query
    meta_epochs: int = 50
    temperature: float = 0.05
    # Temporal splits for meta-training
    train_split_year: int = 2018
    val_split_year: int = 2020
    # State construction
    max_state_articles: int = 500   # max articles to include in state
    # Stochastic embedding
    kl_weight: float = 0.001        # weight for KL regularization
    n_mc_samples: int = 1           # Monte Carlo samples during training


@dataclass
class TELENConfig:
    """Full TELEN configuration."""
    backbone: str = "vinai/phobert-base-v2"
    hidden_dim: int = 768
    max_seq_length: int = 480
    graph: GraphConfig = field(default_factory=GraphConfig)
    hypernetwork: HyperNetworkConfig = field(default_factory=HyperNetworkConfig)
    meta: MetaTrainingConfig = field(default_factory=MetaTrainingConfig)
    output_dir: str = str(CHECKPOINT_DIR)
    seed: int = 42
