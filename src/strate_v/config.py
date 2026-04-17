"""Configuration dataclasses for Strate V (GNN On-Chain)."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from dacite import from_dict, Config as DaciteConfig


@dataclass(frozen=True)
class GraphConfig:
    node_features: int = 8        # in_value_log, out_value_log, in_degree, out_degree, net_flow, is_exchange, tx_count, unique_counterparties
    edge_features: int = 2        # value_log, tx_count
    max_nodes: int = 50_000       # Cap per snapshot (k-hop subgraph)
    k_hop: int = 2                # Subgraph sampling depth around exchange addresses
    chains: list = field(default_factory=lambda: ["btc", "eth"])


@dataclass(frozen=True)
class GNNModelConfig:
    hidden_dim: int = 128
    gnn_dim: int = 256            # Output embedding dim (MXU-aligned: 2x128)
    n_layers: int = 3
    gat_heads: int = 4
    dropout: float = 0.1
    pool_type: str = "mean_max_attn"  # mean || max || attention pooling


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    warmup_epochs: int = 5
    batch_size: int = 32
    precision: str = "16-mixed"
    # Loss weights
    link_weight: float = 1.0
    contrastive_weight: float = 0.5
    flow_weight: float = 0.1
    # Contrastive
    temperature: float = 0.07
    neg_samples: int = 16


@dataclass(frozen=True)
class BigQueryConfig:
    project: str = "bigquery-public-data"
    btc_dataset: str = "crypto_bitcoin"
    eth_dataset: str = "crypto_ethereum"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    max_bytes_billed: int = 1_000_000_000_000  # 1 TB free tier


@dataclass(frozen=True)
class StrateVConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: GNNModelConfig = field(default_factory=GNNModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bigquery: BigQueryConfig = field(default_factory=BigQueryConfig)


def load_config(path: str) -> StrateVConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(
        data_class=StrateVConfig,
        data=config_dict,
        config=DaciteConfig(strict=True),
    )
