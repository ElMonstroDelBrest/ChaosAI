"""Configuration dataclasses for JAX Fin-JEPA.

Mirror of strate_ii/config.py with JAX-specific additions (chunk_size).
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from dacite import from_dict, Config as DaciteConfig


@dataclass(frozen=True)
class Mamba2Config:
    d_model: int = 128
    d_state: int = 16
    n_layers: int = 6
    n_heads: int = 2
    expand_factor: int = 2
    conv_kernel: int = 4
    encoder_type: str = "mamba"
    exo_clock: bool = True
    chunk_size: int = 128  # SSD chunk size — 128 fills one MXU tile exactly


@dataclass(frozen=True)
class PredictorConfig:
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.1
    z_dim: int = 32
    cfm_weight: float = 1.0
    cfm_n_steps: int = 2
    cfm_ot: bool = True
    cfm_ot_batch_size: int = 256


@dataclass(frozen=True)
class MaskingConfig:
    mask_ratio: float = 0.5
    block_size_min: int = 4
    block_size_max: int = 8


@dataclass(frozen=True)
class VICRegConfig:
    inv_weight: float = 25.0
    var_weight: float = 25.0
    cov_weight: float = 1.0
    var_gamma: float = 1.0


@dataclass(frozen=True)
class EMAConfig:
    tau_start: float = 0.996
    tau_end: float = 1.0
    anneal_epochs: int = 100


@dataclass(frozen=True)
class EmbeddingConfig:
    num_codes: int = 1024
    codebook_dim: int = 64
    seq_len: int = 128  # Padded seq_len for ArrayRecord (was 64 in PyTorch)


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_epochs: int = 100
    warmup_epochs: int = 10
    batch_size: int = 1024  # Global batch across 32 TPU chips
    precision: str = "bf16"


@dataclass(frozen=True)
class DataConfig:
    token_dir: str = "data/tokens_v5/"
    arrayrecord_dir: str = "data/arrayrecord/"
    val_split: float = 0.2
    num_workers: int = 4
    prefetch_buffer_size: int = 2


@dataclass(frozen=True)
class StrateIIConfig:
    mamba2: Mamba2Config = field(default_factory=Mamba2Config)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    vicreg: VICRegConfig = field(default_factory=VICRegConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> StrateIIConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(
        data_class=StrateIIConfig,
        data=config_dict,
        config=DaciteConfig(strict=True),
    )
