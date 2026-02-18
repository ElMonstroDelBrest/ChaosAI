"""Configuration dataclasses for Strate I (Spherical VQ-VAE / FSQ tokenizer)."""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from dacite import from_dict, Config as DaciteConfig


@dataclass(frozen=True)
class PatchConfig:
    patch_length: int = 16
    stride: int = 16
    n_channels: int = 5


@dataclass(frozen=True)
class RevINConfig:
    eps: float = 1e-5
    affine: bool = False


@dataclass(frozen=True)
class EncoderConfig:
    in_channels: int = 5
    hidden_channels: int = 128
    latent_dim: int = 64
    n_layers: int = 4
    dilation_base: int = 2
    kernel_size: int = 3


@dataclass(frozen=True)
class DecoderConfig:
    latent_dim: int = 64
    hidden_channels: int = 128
    out_channels: int = 5
    patch_length: int = 16
    n_layers: int = 4
    kernel_size: int = 3


@dataclass(frozen=True)
class CodebookConfig:
    num_codes: int = 1024
    latent_dim: int = 64
    ema_decay: float = 0.99
    eps: float = 1e-5
    dead_threshold: int = 2
    commitment_weight: float = 0.25
    # FSQ (Phase C — v6): if non-empty, use FSQCodebook instead of SphericalCodebook.
    # product(fsq_levels) must equal num_codes.
    # Example: [8, 8, 8, 2] → 1024 codes, 4-dim FSQ grid.
    # Empty list (default) keeps the original SphericalCodebook (VQ-VAE).
    fsq_levels: list = field(default_factory=list)


@dataclass(frozen=True)
class LossConfig:
    huber_delta: float = 1.0
    sdtw_alpha: float = 0.1
    sdtw_gamma: float = 0.1
    commitment_beta: float = 0.25


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_epochs: int = 200
    warmup_epochs: int = 5
    patience: int = 20
    batch_size: int = 64
    num_patches: int = 32
    precision: str = "16-mixed"


@dataclass(frozen=True)
class DataConfig:
    data_path: str = "data/"
    val_split: float = 0.2
    num_workers: int = 4


@dataclass(frozen=True)
class StrateIConfig:
    patch: PatchConfig = field(default_factory=PatchConfig)
    revin: RevINConfig = field(default_factory=RevINConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    codebook: CodebookConfig = field(default_factory=CodebookConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)


def load_config(path: str) -> StrateIConfig:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return from_dict(
        data_class=StrateIConfig,
        data=config_dict,
        config=DaciteConfig(strict=True),
    )
