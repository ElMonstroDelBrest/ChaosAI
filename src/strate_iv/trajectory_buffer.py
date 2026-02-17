"""Trajectory buffer for Strate IV RL training.

Pre-computes episodes from MultiverseGenerator and stores them as .pt files.
The TrajectoryBuffer loads and samples entries for the LatentCryptoEnv.
"""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from . import EPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(ohlcv: Tensor, threshold: float = 1.0) -> str:
    """Classify an OHLCV window as bull, bear, or range.

    Uses normalized log-return of close prices over the window.
    The return is normalized by the window's volatility (std of log-returns)
    so the threshold is in units of sigma.

    Args:
        ohlcv: (T, 5) OHLCV tensor.
        threshold: Number of sigmas for bull/bear classification.

    Returns:
        "bull", "bear", or "range".
    """
    close = ohlcv[:, 3]  # close channel
    if len(close) < 2:
        return "range"

    # Log-returns
    log_ret = torch.log(close[1:] / (close[:-1] + EPS) + EPS)
    cumulative = log_ret.sum().item()
    sigma = log_ret.std().item() + EPS

    # Normalize by sqrt(T) to get a z-score-like measure
    z = cumulative / (sigma * math.sqrt(len(log_ret)))

    if z > threshold:
        return "bull"
    elif z < -threshold:
        return "bear"
    else:
        return "range"


def stratified_sample(
    n_total: int,
    n_episodes: int,
    ohlcv_lookup: Callable[[int], Tensor],
    threshold: float = 1.0,
    seed: int = 42,
) -> list[int]:
    """Sample indices with balanced regime representation.

    Classifies all sequences, then samples ~equal numbers from each regime.
    If a regime has fewer sequences, it oversamples to fill its quota.

    Args:
        n_total: Total number of sequences in dataset.
        n_episodes: Target number of episodes.
        ohlcv_lookup: Function idx -> (T, 5) OHLCV tensor.
        threshold: Z-score threshold for regime classification.
        seed: Random seed.

    Returns:
        List of indices with balanced regime representation.
    """
    rng = random.Random(seed)

    # Classify all sequences
    buckets: dict[str, list[int]] = {"bull": [], "bear": [], "range": []}
    for idx in range(n_total):
        ohlcv = ohlcv_lookup(idx)
        regime = classify_regime(ohlcv, threshold=threshold)
        buckets[regime].append(idx)

    logger.info("Regime distribution (raw %d sequences):", n_total)
    for regime, indices in buckets.items():
        logger.info("  %s: %d (%.1f%%)", regime, len(indices),
                     100 * len(indices) / n_total)

    # Stratified sampling: equal quota per regime
    per_regime = n_episodes // 3
    remainder = n_episodes - per_regime * 3

    sampled = []
    for i, (regime, indices) in enumerate(buckets.items()):
        quota = per_regime + (1 if i < remainder else 0)
        if len(indices) == 0:
            logger.warning("No %s sequences found, skipping", regime)
            continue
        if len(indices) >= quota:
            sampled.extend(rng.sample(indices, quota))
        else:
            # Oversample: repeat + sample remainder
            repeats = quota // len(indices)
            extra = quota % len(indices)
            pool = indices * repeats + rng.sample(indices, extra)
            sampled.extend(pool)

    rng.shuffle(sampled)

    logger.info("Stratified buffer: %d episodes (~%d per regime)",
                len(sampled), per_regime)

    return sampled


@dataclass
class TrajectoryEntry:
    """A single pre-computed episode for RL training.

    All tensors are stored on CPU and moved to device as needed.
    """

    # Context tokens and masks
    context_tokens: Tensor       # (S,) int64
    weekend_mask: Tensor | None  # (S,) float or None

    # Raw OHLCV context for reference
    context_ohlcv: Tensor        # (T, 5) raw OHLCV

    # N future OHLCV trajectories from MultiverseGenerator
    future_ohlcv: Tensor         # (N, N_tgt, patch_len, 5)

    # N future latent representations
    future_latents: Tensor       # (N, N_tgt, d_model)

    # RevIN stats from context
    revin_means: Tensor          # (1, 5)
    revin_stds: Tensor           # (1, 5)

    # Last close price (anchor for returns)
    last_close: float

    # Context encoder pooled representation
    h_x_pooled: Tensor           # (d_model,)


class TrajectoryBuffer:
    """Loads and samples pre-computed TrajectoryEntry objects from disk."""

    def __init__(self, buffer_dir: str) -> None:
        self.buffer_dir = Path(buffer_dir)
        self.entries: list[TrajectoryEntry] = []
        self._load()

    @classmethod
    def from_entries(cls, entries: list[TrajectoryEntry]) -> TrajectoryBuffer:
        """Create a buffer from an in-memory list of entries (no disk I/O)."""
        buf = cls.__new__(cls)
        buf.entries = list(entries)
        return buf

    def _load(self) -> None:
        if not self.buffer_dir.exists():
            return
        pt_files = sorted(self.buffer_dir.glob("*.pt"))
        for f in pt_files:
            data = torch.load(f, map_location="cpu", weights_only=False)
            entry = TrajectoryEntry(**data)
            self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def sample(self) -> TrajectoryEntry:
        """Sample a random trajectory entry."""
        return random.choice(self.entries)

    def add(self, entry: TrajectoryEntry) -> None:
        """Add an entry to the buffer (in-memory only)."""
        self.entries.append(entry)

    def split(
        self, val_ratio: float = 0.2, seed: int = 42
    ) -> tuple[TrajectoryBuffer, TrajectoryBuffer]:
        """Split buffer into train and eval buffers.

        Uses a deterministic shuffle so the split is reproducible.

        Args:
            val_ratio: Fraction of entries for eval (default 20%).
            seed: Random seed for the split.

        Returns:
            (train_buffer, eval_buffer)
        """
        rng = random.Random(seed)
        indices = list(range(len(self.entries)))
        rng.shuffle(indices)

        n_val = max(1, int(len(indices) * val_ratio))
        val_indices = set(indices[:n_val])

        train_entries = [e for i, e in enumerate(self.entries) if i not in val_indices]
        eval_entries = [e for i, e in enumerate(self.entries) if i in val_indices]

        return (
            TrajectoryBuffer.from_entries(train_entries),
            TrajectoryBuffer.from_entries(eval_entries),
        )


class TrajectoryPrecomputer:
    """Pre-computes trajectory entries using MultiverseGenerator + JEPA.

    Usage:
        precomputer = TrajectoryPrecomputer(generator, jepa, config)
        precomputer.run(token_dataset, ohlcv_dataset, output_dir)
    """

    def __init__(self, generator, jepa, n_futures: int = 16, n_tgt: int = 8) -> None:
        self.generator = generator
        self.jepa = jepa
        self.n_futures = n_futures
        self.n_tgt = n_tgt

    @torch.no_grad()
    def compute_entry(
        self,
        token_indices: Tensor,
        weekend_mask: Tensor | None,
        context_ohlcv: Tensor,
    ) -> TrajectoryEntry:
        """Compute a single trajectory entry.

        Args:
            token_indices: (S,) int64 context token indices.
            weekend_mask: (S,) float or None.
            context_ohlcv: (T, 5) raw OHLCV context.

        Returns:
            TrajectoryEntry with all fields populated.
        """
        device = token_indices.device

        # Add batch dimension
        tokens_b = token_indices.unsqueeze(0)          # (1, S)
        wm_b = weekend_mask.unsqueeze(0) if weekend_mask is not None else None
        ohlcv_b = context_ohlcv.unsqueeze(0)           # (1, T, 5)

        S = tokens_b.shape[1]
        target_positions = torch.arange(
            S - self.n_tgt, S, device=device
        ).unsqueeze(0)  # (1, N_tgt)

        # Generate futures via MultiverseGenerator
        result = self.generator.generate(
            tokens_b, wm_b, target_positions, ohlcv_b,
            n_samples=self.n_futures,
        )

        # Extract context encoder representation
        h_x = self.jepa.context_encoder(tokens_b, weekend_mask=wm_b)  # (1, S, d_model)
        h_x_pooled = h_x.mean(dim=1).squeeze(0)  # (d_model,)

        # RevIN stats
        means, stds = self.generator._estimate_context_stats(ohlcv_b)

        last_close = context_ohlcv[-1, 3].item()

        return TrajectoryEntry(
            context_tokens=token_indices.cpu(),
            weekend_mask=weekend_mask.cpu() if weekend_mask is not None else None,
            context_ohlcv=context_ohlcv.cpu(),
            future_ohlcv=result["ohlcv"][:, 0].cpu(),        # (N, N_tgt, patch_len, 5)
            future_latents=result["latents"][:, 0].cpu(),     # (N, N_tgt, d_model)
            revin_means=means[0].cpu(),                        # (1, 5)
            revin_stds=stds[0].cpu(),                          # (1, 5)
            last_close=last_close,
            h_x_pooled=h_x_pooled.cpu(),                       # (d_model,)
        )

    def run(
        self,
        dataset,
        ohlcv_lookup,
        output_dir: str,
        n_episodes: int = 255,
        device: str = "cpu",
        indices: list[int] | None = None,
    ) -> None:
        """Pre-compute and save trajectory entries.

        Args:
            dataset: TokenSequenceDataset yielding {token_indices, weekend_mask}.
            ohlcv_lookup: Callable(idx) -> (T, 5) raw OHLCV tensor.
            output_dir: Directory to save .pt files.
            n_episodes: Number of episodes to generate.
            device: Device to run on.
            indices: Explicit list of dataset indices to use.
                     If None, samples randomly.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if indices is None:
            indices = list(range(len(dataset)))
            if len(indices) > n_episodes:
                indices = random.sample(indices, n_episodes)
            elif len(indices) < n_episodes:
                indices = indices * (n_episodes // len(indices) + 1)
                indices = indices[:n_episodes]

        for i, idx in enumerate(indices):
            sample = dataset[idx]
            token_indices = sample["token_indices"].to(device)
            weekend_mask = sample.get("weekend_mask")
            if weekend_mask is not None:
                weekend_mask = weekend_mask.to(device)

            context_ohlcv = ohlcv_lookup(idx).to(device)

            entry = self.compute_entry(token_indices, weekend_mask, context_ohlcv)

            save_dict = {
                "context_tokens": entry.context_tokens,
                "weekend_mask": entry.weekend_mask,
                "context_ohlcv": entry.context_ohlcv,
                "future_ohlcv": entry.future_ohlcv,
                "future_latents": entry.future_latents,
                "revin_means": entry.revin_means,
                "revin_stds": entry.revin_stds,
                "last_close": entry.last_close,
                "h_x_pooled": entry.h_x_pooled,
            }
            torch.save(save_dict, out_path / f"episode_{i:05d}.pt")

            if (i + 1) % 50 == 0:
                logger.info("[%d/%d] episodes saved", i + 1, n_episodes)
