"""Block masking for Fin-JEPA (numpy, for Grain transform).

Pre-computes masks in the data pipeline to avoid JIT tracing issues.
Masked positions are target tokens; unmasked are context tokens.
"""

import numpy as np


def generate_block_mask(
    seq_len: int,
    mask_ratio: float = 0.5,
    block_size_min: int = 4,
    block_size_max: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single block mask for one sequence.

    Args:
        seq_len: Length of the sequence.
        mask_ratio: Target fraction of positions to mask.
        block_size_min: Minimum block size.
        block_size_max: Maximum block size.
        rng: Numpy random generator (for reproducibility).

    Returns:
        bool array (S,) where True = masked (target).
    """
    if rng is None:
        rng = np.random.default_rng()

    target_masked = int(seq_len * mask_ratio)
    mask = np.zeros(seq_len, dtype=bool)
    masked_count = 0

    available = list(range(seq_len))

    while masked_count < target_masked and available:
        block_size = rng.integers(block_size_min, block_size_max + 1)
        start_idx = rng.integers(0, len(available))
        start = available[start_idx]
        end = min(start + block_size, seq_len)

        mask[start:end] = True
        masked_count = mask.sum()

        available = [i for i in available if not mask[i]]

    return mask


def generate_batch_masks(
    batch_size: int,
    seq_len: int,
    mask_ratio: float = 0.5,
    block_size_min: int = 4,
    block_size_max: int = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate block masks for a batch.

    Args:
        batch_size: Number of sequences.
        seq_len: Length of each sequence.
        mask_ratio: Target fraction of positions to mask.
        block_size_min: Minimum block size.
        block_size_max: Maximum block size.
        rng: Numpy random generator.

    Returns:
        bool array (B, S) where True = masked (target).
    """
    if rng is None:
        rng = np.random.default_rng()

    masks = np.stack([
        generate_block_mask(seq_len, mask_ratio, block_size_min, block_size_max, rng)
        for _ in range(batch_size)
    ])
    return masks
