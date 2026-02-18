"""Pre-tokenize OHLCV data using a trained Strate I model.

Converts raw OHLCV .pt files into token sequences for Strate II.
Includes volatility-based "Apathy Mask": patches with volatility below
the 10th percentile (per asset) get mask=1.0, freezing Mamba-2's state
so it skips boring micro-oscillations.

Usage:
    PYTHONPATH=. python scripts/pretokenize.py \
        --strate_i_config configs/strate_i_binance.yaml \
        --checkpoint checkpoints/strate-i-epoch=09-val/loss/total=-0.0575.ckpt \
        --data_dir data/binance_1h_subset \
        --output_dir data/tokens/ \
        --seq_len 64
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from src.strate_i.config import load_config as load_strate_i_config
from src.strate_i.data.transforms import compute_log_returns, extract_patches
from src.strate_i.lightning_module import StrateILightningModule


def compute_patch_volatility(log_ret_patches: torch.Tensor) -> torch.Tensor:
    """Compute volatility for each patch as std of OHLC log-returns.

    Args:
        log_ret_patches: (N, L, 5) — patches of log-returns (OHLC + volume).

    Returns:
        (N,) — per-patch volatility scalar.
    """
    # Use OHLC channels only (indices 0-3), ignore volume
    ohlc = log_ret_patches[:, :, :4]  # (N, L, 4)
    # Volatility = std across time and channels per patch
    vol = ohlc.reshape(ohlc.shape[0], -1).std(dim=1)  # (N,)
    return vol


def compute_exo_clock(patches: torch.Tensor) -> torch.Tensor:
    """Compute exogenous clock signals (Realized Volatility + Normalized Volume).

    Per-patch RV and Volume, z-scored per asset for zero-mean unit-variance.
    These replace the endogenous L2 vol_clock in Mamba2Encoder, breaking
    the codebook feedback loop.

    Args:
        patches: (N, L, 5) — patches of log-returns (OHLC + volume).

    Returns:
        (N, 2) float32 — [RV, Volume] per patch, z-scored across the asset.
    """
    N = patches.shape[0]

    # Realized Volatility: std of OHLC log-returns per patch
    ohlc = patches[:, :, :4]  # (N, L, 4)
    rv = ohlc.reshape(N, -1).std(dim=1)  # (N,)

    # Normalized Volume: mean absolute volume per patch
    vol = patches[:, :, 4].abs().mean(dim=1)  # (N,)

    # Z-score per asset (entire sequence) — zero-mean, unit-variance
    def _zscore(x: torch.Tensor) -> torch.Tensor:
        mu = x.mean()
        sigma = x.std().clamp(min=1e-6)
        return (x - mu) / sigma

    return torch.stack([_zscore(rv), _zscore(vol)], dim=-1)  # (N, 2)


def compute_apathy_mask(
    volatilities: torch.Tensor, percentile: float = 10.0
) -> torch.Tensor:
    """Create apathy mask: 1.0 for low-volatility patches, 0.0 otherwise.

    Args:
        volatilities: (N,) per-patch volatility.
        percentile: Threshold percentile. Patches below this are "apathetic".

    Returns:
        (N,) float32 mask — 1.0 = frozen (apathetic), 0.0 = active.
    """
    threshold = np.percentile(volatilities.numpy(), percentile)
    mask = (volatilities < threshold).float()
    return mask


def pretokenize(
    strate_i_config_path: str,
    checkpoint_path: str,
    data_dir: str,
    output_dir: str,
    seq_len: int = 64,
    apathy_percentile: float = 10.0,
):
    """Tokenize raw OHLCV data into sequences of discrete tokens.

    For each OHLCV .pt file, creates multiple token sequence files with:
        - token_indices: (S,) int64 — discrete token for each patch
        - weekend_mask: (S,) float32 — apathy mask {0.0, 1.0}
          (field name kept as weekend_mask for pipeline compatibility)

    Generates N // seq_len sequences per pair (non-overlapping windows).
    """
    config = load_strate_i_config(strate_i_config_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load trained Strate I model
    model = StrateILightningModule.load_from_checkpoint(
        checkpoint_path, config=config
    )
    model.eval()
    tokenizer = model.tokenizer

    patch_length = config.patch.patch_length
    stride = config.patch.stride

    data_path = Path(data_dir)
    pt_files = sorted(data_path.glob("*.pt"))
    print(f"Found {len(pt_files)} OHLCV files in {data_dir}")
    print(f"Apathy threshold: {apathy_percentile}th percentile volatility\n")

    total_seqs = 0

    for pt_file in pt_files:
        pair_name = pt_file.stem
        ohlcv = torch.load(pt_file, weights_only=True)  # (T, 5)

        # Log-returns and patches
        log_ret = compute_log_returns(ohlcv)  # (T-1, 5)
        patches = extract_patches(log_ret, patch_length, stride)  # (N, L, 5)

        if patches.shape[0] == 0:
            print(f"  Skipping {pair_name}: no patches extracted")
            continue

        # Tokenize all patches
        with torch.no_grad():
            all_tokens = []
            batch_size = 256
            for i in range(0, patches.shape[0], batch_size):
                batch = patches[i : i + batch_size]
                tokens = tokenizer.tokenize(batch)
                all_tokens.append(tokens)
            token_indices = torch.cat(all_tokens)  # (N,)

        # Compute per-patch volatility and apathy mask
        volatilities = compute_patch_volatility(patches)  # (N,)
        apathy_mask = compute_apathy_mask(volatilities, apathy_percentile)  # (N,)

        # Compute exogenous clock signals (RV + Volume, z-scored)
        exo_clock = compute_exo_clock(patches)  # (N, 2)

        # Stats
        n_patches = token_indices.shape[0]
        n_apathetic = int(apathy_mask.sum().item())
        vol_threshold = np.percentile(volatilities.numpy(), apathy_percentile)
        print(
            f"  {pair_name}: {n_patches} patches, "
            f"vol p10={vol_threshold:.6f}, "
            f"apathetic={n_apathetic}/{n_patches} ({n_apathetic/n_patches:.1%})"
        )

        # Slice into non-overlapping sequences of seq_len
        n_seqs = n_patches // seq_len
        if n_seqs == 0:
            print(f"    → Too few patches for seq_len={seq_len}, skipping")
            continue

        for i in range(n_seqs):
            start = i * seq_len
            end = start + seq_len
            chunk_tokens = token_indices[start:end]
            chunk_mask = apathy_mask[start:end]
            chunk_exo_clock = exo_clock[start:end]  # (seq_len, 2)

            torch.save(
                {
                    "token_indices": chunk_tokens,
                    "weekend_mask": chunk_mask,
                    "exo_clock": chunk_exo_clock,
                },
                output_path / f"{pair_name}_seq{i:04d}.pt",
            )

        total_seqs += n_seqs
        print(f"    → Saved {n_seqs} sequences")

    print(f"\nDone. {total_seqs} total sequences saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenize OHLCV data for Strate II (with apathy mask)"
    )
    parser.add_argument(
        "--strate_i_config", type=str, default="configs/strate_i_binance.yaml"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/binance_1h_subset")
    parser.add_argument("--output_dir", type=str, default="data/tokens/")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument(
        "--apathy_percentile",
        type=float,
        default=10.0,
        help="Percentile threshold for apathy mask (default: 10th)",
    )
    args = parser.parse_args()

    pretokenize(
        strate_i_config_path=args.strate_i_config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        apathy_percentile=args.apathy_percentile,
    )
