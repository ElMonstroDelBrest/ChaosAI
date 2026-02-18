"""Ablation Study: Transformer-FinJEPA vs Mamba-FinJEPA.

Compares two separately-trained FinJEPA checkpoints on:
  Proof A — Temporal Extrapolation: loss on held-out FUTURE sequences (last 20%).
  Proof B — Black Swan Reaction: loss on sequences containing extreme-volatility
             events (top 1% L2 distance between consecutive codebook embeddings).

Mamba has vol_clock conditioning; Transformer does not.
Expected: Mamba generalises better on Proof B (regime-change detection).

Usage:
    python scripts/ablation_transformer_vs_mamba.py \\
        --mamba_ckpt      checkpoints/strate_ii_mamba/best.ckpt \\
        --transformer_ckpt checkpoints/strate_ii_transformer/best.ckpt \\
        --data_path       data/tokens/test.pt \\
        --output_dir      results/ablation/ \\
        [--config         configs/strate_ii.yaml]   # optional, for model arch

    # With explicit model params (override config):
    python scripts/ablation_transformer_vs_mamba.py ... \\
        --d_model 512 --n_layers 12 --n_heads 8 --seq_len 128

Data format expected in data_path (.pt):
    {
        'token_indices': Tensor(N, S)  int64,
        'weekend_mask':  Tensor(N, S)  float32,
    }

Checkpoint format: either
    - PyTorch Lightning: {'state_dict': {'jepa.*': ...}, ...}
    - Direct:            {'model_state_dict': {...}} or raw state_dict
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strate_ii.jepa import FinJEPA


# ─── Helpers ────────────────────────────────────────────────────────────────

def _load_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    """Load raw state dict from a checkpoint, handling multiple formats."""
    raw = torch.load(ckpt_path, map_location=device)

    # PyTorch Lightning format: {'state_dict': {'jepa.encoder.*': ...}}
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        # Strip leading 'jepa.' (LightningModule stores self.jepa = FinJEPA(...))
        sd = {
            (k[len("jepa."):] if k.startswith("jepa.") else k): v
            for k, v in sd.items()
        }
        return sd

    # Direct save: {'model_state_dict': {...}}
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw["model_state_dict"]

    # Raw state dict
    if isinstance(raw, dict) and all(isinstance(v, torch.Tensor) for v in raw.values()):
        return raw

    raise ValueError(
        f"Unrecognised checkpoint format in {ckpt_path}. "
        "Expected Lightning, model_state_dict, or raw state dict."
    )


def load_model(
    ckpt_path: Path,
    encoder_type: str,
    model_kwargs: dict,
    device: torch.device,
) -> FinJEPA:
    """Load a FinJEPA model from a checkpoint."""
    print(f"Loading {encoder_type} model from {ckpt_path} …")
    model = FinJEPA(encoder_type=encoder_type, **model_kwargs).to(device)
    sd = _load_state_dict(ckpt_path, device)
    model.load_state_dict(sd, strict=False)   # strict=False: tolerate target_encoder EMA
    model.eval()
    print(f"  → {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def load_data(data_path: Path, device: torch.device) -> dict:
    """Load token data from a .pt file."""
    print(f"Loading data from {data_path} …")
    data = torch.load(data_path, map_location=device)
    n = data["token_indices"].shape[0]
    print(f"  → {n:,} sequences")
    return data


@torch.no_grad()
def evaluate_loss(model: FinJEPA, dataloader: DataLoader, device: torch.device) -> float:
    """Average FinJEPA total loss over a DataLoader."""
    total, count = 0.0, 0
    for token_indices, weekend_mask in dataloader:
        token_indices = token_indices.to(device)
        weekend_mask = weekend_mask.to(device)
        out = model(token_indices=token_indices, weekend_mask=weekend_mask)
        total += out["loss"].item()
        count += 1
    return total / count if count > 0 else float("nan")


# ─── Proof A: Temporal Extrapolation ────────────────────────────────────────

def run_proof_a(
    mamba: FinJEPA,
    transformer: FinJEPA,
    data: dict,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Loss on temporally held-out FUTURE sequences (last 20% of dataset)."""
    print("\n─── Proof A: Temporal Extrapolation ────────────────────────────")
    token_indices = data["token_indices"]
    weekend_mask  = data["weekend_mask"]

    n = token_indices.shape[0]
    split = int(n * 0.8)
    test_tok  = token_indices[split:]
    test_wknd = weekend_mask[split:]
    print(f"Test set: {len(test_tok):,} sequences (indices {split}–{n-1})")

    ds = TensorDataset(test_tok, test_wknd)
    dl = DataLoader(ds, batch_size=batch_size)

    mamba_loss = evaluate_loss(mamba, dl, device)
    transformer_loss = evaluate_loss(transformer, dl, device)

    print(f"Mamba       loss: {mamba_loss:.6f}")
    print(f"Transformer loss: {transformer_loss:.6f}")
    delta = transformer_loss - mamba_loss
    print(f"Δ (Transformer−Mamba): {delta:+.6f}  "
          f"({'Mamba better' if delta > 0 else 'Transformer better'})")
    return {"mamba_loss": mamba_loss, "transformer_loss": transformer_loss,
            "delta": delta, "n_sequences": len(test_tok)}


# ─── Proof B: Black Swan Events ──────────────────────────────────────────────

def run_proof_b(
    mamba: FinJEPA,
    transformer: FinJEPA,
    data: dict,
    device: torch.device,
    batch_size: int = 32,
    percentile: float = 0.99,
) -> dict:
    """Loss on sequences containing extreme-volatility (Black Swan) steps.

    Black Swan proxy: top (1 − percentile) of L2 distances between consecutive
    codebook embeddings — same signal as Mamba's vol_clock, but used here only
    for *selecting* test sequences, not as model input.
    """
    print("\n─── Proof B: Black Swan Reaction ───────────────────────────────")
    token_indices = data["token_indices"]
    weekend_mask  = data["weekend_mask"]

    # Use Mamba's codebook to compute the vol proxy (both models share codebook)
    codebook = mamba.context_encoder.codebook_embed

    print("Computing volatility proxy (codebook L2 distances) …")
    with torch.no_grad():
        x_embed = codebook(token_indices)             # (N, S, codebook_dim)
        diffs = x_embed[:, 1:, :] - x_embed[:, :-1, :]  # (N, S-1, D)
        dists = diffs.norm(dim=-1)                    # (N, S-1)

        threshold = torch.quantile(dists.reshape(-1).float(), percentile)
        print(f"Black Swan threshold (p={percentile:.2f}): {threshold.item():.4f}")

        has_black_swan = (dists > threshold).any(dim=1)  # (N,)
        bs_indices = has_black_swan.nonzero(as_tuple=True)[0]

    if len(bs_indices) == 0:
        print("WARNING: No Black Swan sequences found — lower percentile or use more data.")
        return {"mamba_loss": float("nan"), "transformer_loss": float("nan"),
                "n_sequences": 0, "threshold": threshold.item()}

    print(f"Black Swan sequences: {len(bs_indices):,} / {len(token_indices):,} "
          f"({100 * len(bs_indices) / len(token_indices):.1f}%)")

    bs_tok  = token_indices[bs_indices]
    bs_wknd = weekend_mask[bs_indices]

    ds = TensorDataset(bs_tok, bs_wknd)
    dl = DataLoader(ds, batch_size=batch_size)

    mamba_loss = evaluate_loss(mamba, dl, device)
    transformer_loss = evaluate_loss(transformer, dl, device)

    print(f"Mamba       loss (Black Swan): {mamba_loss:.6f}")
    print(f"Transformer loss (Black Swan): {transformer_loss:.6f}")
    delta = transformer_loss - mamba_loss
    print(f"Δ (Transformer−Mamba): {delta:+.6f}  "
          f"({'Mamba better' if delta > 0 else 'Transformer better'})")

    return {
        "mamba_loss": mamba_loss,
        "transformer_loss": transformer_loss,
        "delta": delta,
        "threshold": threshold.item(),
        "n_sequences": len(bs_indices),
        "pct_black_swan": float(100 * len(bs_indices) / len(token_indices)),
    }


# ─── CLI ────────────────────────────────────────────────────────────────────

def build_model_kwargs(args, config) -> dict:
    """Merge config + CLI overrides into FinJEPA kwargs."""
    if config is not None:
        cfg = config
        kwargs = dict(
            num_codes=cfg.embedding.num_codes,
            codebook_dim=cfg.embedding.codebook_dim,
            d_model=cfg.mamba2.d_model,
            d_state=cfg.mamba2.d_state,
            n_layers=cfg.mamba2.n_layers,
            n_heads=cfg.mamba2.n_heads,
            expand_factor=cfg.mamba2.expand_factor,
            conv_kernel=cfg.mamba2.conv_kernel,
            seq_len=cfg.embedding.seq_len,
            pred_hidden_dim=cfg.predictor.hidden_dim,
            pred_n_layers=cfg.predictor.n_layers,
            pred_dropout=cfg.predictor.dropout,
            pred_z_dim=cfg.predictor.z_dim,
            cfm_weight=0.0,   # no CFM loss during ablation eval
        )
    else:
        # Defaults matching the GOD-TIER strate_ii.yaml
        kwargs = dict(
            num_codes=1024, codebook_dim=64,
            d_model=512, d_state=64, n_layers=12, n_heads=8,
            expand_factor=2, conv_kernel=4, seq_len=128,
            pred_hidden_dim=1024, pred_n_layers=4, pred_dropout=0.05,
            pred_z_dim=128, cfm_weight=0.0,
        )

    # CLI overrides
    if args.d_model:    kwargs["d_model"] = args.d_model
    if args.n_layers:   kwargs["n_layers"] = args.n_layers
    if args.n_heads:    kwargs["n_heads"] = args.n_heads
    if args.seq_len:    kwargs["seq_len"] = args.seq_len

    return kwargs


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: Transformer-FinJEPA vs Mamba-FinJEPA (Proof A & B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mamba_ckpt",       type=Path, required=True)
    parser.add_argument("--transformer_ckpt", type=Path, required=True)
    parser.add_argument("--data_path",        type=Path, required=True)
    parser.add_argument("--output_dir",       type=Path, required=True)
    parser.add_argument("--config",           type=str,  default=None,
                        help="Path to strate_ii.yaml (for model arch; optional)")
    parser.add_argument("--d_model",  type=int, default=None, help="Override d_model")
    parser.add_argument("--n_layers", type=int, default=None, help="Override n_layers")
    parser.add_argument("--n_heads",  type=int, default=None, help="Override n_heads")
    parser.add_argument("--seq_len",  type=int, default=None, help="Override seq_len")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--percentile", type=float, default=0.99,
                        help="Black Swan threshold percentile (default: 0.99 = top 1%%)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cuda / cpu). Auto-detected if not set.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load YAML config if provided
    config = None
    if args.config:
        from src.strate_ii.config import load_config
        config = load_config(args.config)
        print(f"Config loaded from {args.config}")

    model_kwargs = build_model_kwargs(args, config)
    print(f"Model arch: d_model={model_kwargs['d_model']}, "
          f"n_layers={model_kwargs['n_layers']}, "
          f"n_heads={model_kwargs['n_heads']}, "
          f"seq_len={model_kwargs['seq_len']}")

    data = load_data(args.data_path, device)

    mamba_model       = load_model(args.mamba_ckpt,       "mamba",       model_kwargs, device)
    transformer_model = load_model(args.transformer_ckpt, "transformer", model_kwargs, device)

    # ── Run ablation proofs ──────────────────────────────────────────────────
    proof_a = run_proof_a(
        mamba_model, transformer_model, data, device,
        batch_size=args.batch_size,
    )
    proof_b = run_proof_b(
        mamba_model, transformer_model, data, device,
        batch_size=max(1, args.batch_size // 2),
        percentile=args.percentile,
    )

    # ── Save results ─────────────────────────────────────────────────────────
    results = {
        "model_arch": model_kwargs,
        "proof_a_temporal_extrapolation": proof_a,
        "proof_b_black_swan_reaction": proof_b,
    }
    out_path = args.output_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved → {out_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    w = 38
    sep = "─" * (w + 30)
    print(f"\n{'─'*70}")
    print(f" Ablation Summary: Mamba-FinJEPA vs Transformer-FinJEPA")
    print(sep)
    print(f"{'Test':<{w}} {'Mamba':>10}  {'Transformer':>12}  {'Δ (T−M)':>10}")
    print(sep)
    for label, d in [
        ("Proof A — Temporal Extrapolation", proof_a),
        (f"Proof B — Black Swan (p={args.percentile:.2f})", proof_b),
    ]:
        ml = d.get("mamba_loss", float("nan"))
        tl = d.get("transformer_loss", float("nan"))
        dl = d.get("delta", float("nan"))
        winner = " ← Mamba" if dl > 0 else " ← Transformer" if dl < 0 else ""
        print(f"{label:<{w}} {ml:>10.6f}  {tl:>12.6f}  {dl:>+10.6f}{winner}")
    print(sep)


if __name__ == "__main__":
    main()
