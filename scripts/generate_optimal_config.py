#!/usr/bin/env python3
"""Generate Chinchilla-optimal, MXU-aligned Mamba-2 configs.

Given a target TPU Pod slice and token budget, this script:
  1. Computes the optimal param count via Chinchilla scaling law (N = T / 20)
  2. Finds the closest MXU-128-aligned architecture (d_model, n_heads, n_layers)
  3. Estimates HBM usage per chip with the auto-sharder topology
  4. Generates a production-ready YAML config

Usage:
  python scripts/generate_optimal_config.py --target_pod v5p-128 --total_tokens 20B
  python scripts/generate_optimal_config.py --target_pod v5p-768 --total_tokens 140B
  python scripts/generate_optimal_config.py --target_pod v5p-8 --total_tokens 300M -o configs/custom.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

MXU_TILE = 128  # v5p MXU computes 128×128 matmuls
HBM_PER_CHIP_GB = 95  # v5p HBM per chip
CHINCHILLA_RATIO = 20  # Optimal tokens/params ratio (Hoffmann et al. 2022)
EXPAND_FACTOR = 2  # Mamba-2 standard expand
D_STATE = 16  # SSM state dimension (small, doesn't need scaling)
CONV_KERNEL = 4

# Pod topologies: name → (num_chips, data_dim, fsdp_dim)
POD_TOPOLOGIES = {
    "v5p-8":   (8,   8,   1),
    "v5p-16":  (16,  8,   2),
    "v5p-32":  (32,  8,   4),
    "v5p-64":  (64,  16,  4),
    "v5p-128": (128, 32,  4),
    "v5p-256": (256, 64,  4),
    "v5p-512": (512, 128, 4),
    "v5p-768": (768, 192, 4),
}


# ─────────────────────────────────────────────────────────────────────
# Token parsing
# ─────────────────────────────────────────────────────────────────────

def parse_tokens(s: str) -> int:
    """Parse human-readable token count: '20B', '300M', '1.5T'."""
    s = s.strip().upper()
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(float(s))


# ─────────────────────────────────────────────────────────────────────
# Architecture search
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Mamba2Arch:
    """MXU-aligned Mamba-2 architecture."""
    d_model: int
    d_inner: int
    n_heads: int
    head_dim: int
    n_layers: int
    params_encoder: int  # Mamba-2 encoder params
    params_predictor: int
    params_total: int


def mamba2_params_per_layer(d_model: int, d_inner: int, n_heads: int, d_state: int) -> int:
    """Exact Mamba-2 parameter count per layer.

    Components:
      in_proj:  d_model → (2*d_inner + 2*n_heads*d_state + n_heads)
      out_proj: d_inner → d_model
      conv1d:   d_inner * conv_kernel
      A_log:    n_heads * d_state
      D:        n_heads
      LayerNorm: 2 * d_model (weight + bias)
    """
    in_proj_size = 2 * d_inner + 2 * n_heads * d_state + n_heads
    in_proj = d_model * in_proj_size + in_proj_size  # weight + bias
    out_proj = d_inner * d_model + d_model  # weight + bias
    conv1d = d_inner * CONV_KERNEL + d_inner  # weight + bias
    a_log = n_heads * d_state
    d_param = n_heads
    layer_norm = 2 * d_model  # scale + bias
    return in_proj + out_proj + conv1d + a_log + d_param + layer_norm


def predictor_params(d_inner: int, hidden_dim: int, z_dim: int, n_layers: int = 2) -> int:
    """Predictor parameter count (MLP)."""
    # Input: 2 * d_inner (context + target), output: z_dim
    total = 0
    in_dim = 2 * d_inner
    for i in range(n_layers):
        out_dim = hidden_dim if i < n_layers - 1 else z_dim
        total += in_dim * out_dim + out_dim  # weight + bias
        in_dim = out_dim
    return total


def find_optimal_arch(target_params: int) -> Mamba2Arch:
    """Find the MXU-128-aligned Mamba-2 architecture closest to target_params.

    Strategy:
      1. Enumerate d_model ∈ {128, 256, 384, ..., 8192} (multiples of 128)
      2. For each d_model, compute n_heads to get head_dim=128
      3. Find n_layers to hit target_params
      4. Pick the (d_model, n_layers) combo closest to target
    """
    best: Mamba2Arch | None = None
    best_diff = float("inf")

    for d_model in range(MXU_TILE, 8192 + 1, MXU_TILE):
        d_inner = d_model * EXPAND_FACTOR
        n_heads = d_inner // MXU_TILE
        if n_heads < 1 or d_inner % MXU_TILE != 0:
            continue

        per_layer = mamba2_params_per_layer(d_model, d_inner, n_heads, D_STATE)
        hidden_dim = d_inner  # predictor hidden = d_inner
        z_dim = max(32, d_model // 8)
        pred_params = predictor_params(d_inner, hidden_dim, z_dim)
        embed_params = 1024 * 64  # codebook

        # Solve for n_layers: total ≈ n_layers * per_layer + pred + embed
        remaining = target_params - pred_params - embed_params
        if remaining <= 0:
            continue
        n_layers = max(1, round(remaining / per_layer))

        # Mamba preference: wider > deeper. Cap depth, prefer width.
        # Practical limits: n_layers ∈ [4, 96]
        n_layers = max(4, min(n_layers, 96))

        total = n_layers * per_layer + pred_params + embed_params
        diff = abs(total - target_params)

        if diff < best_diff:
            best_diff = diff
            best = Mamba2Arch(
                d_model=d_model,
                d_inner=d_inner,
                n_heads=n_heads,
                head_dim=MXU_TILE,
                n_layers=n_layers,
                params_encoder=n_layers * per_layer,
                params_predictor=pred_params,
                params_total=total,
            )

    if best is None:
        raise ValueError(f"Could not find architecture for {target_params:,} params")
    return best


# ─────────────────────────────────────────────────────────────────────
# HBM estimation
# ─────────────────────────────────────────────────────────────────────

def estimate_hbm_per_chip(
    arch: Mamba2Arch,
    fsdp_dim: int,
    batch_per_chip: int,
    seq_len: int = 128,
    use_remat: bool = False,
) -> float:
    """Estimate HBM usage per chip in GB.

    Components:
      - Params (bf16): total_params * 2 bytes / fsdp_dim
      - Optimizer state (float32): params * 8 bytes / fsdp_dim (Adam: m + v)
      - Activations (bf16): batch * seq * d_model * n_layers * 2 bytes
        (with remat: only 1 layer at a time)
      - Gradients (bf16): same as params
    """
    bytes_bf16 = 2
    bytes_f32 = 4

    param_bytes = arch.params_total * bytes_bf16 / fsdp_dim
    opt_bytes = arch.params_total * 2 * bytes_f32 / fsdp_dim  # Adam m + v
    grad_bytes = arch.params_total * bytes_bf16 / fsdp_dim

    # Activations: per layer ≈ batch * seq * (d_model + d_inner) * 2 bytes
    act_per_layer = batch_per_chip * seq_len * (arch.d_model + arch.d_inner) * bytes_bf16
    n_active_layers = 1 if use_remat else arch.n_layers
    act_bytes = act_per_layer * n_active_layers

    total_bytes = param_bytes + opt_bytes + grad_bytes + act_bytes
    return total_bytes / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────
# Config generation
# ─────────────────────────────────────────────────────────────────────

def compute_training_hparams(arch: Mamba2Arch, num_chips: int, data_dim: int, fsdp_dim: int):
    """Compute training hyperparameters scaled to architecture size."""
    # Learning rate: μP scaling ∝ 1/sqrt(d_model), anchored at lr=1e-4 for d_model=1024
    lr = 1e-4 * math.sqrt(1024 / arch.d_model)

    # Weight decay scales up with model size
    if arch.params_total < 50e6:
        wd = 0.01
    elif arch.params_total < 500e6:
        wd = 0.02
    elif arch.params_total < 2e9:
        wd = 0.05
    else:
        wd = 0.1

    # EMA tau: slower for larger models
    tau = min(0.999, 0.996 + 0.001 * math.log2(arch.params_total / 15e6))

    # Batch per chip: fit in HBM, try 1024 first then halve
    use_remat = arch.params_total > 200e6
    batch_per_chip = 1024
    while batch_per_chip >= 32:
        hbm = estimate_hbm_per_chip(arch, fsdp_dim, batch_per_chip, use_remat=use_remat)
        if hbm < HBM_PER_CHIP_GB * 0.85:  # 85% safety margin
            break
        batch_per_chip //= 2

    global_batch = batch_per_chip * data_dim

    # Checkpoint interval: more frequent for larger pods (preemption risk)
    ckpt_interval = max(50, 250 - num_chips // 4)

    # Dropout: decrease with scale
    dropout = 0.1 if arch.params_total < 100e6 else (0.05 if arch.params_total < 1e9 else 0.0)

    return {
        "lr": lr,
        "weight_decay": wd,
        "ema_tau": tau,
        "batch_per_chip": batch_per_chip,
        "global_batch": global_batch,
        "use_remat": use_remat,
        "hbm_per_chip_gb": estimate_hbm_per_chip(arch, fsdp_dim, batch_per_chip, use_remat=use_remat),
        "checkpoint_interval": ckpt_interval,
        "dropout": dropout,
    }


def generate_yaml(
    arch: Mamba2Arch,
    pod_name: str,
    total_tokens: int,
    hparams: dict,
    num_chips: int,
    data_dim: int,
    fsdp_dim: int,
) -> str:
    """Generate a complete YAML config string."""
    config = {
        "mamba2": {
            "d_model": arch.d_model,
            "d_state": D_STATE,
            "n_layers": arch.n_layers,
            "n_heads": arch.n_heads,
            "expand_factor": EXPAND_FACTOR,
            "conv_kernel": CONV_KERNEL,
            "encoder_type": "mamba",
            "exo_clock": True,
            "chunk_size": MXU_TILE,
            "use_remat": hparams["use_remat"],
        },
        "predictor": {
            "hidden_dim": arch.d_inner,
            "n_layers": 2,
            "dropout": hparams["dropout"],
            "z_dim": max(32, arch.d_model // 8),
            "cfm_weight": 1.0,
            "cfm_n_steps": 2,
            "cfm_ot": True,
            "cfm_ot_batch_size": 256,
        },
        "masking": {
            "mask_ratio": 0.5,
            "block_size_min": 4,
            "block_size_max": 8 if arch.params_total < 500e6 else 16,
        },
        "vicreg": {
            "inv_weight": 25.0,
            "var_weight": 25.0,
            "cov_weight": 1.0,
            "var_gamma": 1.0,
        },
        "ema": {
            "tau_start": round(hparams["ema_tau"], 4),
            "tau_end": 1.0,
            "anneal_epochs": 100 if arch.params_total < 1e9 else 200,
        },
        "embedding": {
            "num_codes": 1024,
            "codebook_dim": 64,
            "seq_len": 128,
        },
        "training": {
            "lr": round(hparams["lr"], 6),
            "weight_decay": hparams["weight_decay"],
            "max_epochs": 100,
            "warmup_epochs": 10 if arch.params_total < 500e6 else 20,
            "batch_size": hparams["global_batch"],
            "precision": "bf16",
            "grad_clip": 1.0,
            "n_restarts": 4,
            "checkpoint_interval": hparams["checkpoint_interval"],
        },
        "data": {
            "token_dir": "data/tokens_v5/",
            "arrayrecord_dir": "data/arrayrecord/",
            "val_split": 0.2 if arch.params_total < 500e6 else 0.1,
            "num_workers": 0,
            "prefetch_buffer_size": max(128, num_chips * 2),
        },
    }

    # Build header comment
    params_str = format_params(arch.params_total)
    tokens_str = format_params(total_tokens)
    chinchilla_ratio = total_tokens / arch.params_total

    header = f"""\
###############################################################################
# AUTO-GENERATED — Chinchilla-optimal, MXU-128-aligned Mamba-2 config
#
# Target: {pod_name} ({num_chips} chips, {HBM_PER_CHIP_GB} GB HBM each)
# Auto-Sharder: Mesh 2D (data={data_dim}, fsdp={fsdp_dim})
# Token budget: {tokens_str} tokens
#
# Architecture:
#   d_model   = {arch.d_model:>5d}  ({arch.d_model // MXU_TILE} MXU tiles)
#   d_inner   = {arch.d_inner:>5d}  ({arch.d_inner // MXU_TILE} MXU tiles)
#   n_heads   = {arch.n_heads:>5d}  (head_dim = {arch.d_model}*{EXPAND_FACTOR}/{arch.n_heads} = {arch.head_dim})
#   n_layers  = {arch.n_layers:>5d}
#   params    = {params_str}  (encoder: {format_params(arch.params_encoder)})
#
# Chinchilla: {tokens_str} / {params_str} = {chinchilla_ratio:.1f} tok/param (optimal ≈ 20)
# HBM: ~{hparams['hbm_per_chip_gb']:.1f} GB/chip @ batch {hparams['batch_per_chip']}/chip
# Global batch: {hparams['global_batch']:,}
###############################################################################

"""
    return header + yaml.dump(config, default_flow_style=False, sort_keys=False)


def format_params(n: int | float) -> str:
    """Format parameter count: 15M, 1.5B, etc."""
    n = float(n)
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.0f}M"
    elif n >= 1e3:
        return f"{n/1e3:.0f}K"
    return str(int(n))


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate Chinchilla-optimal, MXU-aligned Mamba-2 config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s --target_pod v5p-8   --total_tokens 300M
  %(prog)s --target_pod v5p-32  --total_tokens 3B
  %(prog)s --target_pod v5p-128 --total_tokens 20B
  %(prog)s --target_pod v5p-768 --total_tokens 140B -o configs/custom.yaml
        """,
    )
    parser.add_argument(
        "--target_pod", required=True,
        choices=list(POD_TOPOLOGIES.keys()),
        help="Target TPU Pod slice",
    )
    parser.add_argument(
        "--total_tokens", required=True,
        help="Total token budget (e.g. '20B', '300M', '1.5T')",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output YAML path (default: stdout + configs/scaling/auto_<params>.yaml)",
    )
    args = parser.parse_args()

    # Parse inputs
    total_tokens = parse_tokens(args.total_tokens)
    num_chips, data_dim, fsdp_dim = POD_TOPOLOGIES[args.target_pod]

    # Chinchilla optimal param count
    target_params = total_tokens // CHINCHILLA_RATIO

    print(f"\n{'='*70}")
    print(f"  ChaosAI Config Generator — MXU-128 Aligned")
    print(f"{'='*70}")
    print(f"  Target pod:     {args.target_pod} ({num_chips} chips)")
    print(f"  Token budget:   {format_params(total_tokens)} tokens")
    print(f"  Chinchilla N:   {format_params(target_params)} params (T/20)")
    print(f"  Auto-Sharder:   Mesh 2D (data={data_dim}, fsdp={fsdp_dim})")
    print()

    # Find optimal architecture
    arch = find_optimal_arch(target_params)

    print(f"  Architecture found:")
    print(f"    d_model   = {arch.d_model:>5d}  ({arch.d_model // MXU_TILE} MXU tiles)")
    print(f"    d_inner   = {arch.d_inner:>5d}  ({arch.d_inner // MXU_TILE} MXU tiles)")
    print(f"    n_heads   = {arch.n_heads:>5d}  (head_dim = {arch.head_dim})")
    print(f"    n_layers  = {arch.n_layers:>5d}")
    print(f"    params    = {format_params(arch.params_total)} (target: {format_params(target_params)})")
    print(f"    delta     = {abs(arch.params_total - target_params) / target_params * 100:.1f}% from target")
    print()

    # Compute training hparams
    hparams = compute_training_hparams(arch, num_chips, data_dim, fsdp_dim)

    print(f"  Training config:")
    print(f"    lr            = {hparams['lr']:.2e}")
    print(f"    weight_decay  = {hparams['weight_decay']}")
    print(f"    batch/chip    = {hparams['batch_per_chip']}")
    print(f"    global_batch  = {hparams['global_batch']:,}")
    print(f"    HBM/chip      = {hparams['hbm_per_chip_gb']:.1f} GB / {HBM_PER_CHIP_GB} GB")
    print(f"    remat         = {hparams['use_remat']}")
    print(f"{'='*70}\n")

    # Generate YAML
    yaml_str = generate_yaml(
        arch, args.target_pod, total_tokens, hparams,
        num_chips, data_dim, fsdp_dim,
    )

    # Output
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(f"configs/scaling/auto_{format_params(arch.params_total).lower()}.yaml")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml_str)
    print(f"  Config written to: {out_path}")
    print()

    # Also print to stdout
    print(yaml_str)


if __name__ == "__main__":
    main()
