# Force float32 BEFORE any JAX import (prevents bf16 NaN in L2norm/tanh/FSQ)
import jax
jax.config.update("jax_default_matmul_precision", "float32")

"""JAX-native Strate I training on TPU.

Trains the FSQ tokenizer (Encoder + Decoder + FSQ) on multi-source OHLCV data
directly on TPU v6e-8. No torch_xla dependency.

Architecture (for patch_length=1):
  x (B, 1, 5) → Encoder → z_e (B, 64) → L2norm → FSQ quantize → z_q (B, 64)
  z_q → Decoder → x_hat (B, 1, 5)
  Loss = Huber(x, x_hat)  (Soft-DTW degenerates for L=1)

Usage:
    PYTHONPATH=. python scripts/train_strate_i_jax.py \
        --data_dirs data/raw_1m_parquet data/ohlcv_stocks_daily \
        --epochs 10 --batch_size 8192
"""

import argparse
import json
import logging
import os
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("strate_i_jax")


# ── Data Loading (CPU parallel) ──────────────────────────────────────────────

def _load_file(args):
    """Load and compute log-returns for one file."""
    path, = args
    p = Path(path)
    try:
        if p.suffix == ".pt":
            import torch
            data = torch.load(p, weights_only=True).numpy().astype(np.float32)
        elif p.suffix == ".parquet":
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=["open", "high", "low", "close", "volume"])
            data = t.to_pandas().values.astype(np.float32)
        elif p.suffix == ".npy":
            data = np.load(p).astype(np.float32)
        else:
            return None
        if data.ndim != 2 or data.shape[1] != 5 or data.shape[0] < 3:
            return None
        # Log-returns
        prices = data[:, :4]
        lr = np.log(prices[1:] + 1e-9) - np.log(prices[:-1] + 1e-9)
        vol = data[:, 4]
        mv = vol.mean()
        vt = np.log1p(vol[1:] / max(mv, 1e-8)) if mv > 1e-8 else np.zeros_like(vol[1:])
        log_ret = np.concatenate([lr, vt[:, None]], axis=-1).astype(np.float32)
        # Clip extreme log-returns (±5 = ±99.3% move, beyond = listing/delisting noise)
        log_ret = np.clip(log_ret, -5.0, 5.0)
        return log_ret
    except Exception:
        return None


def load_all_data(data_dirs, workers=64):
    """Load all OHLCV → log-returns, concatenate into (N, 5)."""
    tasks = []
    for d in data_dirs:
        dp = Path(d)
        if not dp.exists():
            continue
        for f in sorted(list(dp.glob("*.pt")) + list(dp.glob("*.npy")) + list(dp.glob("*.parquet"))):
            tasks.append((str(f),))

    log.info("Loading %d files with %d workers...", len(tasks), workers)
    t0 = time.time()
    arrays = []
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(_load_file, tasks, chunksize=32):
            if r is not None and len(r) > 0:
                arrays.append(r)
    all_data = np.concatenate(arrays, axis=0)  # (N_total, 5)
    log.info("Loaded %d candles (%.1f GB) in %.1fs", len(all_data), all_data.nbytes / 1e9, time.time() - t0)
    return all_data


# ── JAX Model ────────────────────────────────────────────────────────────────

def build_model_and_train():
    """Import JAX lazily to avoid TPU lock before data loading."""
    import jax
    import jax.numpy as jnp
    import optax
    from flax import linen as nn
    from flax.training import train_state
    from src.jax_v6.strate_i.tokenizer import Encoder, FSQQuantizer, CausalConv1d, CausalResidualBlock

    class Decoder(nn.Module):
        """Decoder: latent → reconstructed patch. For L=1, effectively an MLP."""
        hidden_channels: int = 128
        out_channels: int = 5
        patch_length: int = 1
        n_layers: int = 4
        kernel_size: int = 3

        @nn.compact
        def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
            """(B, latent_dim) → (B, L, C)."""
            x = nn.Dense(self.hidden_channels, name="input_proj")(z)  # (B, H)
            # Expand to (B, L, H) for conv layers
            x = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.patch_length, self.hidden_channels))
            for i in range(self.n_layers):
                residual = x
                x = nn.gelu(nn.Conv(self.hidden_channels, (self.kernel_size,),
                                    padding="SAME", name=f"block_{i}_conv1")(x))
                x = nn.gelu(nn.Conv(self.hidden_channels, (self.kernel_size,),
                                    padding="SAME", name=f"block_{i}_conv2")(x))
                x = x + residual
            x = nn.Dense(self.out_channels, name="output_proj")(x)  # (B, L, C)
            return x

    class StrateIVAE(nn.Module):
        """Full Strate I: Encoder + FSQ + Decoder."""
        hidden_channels: int = 128
        latent_dim: int = 64
        n_layers: int = 4
        patch_length: int = 1
        fsq_levels: tuple = (8, 8, 8, 2)

        @nn.compact
        def __call__(self, x: jnp.ndarray, deterministic: bool = True):
            """(B, L, 5) → (x_hat, z_e, z_q, indices)."""
            # Encode
            z_e = Encoder(
                hidden_channels=self.hidden_channels,
                latent_dim=self.latent_dim,
                n_layers=self.n_layers,
                name="encoder",
            )(x)  # (B, latent_dim)

            # FSQ quantize
            z_fsq = nn.Dense(len(self.fsq_levels), use_bias=False, name="fsq_proj_in")(z_e)
            levels = jnp.array(self.fsq_levels, dtype=jnp.float32)
            z_unit = (jnp.tanh(z_fsq) + 1.0) / 2.0
            z_scaled = z_unit * (levels - 1.0)
            z_q_nonneg = jnp.round(z_scaled)
            z_q_nonneg = jnp.clip(z_q_nonneg, 0.0, levels - 1.0)
            z_q_centered = z_q_nonneg - (levels - 1.0) / 2.0

            # STE: forward uses quantized, backward flows through z_e
            z_q_latent = nn.Dense(self.latent_dim, use_bias=False, name="fsq_proj_out")(z_q_centered)
            z_q_ste = z_e + jax.lax.stop_gradient(z_q_latent - z_e)

            # Decode
            x_hat = Decoder(
                hidden_channels=self.hidden_channels,
                out_channels=5,
                patch_length=self.patch_length,
                n_layers=self.n_layers,
                name="decoder",
            )(z_q_ste)  # (B, L, 5)

            return x_hat, z_e, z_q_ste

    return StrateIVAE, Decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dirs", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/strate_i_jax_multi")
    parser.add_argument("--io_workers", type=int, default=64)
    args = parser.parse_args()

    # ── Phase 1: Load data (CPU) ──
    all_data = load_all_data(args.data_dirs, args.io_workers)
    N = len(all_data)

    # Shuffle
    rng_np = np.random.default_rng(42)
    perm = rng_np.permutation(N)
    all_data = all_data[perm]

    # Train/val split
    val_n = int(N * 0.1)
    train_data = all_data[val_n:]
    val_data = all_data[:val_n]
    log.info("Train: %d, Val: %d", len(train_data), len(val_data))

    # ── Phase 2: JAX setup ──
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import train_state as ts

    log.info("JAX: %d devices (%s), float32 mode", jax.device_count(), jax.devices()[0].device_kind)

    StrateIVAE, _ = build_model_and_train()
    model = StrateIVAE(
        hidden_channels=128, latent_dim=64, n_layers=4,
        patch_length=1, fsq_levels=(8, 8, 8, 2),
    )

    # Init
    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, 1, 5))
    params = model.init(key, dummy)["params"]
    n_params = sum(x.size for x in jax.tree.leaves(params))
    log.info("Model: %d params (%.2fM)", n_params, n_params / 1e6)

    # Optimizer
    steps_per_epoch = len(train_data) // args.batch_size
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = steps_per_epoch  # 1 epoch warmup

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-7, peak_value=args.lr,
        warmup_steps=warmup_steps, decay_steps=total_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.zero_nans(),
        optax.adamw(schedule, weight_decay=1e-2),
    )
    state = ts.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # ── Huber loss ──
    def huber_loss(pred, target, delta=1.0):
        diff = pred - target
        abs_diff = jnp.abs(diff)
        return jnp.where(abs_diff <= delta, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta)).mean()

    @jax.jit
    def train_step(state, batch):
        batch = batch.astype(jnp.float32)
        def loss_fn(params):
            x_hat, z_e, z_q = model.apply({"params": params}, batch)
            return huber_loss(x_hat.astype(jnp.float32), batch)
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    @jax.jit
    def eval_step(params, batch):
        batch = batch.astype(jnp.float32)
        x_hat, _, _ = model.apply({"params": params}, batch)
        return huber_loss(x_hat.astype(jnp.float32), batch)

    # ── Training loop ──
    log.info("=== Training: %d epochs, %d steps/epoch, %d total ===",
             args.epochs, steps_per_epoch, total_steps)

    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        # Shuffle train data each epoch
        perm = rng_np.permutation(len(train_data))

        train_losses = []
        for step_in_epoch in range(steps_per_epoch):
            idx = perm[step_in_epoch * args.batch_size:(step_in_epoch + 1) * args.batch_size]
            batch = jnp.array(train_data[idx][:, None, :])  # (B, 1, 5)
            state, loss = train_step(state, batch)

            if step_in_epoch % 500 == 0 or step_in_epoch < 10:
                log.info("Epoch %d step %d/%d | loss %.6f", epoch, step_in_epoch, steps_per_epoch, float(loss))
            train_losses.append(float(loss))

        # Validation
        val_losses = []
        for i in range(0, len(val_data), args.batch_size):
            batch = jnp.array(val_data[i:i + args.batch_size][:, None, :])
            if batch.shape[0] < args.batch_size:
                continue
            vl = eval_step(state.params, batch)
            val_losses.append(float(vl))

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else 0
        elapsed = time.time() - t0

        log.info("Epoch %d | train %.6f | val %.6f | %.1fs (%.0f samples/sec)",
                 epoch, train_loss, val_loss, elapsed, len(train_data) / elapsed)

        # Save best (numpy — no orbax dependency, instant save)
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(args.ckpt_dir, "best_params.npz")
            flat_params = {'/'.join(str(p) for p in path): np.array(leaf)
                          for path, leaf in jax.tree.leaves_with_path(state.params)}
            np.savez(ckpt_path, **flat_params)
            log.info("Saved best checkpoint: %s (val=%.6f, %d arrays)", ckpt_path, val_loss, len(flat_params))

    log.info("=== Done. Best val loss: %.6f ===", best_val)


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()
