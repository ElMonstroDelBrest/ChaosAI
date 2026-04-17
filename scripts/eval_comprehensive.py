#!/usr/bin/env python3
"""Comprehensive eval: OOS loss + CFM diversity + latent variance + multiverse fan-out.

Single-run eval sharing one checkpoint load and XLA compilation across all 4 tests.
Fixes from eval_oos.py: __main__ guard, JAX_PLATFORMS for Grain, unbuffered print.

Usage (on TPU VM):
    SCALE_CONFIG=configs/scaling/v6e_54m_gnn_cfm.yaml \
    SCALE_TIER=54m_gnn_cfm \
    TPU_GEN=v6e \
    TRAIN_LOSS=2462 \
    PYTHONPATH=. python3 -u scripts/eval_comprehensive.py
"""

import os
import sys

# ── Anti-hang: __main__ guard (Grain worker subprocess protection) ──
if __name__ != "__main__":
    sys.exit(0)

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding, PartitionSpec as P

from src.jax_v6.config import load_config
from src.jax_v6.data.grain_loader import create_dataloader
from src.jax_v6.jepa import FinJEPA
from src.jax_v6.predictors.flow_predictor import FlowPredictor
from src.jax_v6.training.sharding import create_mesh, shard_batch, shard_train_state
from src.jax_v6.training.train_state import create_checkpoint_manager, create_train_state


# ── Helpers ──────────────────────────────────────────────────────────────────

def p(msg):
    """Unbuffered print (no logging — direct stdout for TPU monitoring)."""
    print(msg, flush=True)


def simple_kmeans(data, k=3, n_iter=50):
    """Simple K-means. data: (N, D). Returns labels (N,)."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(data), k, replace=False)
    centroids = data[idx].copy()
    for _ in range(n_iter):
        dists = np.linalg.norm(data[:, None] - centroids[None], axis=-1)
        labels = dists.argmin(axis=-1)
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = data[mask].mean(axis=0)
    return labels


def _call_ctx_encoder(self, token_indices, weekend_mask=None, block_mask=None,
                      exo_clock=None, gnn_embeddings=None, gnn_mask=None):
    """Helper for model.apply(method=...) — calls context_encoder on bound model."""
    return self.context_encoder(
        token_indices, weekend_mask=weekend_mask, block_mask=block_mask,
        exo_clock=exo_clock, gnn_embeddings=gnn_embeddings, gnn_mask=gnn_mask,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Config + Topology
# ═════════════════════════════════════════════════════════════════════════════
p("=" * 70)
p("  COMPREHENSIVE EVAL — 4 tests in 1 run")
p("=" * 70)
t_global = time.time()

config = load_config(os.environ["SCALE_CONFIG"])
scale = os.environ.get("SCALE_TIER", "54m")
tpu_gen = os.environ.get("TPU_GEN", "v6e")
train_loss = float(os.environ.get("TRAIN_LOSS", "2462"))
ckpt_dir = os.environ.get(
    "CKPT_DIR", os.path.join(os.getcwd(), f"checkpoints/jax_{tpu_gen}/{scale}"),
)

n_devices = len(jax.devices())
device_kind = jax.devices()[0].device_kind

# Auto-scale batch size for smaller pods
config_chips = 64
if n_devices != config_chips:
    per_chip = config.training.batch_size // config_chips
    object.__setattr__(config.training, "batch_size", per_chip * n_devices)

d_model = config.mamba2.d_model
has_cfm = config.predictor.cfm_weight > 0

p(f"  {n_devices}x {device_kind} | batch={config.training.batch_size} | "
  f"d={d_model} L={config.mamba2.n_layers} | CFM={'ON' if has_cfm else 'OFF'} | "
  f"GNN={config.mamba2.gnn_dim}")

# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Model + Checkpoint
# ═════════════════════════════════════════════════════════════════════════════
p("\n[Phase 2] Model + checkpoint...")
t2 = time.time()

mesh = create_mesh()
model = FinJEPA.from_config(config)

B = config.training.batch_size // n_devices
S = config.embedding.seq_len
max_tgt = int(S * config.masking.mask_ratio) + config.masking.block_size_max

dummy_batch = {
    "token_indices": jnp.zeros((B, S), dtype=jnp.int32),
    "weekend_mask": jnp.zeros((B, S), dtype=jnp.float32),
    "exo_clock": jnp.zeros((B, S, 2), dtype=jnp.float32),
    "block_mask": jnp.zeros((B, S), dtype=jnp.bool_),
    "target_positions": jnp.zeros((B, max_tgt), dtype=jnp.int32),
    "target_mask": jnp.ones((B, max_tgt), dtype=jnp.bool_),
}
if config.mamba2.gnn_dim > 0:
    dummy_batch["gnn_embeddings"] = jnp.zeros(
        (B, S, config.mamba2.gnn_dim), dtype=jnp.float32,
    )
    dummy_batch["gnn_mask"] = jnp.zeros((B, S), dtype=jnp.float32)

state = create_train_state(
    model, jax.random.PRNGKey(42), dummy_batch,
    lr=config.training.lr, weight_decay=config.training.weight_decay,
    warmup_steps=0, total_steps=1,
    tau_start=config.ema.tau_start, grad_clip=config.training.grad_clip, n_restarts=1,
)
n_params = sum(x.size for x in jax.tree.leaves(state.params))
p(f"  {n_params:,} params ({n_params / 1e6:.1f}M)")

state = shard_train_state(state, mesh)
ckpt_mgr = create_checkpoint_manager(ckpt_dir, max_to_keep=5)
latest = ckpt_mgr.latest_step()
assert latest is not None, f"No checkpoint in {ckpt_dir}"
state = ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(state))
p(f"  Restored step {latest} ({time.time() - t2:.1f}s)")

# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Val DataLoader
# ═════════════════════════════════════════════════════════════════════════════
p("\n[Phase 3] Val dataloader...")
# Prevent Grain worker subprocesses from initializing TPU
# (main process already has JAX initialized — only affects new processes)
os.environ["JAX_PLATFORMS"] = ""

val_loader = create_dataloader(
    config.data.arrayrecord_dir, split="val",
    batch_size=config.training.batch_size, seq_len=S,
    mask_ratio=config.masking.mask_ratio,
    block_size_min=config.masking.block_size_min,
    block_size_max=config.masking.block_size_max,
    val_ratio=config.data.val_split,
    worker_count=0,  # In-process — safe for eval, val set is small
    prefetch_buffer_size=64,
    gnn_dim=config.mamba2.gnn_dim,
)

# Collect val batches into memory (capped — full val set is 359K records = 701 batches)
MAX_VAL_BATCHES = int(os.environ.get("MAX_VAL_BATCHES", "50"))
val_batches = []
for batch in val_loader:
    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
    batch = shard_batch(batch, mesh)
    val_batches.append(batch)
    if len(val_batches) >= MAX_VAL_BATCHES:
        break

n_val = len(val_batches)
p(f"  {n_val} batches x {config.training.batch_size} = ~{n_val * config.training.batch_size} samples (cap={MAX_VAL_BATCHES})")

if n_val == 0:
    p("ERROR: No val batches! Check data path.")
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════════
# JIT-compiled functions
# ═════════════════════════════════════════════════════════════════════════════

@jax.jit
def eval_fn(params, target_params, batch, rng):
    """Full JEPA forward (with masking) for OOS loss."""
    out = model.apply(
        {"params": params}, batch,
        target_params=target_params, key=rng, deterministic=True,
    )
    return {k: out[k] for k in ("loss", "invariance", "variance", "covariance", "cfm_loss")}


@jax.jit
def encode_context(params, batch):
    """Context encoder WITHOUT masking (full encoding for analysis)."""
    return model.apply(
        {"params": params},
        batch["token_indices"],
        method=_call_ctx_encoder,
        weekend_mask=batch["weekend_mask"],
        block_mask=None,
        exo_clock=batch["exo_clock"],
        gnn_embeddings=batch.get("gnn_embeddings"),
        gnn_mask=batch.get("gnn_mask"),
    )


if has_cfm:
    flow_pred = FlowPredictor(
        d_model=d_model,
        hidden_dim=config.predictor.hidden_dim,
        n_layers=config.predictor.n_layers,
        seq_len=S,
        dropout=config.predictor.dropout,
        ot=config.predictor.cfm_ot,
    )

    @jax.jit
    def sample_cfm(flow_params, h_x, target_positions, key):
        """One CFM trajectory via Euler integration (2 steps)."""
        return flow_pred.sample(flow_params, h_x, target_positions, key=key, n_steps=2)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1: OOS Loss
# ═════════════════════════════════════════════════════════════════════════════
p("\n" + "=" * 70)
p("  TEST 1: Out-of-Sample Loss")
p("=" * 70)
t1 = time.time()

loss_acc = {k: [] for k in ("loss", "invariance", "variance", "covariance", "cfm_loss")}
rng = jax.random.PRNGKey(0)

for i, batch in enumerate(val_batches):
    rng, sk = jax.random.split(rng)
    m = eval_fn(state.params, state.target_params, batch, sk)
    for k in loss_acc:
        loss_acc[k].append(float(m[k]))
    if (i + 1) % 10 == 0 or i == 0:
        p(f"  {i+1:3d}/{n_val} | loss {loss_acc['loss'][-1]:.1f} | "
          f"inv {loss_acc['invariance'][-1]:.1f} | cfm {loss_acc['cfm_loss'][-1]:.3f}")

oos = float(np.mean(loss_acc["loss"]))
ratio = oos / train_loss if train_loss > 0 else float("inf")
verdict_1 = "GOOD" if ratio < 1.2 else ("OK" if ratio < 1.5 else "OVERFIT")
p(f"\n  OOS={oos:.1f} | Train={train_loss:.1f} | Ratio={ratio:.2f}x | {verdict_1}")
p(f"  Time: {time.time() - t1:.1f}s")

# ═════════════════════════════════════════════════════════════════════════════
# TESTS 2+3: CFM Diversity + Latent Variance (shared encoding pass)
# ═════════════════════════════════════════════════════════════════════════════
p("\n" + "=" * 70)
p("  TESTS 2+3: CFM Diversity + Latent Variance")
p("=" * 70)
t23 = time.time()

N_CFM = 50
MAX_CFM_BATCHES = 5  # CFM sampling is expensive (50 per batch) — limit to first 5
h_pooled_all = []
cfm_inter_stds = []
cfm_pairwise_coss = []
rng = jax.random.PRNGKey(1)
first_batch_data = None  # (h_x_np, tgt_np) saved for Test 4

for i, batch in enumerate(val_batches):
    h_x = encode_context(state.params, batch)  # (B, S, d_model)

    # Test 3: accumulate mean-pooled representations (all batches)
    h_pooled_all.append(np.array(jnp.mean(h_x, axis=1)))  # (B, d_model)

    # Test 2: CFM trajectory diversity (first MAX_CFM_BATCHES only)
    if has_cfm and i < MAX_CFM_BATCHES:
        samples = []
        for s in range(N_CFM):
            rng, sk = jax.random.split(rng)
            h_pred = sample_cfm(
                state.params["flow_predictor"], h_x, batch["target_positions"], sk,
            )
            samples.append(np.array(h_pred))  # (B, N_tgt, d_model)

        stk = np.stack(samples)  # (50, B, N_tgt, d_model)

        # inter_traj_std: std across 50 samples, then global mean
        cfm_inter_stds.append(float(stk.std(axis=0).mean()))

        # pairwise cosine: flatten each sample, normalize, cos sim upper triangle
        flat = stk.reshape(N_CFM, -1)
        nrm = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        cos_mat = (flat / nrm) @ (flat / nrm).T
        triu = np.triu_indices(N_CFM, k=1)
        cfm_pairwise_coss.append(float(cos_mat[triu].mean()))

        # Save first batch for Test 4 multiverse fan-out
        if i == 0:
            first_batch_data = (np.array(h_x), np.array(batch["target_positions"]))

    if (i + 1) % 10 == 0 or i == 0:
        s_str = f"cfm_std={cfm_inter_stds[-1]:.4f}" if cfm_inter_stds else ("encoding only" if i >= MAX_CFM_BATCHES else "no CFM")
        p(f"  {i+1:3d}/{n_val} | {s_str}")

# ── Test 3: Latent Variance results ──
h_all = np.concatenate(h_pooled_all)  # (total, d_model)
var_dim = h_all.var(axis=0)  # (d_model,)
active = int((var_dim > 1e-4).sum())
dead = int((var_dim < 1e-6).sum())

# Effective rank via eigenvalue entropy of covariance
cov_mat = np.cov(h_all.T)  # (d_model, d_model)
ev = np.maximum(np.linalg.eigvalsh(cov_mat), 0)
ev_n = ev / (ev.sum() + 1e-12)
entropy = -np.sum(ev_n[ev_n > 0] * np.log(ev_n[ev_n > 0] + 1e-12))
eff_rank = float(np.exp(entropy))
top10 = float(ev[-10:].sum() / (ev.sum() + 1e-12))

p(f"\n  --- Test 3: Latent Variance ---")
p(f"  Samples={h_all.shape[0]} | Active={active}/{d_model} "
  f"({100 * active / d_model:.0f}%) | Dead={dead}")
p(f"  Effective rank={eff_rank:.1f}/{d_model} | Top-10 eigenval={100 * top10:.1f}%")
verdict_3 = "HEALTHY" if active / d_model > 0.9 else (
    "PARTIAL" if active / d_model > 0.5 else "COLLAPSED")
p(f"  Verdict: {verdict_3}")

# ── Test 2: CFM Diversity results ──
if has_cfm and cfm_inter_stds:
    m_std = float(np.mean(cfm_inter_stds))
    m_cos = float(np.mean(cfm_pairwise_coss))
    div_score = 1.0 - m_cos
    p(f"\n  --- Test 2: CFM Diversity ---")
    p(f"  Inter-traj std={m_std:.4f} | Pairwise cos={m_cos:.4f} | Diversity={div_score:.4f}")
    verdict_2 = "COLLAPSED" if m_cos > 0.99 else ("LOW" if m_cos > 0.95 else "HEALTHY")
    p(f"  Verdict: {verdict_2}")
else:
    m_std, m_cos, div_score, verdict_2 = 0.0, 0.0, 0.0, "SKIPPED"
    p("\n  --- Test 2: SKIPPED (no CFM) ---")

p(f"  Time (T2+T3): {time.time() - t23:.1f}s")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4: Multiverse Fan-out
# ═════════════════════════════════════════════════════════════════════════════
p("\n" + "=" * 70)
p("  TEST 4: Multiverse Fan-out")
p("=" * 70)
t4 = time.time()

mv = {}
if has_cfm and first_batch_data is not None:
    h_x_np, tgt_np = first_batch_data  # (B_total, S, d), (B_total, N_tgt)
    B_full = h_x_np.shape[0]
    n_seq = min(8, B_full)
    sel = np.linspace(0, B_full - 1, n_seq, dtype=int)

    # Put selected sequences on device (no explicit sharding — runs on device 0)
    h_x_mv = jnp.array(h_x_np[sel])    # (n_seq, S, d_model)
    tgt_mv = jnp.array(tgt_np[sel])    # (n_seq, N_tgt)

    # Gather flow params to local device for unsharded computation
    flow_params_local = jax.tree.map(
        lambda x: jnp.array(np.array(x)), state.params["flow_predictor"],
    )

    mv_samples = []
    rng = jax.random.PRNGKey(2)
    for s in range(N_CFM):
        rng, mk = jax.random.split(rng)
        h_pred = flow_pred.sample(flow_params_local, h_x_mv, tgt_mv, key=mk, n_steps=2)
        mv_samples.append(np.array(h_pred))
        if s == 0:
            p(f"  First CFM sample compiled ({time.time() - t4:.1f}s)")

    mv_np = np.stack(mv_samples)  # (50, n_seq, N_tgt, d_model)
    N_tgt = mv_np.shape[2]

    # Spread over time: std per temporal position
    spread = mv_np.std(axis=0).mean(axis=(0, 2))  # (N_tgt,)
    sp_start, sp_end = float(spread[0]), float(spread[-1])
    sp_ratio = sp_end / (sp_start + 1e-8)

    # Endpoint analysis (per sequence)
    endpoints = mv_np[:, :, -1, :]  # (50, n_seq, d_model)
    bifs, n_clusts = [], []
    for si in range(n_seq):
        pts = endpoints[:, si, :]  # (50, d_model)

        # Bifurcation index: top eigenvalue of Gram matrix
        pts_c = pts - pts.mean(axis=0)
        gram = pts_c @ pts_c.T / len(pts)
        ev_g = np.linalg.eigvalsh(gram)
        bifs.append(float(ev_g[-1] / (ev_g.sum() + 1e-12)))

        # K-means (k=3) cluster analysis
        labels = simple_kmeans(pts, k=3)
        counts = [int((labels == c).sum()) for c in range(3)]
        n_clusts.append(sum(1 for c in counts if c >= 3))

    m_bif = float(np.mean(bifs))
    m_clust = float(np.mean(n_clusts))

    p(f"  Spread: t=0 {sp_start:.4f} -> t=-1 {sp_end:.4f} (ratio {sp_ratio:.2f}x)")
    p(f"  Bifurcation index: {m_bif:.3f} (1.0=degenerate, <0.5=structured)")
    p(f"  Active clusters: {m_clust:.1f}/3 (>1 = multimodal)")

    mv = {
        "spread_start": sp_start, "spread_end": sp_end, "spread_ratio": sp_ratio,
        "bifurcation_index": m_bif, "avg_clusters": m_clust,
        "per_seq_bifurcation": bifs, "per_seq_clusters": n_clusts,
    }
else:
    p("  SKIPPED (no CFM)")

p(f"  Time: {time.time() - t4:.1f}s")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY + JSON
# ═════════════════════════════════════════════════════════════════════════════
elapsed = time.time() - t_global

p("\n" + "=" * 70)
p("  SUMMARY")
p("=" * 70)
p(f"  Model       : {n_params / 1e6:.1f}M params | d={d_model} L={config.mamba2.n_layers}")
p(f"  Checkpoint  : step {latest}")
p(f"  Val batches : {n_val}")
p(f"  Total time  : {elapsed:.1f}s")
p("-" * 70)
p(f"  T1 OOS Loss      : {oos:.1f} (ratio {ratio:.2f}x) -- {verdict_1}")
p(f"  T2 CFM Diversity : std={m_std:.4f} cos={m_cos:.4f} -- {verdict_2}")
p(f"  T3 Latent Var    : active={active}/{d_model} rank={eff_rank:.1f} -- {verdict_3}")
if mv:
    p(f"  T4 Multiverse    : spread={mv['spread_ratio']:.2f}x "
      f"bif={mv['bifurcation_index']:.3f} clusters={mv['avg_clusters']:.1f}")
else:
    p(f"  T4 Multiverse    : SKIPPED")
p("=" * 70)

# JSON export
results = {
    "config": os.environ.get("SCALE_CONFIG", ""),
    "checkpoint_step": int(latest),
    "n_params": int(n_params),
    "n_val_batches": n_val,
    "elapsed_seconds": round(elapsed, 1),
    "test1_oos_loss": {
        "oos_loss": round(oos, 1),
        "train_loss": round(train_loss, 1),
        "ratio": round(ratio, 3),
        "verdict": verdict_1,
        "per_metric": {
            k: {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2)}
            for k, v in loss_acc.items()
        },
    },
    "test2_cfm_diversity": {
        "inter_traj_std": round(m_std, 4),
        "pairwise_cosine": round(m_cos, 4),
        "diversity_score": round(div_score, 4),
        "verdict": verdict_2,
    },
    "test3_latent_variance": {
        "active_dims": active,
        "dead_dims": dead,
        "effective_rank": round(eff_rank, 1),
        "top10_eigenval_ratio": round(top10, 3),
        "verdict": verdict_3,
    },
    "test4_multiverse": mv,
}

out_dir = os.path.join(os.getcwd(), "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"eval_{scale}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
p(f"\n  Results -> {out_path}")
