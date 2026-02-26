#!/usr/bin/env python3
"""Train Strate V GNN on-chain model in JAX (self-supervised, TPU/GPU).

Usage:
    PYTHONPATH=$PWD python scripts/train_gnn_jax.py \\
      --graph_dirs data/onchain/graphs/eth/ \\
      --max_epochs 50 --lr 1e-3

    # Quick smoke test (1 epoch, 10 steps):
    PYTHONPATH=$PWD python scripts/train_gnn_jax.py \\
      --graph_dirs data/onchain/graphs/eth/ --smoke_test

Training: ~5 min on TPU v6e-8 for 8,528 pairs x 50 epochs (450K params).
Single-device training — no sharding needed for this small model.

Three self-supervised losses (same as PyTorch version):
  1. Link prediction (BCE) — intra-snapshot structure
  2. Temporal contrastive (InfoNCE) — inter-snapshot dynamics
  3. Exchange flow prediction (MSE) — next-hour net inflow/outflow

Loss: 1.0 * L_link + 0.5 * L_contrastive + 0.1 * L_flow
"""

import argparse
import sys
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.jax_v6.strate_v.gnn_model import OnChainGNN
from src.jax_v6.strate_v.data_loader import GraphPairDataset, negative_sampling_jax


# ---------------------------------------------------------------------------
# Train state creation
# ---------------------------------------------------------------------------

def create_train_state(
    key: jax.Array,
    model: OnChainGNN,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    total_steps: int,
) -> train_state.TrainState:
    """Create a Flax TrainState with AdamW + linear warmup + cosine decay.

    Args:
        key: PRNG key for parameter initialization.
        model: OnChainGNN Flax module.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.

    Returns:
        Initialized TrainState.
    """
    # Dummy graph for param init (shape doesn't matter, just needs valid structure)
    dummy = jraph.GraphsTuple(
        nodes=jnp.zeros((10, model.node_features)),
        edges=jnp.zeros((20, model.edge_features)),
        senders=jnp.zeros(20, dtype=jnp.int32),
        receivers=jnp.zeros(20, dtype=jnp.int32),
        n_node=jnp.array([10]),
        n_edge=jnp.array([20]),
        globals=jnp.array([[0.0]]),
    )
    params = model.init(key, dummy, deterministic=True)

    # LR schedule: linear warmup then cosine decay to 0
    warmup_steps = max(warmup_steps, 1)  # avoid division by zero
    decay_steps = max(total_steps - warmup_steps, 1)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, decay_steps),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def loss_fn(
    params,
    model: OnChainGNN,
    graph_t: jraph.GraphsTuple,
    graph_t1: jraph.GraphsTuple,
    rng_key: jax.Array,
    link_weight: float = 1.0,
    contrastive_weight: float = 0.5,
    flow_weight: float = 0.1,
    temperature: float = 0.07,
):
    """Compute combined self-supervised loss for a graph pair (t, t+1).

    Three losses matching the PyTorch version:
      1. Link prediction (BCE) on graph_t edges
      2. Temporal contrastive (InfoNCE) between graph_t and graph_t+1
      3. Exchange flow prediction (MSE) from graph_t embedding -> t+1 flow

    Args:
        params: Model parameters.
        model: OnChainGNN module (static, not traced).
        graph_t: jraph.GraphsTuple at time t.
        graph_t1: jraph.GraphsTuple at time t+1.
        rng_key: PRNG key for negative sampling.
        link_weight: Weight for link prediction loss.
        contrastive_weight: Weight for contrastive loss.
        flow_weight: Weight for flow prediction loss.
        temperature: InfoNCE temperature.

    Returns:
        (total_loss, metrics_dict) tuple.
    """
    # --- Encode both graphs ---
    node_emb_t = model.apply(params, graph_t, deterministic=True, method=model.encode)
    graph_emb_t = model.apply(params, graph_t, deterministic=True)
    graph_emb_t1 = model.apply(params, graph_t1, deterministic=True)

    # --- 1. Link prediction loss (on graph_t) ---
    n_edges = graph_t.senders.shape[0]
    # Cap positive edges to avoid OOM on large graphs
    n_pos = min(n_edges, 512)
    pos_edges = jnp.stack(
        [graph_t.senders[:n_pos], graph_t.receivers[:n_pos]], axis=1
    )

    n_nodes_t = jnp.sum(graph_t.n_node)
    neg_src, neg_dst = negative_sampling_jax(
        rng_key, graph_t.senders, graph_t.receivers, n_nodes_t, n_pos
    )
    neg_edges = jnp.stack([neg_src, neg_dst], axis=1)

    link_loss = model.apply(
        params, node_emb_t, pos_edges, neg_edges,
        method=model.link_prediction_loss,
    )

    # --- 2. Temporal contrastive loss ---
    # For single-pair training: each graph produces (gnn_dim,) embedding.
    # Expand to (1, gnn_dim) for contrastive_loss which expects (B, gnn_dim).
    emb_t = graph_emb_t[None] if graph_emb_t.ndim == 1 else graph_emb_t
    emb_t1 = graph_emb_t1[None] if graph_emb_t1.ndim == 1 else graph_emb_t1
    contrastive_loss = model.apply(
        params, emb_t, emb_t1,
        temperature=temperature,
        method=model.contrastive_loss,
    )

    # --- 3. Flow prediction loss ---
    # Target: exchange_net_flow from graph_t+1 (predict next-hour flow)
    flow_target = graph_t1.globals.squeeze()  # scalar
    flow_emb = graph_emb_t[None] if graph_emb_t.ndim == 1 else graph_emb_t
    flow_target_batched = flow_target[None] if flow_target.ndim == 0 else flow_target
    flow_loss = model.apply(
        params, flow_emb, flow_target_batched,
        method=model.flow_prediction_loss,
    )

    # --- Combined loss ---
    total = (
        link_weight * link_loss
        + contrastive_weight * contrastive_loss
        + flow_weight * flow_loss
    )

    return total, {
        "total": total,
        "link": link_loss,
        "contrastive": contrastive_loss,
        "flow": flow_loss,
    }


# ---------------------------------------------------------------------------
# Train step (JIT-compiled)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(1,))
def train_step(
    state: train_state.TrainState,
    model: OnChainGNN,
    graph_t: jraph.GraphsTuple,
    graph_t1: jraph.GraphsTuple,
    rng_key: jax.Array,
):
    """Single training step: forward + backward + optimizer update.

    Args:
        state: Current TrainState (params + optimizer state).
        model: OnChainGNN module (static arg, not traced).
        graph_t: Graph at time t.
        graph_t1: Graph at time t+1.
        rng_key: PRNG key for negative sampling.

    Returns:
        (updated_state, metrics_dict) tuple.
    """
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, model, graph_t, graph_t1, rng_key,
    )
    state = state.apply_gradients(grads=grads)
    return state, metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Strate V GNN on-chain model (JAX, self-supervised)"
    )
    parser.add_argument(
        "--graph_dirs", nargs="+", default=["data/onchain/graphs/eth/"],
        help="Directories containing .pt graph files (consecutive snapshots).",
    )
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--ckpt_dir", default="checkpoints/strate_v_jax/")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--smoke_test", action="store_true",
        help="Quick test: 1 epoch, 10 train steps, 5 val steps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # --- Device info ---
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)

    # --- Load dataset ---
    print(f"Loading graph pairs from {args.graph_dirs}...", flush=True)
    dataset = GraphPairDataset(args.graph_dirs)
    n_pairs = len(dataset)
    if n_pairs == 0:
        print("ERROR: No graph pairs found. Run scripts/build_graphs.py first.")
        sys.exit(1)
    print(f"Found {n_pairs} consecutive graph pairs.", flush=True)

    # --- Train/val split (80/20, shuffled) ---
    indices = np.arange(n_pairs)
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    n_val = max(1, int(n_pairs * 0.2))
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    print(f"Train: {len(train_indices)} pairs, Val: {len(val_indices)} pairs", flush=True)

    # --- Model ---
    model = OnChainGNN(
        node_features=8,
        edge_features=2,
        hidden_dim=128,
        gnn_dim=256,
        n_layers=3,
        gat_heads=4,
        dropout=0.1,
    )

    # --- Train state ---
    steps_per_epoch = len(train_indices)
    max_epochs = 1 if args.smoke_test else args.max_epochs
    total_steps = steps_per_epoch * max_epochs
    warmup_steps = steps_per_epoch * min(args.warmup_epochs, max_epochs)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_key = jax.random.split(rng)
    state = create_train_state(
        init_key, model, args.lr, args.weight_decay, warmup_steps, total_steps,
    )
    n_params = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"GNN params: {n_params:,}", flush=True)
    print(
        f"Schedule: {max_epochs} epochs x {steps_per_epoch} steps/epoch "
        f"= {total_steps} total steps, warmup={warmup_steps}",
        flush=True,
    )

    # --- Checkpoint directory ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    best_val_loss = float("inf")
    global_step = 0
    t_start_total = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        np.random.shuffle(train_indices)

        # ---- Train ----
        train_losses = []
        for i, idx in enumerate(train_indices):
            if args.smoke_test and i >= 10:
                break

            rng, step_key = jax.random.split(rng)
            graph_t, graph_t1 = dataset[int(idx)]

            state, metrics = train_step(state, model, graph_t, graph_t1, step_key)
            train_losses.append(float(metrics["total"]))
            global_step += 1

            if global_step % args.log_every == 0:
                avg = np.mean(train_losses[-args.log_every:])
                print(
                    f"  step {global_step:>6d} | loss {avg:.4f} | "
                    f"link {float(metrics['link']):.4f} | "
                    f"contr {float(metrics['contrastive']):.4f} | "
                    f"flow {float(metrics['flow']):.4f}",
                    flush=True,
                )

        train_avg = np.mean(train_losses) if train_losses else 0.0

        # ---- Validation ----
        val_losses = []
        val_link, val_contr, val_flow = [], [], []
        for idx in val_indices:
            if args.smoke_test and len(val_losses) >= 5:
                break
            rng, val_key = jax.random.split(rng)
            graph_t, graph_t1 = dataset[int(idx)]
            _, val_metrics = loss_fn(
                state.params, model, graph_t, graph_t1, val_key,
            )
            val_losses.append(float(val_metrics["total"]))
            val_link.append(float(val_metrics["link"]))
            val_contr.append(float(val_metrics["contrastive"]))
            val_flow.append(float(val_metrics["flow"]))

        val_avg = np.mean(val_losses) if val_losses else 0.0
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1:>3d}/{max_epochs} | "
            f"train_loss {train_avg:.4f} | val_loss {val_avg:.4f} | "
            f"val_link {np.mean(val_link):.4f} | "
            f"val_contr {np.mean(val_contr):.4f} | "
            f"val_flow {np.mean(val_flow):.4f} | "
            f"{epoch_time:.1f}s",
            flush=True,
        )

        # ---- Save best checkpoint ----
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            _save_checkpoint(state.params, ckpt_dir, val_avg)

    total_time = time.time() - t_start_total
    print(
        f"\nTraining complete in {total_time:.0f}s. "
        f"Best val_loss: {best_val_loss:.4f}",
        flush=True,
    )


def _save_checkpoint(params, ckpt_dir: Path, val_loss: float):
    """Save model params — orbax if available, pickle fallback.

    Args:
        params: JAX parameter pytree.
        ckpt_dir: Directory to save to.
        val_loss: Validation loss (for logging).
    """
    # Try orbax first (available on TPU VM)
    try:
        import orbax.checkpoint as ocp

        ckpt_path = str(ckpt_dir / "best")
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(ckpt_path, params, force=True)
        print(
            f"  Saved best checkpoint (val_loss={val_loss:.4f}) "
            f"to {ckpt_path} [orbax]",
            flush=True,
        )
        return
    except ImportError:
        pass  # orbax not installed, fall through to pickle
    except Exception as e:
        print(f"  WARNING: orbax save failed ({e}), falling back to pickle.", flush=True)

    # Fallback: pickle
    import pickle

    ckpt_path = ckpt_dir / "best_params.pkl"
    with open(ckpt_path, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    print(
        f"  Saved best params (val_loss={val_loss:.4f}) "
        f"to {ckpt_path} [pickle]",
        flush=True,
    )


if __name__ == "__main__":
    main()
