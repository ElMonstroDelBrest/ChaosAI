#!/bin/bash
# Run ArrayRecord conversion + training on TPU VM (v5p-8).
# Designed to run via nohup on the TPU VM itself.
set -euo pipefail

LOG="$HOME/Financial_IA/tpu_pipeline.log"
exec > "$LOG" 2>&1

cd "$HOME/Financial_IA"
source .venv_tpu/bin/activate

# ── JAX / XLA flags ──────────────────────────────────────────────────
# Compilation cache — persists across preemptions, avoids 5-10min recompile
export JAX_COMPILATION_CACHE_DIR="$HOME/.jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# x64 emulation: DISABLED — float64 is 2× slower on TPU (software emulated).
# Token indices fit in int32 (codebook=1024). If int64 is truly needed somewhere,
# enable selectively with jax.experimental.enable_x64() in Python instead.
# export JAX_ENABLE_X64=1

# LIBTPU_INIT_ARGS — production XLA flags for TPU v5p
# These tell the XLA compiler to be ultra-aggressive on kernel fusion,
# collective overlap, and memory management.
export LIBTPU_INIT_ARGS="
  --xla_tpu_enable_data_parallel_all_reduce_opt=true
  --xla_tpu_data_parallel_opt_different_sized_ops=true
  --xla_tpu_enable_async_collective_fusion=true
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true
  --xla_tpu_enable_async_collective_fusion_multiple_steps=true
  --xla_tpu_overlap_compute_collective_tc=true
  --xla_enable_async_all_gather=true
  --xla_tpu_enable_latency_hiding_scheduler=true:2
  --xla_tpu_perform_spmd_cse_prevention=false
  --xla_tpu_spmd_rng_bit_generator_unsafe=true
  --xla_tpu_megacore_fusion_allow_ags=false
  --xla_tpu_enable_aggressive_loop_fusion=true
  --xla_tpu_enable_experimental_fusion_cost_model=true
"
# Remove newlines — LIBTPU expects a single-line string
LIBTPU_INIT_ARGS=$(echo "$LIBTPU_INIT_ARGS" | tr '\n' ' ' | tr -s ' ')
export LIBTPU_INIT_ARGS

# Silence TensorFlow spam (Grain uses TF internally)
export TF_CPP_MIN_LOG_LEVEL=2

echo "[$(date -u)] === Sync ArrayRecord from GCS ==="
mkdir -p data/arrayrecord
gsutil -m rsync -r gs://fin-ia-bucket/data/arrayrecord/ data/arrayrecord/
echo "[$(date -u)] Synced $(ls data/arrayrecord/*.arrayrecord 2>/dev/null | wc -l) shards"

echo "[$(date -u)] === Training ==="
python3 -u -c "
import sys, logging, time, threading, queue
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger('train')

import jax
import jax.numpy as jnp
import numpy as np

from src.jax_v6.config import StrateIIConfig
from src.jax_v6.training.sharding import create_mesh, shard_batch, shard_params
from src.jax_v6.training.train_state import create_train_state
from src.jax_v6.training.train_step import train_step
from src.jax_v6.jepa import FinJEPA
from src.jax_v6.data.grain_loader import create_dataloader

config = StrateIIConfig()
mesh = create_mesh()
n_devices = len(jax.devices())
log.info('Training on %d %s chips', n_devices, jax.devices()[0].platform.upper())

model = FinJEPA.from_config(config)

# Init with full batch — v5p-8 has 95 GB HBM/chip, no need for tiny batch hack
B = config.training.batch_size
S = config.embedding.seq_len
max_tgt = int(S * 0.5) + 8
dummy_batch = {
    'token_indices': jnp.zeros((B, S), dtype=jnp.int64),
    'weekend_mask': jnp.zeros((B, S), dtype=jnp.float32),
    'exo_clock': jnp.zeros((B, S, 2), dtype=jnp.float32),
    'block_mask': jnp.zeros((B, S), dtype=jnp.bool_),
    'target_positions': jnp.zeros((B, max_tgt), dtype=jnp.int64),
    'target_mask': jnp.ones((B, max_tgt), dtype=jnp.bool_),
}
log.info('Init with batch: B=%d, S=%d, max_tgt=%d', B, S, max_tgt)

key = jax.random.PRNGKey(42)
state = create_train_state(
    model, key, dummy_batch,
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
    tau_start=config.ema.tau_start,
    grad_clip=config.training.grad_clip,
    n_restarts=config.training.n_restarts,
)
state = shard_params(state, mesh)
n_params = sum(x.size for x in jax.tree.leaves(state.params))
log.info('TrainState: %d params (%.1f MB bf16), sharded across %d chips',
         n_params, n_params * 2 / 1e6, n_devices)
log.info('Config: d_model=%d, n_heads=%d, expand=%d, n_layers=%d, batch=%d, seq=%d, grad_clip=%.1f, remat=%s',
         config.mamba2.d_model, config.mamba2.n_heads, config.mamba2.expand_factor,
         config.mamba2.n_layers, config.training.batch_size, config.embedding.seq_len,
         config.training.grad_clip, config.mamba2.use_remat)

train_loader = create_dataloader(
    config.data.arrayrecord_dir, split='train',
    batch_size=config.training.batch_size,
    seq_len=config.embedding.seq_len,
    worker_count=config.data.num_workers,
    prefetch_buffer_size=config.data.prefetch_buffer_size,
)

# Async data prefetch — double-buffering thread overlaps IO with TPU compute
prefetch_q = queue.Queue(maxsize=2)
def _prefetch(loader, mesh):
    try:
        for batch in loader:
            batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
            batch = shard_batch(batch, mesh)
            prefetch_q.put(batch)
    except Exception as e:
        log.error('Prefetch error: %s', e)
    prefetch_q.put(None)  # Sentinel

threading.Thread(target=_prefetch, args=(train_loader, mesh), daemon=True).start()
log.info('Async prefetch started (queue depth=2)')

log.info('=== Training started ===')
step = 0
t0 = time.time()
while True:
    batch = prefetch_q.get()
    if batch is None:
        break

    state, metrics = train_step(state, batch, model)

    step += 1
    if step % 50 == 0:
        elapsed = time.time() - t0
        loss = float(metrics['loss'])
        cfm = float(metrics['cfm_loss'])
        gnorm = float(metrics['grad_norm'])
        log.info('step %d | loss %.4f | cfm %.4f | grad_norm %.2f | %.2f steps/s',
                 step, loss, cfm, gnorm, 50 / elapsed)
        t0 = time.time()

    if step % 500 == 0:
        log.info('Checkpointing at step %d...', step)
        import os, pickle
        ckpt_dir = f'checkpoints/jax_v6/step_{step}'
        os.makedirs(ckpt_dir, exist_ok=True)
        params_np = jax.tree.map(lambda x: np.array(x), state.params)
        target_np = jax.tree.map(lambda x: np.array(x), state.target_params)
        ckpt_path = f'{ckpt_dir}/state.pkl'
        with open(ckpt_path, 'wb') as f:
            pickle.dump({'params': params_np, 'target_params': target_np, 'step': step, 'tau': float(state.tau)}, f)
        ckpt_mb = os.path.getsize(ckpt_path) / 1e6
        log.info('Saved %s (%.1f MB)', ckpt_path, ckpt_mb)
        import subprocess
        subprocess.run(['gsutil', '-m', '-q', 'rsync', '-r',
                        'checkpoints/jax_v6/', 'gs://fin-ia-bucket/checkpoints/jax_v6/'],
                       check=False)

log.info('=== Training complete at step %d ===', step)
"

echo "[$(date -u)] === PIPELINE COMPLETE ==="
