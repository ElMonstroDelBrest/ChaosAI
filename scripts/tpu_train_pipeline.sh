#!/bin/bash
# Run ArrayRecord conversion + training on TPU VM.
# Designed to run via nohup on the TPU VM itself.
set -euo pipefail

LOG="$HOME/Financial_IA/tpu_pipeline.log"
exec > "$LOG" 2>&1

cd "$HOME/Financial_IA"
source .venv_tpu/bin/activate
export JAX_ENABLE_X64=1

echo "[$(date -u)] === ArrayRecord Conversion ==="
python3 scripts/convert_pt_to_arrayrecord.py \
    --input data/tokens_v5/ \
    --output data/arrayrecord/ \
    --seq_len 128

echo "[$(date -u)] === Upload ArrayRecord to GCS ==="
gsutil -m -q rsync -r data/arrayrecord/ gs://fin-ia-bucket/data/arrayrecord/

echo "[$(date -u)] === Training ==="
python3 -u -c "
import sys, logging, time
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

# Create dummy batch for init (shape inference)
B = config.training.batch_size // n_devices  # local batch
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
log.info('Init with dummy batch: B=%d, S=%d, max_tgt=%d', B, S, max_tgt)

key = jax.random.PRNGKey(42)
state = create_train_state(
    model, key, dummy_batch,
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
    tau_start=config.ema.tau_start,
)
state = shard_params(state, mesh)
log.info('TrainState created and sharded across %d chips', n_devices)

train_loader = create_dataloader(
    config.data.arrayrecord_dir, split='train',
    batch_size=config.training.batch_size,
    seq_len=config.embedding.seq_len,
    worker_count=config.data.num_workers,
    prefetch_buffer_size=config.data.prefetch_buffer_size,
)

log.info('=== Training started ===')
step = 0
t0 = time.time()
for batch in train_loader:
    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
    batch = shard_batch(batch, mesh)
    state, metrics = train_step(state, batch, model)

    step += 1
    if step % 50 == 0:
        elapsed = time.time() - t0
        loss = float(metrics['loss'])
        cfm = float(metrics['cfm_loss'])
        log.info('step %d | loss %.4f | cfm %.4f | %.1f steps/s', step, loss, cfm, 50 / elapsed)
        t0 = time.time()

    if step % 500 == 0:
        log.info('Checkpointing at step %d...', step)
        import orbax.checkpoint as ocp
        mgr = ocp.CheckpointManager('checkpoints/jax_v6')
        mgr.save(step, args=ocp.args.StandardSave(state))
        import subprocess
        subprocess.run(['gsutil', '-m', '-q', 'rsync', '-r',
                        'checkpoints/jax_v6/', 'gs://fin-ia-bucket/checkpoints/jax_v6/'],
                       check=False)

log.info('=== Training complete at step %d ===', step)
"

echo "[$(date -u)] === PIPELINE COMPLETE ==="
