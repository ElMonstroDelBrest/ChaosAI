"""Standalone training script for TPU — launched by launch_tpu_*.sh or directly."""
import sys, os, logging, time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("train")

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from src.jax_v6.config import load_config
from src.jax_v6.training.sharding import create_mesh, shard_batch, shard_train_state
from src.jax_v6.training.train_state import create_train_state, create_checkpoint_manager, compute_tau
from src.jax_v6.training.train_step import train_step, eval_step
from src.jax_v6.training.preemption import PreemptionWatcher
from src.jax_v6.training.metrics import compute_mfu, compute_tokens_per_second
from src.jax_v6.jepa import FinJEPA
from src.jax_v6.data.grain_loader import create_dataloader, count_train_records

# ── Config ──
config = load_config(os.environ["SCALE_CONFIG"])
scale = os.environ["SCALE_TIER"]
gcs_bucket = os.environ.get("GCS_BUCKET", "gs://fin-ia-eu")
tpu_type = os.environ.get("TPU_TYPE", "unknown")
resume = os.environ.get("RESUME", "false") == "true"
tpu_gen = os.environ.get("TPU_GEN", "v6e")  # v6e or v5e — for peak TFLOPS

# ── Detect topology ──
n_devices = len(jax.devices())
device_kind = jax.devices()[0].device_kind
assert n_devices >= 4, f"Expected >=4 TPU chips, got {n_devices}"

# Auto-scale batch_size (configs written for 64 chips)
config_chips = 64
if n_devices != config_chips:
    orig = config.training.batch_size
    per_chip = orig // config_chips
    adj = per_chip * n_devices
    log.info("Batch auto-scaled: %d -> %d (%d/chip, %d chips)", orig, adj, per_chip, n_devices)
    object.__setattr__(config.training, "batch_size", adj)

log.info("=== %s | %d chips (%s) | scale=%s ===", tpu_type, n_devices, device_kind, scale)
log.info(
    "Config: d_model=%d, n_layers=%d, n_heads=%d, batch=%d, lr=%.1e, remat=%s",
    config.mamba2.d_model, config.mamba2.n_layers, config.mamba2.n_heads,
    config.training.batch_size, config.training.lr, config.mamba2.use_remat,
)

# ── Compute training schedule from data ──
n_train = count_train_records(config.data.arrayrecord_dir, val_ratio=config.data.val_split)
steps_per_epoch = max(1, n_train // config.training.batch_size)
warmup_steps = config.training.warmup_epochs * steps_per_epoch
total_steps = config.training.max_epochs * steps_per_epoch
log.info("Schedule: %d train records, %d steps/epoch, warmup=%d, total=%d",
         n_train, steps_per_epoch, warmup_steps, total_steps)

# ── Mesh ──
mesh = create_mesh()

# ── Model ──
model = FinJEPA.from_config(config)

B = config.training.batch_size // n_devices
S = config.embedding.seq_len
max_tgt = int(S * 0.5) + 8
dummy_batch = {
    "token_indices": jnp.zeros((B, S), dtype=jnp.int32),
    "weekend_mask": jnp.zeros((B, S), dtype=jnp.float32),
    "exo_clock": jnp.zeros((B, S, 2), dtype=jnp.float32),
    "block_mask": jnp.zeros((B, S), dtype=jnp.bool_),
    "target_positions": jnp.zeros((B, max_tgt), dtype=jnp.int32),
    "target_mask": jnp.ones((B, max_tgt), dtype=jnp.bool_),
}
if config.mamba2.gnn_dim > 0:
    dummy_batch["gnn_embeddings"] = jnp.zeros((B, S, config.mamba2.gnn_dim), dtype=jnp.float32)
    dummy_batch["gnn_mask"] = jnp.zeros((B, S), dtype=jnp.float32)

key = jax.random.PRNGKey(42)
state = create_train_state(
    model, key, dummy_batch,
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    tau_start=config.ema.tau_start,
    grad_clip=config.training.grad_clip,
    n_restarts=config.training.n_restarts,
)

n_params = sum(x.size for x in jax.tree.leaves(state.params))
log.info("Model: %s (%.1fM params)", f"{n_params:,}", n_params / 1e6)
log.info("Per-chip: batch=%d, params=%.0f MB bf16", B, n_params * 2 / 1e6)

# ── Shard ──
state = shard_train_state(state, mesh)
log.info("TrainState sharded across %d chips", n_devices)

# ── Checkpoint ──
ckpt_dir = os.path.join(os.getcwd(), f"checkpoints/jax_{tpu_gen}/{scale}")
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_mgr = create_checkpoint_manager(ckpt_dir, max_to_keep=5)

start_step = 0
if resume:
    latest = ckpt_mgr.latest_step()
    if latest is not None:
        log.info("Restoring checkpoint from step %d...", latest)
        state = ckpt_mgr.restore(latest, args=ocp.args.StandardRestore(state))
        start_step = latest
        log.info("Resumed from step %d", start_step)

# ── Data ──
train_loader = create_dataloader(
    config.data.arrayrecord_dir, split="train",
    batch_size=config.training.batch_size,
    seq_len=config.embedding.seq_len,
    val_ratio=config.data.val_split,
    worker_count=config.data.num_workers,
    prefetch_buffer_size=config.data.prefetch_buffer_size,
    gnn_dim=config.mamba2.gnn_dim,
)

# ── Preemption watcher ──
watcher = PreemptionWatcher()
watcher.start()

# ── Training loop ──
ckpt_interval = config.training.checkpoint_interval
# Disable checkpointing if orbax is incompatible (JAX 0.6.x + orbax 0.11.x)
DISABLE_CKPT = os.environ.get("DISABLE_CKPT", "false") == "true"
mfu_interval = 50
val_interval = 500
PEAK_TFLOPS = {"v6e": 918.0, "v5e": 197.0, "v5p": 459.0}.get(tpu_gen, 918.0)

log.info("=== Training started (%s, scale=%s, step=%d, total=%d) ===",
         tpu_type, scale, start_step, total_steps)
step = start_step
prev_epoch = start_step // steps_per_epoch if steps_per_epoch > 0 else 0
t0 = time.time()

for batch in train_loader:
    if watcher.should_stop():
        log.warning("PREEMPTION at step %d — saving...", step)
        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        import subprocess
        subprocess.run(["gsutil", "-m", "rsync", "-r", ckpt_dir + "/",
            f"{gcs_bucket}/checkpoints/jax_{tpu_gen}/{scale}/"], check=False)
        break

    batch = {k: jnp.array(v) for k, v in batch.items() if not isinstance(v, (str, bytes))}
    batch = shard_batch(batch, mesh)
    state, metrics = train_step(state, batch, model)

    step += 1

    # ── Stop at max steps ──
    if step >= total_steps:
        log.info("Max steps reached (%d)", total_steps)
        break

    # ── Epoch tracking + EMA tau scheduling ──
    epoch = step // steps_per_epoch
    if epoch != prev_epoch:
        new_tau = compute_tau(epoch, config.ema.tau_start, config.ema.tau_end, config.ema.anneal_epochs)
        state = state.replace(tau=float(new_tau))
        log.info("Epoch %d | tau=%.6f", epoch, float(new_tau))
        prev_epoch = epoch

    step_time = time.time() - t0

    if step % mfu_interval == 0:
        avg_step_time = step_time / mfu_interval
        mfu = compute_mfu(n_params, config.training.batch_size, S, avg_step_time, n_devices, peak_tflops=PEAK_TFLOPS)
        tps = compute_tokens_per_second(config.training.batch_size, S, avg_step_time)
        log.info(
            "step %d | loss %.4f | grad %.2f | MFU %.1f%% | %.0f tok/s | %.3fs | ep %d",
            step, float(metrics["loss"]), float(metrics["grad_norm"]),
            mfu * 100, tps, avg_step_time, epoch,
        )
        t0 = time.time()

    if not DISABLE_CKPT and step % ckpt_interval == 0:
        log.info("Checkpoint step %d...", step)
        ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
        ckpt_mgr.wait_until_finished()
        import subprocess
        subprocess.Popen(["gsutil", "-m", "rsync", "-r", ckpt_dir + "/",
            f"{gcs_bucket}/checkpoints/jax_{tpu_gen}/{scale}/"])

watcher.stop()

if not DISABLE_CKPT:
    log.info("Final checkpoint step %d...", step)
    ckpt_mgr.save(step, args=ocp.args.StandardSave(state))
    ckpt_mgr.wait_until_finished()
    import subprocess
    subprocess.run(["gsutil", "-m", "rsync", "-r", ckpt_dir + "/",
        f"{gcs_bucket}/checkpoints/jax_{tpu_gen}/{scale}/"], check=False)
else:
    log.info("Checkpointing disabled (DISABLE_CKPT=true). Step %d not saved.", step)

cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
if cache_dir and os.path.exists(cache_dir):
    subprocess.run(["gsutil", "-m", "rsync", "-r", cache_dir + "/",
        f"{gcs_bucket}/xla_cache/{tpu_gen}_{scale}/"], check=False)

log.info("=== Done (%s, scale=%s, step=%d) ===", tpu_type, scale, step)
