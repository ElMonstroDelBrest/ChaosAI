# JEPA Return Prediction Auxiliary Loss — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a causal next-return prediction head to the JEPA encoder, forcing embeddings to carry directional return signal, then re-pretokenize and retrain JEPA v6.3.

**Architecture:** Store raw price log-returns per timestep in ArrayRecord during pretokenization. The JEPA context encoder (causal Mamba-2) predicts `r_{t+1}` from `h_x[:, t, :]` via a single Dense(d_model→1) head. Loss = z-scored MSE, weighted by `ret_weight=0.01`. All changes backward-compatible via `ret_weight=0.0` default.

**Tech Stack:** JAX/Flax, Grain ArrayRecord, TensorFlow proto, optax, numpy

---

## Overview of changes

| File | Change |
|------|--------|
| `scripts/pretokenize_tpu.py` | Pass `all_candles[:, 0]` → `compute_per_asset_features` → `_write_shard` → store `"returns"` field |
| `src/jax_v6/data/grain_loader.py` | Parse `"returns"` field (optional, fallback zeros) |
| `src/jax_v6/jepa.py` | Add `ret_weight`, `return_head = Dense(1)`, `ret_loss` in forward |
| `src/jax_v6/config.py` | Add `ret_weight: float = 0.0` to `Mamba2Config` |
| `configs/scaling/v6e_38m.yaml` | Add `ret_weight: 0.01` |
| `src/jax_v6/training/train_step.py` | Add `ret_loss` to metrics dict |
| `scripts/run_training.py` | Log `ret_loss` in step output |

---

## Task 1: Store returns in ArrayRecord (pretokenize_tpu.py)

**Files:**
- Modify: `scripts/pretokenize_tpu.py`
- Test: `tests/test_pretokenize_returns.py` (create)

### Step 1: Write the failing test

Create `tests/test_pretokenize_returns.py`:

```python
"""Test that pretokenize_tpu stores returns in ArrayRecord."""
import numpy as np
import pytest
import tempfile, os


def test_write_shard_stores_returns():
    """_write_shard must include a 'returns' field in each record."""
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordWriter, ArrayRecordReader

    # Import the function under test
    import sys
    sys.path.insert(0, ".")
    from scripts.pretokenize_tpu import _write_shard

    seq_len = 4
    n_seqs = 2
    pair_name = "test__BTCUSDT"
    ti = np.zeros((n_seqs, seq_len), dtype=np.int64)
    ec = np.zeros((n_seqs, seq_len, 2), dtype=np.float32)
    am = np.ones((n_seqs, seq_len), dtype=np.float32)
    ret = np.array([[0.01, -0.02, 0.03, 0.0], [-0.01, 0.0, 0.02, -0.03]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        _write_shard((pair_name, ti, ec, am, ret, 0, n_seqs, tmp, seq_len, 2, None))
        shard_path = os.path.join(tmp, f"{pair_name}.arrayrecord")
        assert os.path.exists(shard_path)

        reader = ArrayRecordReader(shard_path)
        record = tf.train.Example()
        record.ParseFromString(reader.read(0))
        features = record.features.feature
        assert "returns" in features, "returns field missing from ArrayRecord"
        returns_stored = list(features["returns"].float_list.value)
        assert len(returns_stored) == seq_len
        np.testing.assert_allclose(returns_stored, ret[0], atol=1e-6)
```

### Step 2: Run the test to verify it fails

```bash
PYTHONPATH=. pytest tests/test_pretokenize_returns.py -v
```
Expected: FAIL — `_write_shard` doesn't accept `ret` yet.

### Step 3: Modify `scripts/pretokenize_tpu.py`

**3a. Update `_write_shard` signature and ArrayRecord writing (line ~251):**

Replace:
```python
def _write_shard(args):
    args_unpacked = args
    pair_name, ti, ec, am, source_id, n_seqs, output_dir, seq_len, exo_clock_dim = args_unpacked[:9]
    explicit_scale_id = args_unpacked[9] if len(args_unpacked) > 9 else None
```

With:
```python
def _write_shard(args):
    args_unpacked = args
    pair_name, ti, ec, am, ret, source_id, n_seqs, output_dir, seq_len, exo_clock_dim = args_unpacked[:10]
    explicit_scale_id = args_unpacked[10] if len(args_unpacked) > 10 else None
```

And inside the loop after `"scale_id"` feature, add:
```python
            "returns": tf.train.Feature(
                float_list=tf.train.FloatList(value=ret[i].tolist())),
```

**3b. Update `compute_per_asset_features` to accept and return `raw_returns` (line ~215):**

Replace signature:
```python
def compute_per_asset_features(token_indices, exo_clocks, assets, seq_len):
```
With:
```python
def compute_per_asset_features(token_indices, exo_clocks, assets, seq_len, raw_returns=None):
```

Inside the loop, after `am = apathy.reshape(n_seqs, seq_len)`, add:
```python
        # Per-timestep price log returns (col 0 = open-to-open)
        if raw_returns is not None:
            ret_raw = raw_returns[start:start + length]  # (length,)
            ret_raw = np.clip(ret_raw, -0.5, 0.5).astype(np.float32)
            ret = ret_raw.reshape(n_seqs, seq_len)
        else:
            ret = np.zeros((n_seqs, seq_len), dtype=np.float32)
```

Change the `results.append(...)` line from:
```python
        results.append((pair_name, ti, ec, am, source_id, n_seqs))
```
To:
```python
        results.append((pair_name, ti, ec, am, ret, source_id, n_seqs))
```

**3c. Update `write_arrayrecords` to unpack 7-tuple (line ~294):**

Replace:
```python
    for name, ti, ec, am, sid, ns in tokenized_data:
        explicit = source_id_to_scale.get(int(sid)) if source_id_to_scale else None
        tasks.append((name, ti, ec, am, sid, ns, output_dir, seq_len, exo_clock_dim, explicit))
```
With:
```python
    for name, ti, ec, am, ret, sid, ns in tokenized_data:
        explicit = source_id_to_scale.get(int(sid)) if source_id_to_scale else None
        tasks.append((name, ti, ec, am, ret, sid, ns, output_dir, seq_len, exo_clock_dim, explicit))
```

**3d. In `main()`, preserve raw_returns before deleting `all_candles` (line ~395 area):**

After `token_indices, exo_clocks = tokenize_batched_tpu(...)`, before `del all_candles`:
```python
    # Preserve price log returns (col 0) before freeing giant array
    raw_returns = all_candles[:, 0].copy()
    del all_candles  # free
```

Update the `compute_per_asset_features` call to pass `raw_returns`:
```python
    tokenized = compute_per_asset_features(token_indices, exo_clocks, assets, args.seq_len,
                                           raw_returns=raw_returns)
    del token_indices, exo_clocks, raw_returns
```

### Step 4: Run the test to verify it passes

```bash
PYTHONPATH=. pytest tests/test_pretokenize_returns.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add tests/test_pretokenize_returns.py scripts/pretokenize_tpu.py
git commit -m "feat: store per-timestep price returns in ArrayRecord for JEPA ret_loss"
```

---

## Task 2: Parse returns in grain_loader.py

**Files:**
- Modify: `src/jax_v6/data/grain_loader.py`
- Test: `tests/test_grain_returns.py` (create)

### Step 1: Write the failing test

Create `tests/test_grain_returns.py`:

```python
"""Test that grain_loader parses returns field correctly."""
import numpy as np
import tensorflow as tf


def _make_serialized_example(seq_len=4, with_returns=True):
    """Create a minimal serialized tf.train.Example with/without returns."""
    feature = {
        "token_indices": tf.train.Feature(int64_list=tf.train.Int64List(value=[0]*seq_len)),
        "weekend_mask": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0]*seq_len)),
        "exo_clock": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]*(seq_len*2))),
        "exo_clock_dim": tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        "pair_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test__BTC"])),
        "original_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len])),
        "source_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        "scale_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
    }
    if with_returns:
        feature["returns"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[0.01, -0.02, 0.03, 0.0]))
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def test_parse_example_with_returns():
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.data.grain_loader import _parse_example
    serialized = _make_serialized_example(seq_len=4, with_returns=True)
    result = _parse_example(serialized, seq_len=4)
    assert "returns" in result
    np.testing.assert_allclose(result["returns"], [0.01, -0.02, 0.03, 0.0], atol=1e-6)


def test_parse_example_without_returns_fallback():
    """Legacy shards without returns field must fall back to zeros."""
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.data.grain_loader import _parse_example
    serialized = _make_serialized_example(seq_len=4, with_returns=False)
    result = _parse_example(serialized, seq_len=4)
    assert "returns" in result
    np.testing.assert_allclose(result["returns"], [0.0]*4, atol=1e-6)
```

### Step 2: Run the test to verify it fails

```bash
PYTHONPATH=. pytest tests/test_grain_returns.py -v
```
Expected: FAIL — `_parse_example` doesn't return `"returns"`.

### Step 3: Modify `src/jax_v6/data/grain_loader.py`

**3a. In `_parse_example` function (~line 44), after the `scale_id` parsing block, add:**

```python
    # Returns: per-timestep log price returns (optional — legacy shards default to zeros)
    if "returns" in features:
        returns = np.array(features["returns"].float_list.value, dtype=np.float32)
    else:
        returns = np.zeros(seq_len, dtype=np.float32)
    result["returns"] = returns
```

**3b. In `ParseAndMask.map` (~line 148), add to the `result` dict:**

```python
        result["returns"] = parsed["returns"]
```

### Step 4: Run the test to verify it passes

```bash
PYTHONPATH=. pytest tests/test_grain_returns.py -v
```
Expected: PASS

### Step 5: Commit

```bash
git add tests/test_grain_returns.py src/jax_v6/data/grain_loader.py
git commit -m "feat: parse returns field in grain_loader (optional, fallback zeros)"
```

---

## Task 3: Add ret_weight to config

**Files:**
- Modify: `src/jax_v6/config.py` (line ~30)
- Modify: `configs/scaling/v6e_38m.yaml`

### Step 1: Write the failing test

Add to `tests/test_grain_returns.py` (or create `tests/test_config_ret_weight.py`):

```python
def test_config_ret_weight_default_zero():
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.config import Mamba2Config
    cfg = Mamba2Config()
    assert cfg.ret_weight == 0.0


def test_config_ret_weight_from_yaml():
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.config import load_config
    cfg = load_config("configs/scaling/v6e_38m.yaml")
    assert cfg.mamba2.ret_weight == 0.01
```

### Step 2: Run the test to verify it fails

```bash
PYTHONPATH=. pytest tests/test_config_ret_weight.py -v
```
Expected: FAIL — `Mamba2Config` has no `ret_weight`.

### Step 3: Modify `src/jax_v6/config.py`

In `Mamba2Config` dataclass, after the `cross_res_R` line (~line 32):
```python
    ret_weight: float = 0.0    # 0 = disabled; >0 enables return prediction auxiliary loss
```

### Step 4: Modify `configs/scaling/v6e_38m.yaml`

Add after `cross_res_weight: 1.0`:
```yaml
  ret_weight: 0.01
```

### Step 5: Run the test to verify it passes

```bash
PYTHONPATH=. pytest tests/test_config_ret_weight.py -v
```
Expected: PASS

### Step 6: Commit

```bash
git add src/jax_v6/config.py configs/scaling/v6e_38m.yaml tests/test_config_ret_weight.py
git commit -m "feat: add ret_weight to Mamba2Config and v6e_38m.yaml"
```

---

## Task 4: Add return prediction head to FinJEPA

**Files:**
- Modify: `src/jax_v6/jepa.py`
- Test: `tests/test_jepa_ret_loss.py` (create)

### Step 1: Write the failing test

Create `tests/test_jepa_ret_loss.py`:

```python
"""Test that FinJEPA computes ret_loss when ret_weight > 0."""
import numpy as np
import pytest


def test_jepa_ret_loss_computed_when_enabled():
    """ret_loss should be non-zero when ret_weight > 0 and returns in batch."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.jepa import FinJEPA
    except ImportError:
        pytest.skip("JAX not installed — AST-only validation")

    model = FinJEPA(
        num_codes=16, codebook_dim=8, d_model=32, d_state=4,
        n_layers=1, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=8, chunk_size=8, ret_weight=0.1,
    )
    B, S = 2, 8
    key = jax.random.PRNGKey(0)
    batch = {
        "token_indices": jnp.zeros((B, S), dtype=jnp.int64),
        "weekend_mask": jnp.ones((B, S), dtype=jnp.float32),
        "block_mask": jnp.zeros((B, S), dtype=bool),
        "target_positions": jnp.zeros((B, 4), dtype=jnp.int64),
        "target_mask": jnp.zeros((B, 4), dtype=bool),
        "returns": jax.random.normal(key, (B, S)) * 0.01,
    }
    params = model.init(key, batch, target_params=None, key=key)["params"]
    outputs = model.apply({"params": params}, batch, target_params=None, key=key)
    assert "ret_loss" in outputs
    assert float(outputs["ret_loss"]) > 0.0, "ret_loss should be non-zero"


def test_jepa_ret_loss_zero_when_disabled():
    """ret_loss should be 0.0 when ret_weight=0 (default)."""
    try:
        import jax
        import jax.numpy as jnp
        from src.jax_v6.jepa import FinJEPA
    except ImportError:
        pytest.skip("JAX not installed")

    model = FinJEPA(
        num_codes=16, codebook_dim=8, d_model=32, d_state=4,
        n_layers=1, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=8, chunk_size=8, ret_weight=0.0,
    )
    B, S = 2, 8
    key = jax.random.PRNGKey(0)
    batch = {
        "token_indices": jnp.zeros((B, S), dtype=jnp.int64),
        "weekend_mask": jnp.ones((B, S), dtype=jnp.float32),
        "block_mask": jnp.zeros((B, S), dtype=bool),
        "target_positions": jnp.zeros((B, 4), dtype=jnp.int64),
        "target_mask": jnp.zeros((B, 4), dtype=bool),
        "returns": jax.random.normal(key, (B, S)) * 0.01,
    }
    params = model.init(key, batch, target_params=None, key=key)["params"]
    outputs = model.apply({"params": params}, batch, target_params=None, key=key)
    assert float(outputs["ret_loss"]) == 0.0
```

### Step 2: Run the test to verify it fails (AST parse only on local machine)

```bash
python -c "import ast; ast.parse(open('src/jax_v6/jepa.py').read()); print('AST OK')"
PYTHONPATH=. pytest tests/test_jepa_ret_loss.py -v 2>&1 | head -20
```
Expected: skip (no JAX locally) — verify AST parse OK.

### Step 3: Modify `src/jax_v6/jepa.py`

**3a. Add `ret_weight` field to `FinJEPA` (~line 70, after `cross_res_R`):**

```python
    # Return prediction auxiliary loss (Option A)
    ret_weight: float = 0.0  # 0 = disabled; >0 enables causal next-return MSE loss
```

**3b. In `from_config` (~line 105, after `cross_res_R` line), add:**

```python
            ret_weight=getattr(config.mamba2, 'ret_weight', 0.0),
```

**3c. In `setup` (~line 108, after `self.output_proj` line), add:**

```python
        # Return prediction head (always created — ~d_model params, negligible)
        self.return_head = nn.Dense(1, name="return_head")
```

**3d. In `__call__`, after `cfm_loss` block and before `total_loss = total_loss + self.cross_res_weight * cross_res_loss`, add:**

```python
        # Return prediction: h_x[:, t, :] -> r_{t+1} (causal — Mamba-2 is unidirectional)
        ret_loss = jnp.float32(0.0)
        if self.ret_weight > 0.0 and "returns" in batch:
            r_pred = self.return_head(h_x[:, :-1, :]).squeeze(-1)  # (B, S-1)
            r_next = batch["returns"][:, 1:].astype(jnp.float32)   # (B, S-1) target
            # Z-score per batch: normalizes scale across asset classes (crypto vs stocks vs forex)
            r_mean = jnp.mean(r_next)
            r_std = jnp.std(r_next) + 1e-6
            r_norm = (r_next - r_mean) / r_std
            r_pred_norm = (r_pred - r_mean) / r_std
            ret_loss = jnp.mean((r_pred_norm - r_norm) ** 2)
            total_loss = total_loss + self.ret_weight * ret_loss
```

**3e. Add `"ret_loss"` to return dict (~line 316):**

```python
            "ret_loss": ret_loss,
```

### Step 4: Verify AST parse

```bash
python -c "import ast; ast.parse(open('src/jax_v6/jepa.py').read()); print('AST OK')"
```
Expected: `AST OK`

### Step 5: Commit

```bash
git add src/jax_v6/jepa.py tests/test_jepa_ret_loss.py
git commit -m "feat: add causal return prediction head to FinJEPA (ret_weight)"
```

---

## Task 5: Propagate ret_loss through train_step and run_training

**Files:**
- Modify: `src/jax_v6/training/train_step.py`
- Modify: `scripts/run_training.py`

### Step 1: Modify `src/jax_v6/training/train_step.py`

In `train_step`, add `"ret_loss"` to the metrics dict (~line 65):
```python
        "ret_loss": outputs.get("ret_loss", jnp.float32(0.0)),
```

In `eval_step`, add:
```python
        "ret_loss": outputs.get("ret_loss", jnp.float32(0.0)),
```

### Step 2: Modify `scripts/run_training.py`

In the MFU log message (~line 254), add `ret_loss`:
```python
        log.info(
            "step %d | loss %.4f | ret %.4f | grad %.2f | MFU %.1f%% | %.0f tok/s | %.3fs | ep %d",
            step, float(metrics["loss"]), float(metrics.get("ret_loss", 0.0)),
            float(metrics["grad_norm"]),
            mfu * 100, tps, avg_step_time, epoch,
        )
```

### Step 3: Verify AST parse both files

```bash
python -c "
import ast
for f in ['src/jax_v6/training/train_step.py', 'scripts/run_training.py']:
    ast.parse(open(f).read())
    print(f'AST OK: {f}')
"
```

### Step 4: Commit

```bash
git add src/jax_v6/training/train_step.py scripts/run_training.py
git commit -m "feat: propagate ret_loss through train_step metrics and run_training log"
```

---

## Task 6: Re-pretokenize on TPU VM → arrayrecord_combined_v3

**Run on TPU VM** (`ssh -i ~/.ssh/google_compute_engine daniel@34.158.185.117`):

```bash
cd ~/Financial_IA
# Pull latest changes
git pull

# Launch re-pretokenize (same sources as v2, adds returns field)
nohup env PYTHONPATH=$PWD python3 scripts/pretokenize_tpu.py \
  --checkpoint checkpoints/strate_i_jax_combined/best_params.npz \
  --output_dir data/arrayrecord_combined_v3/ \
  --seq_len 128 \
  --source_dirs \
    data/crypto_parquet/futures \
    data/crypto_parquet/spot \
    data/ohlcv_stocks_daily \
    data/ohlcv_stocks_1h \
    data/ohlcv_sp500 \
    data/ohlcv_forex \
    data/ohlcv_commodities \
    data/yfinance_parquet \
  --source_scales 0 0 2 1 2 2 2 2 \
  --tpu_batch 2000000 \
  > logs/pretokenize_v3.log 2>&1 &
echo "PID: $!"
```

Expected: ~2 min, 8272 shards in `data/arrayrecord_combined_v3/`, same structure as v2 but with `"returns"` field.

Monitor:
```bash
tail -f logs/pretokenize_v3.log
```

---

## Task 7: Retrain JEPA v6.3 on TPU VM

**Run on TPU VM** after Task 6 completes:

```bash
cd ~/Financial_IA

# Update config to point to v3 data
# Edit configs/scaling/v6e_38m.yaml: arrayrecord_dir → data/arrayrecord_combined_v3/
sed -i 's|arrayrecord_combined_v2|arrayrecord_combined_v3|g' configs/scaling/v6e_38m.yaml

# Launch training (fresh start — v6.3)
nohup env \
  PYTHONPATH=$PWD \
  SCALE_CONFIG=configs/scaling/v6e_38m.yaml \
  SCALE_TIER=38m_v5 \
  GCS_BUCKET=gs://fin-ia-eu \
  TPU_TYPE=v6e-8 \
  TPU_GEN=v6e \
  python3 scripts/run_training.py \
  > logs/train_v6_3.log 2>&1 &
echo "PID: $!"
```

Monitor convergence:
```bash
tail -f logs/train_v6_3.log | grep "step\|ret\|loss"
```

**Success criteria:**
- `ret ` value in log should decrease from ~1.0 toward ~0.3–0.5 during training
- `loss` trajectory similar to v6.2 (starts ~13K, converges to ~2K by step 28K)
- If `ret` stays stuck at 1.0 → increase `ret_weight` to 0.05 in YAML and restart

---

## Task 8: Validate with linear probe (on TPU VM)

After JEPA converges (step ~28K), run the linear probe to confirm R² improved:

```bash
cd ~/Financial_IA
# Recompute RL buffer with v6.3 embeddings
nohup env PYTHONPATH=$PWD python3 scripts/precompute_rl_buffer.py \
  --arrayrecord_dir data/arrayrecord_combined_v3/ \
  --jepa_ckpt checkpoints/jax_v6e/38m_v5/ \
  --output_dir data/rl_buffer_v6_3/ \
  --seq_cutoff_ratio 0.8 \
  --oos_dir data/rl_buffer_v6_3_oos/ \
  > logs/precompute_v6_3.log 2>&1 &
```

Then run OOS eval to check if Sharpe > 0.3:
```bash
PYTHONPATH=$PWD python3 scripts/eval_oos_temporal.py \
  --oos_dir data/rl_buffer_v6_3_oos/ \
  --dqn_ckpt checkpoints/cs_v6_3/best_cs_dqn.npz \
  --episode_len 4 --n_eval 2000 --k_assets 16 \
  --output results/oos_v6_3.json
```

---

## Notes

- `ret_weight=0.01` is conservative. If `ret_loss` doesn't decrease, try `0.05` or `0.1`.
- The `return_head` Dense(1) adds only `d_model + 1 = 513` params — negligible.
- Z-scoring per batch is critical: raw crypto returns ~0.005 std vs stocks ~0.001 std → without normalization, crypto dominates the gradient.
- The `"returns"` field in existing ArrayRecords (v2) defaults to zeros → backward compat preserved, but JEPA on v2 data will get zero gradient on ret_loss (fine — set `ret_weight=0.0` for v2).
