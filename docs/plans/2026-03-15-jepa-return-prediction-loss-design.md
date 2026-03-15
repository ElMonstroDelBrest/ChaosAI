# JEPA Return Prediction Auxiliary Loss — Design

**Goal:** Force the JEPA context encoder to produce embeddings with directional return predictive signal, by adding a supervised next-return prediction head trained jointly with VICReg.

**Architecture:** Minimal surgical change — one Dense head, one loss term, one new ArrayRecord field. All existing code backward-compatible via `ret_weight=0.0` default.

---

## Problem

OOS temporal evaluation shows Sharpe ≈ 0 across all asset classes. Root cause: the ArrayRecord does not store raw returns, so the JEPA is trained purely with VICReg/Barlow Twins — it learns market structure but has no gradient signal pushing it toward return-predictive representations. The DQN then overfits on spurious correlations in embeddings that carry no actual forward-return information.

## Design

### 1. Data Pipeline — `scripts/pretokenize_tpu.py`

During shard writing, extract per-timestep log price returns from the raw candle data and store in ArrayRecord.

**Field**: `"returns"` — float32 array of length `seq_len` = log price return per timestep in the sequence.

- Source: `log_ret[:, 0]` (log price return, already computed in `_load_single_file`)
- Must be passed from CPU preprocessing through to `_write_shard`
- Per-sequence slice: `log_ret[start:end, 0]` → store as flat float list length `seq_len`
- Clip to ±0.5 before storing (removes extreme delisting spikes already seen in data)

### 2. Data Loader — `src/jax_v6/data/grain_loader.py`

Parse `"returns"` field as optional (backward compat with existing ArrayRecords without this field):

```python
if "returns" in features:
    returns = np.array(features["returns"].float_list.value, dtype=np.float32)
else:
    returns = np.zeros(seq_len, dtype=np.float32)
result["returns"] = returns
```

### 3. JEPA Model — `src/jax_v6/jepa.py`

Add `ret_weight: float = 0.0` field to `FinJEPA`. In `setup()`, add `self.return_head = nn.Dense(1)`. In `__call__()`:

```python
ret_loss = jnp.float32(0.0)
if self.ret_weight > 0.0 and "returns" in batch:
    # Causal: h_x[:, t, :] sees tokens 0..t → predict r_{t+1}
    r_pred = self.return_head(h_x[:, :-1, :]).squeeze(-1)  # (B, S-1)
    r_next = batch["returns"][:, 1:]                        # (B, S-1)
    # Z-score per batch to normalize scale across asset classes
    r_std = jnp.std(r_next) + 1e-6
    r_norm = (r_next - jnp.mean(r_next)) / r_std
    r_pred_norm = (r_pred - jnp.mean(r_next)) / r_std
    ret_loss = jnp.mean((r_pred_norm - r_norm) ** 2)
    total_loss = total_loss + self.ret_weight * ret_loss
```

Return `ret_loss` in the output dict.

### 4. Config — `src/jax_v6/config.py`

Add `ret_weight: float = 0.0` to `Mamba2Config` (or `StrateIIConfig`, wherever `cross_res_weight` lives).

### 5. YAML — `configs/scaling/v6e_38m.yaml`

```yaml
ret_weight: 0.01
```

Small λ — keeps VICReg dominant, adds directional gradient. Can tune upward if linear probe R² stays near 0.

### 6. `scripts/run_training.py`

Log `ret_loss` in the training metrics output (alongside `invariance`, `variance`, `cfm_loss`).

---

## Execution

1. Re-pretokenize: `pretokenize_tpu.py` → `data/arrayrecord_combined_v3/` (~2 min on TPU)
2. Retrain JEPA v6.3: `run_training.py` with `v6e_38m.yaml` → `checkpoints/jax_v6e/38m_v5/` (~40 min)
3. Recompute RL buffer with v6.3 embeddings + temporal OOS split
4. Retrain DQN + OOS eval

## Success Criteria

- `ret_loss` decreasing during JEPA training (not stuck at ~1.0)
- Linear probe R² on returns > 0.05 (current baseline ≈ 0)
- OOS Sharpe > 0.3 cross-class

## Non-Goals

- Do not change VICReg weights
- Do not add return prediction to the target encoder (EMA) forward pass
- Do not change the DQN architecture
