# ChaosAI — Time-Series Foundation Model

Self-supervised world model for chaotic dynamical systems. Financial markets as proving ground.

> **DISCLAIMER:** Research only. Not a trading system. Not financial advice.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Architecture

```
Raw OHLCV → [Strate I: FSQ Tokenizer] → discrete codes
          → [Strate II: Mamba-2 JEPA]  → latent embeddings
          → [Strate III: OT-CFM]       → N future trajectories
          → [Strate IV: TD-MPC2 Agent] → trading actions
```

- **Strate I** — Dilated CNN + Finite Scalar Quantization (1024 codes, zero codebook collapse)
- **Strate II** — Mamba-2 SSD + VICReg JEPA + cross-attention macro conditioning (FRED/COT)
- **Strate III** — Optimal Transport Flow Matching, multimodal latent futures
- **Strate IV** — TD-MPC2 + CVaR + Multiverse Crossing (M=30 geodesic perturbations)

## Results

| Model | Params | Dataset | Loss | Strate IV Sharpe |
|-------|--------|---------|------|-----------------|
| v6 | 36.1M | 838M tokens | 1,310 | 2.63 |
| **v6.1** | **36.6M** | **838M tokens** | **908** | **2.78** |

v6.1 adds: cross-attention macro injection, bifurcation-modulated CQL, risk-parity rewards, priority experience replay.

Multiverse Crossing (M=30, fresh data): Lyapunov **-0.73** (stable), Sharpe **2.78**, 0 contested assets.

## Status

| Component | Status |
|-----------|--------|
| Data pipeline (838M tokens, 8,969 assets) | ✅ Done |
| Strate I — FSQ tokenizer (JAX/Flax) | ✅ Done |
| Strate II — Mamba-2 JEPA, auto-sharding, XLA flags | ✅ Done |
| Strate III — OT-CFM stochastic predictor | ✅ Done |
| Strate IV — TD-MPC2 + CVaR + Multiverse Crossing | ✅ Done |
| v6.1 training (100 epochs, TPU v6e) | ✅ Done |
| v6.2 — scale-invariant JEPA (scale_id embedding + cross-res VICReg) | 🔄 Training |
| v6.3 — return prediction auxiliary loss | 🔄 Implemented, retraining |

## Stack

JAX/Flax (TPU-native) + PyTorch (GPU validation). TPU Research Cloud (TRC), zero idle cost via Drive ↔ GCS ↔ TPU data lake.

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync
export PYTHONPATH=$PWD

# Train (TPU v6e)
export SCALE_CONFIG=configs/scaling/v6e_38m_v3.yaml
export GCS_BUCKET=gs://fin-ia-eu
nohup python3 -u scripts/run_training.py > logs/train.log 2>&1 &

# BTC regime prediction
PYTHONPATH=. python3 scripts/predict_btc_1week.py \
    --jepa_ckpt checkpoints/jax_v6e/38m_v3/92112 \
    --strate_i_ckpt checkpoints/strate_i_jax_combined/best_params.npz
```

## License

[AGPL v3](LICENSE)
