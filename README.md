# ChaosAI — Time-Series Foundation Model

Self-supervised world model for chaotic dynamical systems, evaluated on financial markets.

> **DISCLAIMER:** Research only. Not a trading system. Not financial advice.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

## Overview

ChaosAI is a four-stratum pipeline for learning compressed, predictive representations of
non-stationary time series. Each stratum is a recent self-supervised or RL component
(FSQ tokenization, Mamba-2 JEPA, OT Conditional Flow Matching, TD-MPC2), composed so that
each level of abstraction is learnable independently and then cascaded. The system is
domain-agnostic by design; we use crypto and equity markets as a stress test because
their non-stationarity, heavy tails, and weak signal-to-noise make them a worst-case
benchmark for representation robustness.

**What is novel.** The contribution is not any single strate — each of them exists in the
literature. The contribution is **Multiverse Crossing**, an evaluation method for
self-supervised time-series representations. Instead of scoring a single deterministic
forward pass, we generate *M* parallel universes by geodesic perturbation on the tangent
plane of the JEPA representation hypersphere, re-normalize to the unit sphere, and measure
the downstream agreement across universes. The resulting metrics — Lyapunov-style
stability proxy, bifurcation index, cross-universe decision consensus — give a geometric
test of whether a learned representation's downstream behavior is driven by signal or by
noise. We find that representations with Lyapunov < 0 and bifurcation ≈ 0 produce
decisions that are robust to 2% geodesic perturbations; this correlates with
out-of-sample performance on the financial backtest (Sharpe 2.78 vs 1.22 for deterministic).

**Why it matters beyond finance.** Any time-series domain where the cost of a wrong
decision is high (medical diagnosis, climate forecasting, industrial fault detection,
neuroscience) benefits from a per-decision robustness test, not just an average metric.
Multiverse Crossing is model-agnostic once a representation sits on a normalized manifold
(any contrastive / JEPA / SimCLR-style embedding space).

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

Multiverse Crossing (M=30, fresh data): Lyapunov **−0.73** (stable), Sharpe **2.78**, 0/372 contested assets.
Full results in [`results/multiverse_crossing_30u.md`](results/multiverse_crossing_30u.md).

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

JAX/Flax (TPU-native) + PyTorch (GPU validation). Training done on Google
[TPU Research Cloud](https://sites.research.google/trc/) (v6e-64, europe-west4-a),
zero-idle-cost data lake via Drive ↔ GCS ↔ TPU.

Key TPU-side design choices:
- **Auto-sharder** detects v6e / v5e / v5p meshes and routes to optimal (8,8) Tore 2D or (16,4) Tore 3D
- **MXU 128×128 alignment** on every `head_dim` to saturate the matrix unit
- **bf16 activations, fp32 SSD accumulators** — required for numerical stability past step ~2750

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

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for GPU fallback setup, environment
pinning, and a 5-minute smoke test that validates the full pipeline on a single batch.

## References

The architecture is a composition of four recent self-supervised / RL components; our
contribution is their integration and the Multiverse Crossing evaluation.

- **FSQ (Strate I)** — Mentzer, F. et al. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR 2024. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
- **Mamba-2 (Strate II)** — Dao, T. & Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **JEPA** — LeCun, Y. *A Path Towards Autonomous Machine Intelligence.* Meta AI, 2022. [openreview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- **VICReg** — Bardes, A., Ponce, J., LeCun, Y. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.* ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
- **OT-CFM (Strate III)** — Tong, A. et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- **TD-MPC2 (Strate IV)** — Hansen, N., Su, H., Wang, X. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)
- **CVaR RL** — Chow, Y. et al. *Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach.* NeurIPS 2015. [arXiv:1506.02188](https://arxiv.org/abs/1506.02188)
- **KAN** — Liu, Z. et al. *KAN: Kolmogorov-Arnold Networks.* 2024. [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)

## Acknowledgments

Compute provided by Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/)
on v6e-64, europe-west4-a. The TRC grant is a non-commercial research program; this
repository is published under AGPL-3.0 to comply with its open-source requirement.

## License

[AGPL v3](LICENSE)
