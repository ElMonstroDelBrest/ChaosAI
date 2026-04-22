# ChaosAI

**Self-supervised foundation model for chaotic time series, with leakage-free out-of-sample evaluation.**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![JAX](https://img.shields.io/badge/JAX-0.6-orange)](https://jax.readthedocs.io)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org)
[![Status: Research](https://img.shields.io/badge/status-research--only-red)](#disclaimer)

> ⚠️ **Academic research artifact only. Not a trading system, not financial advice,
> not for commercial use. The author disclaims all liability for any use of this
> code or results. See [Disclaimer](#disclaimer).**

## TL;DR

We train a 33M-parameter Mamba-3 JEPA on 838M financial tokens and find that the
**standard cross-sectional out-of-sample evaluation protocol leaks +7 Sharpe units**.
Adding a Conditional Flow Matching (CFM) auxiliary objective during pretraining
**eliminates the leakage gap** while preserving downstream performance.

| Model | Params | In-domain | Within-pretrain "OOS" | Truly fresh OOS | Leakage gap |
|-------|-------:|----------:|----------------------:|----------------:|------------:|
| Mamba-3 JEPA                          |  33M | 5.54 | **6.57** | **−0.45** | **−7.02** |
| Mamba-3 JEPA + CFM                    |  33M | 0.53 |  0.32    | **+0.40** | **+0.08** |
| Mamba-3 JEPA + CFM (scaled, fixed reg) |  60M | 0.39 | −0.05    | **−0.44** | **−0.82** |

Sharpe is computed over 2,000 cross-sectional portfolios of 16 crypto assets each,
trained DQN evaluated on truly held-out post-pretraining data
(2026-03-01 → 2026-04-21). Increasing the holding horizon from 64 to 256 steps
yields Sharpe **1.32 (super-additive scaling)** for the 33M+CFM model.

**Scaling caveat.** Doubling parameters (33M → 60M) at fixed `cfm_weight=1.0`
reintroduces +0.82 Sharpe units of leakage. The CFM regularization weight must
scale with model capacity. See [`results/v6.6_scaling_ablation.md`](results/v6.6_scaling_ablation.md).

## Method

A four-stratum pipeline; each stratum trains independently on the previous one's output.

```
Raw OHLCV → [Strate I:  FSQ Tokenizer]    → discrete codes
          → [Strate II: Mamba-3 JEPA + CFM] → latent embeddings
          → [Strate III: Cross-sectional QR-DQN] → portfolio actions
          → [Strate IV: Multiverse Crossing]   → decision-confidence filter
```

- **Strate I** — Dilated CNN + Finite Scalar Quantization (1024 codes, no codebook collapse)
- **Strate II** — Mamba-3 JEPA (trapezoidal SSM with complex RoPE on B/C, BCNorm), Attention
  Residuals across encoder depth, asymmetric Transformer predictor (d_pred=64 < d_model=512),
  CFM flow predictor with optimal-transport coupling
- **Strate III** — Quantile-regression DQN with CVaR-α critic, soft cross-sectional allocation
- **Strate IV** — Multiverse Crossing — geodesic perturbation consensus on the JEPA
  hypersphere as a tractable surrogate for representation robustness

## Headline contributions

1. **Leakage measurement.** Standard per-asset temporal split inflates cross-sectional
   Sharpe by +7.02 units in foundation-model finance ML when the encoder is pretrained
   on all dates.
2. **CFM as Bayesian-by-construction regularization.** A flow-predictor auxiliary
   objective during pretraining internalizes uncertainty into the encoder, eliminating
   the leakage gap (−7.02 → +0.08).
3. **Multiverse Crossing surrogate.** A geodesic perturbation consensus metric provides
   a tractable computable surrogate for the (Rice-undecidable) representation robustness
   property; CFM-sample variance is more discriminative than tangent-plane noise.

Full results and methodology in [`results/v6.5_cfm_leakage_elimination.md`](results/v6.5_cfm_leakage_elimination.md).

## Repository layout

```
src/jax_v6/               JAX/Flax foundation model + RL pipeline
  encoders/               Mamba-2, Mamba-3, SSD scans (chunked + matrix)
  predictors/             MLP, Transformer, KAN, FlowPredictor (CFM)
  losses/                 VICReg, Barlow Twins, cross-resolution consistency
  strate_iv/              Cross-sectional environment, QR-DQN, MVX
  training/               Sharding, optimizer, train state, metrics

scripts/                  End-to-end pipeline scripts
  train_strate_i_jax.py   Tokenizer training
  pretokenize_tpu.py      Tokenize raw OHLCV → ArrayRecord shards
  run_training.py         JEPA pretraining
  precompute_rl_buffer.py JEPA inference + CFM-based MVX bifurcation
  train_cross_sectional.py  Cross-sectional DQN training
  eval_oos_temporal.py    Held-out OOS evaluation
  eval_oos_mvx.py         OOS evaluation with MVX confidence filter
  probe_fresh_jepa.py     Linear probe (Ridge) on frozen embeddings

configs/scaling/          Pre-set training configurations
results/                  Evaluation JSONs and writeups
tests/                    PyTest unit tests for the JAX modules
```

## Quick start

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync
export PYTHONPATH=$PWD

# Train Mamba-3 JEPA + CFM (TPU v6e-8)
SCALE_CONFIG=configs/scaling/v6e_38m_v5.yaml \
SCALE_TIER=38m_v5 \
TPU_TYPE=v6e-8 \
TPU_GEN=v6e \
GCS_BUCKET=gs://your-bucket \
python scripts/run_training.py

# Precompute RL buffer with CFM-based MVX (M=30 trajectory samples)
python scripts/precompute_rl_buffer.py \
  --raw_dirs <raw_parquet_dirs...> \
  --arrayrecord_dir data/arrayrecord_combined/ \
  --jepa_ckpt_dir checkpoints/jax_v6e/38m_v5/<step>/ \
  --config configs/scaling/v6e_38m_v5.yaml \
  --output_dir data/rl_buffer_v5/ \
  --seq_cutoff_ratio 0.8 \
  --oos_dir data/rl_buffer_v5_oos/ \
  --mvx_cfm 30

# Train cross-sectional DQN
python scripts/train_cross_sectional.py \
  --buffer_dir data/rl_buffer_v5/ \
  --output_dir checkpoints/cross_sectional_v5/ \
  --total_steps 1000000

# Evaluate on truly held-out data + MVX confidence sweep
python scripts/eval_oos_temporal.py \
  --oos_dir data/rl_buffer_v5_fresh/ \
  --dqn_ckpt checkpoints/cross_sectional_v5/best_cs_dqn.npz \
  --episode_len 256 --n_eval 500 --k_assets 16

python scripts/eval_oos_mvx.py \
  --oos_dir data/rl_buffer_v5_fresh/ \
  --dqn_ckpt checkpoints/cross_sectional_v5/best_cs_dqn.npz \
  --thresholds 10.0 2.85 2.80 2.70 2.60
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the GPU fallback path
and the smoke-test protocol.

## Stack

- **Modeling**: JAX 0.6, Flax 0.10, Optax, Diffrax (CFM ODE solver)
- **Data**: Grain + ArrayRecord shards, 838M tokens (8,969 assets: crypto futures+spot 1m,
  US stocks daily/hourly, S&P 500, forex, commodities)
- **Compute**: Google Cloud TPU v6e-8 (Trillium)

## Limitations and scope

This is **a research artifact, not a deployable trading system.** Specifically:

- **All Sharpe figures are backtest-only.** No live trading record exists. Real
  execution would face exchange-side slippage 5–20× larger than the quadratic
  proxy used here, market-impact costs not modeled at retail size, exchange
  fees per rebalance not netted at the institutional rate assumed, and regime
  shifts the model has never seen.
- **Statistical confidence is wide.** The headline Sharpe 1.32 has a 95% confidence
  interval of approximately [0.9, 1.7] given the sample size; out-of-distribution
  performance in unseen market regimes is not guaranteed.
- **Capacity is unmeasured and likely small.** Trading any meaningful size on the
  long-tail of the 8,969-asset universe would cause significant own-order
  market impact.
- **No production engineering.** No risk management, position sizing,
  kill-switches, exchange API integration, monitoring, or operational tooling
  is provided. Building these is non-trivial.
- **No legal review.** In many jurisdictions, providing trading signals or
  managing third-party capital requires regulatory licensing.

The repository is intended for **methodological reproducibility** of the
leakage-detection finding and the CFM-based fix — not for live deployment.

## Numerical safeguards

The implementation enforces three invariants required for stable training at scale:

- **MXU 128×128 alignment** — `head_dim = d_model · expand_factor / n_heads = 128`
  ensures TPU matrix-unit tiles fully fill, avoiding ≥50% compute waste.
- **bf16 with float32 SSD accumulators** — temporal accumulations (`cumsum`, RoPE phase,
  state decays) stay in float32 to prevent the bf16-mantissa NaN that surfaces near
  step ~2,750.
- **Geodesic perturbations** — `multiverse_crossing.perturb_latent` projects noise onto
  the tangent plane of the JEPA representation hypersphere then re-normalizes to keep
  perturbed latents on-manifold.

## References

ChaosAI is the integration of recent self-supervised and reinforcement-learning
components — the contribution is the leakage-detection methodology and the CFM-based
fix, not any single building block.

- **JEPA** — LeCun, Y. *A Path Towards Autonomous Machine Intelligence*. 2022. [openreview](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- **VICReg** — Bardes, A., Ponce, J., LeCun, Y. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR 2022. [arXiv:2105.04906](https://arxiv.org/abs/2105.04906)
- **Mamba-2** — Dao, T. & Gu, A. *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. ICML 2024. [arXiv:2405.21060](https://arxiv.org/abs/2405.21060)
- **Mamba-3** — *Mamba-3: Improved Sequence Modeling Using State Space Principles*. 2026. [arXiv:2603.15569](https://arxiv.org/abs/2603.15569)
- **FSQ** — Mentzer, F. et al. *Finite Scalar Quantization: VQ-VAE Made Simple*. ICLR 2024. [arXiv:2309.15505](https://arxiv.org/abs/2309.15505)
- **OT-CFM** — Tong, A. et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport*. TMLR 2024. [arXiv:2302.00482](https://arxiv.org/abs/2302.00482)
- **CVaR RL** — Chow, Y. et al. *Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach*. NeurIPS 2015. [arXiv:1506.02188](https://arxiv.org/abs/1506.02188)

## Citation

```bibtex
@misc{chaosai2026,
  author       = {Gherasim, George-Daniel},
  title        = {{ChaosAI: Leakage-Free Out-of-Sample Evaluation for Foundation
                   Models on Financial Time Series}},
  year         = {2026},
  howpublished = {\url{https://github.com/ElMonstroDelBrest/ChaosAI}},
}
```

## Acknowledgments

Compute provided by Google's [TPU Research Cloud](https://sites.research.google/trc/)
(v6e-8, europe-west4-a). TRC is a non-commercial research program; this work is
released under AGPL-3.0 in alignment with the program's expectation that
TRC-supported research be shared openly.

## Disclaimer

This software and the associated results are released **strictly for academic
research and educational purposes**. Use is governed by the AGPL-3.0 license
([`LICENSE`](LICENSE)) and by the additional restrictions and disclaimers below.

**No commercial use.** This work was produced under the Google TPU Research
Cloud program, a non-commercial research grant. Use of this code or any
derivative model thereof for commercial purposes — including but not limited
to live trading for profit, paid advisory services, managed accounts, or
licensing to third parties — is **not authorized** and may additionally
violate the terms of the underlying compute grant.

**Not financial advice.** Nothing in this repository constitutes investment
advice, an offer to buy or sell any security, or a solicitation of any kind.
The author is not a registered investment adviser, broker-dealer, or financial
analyst. Reported performance metrics are derived from historical backtests
on past market data and **do not guarantee** future results. Live deployment
would face execution costs, market impact, regulatory constraints, and regime
shifts that are not captured by the backtest.

**No warranty, no liability.** This software is provided "as is", without
warranty of any kind, express or implied, including but not limited to the
warranties of merchantability, fitness for a particular purpose, and
non-infringement. **In no event shall the author be liable for any claim,
damages, or other liability** — including direct, indirect, incidental,
special, consequential, or exemplary damages, or any loss of profit, capital,
or data — arising from, out of, or in connection with the use of this software,
its results, or any decision made on the basis of either.

**User assumes all risk.** Any person who downloads, runs, modifies, or
otherwise relies on this code does so entirely at their own risk and is
solely responsible for compliance with all applicable laws, regulations, and
exchange terms-of-service in their jurisdiction.

## License

[AGPL-3.0](LICENSE)
