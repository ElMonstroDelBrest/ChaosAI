# ChaosAI — Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                          DATA ACQUISITION (Stage 0)                              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  Binance Futures API                      Ethereum DeFi (Dune)                   ║
║  (432 USDT-M pairs)                       (on-chain interactions)                ║
║       │                                         │                                ║
║       ▼                                         ▼                                ║
║  download_massive_data.py              download_onchain_data.py                  ║
║       │                                         │                                ║
║       ├──► data/raw/1m/*.parquet                ├──► data/onchain/raw/eth/       ║
║       ├──► data/raw/1h/*.parquet                │                                ║
║       ├──► data/raw/funding_rates/              │    build_graphs.py             ║
║       └──► data/raw/macro/                      │         │                      ║
║            │                                    │         ▼                      ║
║       convert_parquet_to_pt.py                  │    data/onchain/graphs/        ║
║            │                                    │                                ║
║            ▼                                    │    train_gnn.py                ║
║  data/ohlcv_1m/ (9.7 GB, 432 .pt)               │         │                      ║
║  data/ohlcv_v5/ (174 MB, 432 .pt)               │         ▼                      ║
║                                                 │    checkpoints/strate_v/       ║
║                                                 │                                ║
║                                                 │    compute_gnn_embeddings.py   ║
║                                                 │         │                      ║
║                                                 └────►    ▼                      ║
║                                                      data/onchain/embeddings/    ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                    │                                        │
                    ▼                                        │
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     STRATE I — TOKENIZER (Discrete Codes)                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  data/ohlcv_1m/*.pt ──► train_strate_i.py ──► checkpoints/strate-i-*.ckpt        ║
║                         (FSQ codebook, 1024 codes, dim=64)                       ║
║                                │                                                 ║
║                                ▼                                                 ║
║               pretokenize_to_arrayrecord.py                                      ║
║                    │                    │                                        ║
║                    ▼                    ▼                                        ║
║  data/arrayrecord_1m/          merge_gnn_to_tokens.py ◄── GNN embeddings         ║
║  (362 MB, 253K seqs)                   │                                         ║
║                                        ▼                                         ║
║                              data/arrayrecord_1m_gnn/                            ║
║                              (450 MB, + gnn_embeddings)                          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                    │
                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║              STRATE II — FIN-JEPA (Self-Supervised World Model)                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────┐    ┌──────────────────────────────────┐         ║
║  │  PyTorch (GPU dev)          │    │  JAX/Flax (TPU production)       │         ║
║  │                             │    │                                  │         ║
║  │  train_strate_ii.py         │    │  run_training.py                 │         ║
║  │  configs/strate_ii.yaml     │    │  configs/scaling/v6e_*.yaml      │         ║
║  │                             │    │                                  │         ║
║  │  FinJEPA:                   │    │  FinJEPA (mirror):               │         ║
║  │  ├─ Mamba2Encoder           │    │  ├─ Mamba2Encoder + vol_clock    │         ║
║  │  │  (SSD kernel, exo_clock) │    │  │  (float32 cumsum safety)      │         ║
║  │  ├─ Predictor (z ~ N(0,I))  │    │  ├─ Predictor / KANPredictor     │         ║
║  │  ├─ FlowPredictor (OT-CFM)  │    │  ├─ FlowPredictor / KANFlow      │         ║
║  │  └─ VICReg loss             │    │  └─ VICReg + CFM loss            │         ║
║  │                             │    │                                  │         ║
║  │  EMA target encoder ──────────────► EMA target encoder              │         ║
║  │  Block masking (JEPA-style) │    │  Block masking (JEPA-style)      │         ║
║  │                             │    │                                  │         ║
║  └──► ckpt/strate_ii/*.ckpt    │    └──► ckpt/jax_v6e/{scale}/step/    │         ║
║                                │         (orbax, FSDP sharded)         │         ║
║                                │                                       │         ║
║  Scaling (Chinchilla-optimal): │    Auto-Sharder: v6e(8,8) v5e(8,8)    │         ║
║  26M → 54M → 300M → 1B → 7B    │    v5p(16,4) — topology-aware mesh    │         ║
║                                │                                       │         ║
║  TPU Launch:                   │    launch_tpu_v6e.sh --scale={s,m,l}  │         ║
║                                │    launch_tpu_v5e.sh --scale={s,m,l}  │         ║
║                                │    XLA flags: async fusion, latency   │         ║
║                                │    hiding, aggressive loop fusion     │         ║
║                                │                                       │         ║
║  Training:                     │    Grain async dataloader (GCS→TPU)   │         ║
║  520M tokens, batch=512-16384  │    SGDR cosine restarts, grad_clip    │         ║
║  Loss: VICReg + CFM            │    Checkpoint: orbax → GCS backup     │         ║
║                                │                                       │         ║
║  Best model: 65M (d=768)       │    OOS loss: 2766 (0.92× train)       │         ║
║  Vol regime acc: 61.2%         │    Linear probe: 55.7% balanced acc   │         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                    │
                    │  Frozen JEPA encoder
                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║              STRATE III — MULTIVERSE CROSSING (Stochastic Futures)               ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  For each latent embedding h_x:                                                  ║
║                                                                                  ║
║  1. Geodesic perturbation on hypersphere (not Gaussian noise)                    ║
║     h_x ──► perturb_latent(σ=0.01, M=5) ──► {h₁, h₂, h₃, h₄, h₅}                 ║
║             (tangent plane projection + L2 re-normalization)                     ║
║                                                                                  ║
║  2. FlowPredictor (OT-CFM) generates N futures per universe                      ║
║     hᵢ ──► ODE solve (Euler, n_steps=2) ──► {f¹ᵢ, f²ᵢ, ..., fᴺᵢ}                 ║
║                                                                                  ║
║  3. Convergence analysis across M universes:                                     ║
║     ├── convergence_score:  1/(1 + inter_std/intra_std)  ∈ [0,1]                 ║
║     ├── divergence_rate:    Δ(inter_mv_std)                                      ║
║     ├── bifurcation_index:  exp(entropy(eigenvalues(Gram)))                      ║
║     ├── lyapunov_proxy:     log(inter_std/σ) / t                                 ║
║     └── inter_mv_std:       disagreement across universe means                   ║
║                                                                                  ║
║  Convergence high → universes agree → agent can be aggressive                    ║
║  Convergence low  → universes diverge → agent should be cautious                 ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                    │
                    │  Embeddings + convergence metrics
                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║           STRATE IV — TD-MPC2 AGENT (Planning + Risk Management)                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─ DUAL-SCALE OBSERVATION VECTOR (obs_dim = 3089)  ──────────────────────────┐  ║
║  │                                                                            │  ║
║  │  h_micro  (1536)  ← JEPA encoder on 128 1-min candles (~2h context)        │  ║
║  │  h_macro  (1536)  ← Rolling mean of past 48 windows (~4 days)              │  ║
║  │  conv_micro  (5)  ← Multiverse convergence (micro scale)                   │  ║
║  │  conv_macro  (5)  ← Multiverse convergence (macro scale)                   │  ║
║  │  cross_tf    (1)  ← Cosine similarity micro ↔ macro                        │  ║
║  │  revin_stds  (5)  ← Volatility regime (O,H,L,C,V)                          │  ║
║  │  fwd_return  (1)  ← 32-min forward return                                  │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                    │                                                             ║
║                    ▼                                                             ║
║  ┌─ WORLD MODEL (4.1M params)  ───────────────────────────────────────────────┐  ║
║  │                                                                            │  ║
║  │  LatentEncoder: obs(3089) → MLP(512) → LayerNorm → z(256)                  │  ║
║  │  LatentDynamics: [z(256), a(1)] → MLP(512) + residual → z_next(256)        │  ║
║  │  RewardHead: [z(256), a(1)] → MLP(512) → scalar reward                     │  ║
║  │                                                                            │  ║
║  │  Rollout: z₀ → z₁ → z₂ → z₃ → z₄ → z₅  (H=5 horizon)                       │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                    │                                                             ║
║                    ▼                                                             ║
║  ┌─ MPPI PLANNING  ───────────────────────────────────────────────────────────┐  ║
║  │                                                                            │  ║
║  │  For 6 iterations (Python-unrolled, XLA fused):                            │  ║
║  │    1. Sample K=512 action sequences ~ N(μ, σ²)                             │  ║
║  │    2. Rollout each through world model → returns                           │  ║
║  │    3. Terminal value: CVaR of distributional critic                        │  ║
║  │    4. MPPI softmax re-weight: w = softmax(returns / τ)                     │  ║
║  │    5. Update μ, σ from weighted mean/std                                   │  ║
║  │                                                                            │  ║
║  │  Output: action ∈ [-1, +1]  (-1=full short, 0=flat, +1=full long)          │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                    │                                                             ║
║                    ▼                                                             ║
║  ┌─ DISTRIBUTIONAL CRITIC  ───────────────────────────────────────────────────┐  ║
║  │                                                                            │  ║
║  │  EnsembleCritic: 2× QuantileCritic → min(q₁, q₂)                           │  ║
║  │  32 quantiles → full return distribution (not just mean)                   │  ║
║  │  CVaR(α) = mean(bottom α% quantiles)                                       │  ║
║  │                                                                            │  ║
║  │  Dynamic α: α_min=0.1 (conservative) ← low convergence                     │  ║
║  │             α_max=0.4 (aggressive)   ← high convergence                    │  ║
║  └────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                  ║
║  Training: Offline on pre-computed replay buffer (43K train transitions)         ║
║  Script:   train_strate_iv_jax.py                                                ║
║  Buffer:   precompute_dual_buffer.py → data/dual_buffer/{train,test}.npz         ║
║  Result:   Sharpe +2.19 (median, 419 pairs) vs B&H -2.19                         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
                    │
                    ▼
╔══════════════════════════════════════════════════════════════════════════════════╗
║                      EVALUATION & BACKTESTING                                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─ Representation Quality ──────┐  ┌─ Trading Performance  ─────────────────┐   ║
║  │                               │  │                                        │   ║
║  │  eval_oos.py                  │  │  backtest_linear.py (v2)               │   ║
║  │  ├─ OOS loss: 2766            │  │  ├─ S1: Vol-Target     Sharpe -10.0    │   ║
║  │  └─ 0.92× train loss          │  │  ├─ S2: K-Means Regime Sharpe -10.9    │   ║
║  │                               │  │  ├─ S3: Anomaly Filter Sharpe  -5.4    │   ║
║  │  linear_probe.py              │  │  └─ Buy & Hold         Sharpe  -1.9    │   ║
║  │  ├─ Direction:  50.3%         │  │                                        │   ║
║  │  ├─ Volatility: 61.2% ✓       │  │  backtest_strate_iv.py                 │   ║
║  │  └─ Large move: 55.7% ✓       │  │  ├─ TD-MPC2            Sharpe  +2.19   │   ║
║  │                               │  │  └─ Buy & Hold         Sharpe  -2.19   │   ║
║  └───────────────────────────────┘  └────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════════════════╗
║                      INFRASTRUCTURE & DATA LAKE                                  ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌──────────────────┐     ┌─────────────────┐     ┌───────────────────────┐      ║
║  │  Google Drive    │     │  GCS Bucket     │     │  TPU VM               │      ║
║  │  (30 TB cold)    │◄───►│  gs://fin-ia-eu │◄───►│  fin-ia-v6e           │      ║
║  │  $0/mois         │     │  europe-west4   │     │  europe-west4-a       │      ║
║  │                  │     │  Standard       │     │  v6e-8 (8 chips)      │      ║
║  │  01_raw_ohlcv/   │     │  Staged during  │     │  32 GB HBM/chip       │      ║
║  │  02_tokens/      │     │  training only  │     │                       │      ║
║  │  03_arrayrecords/│     │  → $0 at rest   │     │  JAX 0.6.x + Flax     │      ║
║  │  04_checkpoints/ │     │                 │     │  orbax + grain        │      ║
║  │  05_results/     │     │  FREE transfer  │     │  torch_xla (Strate I) │      ║
║  │                  │     │  (same region)  │     │                       │      ║
║  └──────────────────┘     └─────────────────┘     └───────────────────────┘      ║
║                                                                                  ║
║  trc_data_manager.sh:  stage | backup | cleanup | status                         ║
║  Coût GCP actuel: $0/mois (tout sur Drive, GCS vidé après training)              ║
║                                                                                  ║
║  Quota TRC: 64× v6e spot (eu-west4-a) | 64× v5e spot (eu-west4-b)                ║
║  Scaling:   65M (done) → 300M (v6e-64) → 1B (v6e-64) → 7B (v5p-768)              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

## Execution Flow — Quick Reference

```
# Full pipeline (from scratch)
1. python scripts/download_massive_data.py --interval 1m                    # ~4h
2. python scripts/train_strate_i.py --config configs/strate_i_1m.yaml       # ~1h GPU
3. python scripts/pretokenize_to_arrayrecord.py                             # ~10 min
4. ./scripts/trc_data_manager.sh stage                                      # Drive→GCS
5. ./scripts/launch_tpu_v6e.sh --scale=s                                    # Train JEPA on TPU
6. python scripts/precompute_dual_buffer.py                                 # ~5 min
7. python scripts/train_strate_iv_jax.py                                    # ~30 min
8. python scripts/backtest_strate_iv.py                                     # ~5 min

# Evaluation only (from existing checkpoint)
1. python scripts/eval_oos.py                                               # OOS loss
2. python scripts/linear_probe.py                                           # Representation quality
3. python scripts/backtest_linear.py                                        # Latent-state strategies
4. python scripts/backtest_strate_iv.py                                     # RL agent performance
```
