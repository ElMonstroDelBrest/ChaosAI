# ChaosAI — General Time-Series Foundation Model

> **Architecture:** Exo-Clocked Mamba-2 SSD + OT-CFM + VICReg JEPA + Multiverse Crossing
> **Framework:** JAX/Flax (TPU-native) + PyTorch (GPU validation)
> **Current model:** v6.1 — 36.6M params, **loss 908** (12 epochs, 838M tokens, 8,969 assets)
> **Approach:** Guerrilla Research — zero idle cost, Drive-backed Data Lake, TRC TPUs

ChaosAI is a self-supervised foundation model for chaotic dynamical systems. The architecture is designed to scale to 7B parameters (T-Shirt XL) on a full TPU v5p-768 pod — but we're not there yet, and that's deliberate.

We're currently validating **scaling laws** on the 36M→150M tier to prove the architecture works before committing larger compute. Every design decision — from MXU-aligned head dimensions to the Auto-Sharder topology mapping — is built so that the same codebase runs unchanged from 15M to 7B.

The infrastructure reflects the same philosophy: **constraint-driven efficiency**. The Data Lake architecture (Google Drive ↔ GCS ↔ TPU) is not a workaround — it's a deliberate strategy. We can scale the training dataset by 10x or 100x to validate model alpha without VC funding or five-figure monthly cloud bills. GCS is hot storage: data stages in before training, results back up to Drive after, and the bucket returns to $0. This is what guerrilla research looks like when you're optimizing for insight per dollar, not for appearances.

> **DISCLAIMER: RESEARCH PURPOSE ONLY**
>
> This is an open-source research project exploring self-supervised learning on chaotic dynamical systems, using financial time series as a high-noise benchmark. **It is NOT a trading system.**
>
> - Nothing in this repository constitutes financial advice.
> - The authors are not responsible for any financial losses incurred by using this code.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![JAX](https://img.shields.io/badge/JAX-TPU%20v6e--8-blue?logo=google&logoColor=white)](https://github.com/google/jax)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org/)

## 1. Why Financial Markets?

Financial markets are the most adversarial, non-stationary, and noise-dominated dynamical system publicly accessible at high frequency. They exhibit fat-tailed distributions, regime shifts, correlated chaos across 8,000+ instruments, and adversarial feedback loops.

We chose this domain as a **proving ground** — if a self-supervised architecture can learn meaningful latent representations here, it transfers to less adversarial time-series domains (biomedical signals, geophysics, audio, video dynamics) where the signal-to-noise ratio is orders of magnitude better.

## 2. Architecture (4 Strata)

### Strate I — Perception (SymLog + FSQ Tokenizer)
Compresses raw OHLCV candles into discrete codes via Dilated Causal CNN + **SymLog** (preserves fat tails) + **Finite Scalar Quantization** (1024 codes, zero codebook collapse). JAX-native port trained on 838M candles at 3.5M samples/sec on TPU.

### Strate II — World Model (Exo-Clocked Mamba-2 JEPA, v6.1)
Self-supervised Joint-Embedding Predictive Architecture with:
- **Mamba-2 SSD** (State Space Duality) — chunked dense matmuls replacing selective scan. Custom JAX kernel optimized for 128×128 MXU tiles. float32 accumulation prevents NaN from bf16 mantissa truncation.
- **Volume-Conditioned Delta** — the SSM step `dt` is modulated by realized volatility via `tanh`-bounded bias, giving the model an adaptive temporal clock (volatile regimes processed at finer granularity). Bounded with `dt_max_delta * tanh(raw)` to prevent state explosion.
- **Cross-Attention Macro Injection** (v6.1) — macro context (FRED + COT, top-48 signals) conditions the first token via single-head cross-attention. Replaces the legacy gated additive injection. Zero-init output projection for backward compatibility.
- **VICReg** self-supervised objective with block masking — no labels, no lookahead bias.
- **KAN Predictor** — optional B-spline activation layers (Cox-de Boor via `jax.lax.scan`) as drop-in replacement for MLP predictor.

### Strate III — Multiverse Predictor (OT-CFM)
**Optimal Transport Continuous Flow Matching** — Sinkhorn coupling + diffrax ODE integration to generate multimodal future trajectories in latent space. Replaces the MLP predictor with a velocity field trained via OT interpolants.

### Strate IV — Agent (TD-MPC2 + CVaR + Multiverse Crossing, v6.1)
Model-based RL agent that plans inside the differentiable latent space:
- **TD-MPC2** with MPPI tree search in latent space (unrolled for XLA kernel fusion)
- **Distributional critic** (ensemble quantile) optimizing **CVaR** — dynamic alpha driven by multiverse convergence
- **Bifurcation-Modulated CQL** (v6.1) — offline RL regularization scales with `bifurcation_index`: `effective_alpha = cql_alpha × (1 + scale × mean_bif)`. High chaos → more conservative.
- **Multiverse Crossing** — M=30 geodesic perturbations on the JEPA hypersphere. Convergence analysis (eigenvalue entropy of Gram matrix, Lyapunov proxy) drives CVaR alpha dynamically.
- **Cross-Sectional DQN** — market-neutral long/short strategy across K assets. Score-weighted soft allocation, quadratic slippage, **risk-parity reward shaping** (v6.1), **Priority Experience Replay** weighted by bifurcation_index (v6.1).

## 3. JAX/TPU Implementation

The primary training path is JAX-native (`src/jax_v6/`), designed for TPU pods.

### TPU-Native Design
- **Chunked SSD kernel** (`ssd.py`) — bf16 compute with **float32 accumulation** (cumsum, h_final) to prevent NaN from mantissa truncation. Weekend gating, causal masking. Fills MXU tiles exactly (chunk_size=128, head_dim=128).
- **Auto-Sharder Topology-Aware** (`sharding.py`) — 2D mesh `('data', 'fsdp')` via `create_device_mesh`, maps virtual axes onto physical TPU torus minimizing ICI hops. Auto-detects topology from v6e-8 `(8,1)` to v5p-768 `(192,4)`.
- **GSPMD** parallelism via `jax.sharding.Mesh` + `NamedSharding`. Zero `pmap`.
- **InMemoryLoader** — loads all records with 128 threads into RAM (5.9M records in 5.9s), then serves batches from memory at <1ms/batch. Eliminates CPU data loading bottleneck entirely.
- **Grain** async multi-host data pipeline — ArrayRecord format, prefetch buffer depth 64.
- **Diffrax** ODE integration with Sinkhorn OT coupling (JIT-compatible, no scipy).
- **Selective remat** — gradient checkpointing configurable per policy: `"full"` (recompute all activations) or `"dots"` (save dot products only). Auto-selected per hardware.
- **Production XLA flags** (`LIBTPU_INIT_ARGS`) — async collective fusion, latency hiding scheduler, aggressive loop fusion.

### T-Shirt Scaling (MXU 128×128 aligned)

All configs enforce `head_dim = d_model × expand / n_heads = 128` for perfect MXU tile filling.

| Size | Params | d_model | Layers | Heads | Pod | Chinchilla Tokens |
|------|--------|---------|--------|-------|-----|-------------------|
| **XS** | 3.9M | 128 | 8 | 2 | v6e-8 | 80M |
| **38M** | 36.6M | 512 | 20 | 8 | v6e-8 | 730M |
| **S** | 50M | 512 | 28 | 8 | v6e-64 | 1B |
| **M** | 300M | 1,024 | 24 | 16 | v6e-64 | 6B |
| **L** | 1B | 2,048 | 48 | 32 | v5p-128 | 20B |
| **XL** | 7B | 4,096 | 32 | 64 | v5p-768 | 140B |

Generate custom configs: `python scripts/generate_optimal_config.py --target_pod v6e-64 --total_tokens 6B`

## 4. Training Results

### v6.1 — 36.6M params, 838M tokens, TPU v6e-8 (March 2026)

**Dataset:** 8,969 assets — 537 Binance Futures 1min + 140 Spot 1min + 68 yfinance + 7,631 US stocks daily + 505 S&P 500 + 15 forex + 8 commodities + 65 ETFs hourly
**Chinchilla ratio:** 838M / 36.6M = 22.9× (near-optimal)
**Config:** `configs/scaling/v6e_38m_v3.yaml` — d_model=512, n_layers=20, n_heads=8, macro_dim=48, **use_macro_cross_attn=true**

| Step | Loss | Epoch | Note |
|------|------|-------|------|
| 100 | 14,229 | 0 | XLA compilation |
| 10,000 | 5,757 | 0 | |
| 20,000 | 4,502 | 1 | |
| 30,000 | 3,259 | 2 | |
| 46,056 | 1,840 | 4 | Cross-attn learning |
| 57,700 | 1,653 | 5 | |
| 64,400 | 1,441 | 6 | |
| 92,112 | **908** | 8 | v6 baseline: 1,310 |

**Key metrics:** 802K tok/s, 0.082s/step, MFU 2.4%, 8 chips
**Checkpoint:** `checkpoints/jax_v6e/38m_v3/92112` (GCS: `gs://fin-ia-eu/checkpoints/jax_v6e/38m_v3/`)

### v6 — 36.1M params, 838M tokens (baseline, March 2026)

| Step | Loss | Note |
|------|------|------|
| 46,056 | **1,310** | Legacy gated macro injection |

### Strate IV — Cross-Sectional DQN (March 2026)

| Buffer | Sharpe | Note |
|--------|--------|------|
| v6 embeddings (22.4M) | 1.22 | Baseline |
| v6 embeddings (38M) | 2.63 | +115% |
| Fresh data (fév-mars 2026, Iran crisis) | **2.78** | +128% |

**Multiverse Crossing (M=30 universes, σ=0.02):** Lyapunov **-0.73** (stable), bifurcation index 0.003, 0 contested assets.
Rock-solid longs (30/30 universes): CGPT, LAB, TURBO
Rock-solid shorts (30/30 universes): BNT, AERGO, CETUS

### BTC Regime Prediction (March 12, 2026)

Live pipeline: 7-day BTC 1-min data → Strate I tokenization → JEPA 38M encoding → M=30 Multiverse Crossing.

**Result:** CONSENSUS regime (convergence=0.964), Lyapunov=-3.39 (stable), NEUTRAL signal (50/50 long/short split), BTC $66,263→$69,538 (+4.9% 7d context).
See `scripts/predict_btc_1week.py` to run your own inference.

## 5. Data Lake & FinOps — Guerrilla Research Infrastructure

Most ML research assumes unlimited cloud budgets. We don't. ChaosAI runs on a **zero-idle-cost** infrastructure designed to scale datasets by 100x without scaling bills.

The 3-tier Data Lake is the core of this strategy:

```
Google Drive (30 TB Cold, free)  ←→  GCS Bucket (Hot, paid)  ←→  TPU VM (Compute)
         rclone 64×parallel              gsutil / Grain
```

### Data Manager (`scripts/trc_data_manager.sh`)

| Command | Direction | Description |
|---------|-----------|-------------|
| `stage` | Drive → GCS | High-perf rclone transfer: 64 parallel streams, 128M chunks, `--fast-list`. Region safety check before transfer. |
| `backup` | GCS → Drive | Archives checkpoints to `.tar.gz` before upload (Orbax generates thousands of small files per chip). |
| `cleanup` | GCS → /dev/null | Wipes all data + checkpoints from GCS. Resets billing to $0. |
| `status` | — | Per-prefix storage breakdown + estimated monthly cost + region match verification. |

### Cost Model

| Resource | Cost | Notes |
|----------|------|-------|
| Google Drive (Cold) | $0/month | Included in Google One / Workspace |
| GCS Standard storage | ~$0.02/GB/month | Emptied after each run → $0 when idle |
| GCS ↔ TPU transfer | $0 | **Only if same region** — inter-region egress is billed per GB |
| GCS ↔ Drive transfer | $0 (egress) | rclone uses Drive API, no GCS egress charges |

### GCS Data Lake (`gs://fin-ia-eu/data/`)

| Path | Contents | Size |
|------|----------|------|
| `raw/futures_1m_parquet/` | 537 Binance Futures 1min (2019→2026) | ~27 GB |
| `raw/spot_1m_parquet/` | 140 Binance Spot 1min (2017→2026) | ~12 GB |
| `raw/yfinance_parquet/` | 68 ETFs, forex, commodities daily | ~18 MB |
| `arrayrecord_combined/` | 8,969 shards, 5.9M records | ~8.4 GB |

### Standard Workflow
```bash
# Before training — stage data to GCS
./scripts/trc_data_manager.sh stage

# Train on TPU
bash scripts/launch_tpu_v6e.sh --scale=38m

# After training — backup then wipe GCS
./scripts/trc_data_manager.sh backup --latest
./scripts/trc_data_manager.sh cleanup --force
```

## 6. Infrastructure & Cost

| Resource | Value | Notes |
|----------|-------|-------|
| **GCP Project** | `financial-ai-487700` (ChaosAI) | |
| **GCS Bucket** | `gs://fin-ia-eu` | Region: `europe-west4`, Class: Standard |
| **TPU Zone** | `europe-west4-a` | Co-located with bucket → zero egress |
| **Compute** | TPU Research Cloud (TRC) | Spot v6e-64 / v5e-64 pods |
| **Current cost** | **$0/month** | All data on Drive, GCS emptied after training |

### TPU Scripts

| Script | TPU | Zone | HBM/chip | Configs |
|--------|-----|------|----------|---------|
| `launch_tpu_v6e.sh` | v6e-64 | europe-west4-a | 32 GB | `v6e_s/m/l.yaml` |
| `launch_tpu_v5e.sh` | v5e-64 | europe-west4-b | 16 GB | `v5e_s/m/l.yaml` |

### Scaling Roadmap

| Phase | Scale | Pod | Status |
|-------|-------|-----|--------|
| Pipeline validation | 3.9M | v6e-8 | Done |
| v6 baseline | 36.1M | v6e-8 | Done — loss 1,310 |
| **v6.1 cross-attn macro** | **36.6M** | **v6e-8** | **Done — loss 908** |
| Scaling laws (S) | 50M | v6e-64 | Next |
| Scaling laws (M) | 300M | v6e-64 | Planned |
| Full scale | 1B+ | v5p-128+ | Target |

## 7. Project Structure

```
ChaosAI/
├── src/
│   ├── strate_i/              # SymLog + FSQ tokenizer (PyTorch)
│   ├── strate_ii/             # FinJEPA — Mamba-2 + OT-CFM (PyTorch)
│   ├── strate_iv/             # TD-MPC2 + CVaR agent (PyTorch)
│   └── jax_v6/                # JAX/TPU-native implementation (primary)
│       ├── encoders/          # Chunked SSD (float32 accum), Mamba-2 + cross-attn macro
│       ├── predictors/        # Predictor + FlowPredictor + KANPredictor
│       ├── losses/            # VICReg + Barlow Twins (float32 covariance)
│       ├── training/          # TrainState, Auto-Sharder 2D mesh, JIT train step
│       ├── data/              # InMemoryLoader + Grain async pipeline (ArrayRecord)
│       ├── strate_i/          # JAX-native FSQ tokenizer (3.5M samples/sec on TPU)
│       ├── strate_iv/         # TD-MPC2 JAX + Multiverse Crossing
│       │   ├── multiverse_crossing.py  # Geodesic perturbations + convergence analysis
│       │   ├── tdmpc2.py      # Agent: MPPI + bifurcation-modulated CQL (v6.1)
│       │   ├── env_allweather.py       # All-weather DQN environment
│       │   ├── env_cross_sectional.py  # Market-neutral cross-sectional DQN (v6.1 risk-parity)
│       │   └── replay_buffer.py        # Async double-buffered jax.device_put
│       ├── config.py          # All hyperparameters (frozen dataclasses + YAML + dacite)
│       └── jepa.py            # FinJEPA model (encoder + predictor + CFM)
├── scripts/
│   ├── predict_btc_1week.py   # Live BTC regime prediction (Binance → JEPA → Multiverse)
│   ├── train_cross_sectional.py  # Cross-sectional DQN training (PER + risk-parity, v6.1)
│   ├── precompute_rl_buffer.py   # RL buffer with bifurcation_index (v6.1)
│   ├── trc_data_manager.sh    # Data lake orchestrator (stage/backup/cleanup/status)
│   ├── launch_tpu_v6e.sh      # v6e-64 launch (europe-west4-a)
│   ├── launch_tpu_v5e.sh      # v5e-64 launch (europe-west4-b)
│   ├── run_training.py        # Main JAX training loop (env-var driven)
│   ├── download_bulk_free.py  # High-speed async downloader (aiohttp, ~1 Gbps)
│   ├── build_graphs.py        # GNN graph construction for on-chain embeddings
│   └── generate_optimal_config.py  # Chinchilla-optimal config generator
├── configs/
│   └── scaling/               # v6e: xs/26m/30m/38m_v2/38m_v3/54m/s/m/l
│                              # v5e: s/m/l  |  legacy: s_15m/m_150m/l_1b/xl_7b
└── results/
    ├── multiverse_crossing_30u.json  # 30-universe Multiverse Crossing results
    └── btc_prediction.json           # Latest BTC regime prediction
```

## 8. Quick Start (TPU v6e)

```bash
# 0. Stage data from Drive → GCS
./scripts/trc_data_manager.sh stage

# 1. Create TPU v6e-8 VM (same region as bucket)
gcloud compute tpus tpu-vm create fin-ia-v6e \
    --zone=europe-west4-a --accelerator-type=v6e-8 \
    --version=v2-alpha-tpuv6e --spot

# 2. Deploy code + setup dependencies
gcloud compute tpus tpu-vm scp --recurse . fin-ia-v6e:~/Financial_IA --zone=europe-west4-a
gcloud compute tpus tpu-vm ssh fin-ia-v6e --zone=europe-west4-a \
    --command="bash Financial_IA/scripts/setup_tpu_vm.sh"

# 3. Launch training (v6.1 with cross-attention macro)
export SCALE_CONFIG=configs/scaling/v6e_38m_v3.yaml
export SCALE_TIER=38m_v3
export GCS_BUCKET=gs://fin-ia-eu
export MACRO_CONTEXT_PATH=data/macro/macro_context.npz
nohup python3 -u scripts/run_training.py > logs/train.log 2>&1 &

# 4. Live BTC regime prediction
PYTHONPATH=. python3 scripts/predict_btc_1week.py \
    --jepa_ckpt checkpoints/jax_v6e/38m_v3/92112 \
    --strate_i_ckpt checkpoints/strate_i_jax_combined/best_params.npz

# 5. After training: backup checkpoints, wipe GCS
./scripts/trc_data_manager.sh backup --latest
./scripts/trc_data_manager.sh cleanup --force
```

## 9. Tech Stack

| Component | Technology |
|---|---|
| Training (primary) | JAX + Flax + Optax |
| SSM kernel | Custom Chunked SSD (bf16 + f32 accum) |
| ODE integration | Diffrax |
| Optimal transport | Sinkhorn (JIT-compatible) |
| Data pipeline | InMemoryLoader + Google Grain + ArrayRecord |
| Parallelism | GSPMD (NamedSharding, Auto-Sharder) |
| Checkpointing | Orbax |
| Training (GPU legacy) | PyTorch + Lightning |
| RL agent | TD-MPC2 + Multiverse Crossing (JAX/Flax) |
| Data Lake | Google Drive (Cold, free) + GCS (Hot) + rclone |
| FinOps | `trc_data_manager.sh` (stage / backup / cleanup / status) |
| Infrastructure | GCP TPU v6e / v5e (TRC program) |

## 10. v6.1 Improvements

Five architectural improvements shipped in v6.1 (March 2026):

| Feature | Description | Impact |
|---------|-------------|--------|
| **Cross-Attention Macro** | Macro context (FRED+COT) conditions prefix token via single-head cross-attn instead of broadcast gated addition | -31% JEPA loss vs v6 |
| **Bifurcation-Modulated CQL** | CQL penalty scales with `bifurcation_index` (eigenvalue entropy) — chaos → conservative | More robust offline RL |
| **Selective Remat** | `remat_policy: "full"` or `"dots"` — finer control over activation checkpointing | Memory/speed tradeoff |
| **Risk-Parity Reward** | Divide per-asset returns by vol proxy before cross-sectional z-scoring | Regime-invariant signal |
| **Priority Experience Replay** | Sample chaotic transitions (high `bifurcation_index`) ~3× more | Better regime coverage |

## 11. Multiverse Crossing — Key Innovation

The **Multiverse Crossing** mechanism is a novel approach to uncertainty quantification for decision-making under chaos:

1. **Perturb**: M=30 copies of the JEPA context `h_x` are created with geodesic perturbations on the representation hypersphere (tangent-plane projection + L2 re-normalization — no naive additive noise that escapes the manifold)
2. **Propagate**: Each perturbed context generates futures via the OT-CFM predictor
3. **Analyze convergence**: Inter-universe agreement measured via:
   - **Convergence score** ∈ [0,1]: ratio of intra-universe noise to inter-universe divergence
   - **Bifurcation index**: eigenvalue entropy of the Gram matrix — effective number of distinct regimes
   - **Lyapunov proxy**: exponential divergence rate from initial perturbation
4. **Act dynamically**: CVaR alpha scales with convergence — agree → aggressive, disagree → conservative/flat

**Empirical results (fresh data, Iran crisis, March 2026):** Sharpe **2.78** with ROCK SOLID consensus across 30 universes (CGPT, LAB, TURBO long; BNT, AERGO, CETUS short). Lyapunov **-0.73** indicates stable attractor, zero contested assets.

## 12. References

- LeCun, *A Path Towards Autonomous Machine Intelligence* (JEPA, 2022)
- Gu & Dao, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2 / SSD, 2024)
- Lipman et al., *Flow Matching for Generative Modeling* (CFM, 2022)
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport* (OT-CFM, 2023)
- Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control* (2024)
- Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple* (FSQ, 2023)
- Bardes et al., *VICReg: Variance-Invariance-Covariance Regularization* (2022)
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023)
- Liu et al., *KAN: Kolmogorov–Arnold Networks* (2024)

## License

[GNU Affero General Public License v3.0](LICENSE) — any use of this code, including as a network service, requires publishing the complete source of derivative works under the same license.
