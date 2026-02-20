# ChaosAI — General Time-Series Foundation Model

> **Architecture:** Exo-Clocked Mamba-2 SSD + OT-CFM + VICReg JEPA + Multiverse Crossing
> **Framework:** JAX/Flax (TPU-native) + PyTorch (GPU validation)
> **Scale:** 15M → 7B params, T-Shirt sizing S/M/L/XL (TPU v5p-8 → v5p-768)
> **Status:** Production-ready — Data Lake on Drive, zero-cost GCS when idle

> **DISCLAIMER: RESEARCH PURPOSE ONLY**
>
> This is an open-source research project exploring self-supervised learning on chaotic dynamical systems, using financial time series as a high-noise benchmark. **It is NOT a trading system.**
>
> - Nothing in this repository constitutes financial advice.
> - The authors are not responsible for any financial losses incurred by using this code.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![JAX](https://img.shields.io/badge/JAX-TPU%20v5p--8-blue?logo=google&logoColor=white)](https://github.com/google/jax)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org/)

## 1. Why Financial Markets?

Financial markets are the most adversarial, non-stationary, and noise-dominated dynamical system publicly accessible at high frequency. They exhibit fat-tailed distributions, regime shifts, correlated chaos across 400+ instruments, and adversarial feedback loops.

We chose this domain as a **proving ground** — if a self-supervised architecture can learn meaningful latent representations here, it transfers to less adversarial time-series domains (biomedical signals, geophysics, audio, video dynamics) where the signal-to-noise ratio is orders of magnitude better.

## 2. Architecture (4 Strata)

### Strate I — Perception (SymLog + FSQ Tokenizer)
Compresses raw OHLCV candles into discrete codes via Dilated Causal CNN + **SymLog** (preserves fat tails) + **Finite Scalar Quantization** (1024 codes, zero codebook collapse).

### Strate II — World Model (Exo-Clocked Mamba-2 JEPA)
Self-supervised Joint-Embedding Predictive Architecture with:
- **Mamba-2 SSD** (State Space Duality) — chunked dense matmuls replacing selective scan. Custom JAX kernel optimized for 128×128 MXU tiles.
- **Volume-Conditioned Delta** — the SSM discretization step `dt` is modulated by realized volatility, giving the model an adaptive temporal clock (processes volatile regimes at finer granularity).
- **VICReg** self-supervised objective with block masking — no labels, no lookahead bias.

### Strate III — Multiverse Predictor (OT-CFM)
**Optimal Transport Continuous Flow Matching** — Sinkhorn coupling + diffrax ODE integration to generate multimodal future trajectories in latent space. Replaces the MLP predictor with a velocity field trained via OT interpolants.

### Strate IV — Agent (TD-MPC2 + CVaR + Multiverse Crossing)
Model-based RL agent that plans inside the differentiable latent space:
- **TD-MPC2** with MPPI tree search in latent space (unrolled for XLA kernel fusion)
- **Distributional critic** (ensemble quantile) optimizing **CVaR** — dynamic alpha driven by multiverse convergence
- **Multiverse Crossing** — M=5 perturbed copies of the JEPA context, each generating N=16 futures. Inter-universe convergence analysis (eigenvalue bifurcation, Lyapunov proxy) drives risk dynamically: convergence → aggressive, divergence → conservative/no trade

## 3. JAX/TPU Implementation

The primary training path is JAX-native (`src/jax_v6/`), designed for TPU pods.

### TPU-Native Design
- **Chunked SSD kernel** (`ssd.py`) — bf16 compute with **float32 accumulation** (cumsum, h_final) to prevent NaN from mantissa truncation. Weekend gating, causal masking. Fills MXU tiles exactly (chunk_size=128, head_dim=128).
- **Auto-Sharder Topology-Aware** (`sharding.py`) — 2D mesh `('data', 'fsdp')` via `create_device_mesh`, maps virtual axes onto physical TPU 3D torus minimizing ICI hops. Auto-detects topology from v5p-8 `(8,1)` to v5p-768 `(192,4)`. Params/optimizer → FSDP, batch → data-parallel, RNG → per-replica.
- **GSPMD** parallelism via `jax.sharding.Mesh` + `NamedSharding`. Zero `pmap`.
- **Grain** async multi-host data pipeline — ArrayRecord format, prefetch buffer depth 128, double-buffered async `jax.device_put`.
- **Diffrax** ODE integration with Sinkhorn OT coupling (JIT-compatible, no scipy).
- **Conditional remat** — gradient checkpointing on/off per hardware (v6e: on, v5p: off).
- **Production XLA flags** (`LIBTPU_INIT_ARGS`) — 13 flags: async collective fusion, latency hiding scheduler, aggressive loop fusion, experimental cost model.

### Validated on TPU v6e-8 (February 2026)
The v6e (Trillium) is designed for inference, not training — with only 31 GB HBM per chip and oversized MXU relative to memory bandwidth. It was the only TPU type available to us without TRC quota. We ran on it anyway to validate the full JAX pipeline end-to-end and prepare the codebase for proper training hardware (v5p).

- 3.9M params (d_model=256), 8 Trillium chips
- 2,700 training steps: loss 7448 → 4839 (-35%), CFM loss 2.17 → 1.45 (-33%)
- 0.3 steps/s with remat + synchronous data loading
- ~5% MXU utilization (memory-bound at 29/31 GB HBM)

### T-Shirt Scaling (MXU 128×128 aligned)

All configs enforce `head_dim = d_model × expand / n_heads = 128` for perfect MXU tile filling.

| Size | Params | d_model | Layers | Heads | Pod | Batch | Chinchilla Tokens |
|------|--------|---------|--------|-------|-----|-------|-------------------|
| **S** | 15M | 256 | 12 | 4 | v5p-8 | 2,048 | 300M |
| **M** | 150M | 1,024 | 24 | 16 | v5p-32 | 8,192 | 3B |
| **L** | 1B | 2,048 | 48 | 32 | v5p-128 | 16,384 | 20B |
| **XL** | 7B | 4,096 | 32 | 64 | v5p-768 | 49,152 | 140B |

Generate custom configs: `python scripts/generate_optimal_config.py --target_pod v5p-32 --total_tokens 20B`

## 4. Training Results (v6e-8)

![Training Curves — TPU v6e-8](results/training_curves.png)

| Metric | Start | End (step 2700) | Change |
|---|---|---|---|
| Total loss | 7448 | 4839 | -35% |
| CFM loss | 2.17 | 1.45 | -33% |
| Steps/s | 0.3 | 0.3 | stable |

Training diverged at step 2750 (NaN) due to missing gradient clipping — now fixed with `optax.clip_by_global_norm(1.0)` and float32 accumulation in the SSD kernel.

## 5. Data Lake & FinOps

We use a 3-tier data lake architecture to manage large datasets (30 TB+) while keeping costs at zero when idle.

```
Google Drive (30 TB Cold, free)  ←→  GCS Bucket (Hot, paid)  ←→  TPU VM (Compute)
         rclone 64×parallel              gsutil / Grain
```

### Data Manager (`scripts/trc_data_manager.sh`)

| Command | Direction | Description |
|---------|-----------|-------------|
| `stage` | Drive → GCS | High-perf rclone transfer: 64 parallel streams, 128M chunks, `--fast-list`. Region safety check before transfer. |
| `backup` | GCS → Drive | Archives checkpoints to `.tar.gz` before upload (Orbax generates thousands of small files per TPU chip — sending raw would hit Drive API rate limits). |
| `cleanup` | GCS → /dev/null | Wipes all data + checkpoints from GCS. Confirmation prompt (`--force` for automation). Resets billing to $0. |
| `status` | — | Per-prefix storage breakdown + estimated monthly cost + region match verification. |

### Cost Model

| Resource | Cost | Notes |
|----------|------|-------|
| Google Drive (Cold) | $0/month | Included in Google One / Workspace |
| GCS Standard storage | ~$0.02/GB/month | Emptied after each training run → $0 when idle |
| GCS ↔ TPU transfer | $0 | **Only if same region** — inter-region egress is billed per GB |
| GCS ↔ Drive transfer | $0 (egress) | rclone uses Drive API, no GCS egress charges |

### Drive Folder Structure
```
ChaosAI_DataLake/
├── 01_raw_ohlcv/                              # Raw OHLCV candles (Binance Futures 1h)
├── 02_tokens/                                 # Tokenized sequences (FSQ codes)
├── 03_training_ready/
│   └── arrayrecords_v5/                       # 433 ArrayRecord shards (Grain-ready)
├── 04_checkpoints/
│   └── jax_v6_v6e/                            # Archived checkpoints (tar.gz per run)
│       └── checkpoints_jax_v6_v6e_all_steps.tar.gz
└── 05_results/                                # Training curves, ablation results
```

### Standard Workflow
```bash
# Before training — stage data to GCS (~2 min for 60 MB, scales linearly)
./scripts/trc_data_manager.sh stage

# Train on TPU (hours to days)
nohup bash scripts/launch_tpu_v5p.sh m &

# After training — save checkpoints to Drive, then wipe GCS
./scripts/trc_data_manager.sh backup --latest --scale m
./scripts/trc_data_manager.sh cleanup --force

# Verify: $0/month
./scripts/trc_data_manager.sh status
```

## 6. Infrastructure & Cost

| Resource | Value | Notes |
|----------|-------|-------|
| **GCP Project** | `financial-ai-487700` (ChaosAI) | |
| **GCS Bucket** | `gs://fin-ia-bucket` | Region: `europe-west4`, Class: Standard |
| **TPU Zone** | `europe-west4-a` | Co-located with bucket → zero egress |
| **Compute** | TPU Research Cloud (TRC) | Preemptible v5p pods, free via application |
| **Current cost** | **$0/month** | All data on Drive, GCS emptied after training |

### GCP Resource Audit (February 2026)

All legacy resources have been cleaned up:

| Resource | Status | Monthly savings |
|----------|--------|-----------------|
| `financial-ia-ingest` (e2 VM + 50 GB SSD) | Deleted | -$8.50 |
| `financial-ia-training` (H100 VM + 200 GB SSD) | Deleted | -$34.00 |
| GCS bucket data (285 MB) | Moved to Drive | -$0.01 |
| **Total** | **All on Drive** | **-$42.50/month** |

## 7. Long-Term Vision

The architecture is domain-agnostic — the tokenizer converts raw signals to discrete codes, and everything downstream operates on abstract token sequences. Planned extensions:

- **Biomedical signals** — EEG/ECG anomaly detection (seizure onset, arrhythmia)
- **Geophysics** — gravitational wave detection (LIGO), seismic precursors
- **Audio** — bioacoustics, speech dynamics
- **Video** — frame-level latent prediction via JEPA spatial-temporal masking

## 8. Project Structure

```
ChaosAI/
├── src/
│   ├── strate_i/              # SymLog + FSQ tokenizer (PyTorch)
│   ├── strate_ii/             # FinJEPA v6 — Mamba-2 + OT-CFM (PyTorch)
│   ├── strate_iv/             # TD-MPC2 + CVaR agent (PyTorch)
│   └── jax_v6/                # JAX/TPU-native implementation
│       ├── encoders/          # Chunked SSD (float32 accum), Mamba-2, causal conv
│       ├── predictors/        # Predictor + FlowPredictor (Diffrax + Sinkhorn)
│       ├── losses/            # VICReg (float32 covariance)
│       ├── training/          # TrainState, Auto-Sharder 2D mesh, JIT train step
│       ├── data/              # Grain async multi-host loader (ArrayRecord)
│       ├── strate_iv/         # TD-MPC2 JAX + Multiverse Crossing
│       │   ├── multiverse_crossing.py  # Convergence analysis (M=5 universes)
│       │   ├── tdmpc2.py      # Agent: MPPI + dynamic CVaR (Flax)
│       │   ├── world_model.py # Latent dynamics + reward head (Flax)
│       │   ├── critic.py      # Ensemble quantile critic (Flax)
│       │   ├── actor.py       # Policy network (Flax)
│       │   ├── env.py         # Gymnasium env with convergence obs
│       │   └── replay_buffer.py  # Async double-buffered jax.device_put
│       ├── config.py          # All hyperparameters (StrateIIConfig + StrateIVJAXConfig)
│       └── jepa.py            # FinJEPA model (encoder + predictor + CFM)
├── scripts/
│   ├── trc_data_manager.sh    # Data lake orchestrator (stage/backup/cleanup/status)
│   ├── launch_tpu_v5p.sh      # Full TPU v5p pipeline (T-Shirt scale selector)
│   ├── tpu_train_pipeline.sh  # Training launcher (XLA flags, async prefetch, GCS sync)
│   ├── generate_optimal_config.py  # Chinchilla-optimal config generator
│   ├── setup_tpu_vm.sh        # TPU VM dependency setup
│   └── convert_pt_to_arrayrecord.py
├── configs/
│   ├── scaling/               # T-Shirt configs: s_15m, m_150m, l_1b, xl_7b
│   ├── mini_10m.yaml          # 10M mini-model (Strate IV multiverse validation)
│   └── strate_*.yaml          # Per-strate configurations
└── infra/                     # GCP startup scripts
```

## 9. Quick Start (TPU)

```bash
# 0. Stage data from Drive → GCS (from local machine with rclone configured)
./scripts/trc_data_manager.sh stage

# 1. Create TPU VM (same region as bucket — europe-west4!)
gcloud compute tpus tpu-vm create chaosai \
    --zone=europe-west4-a --accelerator-type=v5p-8 \
    --version=tpu-vm-tf-2.16.1-pjrt --preemptible

# 2. Deploy code + setup dependencies
gcloud compute tpus tpu-vm scp --recurse . chaosai:~/Financial_IA --zone=europe-west4-a
gcloud compute tpus tpu-vm ssh chaosai --zone=europe-west4-a \
    --command="bash Financial_IA/scripts/setup_tpu_vm.sh"

# 3. Launch training (T-Shirt scale: s/m/l/xl or legacy 184m/500m/1_5b/3b)
gcloud compute tpus tpu-vm ssh chaosai --zone=europe-west4-a \
    --command="nohup bash Financial_IA/scripts/launch_tpu_v5p.sh 184m &"

# 4. After training: backup checkpoints to Drive, wipe GCS
./scripts/trc_data_manager.sh backup --latest --scale m
./scripts/trc_data_manager.sh cleanup --force

# 5. Generate custom scaling config
python scripts/generate_optimal_config.py --target_pod v5p-32 --total_tokens 20B
```

## 10. Tech Stack

| Component | Technology |
|---|---|
| Training (primary) | JAX + Flax + Optax |
| SSM kernel | Custom Chunked SSD (bf16 + f32 accum) |
| ODE integration | Diffrax |
| Optimal transport | Sinkhorn (JIT-compatible) |
| Data pipeline | Google Grain + ArrayRecord |
| Parallelism | GSPMD (NamedSharding, Auto-Sharder) |
| Checkpointing | Orbax |
| Training (GPU legacy) | PyTorch + Lightning |
| RL agent | TD-MPC2 + Multiverse Crossing (JAX/Flax) |
| Data Lake | Google Drive (Cold, free) + GCS (Hot) + rclone |
| FinOps | `trc_data_manager.sh` (stage / backup / cleanup / status) |
| Infrastructure | GCP TPU v5p-8 → v5p-768 (TRC program) |

## 11. References

- LeCun, *A Path Towards Autonomous Machine Intelligence* (JEPA, 2022)
- Gu & Dao, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2 / SSD, 2024)
- Lipman et al., *Flow Matching for Generative Modeling* (CFM, 2022)
- Tong et al., *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport* (OT-CFM, 2023)
- Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control* (2024)
- Mentzer et al., *Finite Scalar Quantization: VQ-VAE Made Simple* (FSQ, 2023)
- Bardes et al., *VICReg: Variance-Invariance-Covariance Regularization* (2022)
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023)

## 12. Multiverse Crossing — Key Innovation

The **Multiverse Crossing** mechanism is a novel approach to uncertainty quantification for decision-making under chaos:

1. **Perturb**: M=5 copies of the JEPA context `h_x` are created with small Gaussian perturbations `ε ~ N(0, σ²I)`, simulating nearby initial conditions
2. **Propagate**: Each perturbed context generates N=16 future trajectories via the OT-CFM predictor → 80 total trajectories per episode
3. **Analyze convergence**: At each decision step, inter-universe agreement is measured via:
   - **Convergence score** ∈ [0,1]: ratio of intra-universe noise to inter-universe divergence
   - **Bifurcation index**: effective number of distinct regimes (eigenvalue entropy of Gram matrix)
   - **Lyapunov proxy**: exponential divergence rate from initial perturbation
4. **Act dynamically**: CVaR alpha scales with convergence — agree → aggressive, disagree → conservative/no trade

This is directly inspired by **sensitivity to initial conditions** in chaotic dynamical systems. Instead of a single forecast, the agent observes how its predictions diverge across parallel universes to calibrate risk in real time.

## License

[GNU Affero General Public License v3.0](LICENSE) — any use of this code, including as a network service, requires publishing the complete source of derivative works under the same license.
