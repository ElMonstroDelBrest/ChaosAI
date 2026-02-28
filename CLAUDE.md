# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Environment

```bash
uv venv && source .venv/bin/activate
uv sync                    # core deps (PyTorch, mamba-ssm, gymnasium)
uv sync --extra dev        # + pytest, pytest-cov
uv sync --extra gcp        # + google-cloud-storage, pyarrow, aiohttp
export PYTHONPATH="$PWD"   # required — src/ uses relative imports from root
```

JAX/TPU dependencies (jax, flax, optax, grain, diffrax, orbax) are installed separately on TPU VMs via `scripts/setup_tpu_vm.sh`. They are not in pyproject.toml.

## Tests

```bash
pytest                              # all tests (testpaths=["tests"], -v --tb=short)
pytest tests/test_revin.py -v       # single test file
pytest tests/test_jepa.py -k "test_forward" -v  # single test function
pytest --cov=src tests/             # with coverage
```

Tests are PyTorch-only (no JAX required locally). The JAX code (`src/jax_v6/`) can be validated without JAX installed via AST parsing:
```bash
python -c "import ast; [ast.parse(open(f).read()) for f in __import__('glob').glob('src/jax_v6/**/*.py', recursive=True)]; print('OK')"
```

## Architecture — 4 Strata Pipeline

The model is a cascaded pipeline where each stratum consumes the output of the previous one:

```
Raw OHLCV → [Strate I: FSQ Tokenizer] → discrete codes
         → [Strate II: Mamba-2 JEPA]  → latent embeddings
         → [Strate III: OT-CFM]       → N future trajectories
         → [Strate IV: TD-MPC2 Agent] → trading actions
```

**Two parallel implementations exist:**
- `src/strate_i/`, `src/strate_ii/`, `src/strate_iv/` — PyTorch + Lightning (GPU dev/validation)
- `src/jax_v6/` — JAX/Flax (TPU production training)

These are independent codebases with the same architecture. Do not mix frameworks in a single module.

### Key Model Files

| Component | PyTorch | JAX/Flax |
|-----------|---------|----------|
| Mamba-2 SSD kernel | `src/strate_ii/mamba2_block.py` | `src/jax_v6/encoders/ssd.py` + `mamba2_block.py` |
| JEPA (encoder + predictor) | `src/strate_ii/jepa.py` | `src/jax_v6/jepa.py` |
| VICReg loss | `src/strate_ii/vicreg.py` | `src/jax_v6/losses/vicreg.py` |
| CFM predictor | `src/strate_ii/flow_predictor.py` | `src/jax_v6/predictors/flow_predictor.py` |
| KAN predictor / velocity field | — | `src/jax_v6/predictors/kan_layer.py` |
| TD-MPC2 agent | `src/strate_iv/tdmpc2.py` | `src/jax_v6/strate_iv/tdmpc2.py` |
| Multiverse Crossing | — | `src/jax_v6/strate_iv/multiverse_crossing.py` |
| Auto-Sharder (GSPMD) | — | `src/jax_v6/training/sharding.py` |
| Strate I tokenizer (JAX) | `src/strate_i/tokenizer.py` | `src/jax_v6/strate_i/tokenizer.py` |
| Strate I transforms (JAX) | `src/strate_i/data/transforms.py` | `src/jax_v6/strate_i/transforms.py` |
| Macro context loader | — | `src/jax_v6/data/macro_loader.py` |

### Config System

Frozen dataclasses + YAML + dacite. Each strate has its own config:

```python
from src.strate_ii.config import load_config
config = load_config("configs/strate_ii.yaml")  # returns StrateIIConfig (frozen dataclass)
config.mamba2.d_model  # 512
```

- `src/strate_i/config.py` — `StrateIConfig` (tokenizer)
- `src/strate_ii/config.py` — `StrateIIConfig` (JEPA + predictor + VICReg)
- `src/strate_iv/config.py` — `StrateIVConfig` (env + PPO + TD-MPC2)
- `src/jax_v6/config.py` — `StrateIIConfig` (JAX mirror) + `StrateIVJAXConfig` (multiverse crossing)
- `configs/scaling/*.yaml` — T-Shirt size presets: v5p (S=15M→XL=7B), v6e (S=50M→L=1B), v5e (S=50M→L=1B)

To add a hyperparameter: add field to dataclass → add to YAML → use via `config.section.field`.

## Training Pipeline

Sequential phases, each depends on the previous:

```bash
# Phase 1: Download data (432 Binance Futures pairs, 1h candles)
python scripts/download_massive_data.py --interval 1h --output_dir data/raw

# Phase 2: Train Strate I tokenizer
python scripts/train_strate_i.py --config configs/strate_i.yaml

# Phase 3a: Pre-tokenize all OHLCV with trained Strate I (old — writes .pt files, huge disk usage)
python scripts/pretokenize.py --checkpoint checkpoints/strate-i-*.ckpt

# Phase 3b: Pre-tokenize directly to ArrayRecord (preferred — 1670x more compact)
python scripts/pretokenize_to_arrayrecord.py --checkpoint checkpoints/strate-i-*.ckpt \
    --data_dir data/ohlcv_1m/ --output_dir data/arrayrecord_1m/ --seq_len 128

# Phase 4: Train Strate II world model (self-supervised)
python scripts/train_strate_ii.py --config configs/strate_ii.yaml --compile

# Phase 5: Pre-compute multiverse trajectory buffer for RL
python scripts/precompute_trajectories.py --strate_i_checkpoint ... --strate_ii_checkpoint ...

# Phase 6: Train Strate IV agent
python scripts/train_strate_iv.py --mode tdmpc2  # or --mode ppo

# Full automated pipeline:
./scripts/train_v5_pipeline.sh --skip-download --start-phase=2
```

### TPU Training (JAX)

```bash
# Data management (Drive ↔ GCS)
./scripts/trc_data_manager.sh stage              # Drive → GCS
./scripts/trc_data_manager.sh cleanup --force    # GCS → $0/month

# v6e-64 (europe-west4-a, 32 GB HBM/chip, Tore 2D 8×8)
./scripts/launch_tpu_v6e.sh --scale=s            # ~50M quick test
./scripts/launch_tpu_v6e.sh --scale=m            # ~300M run principal
./scripts/launch_tpu_v6e.sh --scale=l --resume   # ~1B, resume from GCS

# v5e-64 (europe-west4-b, 16 GB HBM/chip, Tore 2D)
./scripts/launch_tpu_v5e.sh --scale=s            # ~50M quick test
./scripts/launch_tpu_v5e.sh --scale=m            # ~200M run principal

# v5p (legacy, quota non disponible actuellement)
nohup bash scripts/launch_tpu_v5p.sh --scale=m &

# Override pod size (sharding auto-adapte)
TPU_TYPE=v6e-16 ./scripts/launch_tpu_v6e.sh --scale=s

# Generate custom scaling config
python scripts/generate_optimal_config.py --target_pod v5p-32 --total_tokens 20B
```

## Critical Design Constraints

**MXU 128x128 alignment (TPU):** All model dimensions must produce `head_dim = d_model * expand_factor / n_heads = 128`. This fills the TPU MXU tiles exactly. Breaking this wastes 50%+ of compute.

**bf16 numerical stability:** The SSD kernel accumulates state via cumsum. bf16 has only ~7 mantissa bits, causing NaN around step 2750. All temporal accumulations (cumsum, h_final decay) use float32 intermediates — do not change this.

**Clock modulation bounds:** Exo-clock and vol-clock bias on dt (pre-softplus) is bounded via `dt_max_delta * tanh(raw)` (default ±2.0). This prevents SSM state explosion or gradient collapse. Do not remove the tanh constraint.

**On-manifold perturbation:** `multiverse_crossing.py:perturb_latent()` uses geodesic perturbation (tangent-plane projection + L2 re-normalization) to keep perturbed latents on the JEPA representation hypersphere. Do not replace with naive additive noise.

**GCS region co-location:** Bucket `gs://fin-ia-eu` and TPU zone must be in the same region (`europe-west4`). Inter-region egress is billed per GB on every Grain batch load.

**PYTHONPATH required:** All imports assume `PYTHONPATH=$PWD`. Without it, `from src.strate_ii.config import ...` will fail.

**KAN layer (B-spline):** `src/jax_v6/predictors/kan_layer.py` — drop-in replacements for `Predictor` (`KANPredictor`) and `FlowPredictor` (`KANFlowPredictor`). Cox-de Boor recurrence via `jax.lax.scan` with static shapes; all basis computation in float32 (same bf16 upcast strategy as SSD kernel). `out_features` must be a multiple of 128 (MXU alignment). Grid update (adaptive knot placement) via pure function `update_grid(params, x_samples, grid_size)` — call every ~200 steps. Do not replace the `lax.scan` with a Python loop: the Python loop over `order` is only for the static `t_r_stack` precompute at trace time.

## Data

- `data/raw_1m_parquet/` — 419 `.parquet` files (Binance Futures 1m candles, ~25 GB, 509M candles)
- `data/ohlcv_stocks_daily/` — 7,631 `.pt` files (US stocks daily, ~547 MB)
- `data/ohlcv_sp500/` — 505 `.pt` files (S&P 500 daily)
- `data/ohlcv_forex/` — 15 `.pt` files (major forex pairs)
- `data/ohlcv_commodities/` — 8 `.pt` files
- `data/ohlcv_stocks_1h/` — 65 `.pt` files (ETFs/indices hourly)
- `data/arrayrecord_multi/` — Multi-source ArrayRecord shards (8,650 shards, 3.98M records, 509M tokens)
- `data/macro/macro_context.npz` — 2,116 macro signals aligned (7,552 timesteps × 2,116 signals)
- `data/trajectory_buffer_v5/` — pre-computed RL trajectories

Data is stored on Google Drive (`drive:ChaosAI_DataLake/`) and staged to GCS only during training. See `scripts/trc_data_manager.sh`.

## Infrastructure

- **GCP Project:** `financial-ai-487700`
- **GCS Bucket:** `gs://fin-ia-eu` (europe-west4, Standard)
- **TPU Zone principale:** `europe-west4-a` ← co-localisé avec GCS → transfert **GRATUIT**
- **IaC:** `infra/` (Terraform)

### Quota TRC actuel (email reçu 2026-02-23, valide 30 jours)

| Chips | Type | Zone | Co-localisé GCS ? | Coût egress |
|-------|------|------|-------------------|-------------|
| 64 spot | **v6e** | **europe-west4-a** | ✅ OUI | **Gratuit** ← USE THIS |
| 64 spot | v5e | europe-west4-**b** | ⚠️ Zone différente (-b vs -a) | Gratuit (même région) |
| 64 spot | v6e | us-east1-d | ❌ NON | ~$0.08/GB inter-région |
| 64 spot | v5e | us-central1-a | ❌ NON | ~$0.08/GB inter-région |
| 32 spot | v4 | us-central2-b | ❌ NON | ~$0.08/GB inter-région |
| 32 on-demand | v4 | us-central2-b | ❌ NON | ~$0.08/GB inter-région |

**⚠️ Important — v5e ≠ v5p :**
- Les scripts `launch_tpu_v5p.sh` ciblent `--version tpu-vm-v5p-rev47` (architecture différente)
- Le quota TRC est pour **v5e** (TPU v5 "efficient") → `--version tpu-vm-v5e-*`
- **Ne pas utiliser `launch_tpu_v5p.sh` avec le quota v5e** — mauvaise version VM
- Pour v5e-64 zone `europe-west4-b` : même sharding (8,8) que v6e (tore 2D similaire), mais HBM différent

**Règle absolue : toujours créer les TPUs dans les zones co-localisées avec GCS (europe-west4-*).**
Les zones us-* déclenchent de l'egress inter-région facturé à chaque batch Grain.

### Scripts TPU par génération

| Script | TPU | Zone | HBM/chip | Peak BF16/chip | Mesh 64 chips | Configs |
|--------|-----|------|----------|----------------|---------------|---------|
| `launch_tpu_v6e.sh` | v6e-64 | europe-west4-a | 32 GB | ~918 TFLOPS | (8,8) Tore 2D | `v6e_s/m/l.yaml` |
| `launch_tpu_v5e.sh` | v5e-64 | europe-west4-b | 16 GB | ~197 TFLOPS | (8,8) Tore 2D | `v5e_s/m/l.yaml` |
| `launch_tpu_v5p.sh` | v5p-32 | europe-west4-a | 95 GB | ~459 TFLOPS | (16,4) Tore 3D | `s_15m..xl_7b.yaml` |

**Auto-Sharder** (`sharding.py`) : détecte automatiquement la génération TPU via `device_kind` et route vers le mesh optimal (Tore 2D pour v6e/v5e, Tore 3D pour v5p). `TPU_TYPE` overridable via env var.

**Disponibilité spot TPU** : les créneaux les moins demandés sont **1h-7h CET** (nuit EU) et les **week-ends**. Le pire : 14h-22h CET (labs EU + US East actifs simultanément).

## Training Runs Log

### Run: Multi-source Chinchilla-optimal on TPU v6e-8 (2026-02-28)

**Config**: `configs/scaling/v6e_multi.yaml` — d_model=512, n_layers=12, n_heads=8, 22.4M params
**Data**: 8,650 assets (419 Binance Futures 1min + 7,631 stocks daily + 505 S&P 500 + 15 forex + 8 commodities + 65 ETFs hourly + 8 crypto YF)
**Tokens**: 509M (Chinchilla ratio: 509M / 22.4M = 22.7× ≈ optimal 20×)
**TPU**: `fin-ia-v6e` (v6e-8, 8 chips, europe-west4-a), Mesh 2D (8,1) DP pur
**Schedule**: batch=512, 7,275 steps/epoch, 21,825 total steps, warmup=7,275 (1 epoch), SGDR n_restarts=2
**Features**: exo_clock_dim=4 (RV, Volume, VIX pad, CreditSpread pad), macro_dim=48 (top-k COT signals, gated fusion bias=-2.0)

**Full pipeline executed on TPU VM (84 min total)**:
1. Strate I JAX retrained on multi-source (3 epochs, val=0.069, 7 min, 3.5M samples/sec)
2. TPU-native pretokenize: 509M candles in 12.3s (41.5M candles/sec), 8,650 shards in 1.7 min
3. Strate II 22.4M Fin-JEPA training: 21,825 steps in 75 min, 0.20s/step, 322K tok/s

**Results**:
| Step | Loss | Epoch | Note |
|------|------|-------|------|
| 50 | 14,422 | 0 | XLA compilation |
| 2,000 | 7,439 | 0 | |
| 5,000 | 5,092 | 0 | |
| 8,000 | 3,311 | 1 | |
| 10,000 | 2,706 | 1 | |
| 14,000 | 1,997 | 1 | **Best** |
| 18,000 | 2,200 | 2 | SGDR restart bump |
| 21,825 | ~2,003 | 2 | Final |

**Comparison with previous 30M run (old tokenizer, 166M tokens)**:
- Previous final loss: 6,400 → New final loss: 2,003 (**-68.7%, ÷3.2**)
- Key improvements: re-trained tokenizer on all asset classes, 3× more tokens (509M vs 166M), macro context fusion

**Key scripts**:
- `scripts/train_strate_i_jax.py` — JAX-native Strate I training (0.81M params, 3.5M samples/sec on TPU)
- `scripts/pretokenize_tpu.py` — TPU-native pretokenizer (41.5M candles/sec, fully batched)
- `scripts/build_macro_graph.py` — Aggregates FRED/COT/alt signals into macro_context.npz

**Strate I JAX port**: `src/jax_v6/strate_i/tokenizer.py` — full encoder + FSQ in Flax, loads from PyTorch .ckpt or JAX .npz checkpoints. Weight conversion handles Conv1d (out,in,k) → JAX (k,in,out) and Linear (out,in) → JAX (in,out) transpositions.

**Data quality fix**: Raw parquet log-returns clipped to ±5 (listing/delisting noise: |log_ret| > 5 = >99.3% move in 1 candle). 130K extreme samples out of 509M.

---

### Run: 30M 1-minute candles on TPU v6e-8 (2026-02-24)

**Config**: `configs/scaling/v6e_30m.yaml` — d_model=512, n_layers=10, n_heads=8, 18.9M params
**Data**: 432 Binance Futures pairs, 1min candles → 253,914 ArrayRecord sequences (seq_len=128)
**TPU**: `fin-ia-v6e` (v6e-8, 8 chips, europe-west4-a), Mesh 2D (8,1) DP pur
**Schedule**: batch=512 (auto-scaled from 4096), 446 steps/epoch, 22,300 total steps, warmup=2,230 steps
**LR**: 3e-4, SGDR with n_restarts=2, weight_decay=0.02, grad_clip=1.0

**Pipeline executed on TPU VM**:
1. Strate I tokenizer trained on TPU via torch_xla (25 epochs, best loss=0.0260)
2. `scripts/pretokenize_to_arrayrecord.py` — combined pretokenize+ArrayRecord (bypasses .pt intermediates: 362 MB vs 604 GB)
3. Strate II 30M Fin-JEPA training via `scripts/run_training.py`

**Results (run 1, DISABLE_CKPT=true)**:
| Step | Loss | Epoch | Note |
|------|------|-------|------|
| 50 | 13,890 | 0 | XLA compilation |
| 500 | 10,441 | 1 | |
| 2,500 | ~4,500 | 5 | |
| 5,000 | 3,420 | 11 | |
| 8,950 | 2,957 | 20 | Previous run diverged here (7,275) |
| 15,000 | ~3,050 | 34 | SGDR restart bump |
| 22,300 | ~2,400 | 49 | Final (not saved — DISABLE_CKPT) |

**Results (run 2, with checkpoints, resumed from step 500)**:
- Same convergence trajectory as run 1
- Checkpoints saved every 500 steps to `checkpoints/jax_v6e/30m/` (orbax, max_to_keep=5)
- Checkpoints synced to GCS `gs://fin-ia-eu/checkpoints/jax_v6e/30m/`
- Training time: ~65 min total

**Bugs fixed this session**:
1. **LR schedule wrong** (`run_training.py`): `warmup_steps`/`total_steps` not passed to `create_train_state()` → used defaults (1000/100000) instead of actual (2230/22300). Root cause of loss divergence at step ~1800.
2. **No max_steps** (`run_training.py`): training loop ran forever. Added `if step >= total_steps: break`.
3. **EMA tau never updated** (`run_training.py`): `compute_tau()` existed but was never called. Added epoch tracking + tau updates.
4. **val_ratio mismatch** (`run_training.py`): dataloader used default 0.2 instead of config's 0.1.
5. **n_restarts too aggressive** (`v6e_30m.yaml`): 4→2 restarts (1300 steps/cycle → 6700 steps/cycle).
6. **Disk full from .pt intermediates**: `pretokenize.py` saves ~2 MB per sequence (604 GB for 432 pairs). Created `pretokenize_to_arrayrecord.py` that writes directly to ArrayRecord (362 MB).
7. **Orbax checkpoint crash** (JAX 0.6.x + orbax 0.11.33): `jax.sharding.set_mesh()` doesn't return a context manager in JAX 0.6.x. Patched `replica_slices.py:99` — replaced `with jax.sharding.set_mesh(mesh):` with `if True:`.
8. **Orbax absolute path required**: `ckpt_dir` was relative → orbax ValueError. Fixed with `os.path.join(os.getcwd(), ...)`.

**Key scripts on TPU VM**:
- `~/launch_30m_save.sh` — launch training with checkpointing (DISABLE_CKPT=false)
- `~/launch_30m_resume.sh` — resume training from latest checkpoint (RESUME=true)

**Orbax patch (must reapply after pip install)**:
```
File: ~/.local/lib/python3.10/site-packages/orbax/checkpoint/_src/serialization/replica_slices.py
Line 99: Replace "with jax.sharding.set_mesh(mesh):" with "if True:  # patched: skip set_mesh (JAX 0.6.x compat)"
```

### TPU VM Dependencies (installed 2026-02-24)

```bash
pip install 'torch~=2.6.0' 'torch_xla[tpu]~=2.6.0' \
  -f https://storage.googleapis.com/libtpu-releases/index.html \
  -f https://storage.googleapis.com/libtpu-wheels/index.html
pip install pytorch-lightning 'jax[tpu]' flax optax grain orbax-checkpoint diffrax
pip install tensorflow-cpu array_record dacite tslearn
# Then apply orbax patch above
```
