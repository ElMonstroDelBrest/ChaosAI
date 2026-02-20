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
| TD-MPC2 agent | `src/strate_iv/tdmpc2.py` | `src/jax_v6/strate_iv/tdmpc2.py` |
| Multiverse Crossing | — | `src/jax_v6/strate_iv/multiverse_crossing.py` |
| Auto-Sharder (GSPMD) | — | `src/jax_v6/training/sharding.py` |

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
- `configs/scaling/*.yaml` — T-Shirt size presets (S=15M, M=150M, L=1B, XL=7B)

To add a hyperparameter: add field to dataclass → add to YAML → use via `config.section.field`.

## Training Pipeline

Sequential phases, each depends on the previous:

```bash
# Phase 1: Download data (432 Binance Futures pairs, 1h candles)
python scripts/download_massive_data.py --interval 1h --output_dir data/raw

# Phase 2: Train Strate I tokenizer
python scripts/train_strate_i.py --config configs/strate_i.yaml

# Phase 3: Pre-tokenize all OHLCV with trained Strate I
python scripts/pretokenize.py --checkpoint checkpoints/strate-i-*.ckpt

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

# Launch training (T-Shirt scale: s/m/l/xl or 184m/500m/1_5b/3b)
nohup bash scripts/launch_tpu_v5p.sh m &

# Generate custom scaling config
python scripts/generate_optimal_config.py --target_pod v5p-32 --total_tokens 20B
```

## Critical Design Constraints

**MXU 128x128 alignment (TPU):** All model dimensions must produce `head_dim = d_model * expand_factor / n_heads = 128`. This fills the TPU MXU tiles exactly. Breaking this wastes 50%+ of compute.

**bf16 numerical stability:** The SSD kernel accumulates state via cumsum. bf16 has only ~7 mantissa bits, causing NaN around step 2750. All temporal accumulations (cumsum, h_final decay) use float32 intermediates — do not change this.

**Clock modulation bounds:** Exo-clock and vol-clock bias on dt (pre-softplus) is bounded via `dt_max_delta * tanh(raw)` (default ±2.0). This prevents SSM state explosion or gradient collapse. Do not remove the tanh constraint.

**On-manifold perturbation:** `multiverse_crossing.py:perturb_latent()` uses geodesic perturbation (tangent-plane projection + L2 re-normalization) to keep perturbed latents on the JEPA representation hypersphere. Do not replace with naive additive noise.

**GCS region co-location:** Bucket `gs://fin-ia-bucket` and TPU zone must be in the same region (`europe-west4`). Inter-region egress is billed per GB on every Grain batch load.

**PYTHONPATH required:** All imports assume `PYTHONPATH=$PWD`. Without it, `from src.strate_ii.config import ...` will fail.

## Data

- `data/ohlcv_v5/` — 432 `.pt` files (Binance Futures 1h candles)
- `data/tokens_v5/` — pre-tokenized sequences from Strate I
- `data/arrayrecord/` — ArrayRecord shards for JAX/Grain (TPU training)
- `data/trajectory_buffer_v5/` — pre-computed RL trajectories

Data is stored on Google Drive (`drive:ChaosAI_DataLake/`) and staged to GCS only during training. See `scripts/trc_data_manager.sh`.

## Infrastructure

- **GCP Project:** `financial-ai-487700`
- **GCS Bucket:** `gs://fin-ia-bucket` (europe-west4, Standard)
- **TPU Zone:** `europe-west4-a`
- **Compute:** TRC program (preemptible v5p pods)
- **IaC:** `infra/` (Terraform)
