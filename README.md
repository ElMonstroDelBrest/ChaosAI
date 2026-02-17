# Financial-IA: Latent Market Intelligence

**A four-stage deep learning pipeline that tokenizes raw OHLCV market data into discrete latent regimes, learns temporal dynamics via self-supervised prediction, and trains an RL agent to trade directly in latent space.**

## Architecture

The system is organized into four sequential *Strates* (stages), each building on the previous:

### Strate I — Spherical VQ-VAE Tokenizer
Converts raw OHLCV price patches into a discrete vocabulary of market micro-regimes using a **Spherical VQ-VAE** with:
- Causal dilated convolutional encoder projecting onto the unit hypersphere
- EMA-updated codebook with dead-code revival
- RevIN (Reversible Instance Normalization) for distribution shift robustness
- Huber + Soft-DTW reconstruction loss

### Strate II — Fin-JEPA (Joint-Embedding Predictive Architecture)
Self-supervised temporal model over token sequences using **Mamba-2** (selective state-space) blocks:
- Context encoder + EMA target encoder (I-JEPA paradigm)
- Block masking strategy for learning predictive representations
- VICReg regularization to prevent representation collapse

### Strate III — Stochastic Multiverse Predictor
Extends Fin-JEPA with a latent noise variable to generate **N divergent future trajectories**:
- Samples from the learned latent space to produce probabilistic forecasts
- Output projection maps back to Strate I codebook space
- Captures irreducible market uncertainty rather than collapsing to a single path

### Strate IV — Latent Regime RL
PPO agent operating on **latent observations** (not raw prices):
- Step-aware observation space: JEPA context, future mean/std, realized returns, position
- PnL reward with transaction cost friction
- Continuous action space for portfolio positioning (long/short/flat)
- Trained with `stable-baselines3` + `gymnasium`

## Key Results

- **1024-code** spherical codebook with >90% utilization and high perplexity
- **Mamba-2** context encoder scales to long token sequences with linear complexity
- **VICReg** prevents mode collapse in the self-supervised objective
- **H100-optimized** training: `torch.compile` (max-autotune), BF16, TF32 matmuls
- End-to-end pipeline from raw Binance 1h data to RL-based trading decisions

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/Financial_IA.git
cd Financial_IA
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest

# Train Strate I (tokenizer) on synthetic data
python scripts/train_strate_i.py --config configs/strate_i.yaml --synthetic

# Train Strate II (Fin-JEPA)
python scripts/train_strate_ii.py --config configs/strate_ii.yaml --synthetic

# Train Strate IV (RL agent)
python scripts/train_strate_iv.py --config configs/strate_iv.yaml
```

## Project Structure

```
Financial_IA/
├── src/
│   ├── common/              # Shared math utilities
│   ├── strate_i/            # Spherical VQ-VAE tokenizer
│   │   ├── encoder.py       # Causal dilated conv encoder
│   │   ├── decoder.py       # Patch reconstruction decoder
│   │   ├── codebook.py      # Spherical codebook with EMA
│   │   ├── losses.py        # Huber + Soft-DTW + commitment
│   │   ├── revin.py         # Reversible Instance Normalization
│   │   ├── tokenizer.py     # Public facade (RevIN + VQ-VAE)
│   │   └── lightning_module.py
│   ├── strate_ii/           # Fin-JEPA temporal model
│   │   ├── jepa.py          # Joint-Embedding Predictive Architecture
│   │   ├── mamba2_block.py  # Mamba-2 selective state-space block
│   │   ├── masking.py       # Block masking strategy
│   │   ├── vicreg.py        # VICReg regularization
│   │   ├── predictor.py     # Stochastic predictor (Strate III)
│   │   └── multiverse.py    # Multi-future trajectory sampling
│   └── strate_iv/           # Latent Regime RL
│       ├── env.py           # LatentCryptoEnv (Gymnasium)
│       ├── reward.py        # PnL reward with transaction costs
│       └── trajectory_buffer.py
├── configs/                 # YAML configurations per strate
├── scripts/                 # Training, data ingestion, and utilities
├── tests/                   # Unit tests (pytest)
├── infra/                   # Terraform GCP infrastructure
└── pyproject.toml
```

## Tech Stack

- **PyTorch** + **PyTorch Lightning** — training framework
- **Mamba-2** — selective state-space model for temporal encoding
- **stable-baselines3** + **Gymnasium** — RL training
- **tslearn** — Soft-DTW loss for time series reconstruction
- **GCP + H100** — production-scale training infrastructure
- **Terraform** — infrastructure-as-code for cloud deployment

## Accelerated Development

This project was architected by Daniel and implemented using an **AI-Augmented Workflow** (Claude 4.6 Opus / Gemini 3 Pro) to simulate a full R&D team interaction. This methodology allowed for H100 scale-up and rigorous testing in a condensed timeframe.

## References

- van den Oord et al., *Neural Discrete Representation Learning* (VQ-VAE, 2017)
- LeCun, *A Path Towards Autonomous Machine Intelligence* (JEPA framework, 2022)
- Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun & Ballas, *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023)
- Bardes et al., *VICReg: Variance-Invariance-Covariance Regularization* (2022)
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (2023)
- Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting* (RevIN, 2022)
- Schulman et al., *Proximal Policy Optimization Algorithms* (PPO, 2017)

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE). Any use of this code — including as a network service — requires publishing the complete source of derivative works under the same license.
