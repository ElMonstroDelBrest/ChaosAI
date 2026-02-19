# Financial-IA — World Model Trading Agent

> **Version:** v6.0-FINAL (Grade Institutionnel)
> **Architecture:** Exo-Clocked Mamba-2 + OT-CFM + TD-MPC2
> **Infrastructure:** GCP H100 Spot (Zero Waste Optimized)
> **Migration en cours:** JAX/TPU v4-32 Pod Slice (TRC Ready)

> **DISCLAIMER: EDUCATIONAL PURPOSE ONLY**
>
> This software is a research project exploring the application of Deep Learning (JEPA, VQ-VAE, PPO) to financial time series. **It is NOT a trading bot ready for production use.**
>
> - Nothing in this repository constitutes financial advice.
> - Past performance (simulated or real) is not indicative of future results.
> - The authors are not responsible for any financial losses incurred by using this code.
> - Use at your own risk and strictly within the legal frameworks of your jurisdiction.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-TPU%20v4--32-blue?logo=google&logoColor=white)](https://github.com/google/jax)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org/)

## 1. Philosophie & Objectifs

Ce projet vise a creer une "Intelligence Artificielle Financiere" fondee sur l'Inference Active. Contrairement aux approches predictives classiques (regression du prix futur), nous adoptons une approche **World Model** :
- **L'IA ne predit pas une seule verite**, mais simule une distribution continue de futurs possibles (Strate II/III).
- **L'Agent (Strate IV)** planifie dans ce "multivers" latent, motive par la maximisation du PnL sous contrainte stricte de risque (survie).
- **Hypothese Scientifique v6 :** La modulation de l'equation differentielle de Mamba par des variables micro-structurelles exogenes (volume/volatilite) couplee au Transport Optimal (OT-CFM) surpasse l'architecture Transformer face a la non-ergodicite financiere.

## 2. Architecture Technique (The v6 Stack)

Le systeme est divise en "Strates" cognitives :

### Strate I : Perception Causal-Strict (SymLog + FSQ)
*Role : Compression homomorphique et discretisation sans perte topologique.*
- **Modele :** Dilated Causal CNN 1D + **SymLog** + **FSQ (Finite Scalar Quantization)**.
- **Fonction :** Transforme les donnees brutes continues en une representation discrete. **SymLog** garantit que les chocs de volatilite extremes (Fat Tails) s'integrent dans la grille FSQ sans ecretage (clipping). Zero *look-ahead bias*.

### Strate II & III : World Model & Multiverse (FinJEPA v6)
*Role : Modelisation temporelle heteroscedastique et physique.*
- **Architecture :** **Exo-Clocked Mamba-2**.
- **Innovation v6 :** Le pas de discretisation temporelle `dt` n'est plus appris aveuglement mais est conditionne par un flux exogene (**Volatilite Realisee et Volume**). Lors des chocs de liquidite, la matrice du modele se reconfigure mecaniquement (Time Dilation).
- **Predicteur (Multivers) :** **OT-CFM (Optimal Transport Continuous Flow Matching)**. Remplace le predicteur MLP par une ODE optimisee par Transport Optimal. Les trajectoires latentes deviennent droites, permettant un echantillonnage ultra-rapide en 1 ou 2 steps d'Euler.

### Strate IV : Agent (Distributional Model-Based RL)
*Role : Inference Active et Controle du Risque.*
- **Algo :** **TD-MPC2** (Temporal Difference Model Predictive Control v2).
- **Critique :** L'agent deploie son arbre de recherche (MCTS) a l'interieur de l'espace latent differentiable et optimise la **CVaR 5%** (Conditional Value at Risk) pour survivre aux 5% des futurs les plus catastrophiques generes par la Strate III.

## 3. Migration TPU v4-32 (JAX Native)

Une branche parallele `src/jax_v6/` a ete developpee pour exploiter le **TPU Research Cloud (TRC)** de Google.

### Architecture "TPU-Native"
- **Framework :** JAX + Flax (Purely Functional).
- **Kernel Mamba-2 :** Implementation **Chunked SSD** (State Space Duality) optimisee pour les MXU (Matrix Units). Remplace le scan associatif lineaire par des multiplications de blocs matriciels denses.
- **Parallelisme :** **GSPMD** (Global Single Program Multiple Data) via `jax.sharding.Mesh` et `NamedSharding`. *Zero `pmap`.*
- **Data Pipeline :** Conversion des donnees PyTorch (`.pt`) vers **ArrayRecord** (binaire) pour ingestion massive via **Google Grain** (Async Multi-Host, bypass complet du CPU Python).
- **Flow Matching :** Integration ODE via **Diffrax** (Adjoint Backprop) et couplage Optimal Transport via un solveur de **Sinkhorn** JIT-compatible.

### Statut de la Migration
- [x] Script de conversion ArrayRecord (`scripts/convert_pt_to_arrayrecord.py`)
- [x] Kernel SSD JAX (`src/jax_v6/encoders/ssd.py`)
- [x] Flow Predictor Diffrax (`src/jax_v6/predictors/flow_predictor.py`)
- [x] Training Loop GSPMD (`src/jax_v6/training/train_step.py`)
- [x] Cross-validation numerique PyTorch/JAX (max err: 5.6e-05 < 1e-4)

## 4. Infrastructure "Zero Waste" (H100 Backup)

L'infrastructure PyTorch (H100) reste active comme "Ground Truth" de validation mathematique :
- **Precision Numerique :** Entrainement exclusif en **BFloat16** (le FP16 classique est banni pour eviter les `NaN` dus aux Fat Tails) et **TF32**.
- **Bouclier Anti-Preemption :** Demon `spot_watcher.py` scrutant l'API GCP. Au preavis de 30s avant la coupure Spot, la VRAM est *dumpee* en asynchrone sur le SSD NVMe puis uploadee sur GCS.
- **Optimisation IO :** Le dataset et le cache de compilation (`TRITON_CACHE`) sont deplaces sur le SSD NVMe (RAID0 a >7 Go/s) au boot.

## 5. Defis Scientifiques & Materiels Restants (v6)

Bien que la dette technique de la v5 ait ete eradiquee (l'OT-CFM resout la lenteur d'inference, l'Exo-Clock resout la boucle L2), les nouveaux defis se concentrent sur le hardware TPU et la theorie de l'optimisation :

### Defi 1 : Stabilite End-to-End (BPTT)
Nous empilons trois technologies de pointe. Faire retropropager le gradient de la Policy (Strate IV) a travers l'ODE Diffrax (Strate III) jusqu'a l'encodeur Mamba (Strate II) via *Backpropagation Through Time* necessitera un reglage chirurgical du *Gradient Clipping* adaptatif pour eviter l'evanouissement ou l'explosion des gradients.

### Defi 2 : Pression Memoire HBM (Arbre MCTS JAX)
Deployer l'arbre de recherche de TD-MPC2 *entierement via `jax.vmap`* (pour eviter les transferts CPU/TPU) va consommer une quantite colossale de memoire HBM. Un tuning fin de l'horizon de planification et du nombre d'echantillons du multivers (N) sera vital pour eviter le *Out Of Memory* sur les puces.

### Defi 3 : Precision Numerique des ODEs (TPU BFloat16)
Les noyaux TPU (MXUs) imposent l'usage du `BFloat16`. Bien que stable pour le Deep Learning, la resolution d'equations differentielles (ODE) accumule des erreurs d'arrondi numerique. Il faudra potentiellement implementer un loss scaling dynamique localise pour `Diffrax`.

## 6. Runbook v6 (Validation)

### Tolerance aux Pannes (H100)
```bash
export TORCH_CUDNN_V8_API_ENABLED=1
nohup python scripts/train_end_to_end_v6.py \
    --compile-mode max-autotune \
    --precision bf16 \
    --spot-resume-gcs gs://fin-ia-bucket/checkpoints/v6/ \
    > /mnt/local_ssd/training_v6.log 2>&1 &
```

### Validation Mathematique H100 vs TPU (Research)

Comparer numeriquement les tenseurs JAX et PyTorch pour valider l'isometrie du portage avant l'entrainement massif :

```bash
python scripts/cross_validate_jax_pytorch.py \
    --pytorch_ckpt checkpoints/strate_ii/strate_ii-epoch=034*.ckpt \
    --tolerance 1e-4
```

## 7. Project Structure

```
Financial_IA/
├── src/
│   ├── common/              # Shared math utilities
│   ├── strate_i/            # SymLog + FSQ tokenizer
│   │   ├── encoder.py       # Causal dilated conv encoder
│   │   ├── decoder.py       # Patch reconstruction decoder
│   │   ├── fsq_codebook.py  # Finite Scalar Quantization
│   │   ├── losses.py        # Huber + Soft-DTW + commitment
│   │   ├── revin.py         # Reversible Instance Normalization
│   │   └── lightning_module.py
│   ├── strate_ii/           # FinJEPA v6 (Exo-Clocked Mamba-2 + OT-CFM)
│   │   ├── jepa.py          # Joint-Embedding Predictive Architecture
│   │   ├── mamba2_block.py  # Mamba-2 with Exo-Clock modulation
│   │   ├── encoder.py       # Mamba-2 encoder stack
│   │   ├── flow_predictor.py # OT-CFM (Optimal Transport Flow Matching)
│   │   ├── masking.py       # Block masking strategy
│   │   ├── vicreg.py        # VICReg regularization
│   │   └── predictor.py     # Stochastic predictor (Strate III)
│   ├── strate_iv/           # TD-MPC2 + CVaR agent
│   │   ├── env.py           # LatentCryptoEnv (Gymnasium)
│   │   ├── tdmpc2.py        # TD-MPC2 agent (MPPI + CVaR planning)
│   │   ├── world_model.py   # Latent dynamics model
│   │   ├── distributional_critic.py  # Ensemble quantile critic
│   │   └── replay_buffer.py
│   └── jax_v6/              # JAX/TPU v4-32 migration
│       ├── encoders/        # Chunked SSD, Mamba-2 block/encoder
│       ├── predictors/      # Predictor + Flow (Diffrax + Sinkhorn)
│       ├── losses/          # VICReg (float32 covariance)
│       ├── training/        # TrainState, GSPMD sharding, JIT step
│       └── data/            # Grain async multi-host loader
├── configs/                 # YAML configurations per strate
├── scripts/                 # Training, data ingestion, cross-validation
├── infra/                   # GCP startup scripts
└── pyproject.toml
```

## 8. Tech Stack

- **PyTorch** + **PyTorch Lightning** — training framework (H100)
- **JAX** + **Flax** + **Diffrax** — TPU-native migration
- **Mamba-2** — selective state-space model for temporal encoding
- **TD-MPC2** — model-based RL with distributional critics
- **GCP H100 Spot** + **TPU v4-32 (TRC)** — training infrastructure

## 9. Accelerated Development

This project was architected by Daniel and implemented using an **AI-Augmented Workflow** (Claude 4.6 Opus / Gemini 3 Pro) to simulate a full R&D team interaction. This methodology allowed for H100 scale-up and rigorous testing in a condensed timeframe.

## 10. References

- LeCun, *A Path Towards Autonomous Machine Intelligence* (JEPA framework, 2022)
- Gu & Dao, *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2 / SSD, 2024)
- Lipman, Chen, Ben-Hamu, Nickel & Le, *Flow Matching for Generative Modeling* (CFM, 2022)
- Tong, Malkin, Fatras, Atanackovic, Zhang, Bengio & LeCun, *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport* (OT-CFM, 2023)
- Hansen, Su, Wang, *TD-MPC2: Scalable, Robust World Models for Continuous Control* (2024)
- Mentzer, Minnen, Agustsson & Tschannen, *Finite Scalar Quantization: VQ-VAE Made Simple* (FSQ, 2023)
- Bardes, Ponce & LeCun, *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning* (2022)
- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023)
- Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting* (RevIN, 2022)

## License

This project is licensed under the [GNU Affero General Public License v3.0](LICENSE). Any use of this code — including as a network service — requires publishing the complete source of derivative works under the same license.
