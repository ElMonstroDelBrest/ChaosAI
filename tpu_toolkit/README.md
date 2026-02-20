# tpu_toolkit — Recettes TPU extraites de ChaosAI

Optimisations validées sur TPU v6e-8 (ChaosAI, Février 2026).
À copier dans les prochains projets JAX/TPU.

## Fichiers

| Fichier | Contenu |
|---|---|
| `sharding.py` | Auto-Sharder GSPMD topologique (v5p-8 → v5p-768) |
| `numerics.py` | Stabilité bf16/f32 : cumsum, covariance, clock bounds |
| `optimizer.py` | AdamW + SGDR + zero_nans + donate_argnums |
| `xla_flags.py` | 13 flags LIBTPU_INIT_ARGS de production |
| `grain_pipeline.py` | Pipeline Grain async multi-host (ArrayRecord) |
| `replay_buffer.py` | Ring buffer double-buffering async H2D |
| `mxu_alignment.py` | Vérification head_dim=128, T-Shirt configs, audit |

## Usage rapide

```python
# 1. Flags XLA — avant tout import JAX
from tpu_toolkit.xla_flags import apply_xla_flags
apply_xla_flags(hardware="v5p")

import jax

# 2. Mesh GSPMD topologique
from tpu_toolkit.sharding import create_mesh, shard_train_state, shard_batch
mesh = create_mesh()
state = shard_train_state(state, mesh)

# 3. Vérifier l'alignement MXU
from tpu_toolkit.mxu_alignment import audit_config
audit_config(d_model=1024, n_heads=16, n_layers=24)

# 4. Optimiseur avec SGDR
from tpu_toolkit.optimizer import create_optimizer
tx = create_optimizer(lr=2e-4, total_steps=50_000, n_restarts=4, b2=0.95)

# 5. Données Grain
from tpu_toolkit.grain_pipeline import create_dataloader
loader = create_dataloader(shard_paths, transform, batch_size=8192)

# 6. Train step avec buffer donation
from tpu_toolkit.optimizer import make_train_step
train_step = make_train_step(model.apply, loss_fn)
```

## Règles critiques

1. **bf16 cumsum → float32** (sinon NaN ~step 2750 sur séquences longues)
2. **head_dim = 128** (MXU 128×128, sinon 25-50% de gaspillage compute)
3. **clock bias → tanh borné** (sinon SSM diverge sur volatilité élevée)
4. **fsdp_dim = 4** (= 1 tray v5p, communications intra-tray uniquement)
5. **donate_argnums** sur le train_step (zero-copy pour les params)
6. **GCS + TPU = même région** (sinon egress facturé par batch Grain)
