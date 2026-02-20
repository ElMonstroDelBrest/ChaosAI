# tpu_toolkit

Recettes JAX/TPU extraites et généralisées depuis **ChaosAI** — validées sur TPU v6e-8 (Trillium) en février 2026, conçues pour v5p.

7 modules indépendants, zéro dépendance entre eux, zéro couplage à ChaosAI. À copier tel quel dans n'importe quel projet JAX/TPU.

---

## Installation

Aucun package à installer. Copier le dossier `tpu_toolkit/` à la racine du projet et ajouter `PYTHONPATH=$PWD`.

Dépendances requises (à installer sur le TPU VM) :
```
jax[tpu]
flax
optax
grain-nightly
orbax-checkpoint
```

---

## Démarrage rapide

```python
# ① Appliquer les flags XLA AVANT tout import jax
from tpu_toolkit.xla_flags import apply_xla_flags
apply_xla_flags(hardware="v5p")  # ou "v6e"

import jax

# ② Vérifier l'architecture avant de lancer quoi que ce soit
from tpu_toolkit.mxu_alignment import audit_config
audit_config(d_model=1024, n_heads=16, n_layers=24)
# → affiche alignement MXU, param count, Chinchilla tokens, remat advice

# ③ Mesh GSPMD topologique
from tpu_toolkit.sharding import create_mesh, shard_train_state, shard_batch
mesh = create_mesh()  # auto-détecte v5p-8 → v5p-768

# ④ Optimiseur avec SGDR + zero_nans
from tpu_toolkit.optimizer import create_optimizer
tx = create_optimizer(lr=2e-4, total_steps=50_000, n_restarts=4, b2=0.95)

# ⑤ Train step JIT avec buffer donation (zero-copy)
from tpu_toolkit.optimizer import make_train_step
train_step = make_train_step(model.apply, loss_fn)

# ⑥ Data pipeline Grain multi-host
from tpu_toolkit.grain_pipeline import create_dataloader
loader = create_dataloader(shard_paths, transform, batch_size=8192)

# ⑦ Boucle d'entraînement
state = shard_train_state(state, mesh)
for batch in loader:
    batch = shard_batch(batch, mesh)
    state, metrics = train_step(state, batch)
```

---

## Modules

### `sharding.py` — Auto-Sharder GSPMD topologique

**Problème résolu :** configurer manuellement le maillage GSPMD pour chaque taille de pod est fastidieux et source d'erreurs. Sur v5p-768, un mauvais mapping des axes sur la topologie physique peut doubler les communications ICI.

**Solution :** `create_mesh()` auto-détecte le nombre de chips et applique le maillage optimal en une ligne. La règle `fsdp_dim=4` correspond exactement aux 4 chips d'un tray v5p qui partagent les liens ICI les plus rapides (~340 Go/s). Les communications FSDP (all-gather/reduce-scatter) restent intra-tray.

```python
from tpu_toolkit.sharding import create_mesh, shard_train_state, shard_batch, shard_rng

mesh = create_mesh()
# → Mesh: 32 devices → (data=8, fsdp=4) | platform=TPU

# Sharder le state complet (params + opt_state → FSDP, step/rng → répliqué)
state = shard_train_state(state, mesh)

# Sharder un batch (axe batch → data-parallel)
batch = shard_batch({"x": x, "y": y}, mesh)

# RNG indépendant par replica (dropout différent sur chaque chip)
rng = shard_rng(jax.random.PRNGKey(42), mesh)
```

**Topologies supportées :**

| Pod | Chips | Mesh | FSDP | DP |
|-----|-------|------|------|-----|
| v5p-8 | 8 | (8, 1) | Désactivé | 8 replicas |
| v5p-32 | 32 | (8, 4) | 4 chips/tray | 8 replicas |
| v5p-128 | 128 | (32, 4) | 4 chips/tray | 32 replicas |
| v5p-768 | 768 | (192, 4) | 4 chips/tray | 192 replicas |

---

### `numerics.py` — Stabilité numérique bf16/float32

**Problème résolu :** bf16 a 7 bits de mantisse (erreur relative ~1%). Sur des séquences longues (128 pas), un `cumsum` en bf16 accumule ~0.1 d'erreur absolue. Cette erreur est amplifiée exponentiellement dans `exp(A * cs)` et provoque des NaN de façon **déterministe** autour du step 2750 en entraînement.

**Cause expérimentale :** observé sur ChaosAI v6e-8. Le NaN apparaît exactement quand les gradients ont assez amplifié la dérive accumulée pour sortir du domaine représentable bf16.

**Solution :** upcast **sélectif** — uniquement sur les accumulations temporelles. Les matmuls restent en bf16 (Tensor Cores). Gain de précision sans perte de performance.

```python
from tpu_toolkit.numerics import (
    stable_cumsum,        # cumsum float32, résultat casté dans dtype d'entrée
    stable_cumsum_cast,   # idem avec cast automatique
    stable_exp,           # exp en float32
    stable_decay,         # exp(A * cs) en float32
    stable_matmul_accum,  # matmul avec accumulation float32
    stable_covariance,    # matrice de covariance float32
    bounded_clock_bias,   # tanh(raw) * max_delta pour biais adaptatif SSM
)

# Exemple dans un SSM (remplace jnp.cumsum direct)
cs = stable_cumsum_cast(dt, axis=2)  # float32 en interne, bf16 en sortie

# Exemple covariance VICReg (TOUJOURS float32)
cov = stable_covariance(z_embeddings)  # (D, D) float32

# Biais d'horloge borné (évite l'explosion SSM)
clock_bias = bounded_clock_bias(linear_output, max_delta=2.0)
dt = dt_raw + clock_bias  # biais ∈ (-2, 2) garanti
```

**Checklist rapide :**
```
✅ matmuls          → bf16 (Tensor Cores)
✅ activations      → bf16 (relu, gelu, silu)
⚠️  cumsum          → float32 (NaN sinon sur L ≥ 64)
⚠️  covariance      → float32 (perte de rang sinon)
⚠️  exp(A * cs)     → float32 (overflow sinon)
⚠️  états SSM h_t   → float32 pour l'accumulation
⚠️  biais adaptatif → borner avec tanh(raw) * max_delta
```

---

### `optimizer.py` — AdamW + SGDR + zero_nans

**Problème résolu :** l'optimiseur standard (`optax.adamw`) ne protège pas contre les NaN, et la décroissance cosinus simple peut rester coincée dans des minima locaux sur des longues séquences.

**Solution :** chaîne d'optimisation avec trois couches de sécurité et un scheduler avec warm restarts.

```python
from tpu_toolkit.optimizer import create_optimizer, ema_update, make_train_step

# Optimiseur complet
tx = create_optimizer(
    lr=2e-4,
    weight_decay=0.01,
    warmup_steps=1_000,
    total_steps=50_000,
    grad_clip=1.0,
    b2=0.95,        # 0.95 pour SSM (plus réactif), 0.999 standard
    n_restarts=4,   # SGDR: cycles cosinus avec période doublante
)
# Chaîne interne: clip_by_global_norm → zero_nans → adamw

# EMA target encoder (JEPA, TD-MPC2...)
target_params = ema_update(target_params, online_params, tau=0.996)

# Train step avec buffer donation (zero-copy sur les params)
train_step = make_train_step(model.apply, loss_fn)
state, metrics = train_step(state, batch)

# LR scaling μP pour différentes tailles de modèle
from tpu_toolkit.optimizer import lr_for_d_model
lr = lr_for_d_model(base_lr=3e-4, base_d=256, target_d=1024)
# → 1.5e-4 (lr ∝ 1/sqrt(d_model))
```

**Pourquoi b2=0.95 pour les SSM :** AdamW avec b2=0.999 met un retard moyen de ~1000 pas sur les gradients. Les SSM ont des paramètres (B, C, Δ) qui doivent s'adapter rapidement aux transitions de régime. b2=0.95 → retard moyen de ~20 pas.

**Pourquoi SGDR :** les warm restarts périodiques permettent à l'optimiseur de sortir des minima locaux étroits en ré-augmentant le LR. Chaque cycle est 2× plus long que le précédent (T, 2T, 4T...).

---

### `xla_flags.py` — Flags de production LIBTPU

**Problème résolu :** XLA n'active pas par défaut toutes les optimisations disponibles pour les TPU. Les flags doivent être définis **avant** l'import de JAX.

```python
from tpu_toolkit.xla_flags import apply_xla_flags

# À mettre en tout premier dans le script d'entraînement
apply_xla_flags(hardware="v5p")   # ou "v6e"
apply_xla_flags(hardware="v5p", debug=True)  # + profiling verbeux

import jax  # seulement après
```

**Flags activés et leur effet :**

| Flag | Effet | Gain estimé |
|------|-------|-------------|
| `async_collective_fusion` | Overlap all-reduce/all-gather avec compute | +10-20% sur v5p-128+ |
| `async_collective_fusion_multiple_steps` | Fusionne plusieurs collectives consécutives | +5% |
| `aggressive_loop_fusion` | Fusionne les ops intra-chunk SSD en un kernel | +15% pour SSM |
| `enable_async_all_gather` | All-gather non-bloquant pour FSDP | +8% |
| `enable_async_reduce_scatter` | Reduce-scatter non-bloquant | +8% |
| `overlap_compute_collective` | Latency hiding scheduler | +5-10% |
| `bf16_reduction=false` | All-reduce en float32 (gradients précis) | Qualité |

**Tips de debugging XLA :**
```python
from tpu_toolkit.xla_flags import DEBUGGING_TIPS
print(DEBUGGING_TIPS)
```

---

### `grain_pipeline.py` — Pipeline Grain multi-host

**Problème résolu :** écrire un pipeline de données multi-host correct pour TPU nécessite de coordonner la distribution des shards entre hosts, le preprocessing async, et l'alignement des batch sizes.

**Solution :** template avec `ShardByJaxProcess()` (distribution automatique), split train/val par hash MD5 (déterministe, reproductible), et worker count auto-calibré.

```python
from tpu_toolkit.grain_pipeline import (
    create_dataloader,
    hash_split,
    BaseTransform,
    to_device_async,
    load_shards_from_manifest,
)

# Split déterministe (pas de seed à gérer)
for shard_path in all_shard_paths:
    split = hash_split(shard_path.stem, val_ratio=0.2)
    # → "train" ou "val", identique à chaque exécution

# Implémenter son propre transform
class MyTransform(BaseTransform):
    def map(self, record: bytes) -> dict:
        # Parsing + preprocessing en numpy (pas de JAX ici!)
        data = parse_my_format(record)
        return {"x": data["features"], "y": data["label"]}

# Créer le DataLoader
train_loader = create_dataloader(
    shard_paths=train_shards,
    transform=MyTransform(),
    batch_size=8192,          # global — divisé par n_hosts automatiquement
    split="train",
    worker_count=0,           # 0 = auto (min(cpu_count, 32))
    prefetch_buffer_size=128, # >= worker_count pour ne pas affamer les TPU
)

# Boucle avec transfer H2D asynchrone
for batch in train_loader:
    batch = to_device_async(batch, sharding=data_sharding(mesh))
    state, metrics = train_step(state, batch)
    # H2D du batch suivant se fait en arrière-plan pendant train_step
```

**Pourquoi ArrayRecord :** format binaire optimisé pour Grain. Accès aléatoire O(1), compatible multi-process, lit depuis GCS avec les performances du backbone réseau Google interne.

**Règle région :** le bucket GCS doit être dans **la même région** que le TPU. Si bucket `europe-west4` + TPU `us-central1`, chaque batch de Grain génère des frais d'egress GCS (~$0.12/Go). Sur 3 jours d'entraînement à 1 Go/s, c'est ~$30 000 de frais inattendus.

---

### `replay_buffer.py` — Ring buffer double-buffering async

**Problème résolu :** `jnp.array(numpy_array)` est **synchrone** — bloque le thread hôte ~100-500μs par appel. Pour le RL avec sample fréquent, cet overhead s'accumule.

**Solution :** `jax.device_put()` est **asynchrone** — retourne immédiatement un DeviceArray future. Le double-buffering précharge le batch N+1 pendant que le TPU calcule sur N.

```python
from tpu_toolkit.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=100_000, obs_dim=422, action_dim=1)

# Remplir
buf.add(obs, action, reward, next_obs, done)

# Sample synchrone (debug)
batch = buf.sample(batch_size=256)

# Sample async avec double-buffering (production)
batch = buf.sample_async(batch_size=256)
# → retourne immédiatement, H2D se fait en arrière-plan

# Attendre avant de commencer
if buf.is_ready(warmup=1000):
    batch = buf.sample_async(256)
    metrics = agent.update(batch)
```

**Diagramme de recouvrement :**
```
Step N:   [TPU: calcule batch N   ][Host: H2D batch N+1 →→→]
Step N+1: [TPU: calcule batch N+1 ][Host: H2D batch N+2 →→→]
```
Sur v5p (1 step ~1s), la latence H2D (~50μs pour 1Mo) est complètement masquée.

---

### `mxu_alignment.py` — Alignement MXU et configs T-Shirt

**Problème résolu :** une dimension de tête (`head_dim`) qui n'est pas un multiple de 128 laisse la tuile MXU partiellement vide. XLA padde en interne mais les cycles de remplissage sont gaspillés.

```python
from tpu_toolkit.mxu_alignment import (
    audit_config,      # audit complet en une ligne
    check_mxu_alignment,
    get_tshirt,        # configs pré-calculées S/M/L/XL
    estimate_params,
    chinchilla_tokens,
    remat_needed,
)

# Audit avant de lancer quoi que ce soit
audit_config(d_model=1024, n_heads=16, n_layers=24)
# ============================================================
# AUDIT CONFIG TPU
# ============================================================
# ✓ MXU: ✓ Parfaitement aligné (head_dim=128)
#   Params estimés: 148.2M
#   Tokens Chinchilla: 2.96B
#   Remat recommandé: NON (mémoire confortable)
#   Taille modèle bf16: 0.3 Go
#   Taille opt state f32: 1.2 Go
#   Total params+opt: 1.5 Go / 95 Go HBM
# ============================================================

# Config T-Shirt pré-validée
cfg = get_tshirt("M")
# → {"d_model": 1024, "n_heads": 16, "n_layers": 24, "head_dim": 128, ...}

# Estimation rapide
n_params = estimate_params(d_model=1024, n_layers=24)
tokens = chinchilla_tokens(n_params)  # 20 × N_params
need_remat = remat_needed(d_model=1024, n_layers=24, seq_len=128,
                          batch_per_chip=256, hbm_gb=95.0)
```

**Règle MXU :**
```
head_dim = d_model × expand_factor / n_heads = 128

head_dim=64  → 50% gaspillage  ← À ÉVITER
head_dim=96  → 25% gaspillage  ← À ÉVITER
head_dim=128 → 0% gaspillage   ← CIBLE
head_dim=256 → 0% gaspillage   ← OK (2 tuiles)
```

**Configs T-Shirt validées :**

| Taille | Params | d_model | Têtes | Couches | Pod | Batch/chip |
|--------|--------|---------|-------|---------|-----|-----------|
| S | 15M | 256 | 4 | 12 | v5p-8 | 1024 |
| M | 150M | 1024 | 16 | 24 | v5p-32 | 256 |
| L | 1B | 2048 | 32 | 48 | v5p-128 | 128 |
| XL | 7B | 4096 | 64 | 32 | v5p-768 | 256 |

---

## Règles critiques — à ne jamais oublier

```
1. apply_xla_flags() AVANT import jax
2. cumsum → float32 (NaN au step ~2750 sinon)
3. head_dim = 128 (MXU 128×128)
4. fsdp_dim = 4 (= 1 tray v5p, ICI intra-tray)
5. donate_argnums=(0,1) sur train_step (zero-copy)
6. GCS bucket + TPU = même région (egress $0 sinon $$$)
7. worker_count ≤ 32 (rendements décroissants au-delà)
8. b2=0.95 pour SSM, 0.999 pour Transformer
9. tanh(raw) * max_delta pour tout biais adaptatif SSM
10. Covariance, all-reduce gradients → float32
```

---

## Origine

Extrait de [ChaosAI](https://github.com/ElMonstroDelBrest/ChaosAI) — apprentissage auto-supervisé sur systèmes dynamiques chaotiques.
Validé sur TPU v6e-8, 3.93M paramètres, 2700 steps, perte -35%.
