"""Règles d'alignement MXU pour TPU v5p/v6e.

La Matrix Multiplication Unit (MXU) des TPU traite des tuiles de 128×128.
Tout tenseur dont la dimension n'est pas un multiple de 128 est paddé
en interne par XLA → les cycles de remplissage sont gaspillés.

Règle principale pour les SSM/Transformers:
    head_dim = d_model * expand_factor / n_heads = 128

Si cette règle est violée:
    head_dim=64  → 50% de gaspillage MXU (tuile 128×128 à moitié vide)
    head_dim=256 → OK (2 tuiles complètes)
    head_dim=96  → 25% de gaspillage

Formule de vérification:
    assert (d_model * expand_factor) % (n_heads * 128) == 0
"""

from __future__ import annotations
import math


# ─────────────────────────────────────────────────────────────────────────────
# Vérification d'alignement
# ─────────────────────────────────────────────────────────────────────────────

def check_mxu_alignment(
    d_model: int,
    n_heads: int,
    expand_factor: int = 2,
    chunk_size: int = 128,
    mxu_tile: int = 128,
) -> dict:
    """Vérifie l'alignement MXU pour une architecture Mamba-2.

    Returns:
        dict avec 'head_dim', 'aligned', 'waste_pct', 'recommendation'.
    """
    d_inner = d_model * expand_factor
    head_dim = d_inner // n_heads if n_heads > 0 else d_inner

    remainder = head_dim % mxu_tile
    if remainder == 0:
        waste_pct = 0.0
        aligned = True
    else:
        waste_pct = (mxu_tile - remainder) / mxu_tile * 100
        aligned = False

    # Vérifier aussi chunk_size
    chunk_aligned = (chunk_size % mxu_tile == 0)

    result = {
        "d_model": d_model,
        "d_inner": d_inner,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "chunk_size": chunk_size,
        "mxu_tile": mxu_tile,
        "aligned": aligned and chunk_aligned,
        "head_dim_waste_pct": waste_pct,
        "chunk_aligned": chunk_aligned,
    }

    if not aligned:
        # Suggérer le n_heads qui donne head_dim=128
        ideal_n_heads = d_inner // mxu_tile
        result["recommendation"] = (
            f"head_dim={head_dim} n'est pas aligné. "
            f"Utiliser n_heads={ideal_n_heads} → head_dim={mxu_tile}."
        )
    elif not chunk_aligned:
        result["recommendation"] = (
            f"chunk_size={chunk_size} n'est pas un multiple de {mxu_tile}. "
            f"Utiliser chunk_size=128 ou 256."
        )
    else:
        result["recommendation"] = f"✓ Parfaitement aligné (head_dim={head_dim})"

    return result


def assert_mxu_aligned(d_model: int, n_heads: int, expand_factor: int = 2) -> None:
    """Lève ValueError si l'architecture n'est pas MXU-aligned."""
    r = check_mxu_alignment(d_model, n_heads, expand_factor)
    if not r["aligned"]:
        raise ValueError(r["recommendation"])


# ─────────────────────────────────────────────────────────────────────────────
# T-Shirt sizing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Configs validées ChaosAI (toutes MXU-aligned: head_dim=128)
TSHIRT_CONFIGS = {
    "nano": {    # Pour tests locaux CPU/GPU
        "d_model": 128, "n_heads": 2, "n_layers": 4, "expand_factor": 2,
        "chunk_size": 128, "head_dim": 128,
        "batch_per_chip": 64, "pod": "v5p-8",
    },
    "S": {       # 15M params
        "d_model": 256, "n_heads": 4, "n_layers": 12, "expand_factor": 2,
        "chunk_size": 128, "head_dim": 128,
        "batch_per_chip": 1024, "pod": "v5p-8",
    },
    "M": {       # 150M params
        "d_model": 1024, "n_heads": 16, "n_layers": 24, "expand_factor": 2,
        "chunk_size": 128, "head_dim": 128,
        "batch_per_chip": 256, "pod": "v5p-32",
    },
    "L": {       # 1B params
        "d_model": 2048, "n_heads": 32, "n_layers": 48, "expand_factor": 2,
        "chunk_size": 128, "head_dim": 128,
        "batch_per_chip": 128, "pod": "v5p-128",
    },
    "XL": {      # 7B params
        "d_model": 4096, "n_heads": 64, "n_layers": 32, "expand_factor": 2,
        "chunk_size": 128, "head_dim": 128,
        "batch_per_chip": 256, "pod": "v5p-768",
    },
}


def get_tshirt(size: str) -> dict:
    """Retourne la config T-Shirt avec vérification d'alignement."""
    if size not in TSHIRT_CONFIGS:
        raise ValueError(f"Size '{size}' inconnu. Choisir parmi: {list(TSHIRT_CONFIGS)}")
    cfg = TSHIRT_CONFIGS[size].copy()
    # Vérification
    assert_mxu_aligned(cfg["d_model"], cfg["n_heads"], cfg["expand_factor"])
    return cfg


def estimate_params(d_model: int, n_layers: int, expand_factor: int = 2,
                    d_state: int = 16, n_heads: int = None) -> int:
    """Estimation grossière du nombre de paramètres Mamba-2.

    Formule approximative:
        Par bloc Mamba-2:
            in_proj:  d_model × (2*d_inner + 2*n_heads*d_state + n_heads)
            conv:     d_inner × conv_kernel
            out_proj: d_inner × d_model
            A_log:    n_heads × d_state
        Total: n_layers × params_par_bloc
    """
    if n_heads is None:
        n_heads = (d_model * expand_factor) // 128
    d_inner = d_model * expand_factor
    in_proj_size = 2 * d_inner + 2 * n_heads * d_state + n_heads
    params_per_block = (
        d_model * in_proj_size  # in_proj
        + d_inner * 4           # conv kernel=4
        + d_inner * d_model     # out_proj
        + n_heads * d_state     # A_log
    )
    embedding = 1024 * 64 + 64 * d_model  # codebook + projection
    return n_layers * params_per_block + embedding


def chinchilla_tokens(n_params: int, ratio: float = 20.0) -> int:
    """Tokens optimaux Chinchilla: T = ratio × N."""
    return int(n_params * ratio)


def remat_needed(d_model: int, n_layers: int, seq_len: int,
                 batch_per_chip: int, hbm_gb: float) -> bool:
    """Estime si le gradient checkpointing (remat) est nécessaire.

    Heuristique: activations ≈ n_layers × batch × seq_len × d_model × 4 bytes.
    Si > 50% HBM → remat recommandé.
    """
    activations_bytes = n_layers * batch_per_chip * seq_len * d_model * 4  # float32
    activations_gb = activations_bytes / 1e9
    threshold_gb = hbm_gb * 0.5
    return activations_gb > threshold_gb


# ─────────────────────────────────────────────────────────────────────────────
# Vérification en une ligne
# ─────────────────────────────────────────────────────────────────────────────

def audit_config(d_model, n_heads, n_layers, expand_factor=2, seq_len=128,
                 batch_per_chip=256, hbm_gb=95.0, d_state=16) -> None:
    """Audit complet d'une config avant lancement.

    Usage:
        audit_config(d_model=1024, n_heads=16, n_layers=24)
    """
    print("=" * 60)
    print("AUDIT CONFIG TPU")
    print("=" * 60)

    r = check_mxu_alignment(d_model, n_heads, expand_factor)
    status = "✓" if r["aligned"] else "✗"
    print(f"{status} MXU: {r['recommendation']}")

    n_params = estimate_params(d_model, n_layers, expand_factor, d_state, n_heads)
    print(f"  Params estimés: {n_params/1e6:.1f}M")
    print(f"  Tokens Chinchilla: {chinchilla_tokens(n_params)/1e9:.1f}B")

    remat = remat_needed(d_model, n_layers, seq_len, batch_per_chip, hbm_gb)
    remat_str = "OUI (mémoire serrée)" if remat else "NON (mémoire confortable)"
    print(f"  Remat recommandé: {remat_str}")

    model_size_gb = n_params * 2 / 1e9  # bf16
    opt_size_gb = n_params * 8 / 1e9    # float32 m1+m2
    total_gb = model_size_gb + opt_size_gb
    print(f"  Taille modèle bf16: {model_size_gb:.1f} Go")
    print(f"  Taille opt state f32: {opt_size_gb:.1f} Go")
    print(f"  Total params+opt: {total_gb:.1f} Go / {hbm_gb:.0f} Go HBM")

    print("=" * 60)
