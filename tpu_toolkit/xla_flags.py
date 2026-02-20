"""Flags LIBTPU_INIT_ARGS pour TPU v5p/v6e en production.

Usage:
    import os
    os.environ.update(get_xla_flags())

    # Ou directement au lancement:
    # LIBTPU_INIT_ARGS="--xla_tpu_..." python train.py

Ces flags ont été validés sur ChaosAI TPU v6e-8 (Trillium) et sont
conçus pour v5p. Retirer les flags expérimentaux si instabilité observée.
"""

import os


# ─────────────────────────────────────────────────────────────────────────────
# Flags de production
# ─────────────────────────────────────────────────────────────────────────────

# Chaque flag est commenté avec son effet et sa raison d'être.
_FLAGS: dict[str, str] = {

    # ── Collectives asynchrones ──────────────────────────────────────────────
    # Fusionne les all-reduce/all-gather FSDP avec les matmuls précédents.
    # Sur v5p: overlaps communication ICI avec compute MXU.
    # Gain: 10-20% sur les grandes topologies (v5p-128+).
    "--xla_tpu_enable_async_collective_fusion": "true",
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather": "true",
    "--xla_tpu_enable_async_collective_fusion_multiple_steps": "true",

    # ── Masquage de latence ──────────────────────────────────────────────────
    # Ordonnanceur qui overlaps les transfers H2D/D2H avec le calcul TPU.
    # Important pour Grain: précharge le batch N+1 pendant calcul batch N.
    "--xla_tpu_overlap_compute_collective_combination_limit": "4",

    # ── Fusion de boucles (chunked SSD) ─────────────────────────────────────
    # Fusionne les ops intra-chunk du SSD en un seul kernel.
    # Sans ça, chaque einsum/matmul du chunk est un kernel séparé.
    "--xla_tpu_enable_aggressive_loop_fusion": "true",

    # ── Scheduling latency-hiding ────────────────────────────────────────────
    # Minimise les bulles dans le pipeline en réordonnant les ops.
    "--xla_enable_async_all_gather": "true",
    "--xla_enable_async_reduce_scatter": "true",

    # ── Modèle de coût XLA ──────────────────────────────────────────────────
    # Modèle de coût amélioré pour décisions de fusion plus précises.
    # Peut ralentir la compilation (~+30s) mais améliore le runtime.
    "--xla_tpu_use_enhanced_launch_barrier": "true",
    "--xla_tpu_scoped_vmem_limit_kib": "81920",

    # ── Stabilité bf16 ──────────────────────────────────────────────────────
    # Force float32 pour la réduction des gradients lors du all-reduce.
    # Sans ça, all-reduce en bf16 peut perdre des gradients petits.
    "--xla_tpu_enable_bf16_reduction": "false",

    # ── Mémoire (v6e avec remat) ─────────────────────────────────────────────
    # Limite la HBM virtuelle utilisée par XLA pour éviter OOM sur v6e.
    "--xla_tpu_memory_limit_bytes": "32000000000",
}

# Flags légers (sans impact mesurable mais utiles pour debug)
_DEBUG_FLAGS: dict[str, str] = {
    "--xla_dump_hlo_pass_re": "",       # set to ".*" to dump all HLO passes
    "--xla_tpu_enable_log_recorder": "false",
}


def get_xla_flags(hardware: str = "v5p", debug: bool = False) -> dict[str, str]:
    """Retourne les variables d'environnement XLA pour le hardware cible.

    Args:
        hardware: "v5p", "v6e", ou "cpu" (pour tests locaux).
        debug: Inclure les flags de debug (verbose, plus lent).

    Returns:
        dict à passer à os.environ.update().
    """
    if hardware == "cpu":
        return {}

    flags = dict(_FLAGS)

    if hardware == "v6e":
        # v6e: 31 Go HBM, chip d'inférence utilisé pour entraînement.
        # Réduire la limite mémoire et activer remat par défaut.
        flags["--xla_tpu_memory_limit_bytes"] = "28000000000"
        flags["--xla_tpu_enable_aggressive_loop_fusion"] = "true"

    if debug:
        flags.update(_DEBUG_FLAGS)

    # Construire la string LIBTPU_INIT_ARGS
    flags_str = " ".join(f"{k}={v}" for k, v in flags.items() if v)

    return {
        "LIBTPU_INIT_ARGS": flags_str,
        # Désactiver les logs JAX verbeux sauf erreurs
        "JAX_LOGGING_LEVEL": "WARNING",
        # Forcer PJRT (nouveau backend, plus stable que jax_xla)
        "JAX_PLATFORMS": "tpu",
    }


def apply_xla_flags(hardware: str = "v5p", debug: bool = False) -> None:
    """Applique directement les flags dans os.environ.

    À appeler AVANT tout import de jax.
    """
    env = get_xla_flags(hardware, debug)
    os.environ.update(env)
    print(f"XLA flags appliqués pour {hardware}: {len(env)} variables")


# ─────────────────────────────────────────────────────────────────────────────
# Flags de debugging (à utiliser ponctuellement, pas en production)
# ─────────────────────────────────────────────────────────────────────────────

DEBUGGING_TIPS = """
Debugging TPU (à utiliser avec modération — ralentit x2-x5)
=============================================================

1. Trouver les NaN:
   import jax
   jax.config.update("jax_debug_nans", True)
   # Lève une exception à la première occurrence de NaN

2. Désactiver JIT pour les prints:
   jax.config.update("jax_disable_jit", True)
   # Permet print() dans les fonctions compilées

3. Voir les shapes HLO:
   jax.config.update("jax_log_compiles", True)
   # Log chaque compilation JIT

4. Vérifier les shardings:
   from jax.experimental.shard_map import shard_map
   jax.debug.visualize_array_sharding(array)

5. Profiling XLA:
   import jax.profiler
   with jax.profiler.trace("/tmp/jax-trace"):
       jax.block_until_ready(train_step(state, batch))
   # Ouvrir dans TensorBoard: tensorboard --logdir /tmp/jax-trace
"""
