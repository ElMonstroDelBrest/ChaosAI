"""Patterns de stabilité numérique bf16/float32 pour TPU.

Problème fondamental: bf16 = 7 bits de mantisse vs 23 pour float32.
Erreur relative: ~1e-2 pour bf16 vs ~6e-8 pour float32.

Sur TPU, bf16 est utilisé pour les matmuls (Tensor Cores).
float32 est utilisé pour les ACCUMULATIONS TEMPORELLES.

Règle: si une valeur est sommée sur beaucoup de termes (cumsum, running
mean, covariance), elle doit être en float32. Le reste peut rester en bf16.

Conséquence observée dans ChaosAI: NaN déterministe au pas 2750 causé
par un cumsum bf16 sur 128 pas. Corrigé par upcast sélectif.
"""

import jax.numpy as jnp
from jax import Array


# ─────────────────────────────────────────────────────────────────────────────
# Règle 1 : cumsum TOUJOURS en float32
# ─────────────────────────────────────────────────────────────────────────────

def stable_cumsum(x: Array, axis: int = -1) -> Array:
    """Cumsum en float32 même si x est bf16.

    CRITIQUE dans les SSM: cs = cumsum(dt) sur L=128 pas.
    En bf16: erreur absolue de ~0.1 sur le cumsum final (pour dt~0.1).
    En float32: erreur ~1e-7. La différence cause des NaN dans exp(A*cs).

    Usage: cs = stable_cumsum(dt_c, axis=2).astype(dt_c.dtype)
    """
    return jnp.cumsum(x.astype(jnp.float32), axis=axis)


def stable_cumsum_cast(x: Array, axis: int = -1) -> Array:
    """Cumsum float32, résultat casté dans le dtype d'entrée."""
    return stable_cumsum(x, axis).astype(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Règle 2 : exponentielles stables
# ─────────────────────────────────────────────────────────────────────────────

def stable_exp(x: Array) -> Array:
    """exp en float32 pour éviter overflow/underflow bf16.

    bf16: exp sature à inf à partir de ~88.0, et underflow à 0 à partir de ~-88.
    float32: limites à ±88.7. Même limite mais précision bien meilleure.

    Usage pour les déclins SSM:
        decay = stable_exp(A * cs_end)  # (N,) float32
        h_final = decay[:, None] * h_init  # cast back après
    """
    return jnp.exp(x.astype(jnp.float32))


def stable_decay(A: Array, cs: Array) -> Array:
    """Facteur de déclin exp(A * cs) en float32.

    Pour SSM: A < 0, cs >= 0, donc exp(A*cs) <= 1 (décroissant, stable).
    Néanmoins, on upcast pour éviter la perte de précision sur les petites valeurs.
    """
    return stable_exp(A.astype(jnp.float32) * cs.astype(jnp.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Règle 3 : matmuls d'accumulation en float32
# ─────────────────────────────────────────────────────────────────────────────

def stable_matmul_accum(a: Array, b: Array) -> Array:
    """Matmul avec accumulation en float32.

    Usage: h_accum = stable_matmul_accum(B_hat.T, x)
    Quand le résultat est accumulé dans un état persistent (h_final),
    float32 évite la dérive progressive.
    """
    return (a.astype(jnp.float32) @ b.astype(jnp.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Règle 4 : covariance TOUJOURS en float32
# ─────────────────────────────────────────────────────────────────────────────

def stable_covariance(z: Array, mask: Array | None = None) -> Array:
    """Matrice de covariance en float32.

    Même en float32, la covariance sur des distributions à queues épaisses
    (marchés financiers) peut être numériquement problématique. bf16 la rend
    carrément inutilisable (perd la distinction Momentum vs Volatilité).

    Returns: (D, D) matrice de covariance float32.
    """
    z_f32 = z.astype(jnp.float32)
    if mask is not None:
        w = mask[:, None].astype(jnp.float32)
        n = jnp.maximum(jnp.sum(mask).astype(jnp.float32), 2.0)
        mean = jnp.sum(z_f32 * w, axis=0) / n
        zc = (z_f32 - mean) * w
        return (zc.T @ zc) / (n - 1)
    else:
        n = z_f32.shape[0]
        zc = z_f32 - jnp.mean(z_f32, axis=0)
        return (zc.T @ zc) / (n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Règle 5 : bornage des biais adaptatifs (SSM clocks)
# ─────────────────────────────────────────────────────────────────────────────

def bounded_clock_bias(raw: Array, max_delta: float = 2.0) -> Array:
    """Borne un biais additif sur dt (pré-softplus) via tanh.

    PROBLÈME: dt_raw + unbounded_bias → si bias >> 0, softplus(dt_raw+bias)
    explose → état SSM diverge. Si bias << 0 → dt ≈ 0 → vanishing gradients.

    SOLUTION: bias = max_delta * tanh(raw) ∈ (-max_delta, +max_delta)
    Après softplus, cela translate dt dans un facteur multiplicatif borné.

    Usage:
        clock_bias = bounded_clock_bias(linear(clock_signal))
        dt = dt_raw + clock_bias
        dt_disc = jax.nn.softplus(dt)
    """
    return max_delta * jnp.tanh(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Règle 6 : zero_nans dans le pipeline d'optimisation
# ─────────────────────────────────────────────────────────────────────────────

def safe_norm(grads, max_norm: float = 1.0):
    """Norme globale des gradients, robuste aux NaN.

    Usage avec optax:
        import optax
        tx = optax.chain(
            optax.clip_by_global_norm(max_norm),
            optax.zero_nans(),   # NaN → 0 AVANT l'update AdamW
            optax.adamw(lr, b2=0.95),
        )
    Note: zero_nans() doit être APRÈS clip_by_global_norm pour que la norme
    calculée soit correcte, mais AVANT adamw pour ne pas corrompre les moments.
    """
    import optax
    return optax.global_norm(grads)


# ─────────────────────────────────────────────────────────────────────────────
# Résumé des règles sous forme de checklist
# ─────────────────────────────────────────────────────────────────────────────

BF16_SAFETY_CHECKLIST = """
bf16 sur TPU — checklist de stabilité numérique
================================================

✅ MATMULS         → bf16 (Tensor Cores, pas de problème)
✅ ACTIVATIONS     → bf16 (relu, gelu, silu — bornés, pas de dérive)

⚠️  CUMSUM         → float32 OBLIGATOIRE (NaN au step ~2750 sinon)
⚠️  COVARIANCE     → float32 OBLIGATOIRE (perte de rang sinon)
⚠️  EXPONENTIELLES → float32 si l'argument peut être > ±10
⚠️  ETATS PERSISTANTS (h_final SSM) → float32 pour l'accumulation
⚠️  BIAIS ADAPTATIFS → borner avec tanh(raw) * max_delta

🔧 PIPELINE OPTIMIZER:
   optax.clip_by_global_norm(1.0)
   optax.zero_nans()
   optax.adamw(lr, b2=0.95)  # b2=0.95 pour SSM (plus réactif que 0.999)
"""
