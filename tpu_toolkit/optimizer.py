"""Optimiseur TPU-ready: AdamW + SGDR + zero_nans + gradient clipping.

Recette validée sur ChaosAI TPU v6e-8, 2700 steps sans divergence
(avec l'upcast bf16 dans le modèle — voir numerics.py).

Différences vs. configuration standard:
  - b2=0.95 au lieu de 0.999 → plus réactif pour SSM
  - zero_nans() → survie aux explosions bf16 isolées
  - SGDR warm restarts → échappe les minima locaux
  - donate_argnums dans train_step → zero-copy sur TPU
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import Any

PyTree = Any


# ─────────────────────────────────────────────────────────────────────────────
# Optimiseur
# ─────────────────────────────────────────────────────────────────────────────

def create_optimizer(
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
    warmup_steps: int = 1000,
    total_steps: int = 100_000,
    grad_clip: float = 1.0,
    b2: float = 0.95,
    n_restarts: int = 4,
) -> optax.GradientTransformation:
    """AdamW avec warmup linéaire + SGDR + clip + zero_nans.

    Args:
        lr: Learning rate peak.
        weight_decay: AdamW weight decay.
        warmup_steps: Pas de warmup linéaire 0 → lr.
        total_steps: Total de pas d'entraînement.
        grad_clip: Norme maximale des gradients (0 = désactivé).
        b2: Moment second AdamW. 0.95 pour SSM, 0.999 standard.
        n_restarts: Nombre de cycles SGDR (0 = cosine simple).

    Returns:
        optax.GradientTransformation chaîné.
    """
    train_steps = total_steps - warmup_steps

    # Schedule: warmup linéaire + SGDR cosines avec période doublante
    if n_restarts > 0:
        first_cycle = max(train_steps // (2 ** n_restarts - 1), 1)
        schedules = [optax.linear_schedule(0.0, lr, warmup_steps)]
        boundaries = [warmup_steps]
        cycle_len = first_cycle
        for i in range(n_restarts):
            schedules.append(optax.cosine_decay_schedule(lr, cycle_len))
            if i < n_restarts - 1:
                boundaries.append(boundaries[-1] + cycle_len)
            cycle_len *= 2
        schedule = optax.join_schedules(schedules, boundaries)
    else:
        schedule = optax.join_schedules(
            [optax.linear_schedule(0.0, lr, warmup_steps),
             optax.cosine_decay_schedule(lr, train_steps)],
            [warmup_steps],
        )

    # Chaîne: clip → zero_nans → adamw
    # Ordre important: clip calcule la norme AVANT zero_nans pour une norme juste.
    # zero_nans est AVANT adamw pour ne pas corrompre les moments m1/m2.
    components = []
    if grad_clip > 0.0:
        components.append(optax.clip_by_global_norm(grad_clip))
    components.append(optax.zero_nans())
    components.append(optax.adamw(learning_rate=schedule, weight_decay=weight_decay, b2=b2))

    return optax.chain(*components)


# ─────────────────────────────────────────────────────────────────────────────
# EMA update (target encoder JEPA)
# ─────────────────────────────────────────────────────────────────────────────

def ema_update(target: PyTree, online: PyTree, tau: float) -> PyTree:
    """EMA: target = tau * target + (1 - tau) * online.

    tau=0.996 → retard moyen de ~250 steps (bon pour pseudo-labels JEPA).
    tau=0.005 → TD3-style (critic target pour RL).
    """
    return jax.tree.map(lambda t, o: tau * t + (1.0 - tau) * o, target, online)


def cosine_tau(epoch: int, tau_start: float, tau_end: float, anneal_epochs: int) -> float:
    """Annelage cosinus du momentum EMA de tau_start vers tau_end."""
    if epoch >= anneal_epochs:
        return tau_end
    progress = epoch / anneal_epochs
    return tau_end - (tau_end - tau_start) * (1.0 + jnp.cos(jnp.pi * progress)) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Train step avec donate_argnums
# ─────────────────────────────────────────────────────────────────────────────

def make_train_step(model_apply, loss_fn):
    """Fabrique un train_step JIT avec buffer donation.

    donate_argnums=(0, 1): JAX peut réutiliser la mémoire de state et batch
    pour les nouvelles valeurs. Sur v5p-128 (32 Go params), ça évite une
    double allocation de ~64 Go au moment de la mise à jour.

    Usage:
        train_step = make_train_step(model.apply, my_loss_fn)
        state, metrics = train_step(state, batch)

    loss_fn signature: (params, batch, rng) -> (scalar_loss, aux_dict)
    """
    @partial(jax.jit, donate_argnums=(0, 1))
    def _step(state, batch):
        rng, step_rng = jax.random.split(state.rng)
        (loss, aux), grads = jax.value_and_grad(
            loss_fn, has_aux=True,
        )(state.params, batch, step_rng)
        grad_norm = optax.global_norm(grads)
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng=rng)
        return state, {"loss": loss, "grad_norm": grad_norm, **aux}
    return _step


# ─────────────────────────────────────────────────────────────────────────────
# LR scaling rules pour T-Shirt sizing
# ─────────────────────────────────────────────────────────────────────────────

def lr_for_d_model(base_lr: float, base_d: int, target_d: int) -> float:
    """μP-inspired LR scaling: lr ∝ 1/sqrt(d_model).

    Validé empiriquement:
        d=256  → lr=3e-4
        d=1024 → lr=1.5e-4  (≈ 3e-4 * sqrt(256/1024))
        d=4096 → lr=7.5e-5

    Usage: lr = lr_for_d_model(3e-4, base_d=256, target_d=d_model)
    """
    import math
    return base_lr * math.sqrt(base_d / target_d)
