"""Cross-Sectional Market-Neutral DQN training.

Same Q-network (682K params) as single-asset DQN. The Q-net scores K assets
in batch → directional scores → ranking → Long/Short/Flat allocation.
Market-neutral by construction (equal Long + Short count).

Key differences from single-asset:
  - Each collect step samples K assets per portfolio, N portfolios in parallel
  - Transitions are per-asset (obs, action=position, reward, next_obs, done)
  - Exploration = Gaussian noise on scores before ranking (preserves neutrality)
  - Eval metric = portfolio Sharpe (mean episode return / std)

Data flow per collect:
  sample_portfolio_windows → (N, K, win_len, ...) numpy
  batched_collect (vmap) → (N, episode_len, K, ...) transitions
  Flatten → N × episode_len × K transitions → replay buffer
  DQN update (unchanged from dqn_agent.py)

Usage:
    PYTHONPATH=. python scripts/train_cross_sectional.py \
        --buffer_dir data/rl_buffer_v2/ --total_steps 1000000

    # Smoke test
    PYTHONPATH=. python scripts/train_cross_sectional.py \
        --buffer_dir data/rl_buffer_v2/ --total_steps 500 --n_portfolios 2 --k_assets 4
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_cs")


def load_buffer(buffer_dir: str, episode_len: int, val_ratio: float = 0.2):
    """Load RL buffer, split train/val by asset hash."""
    import hashlib
    with open(os.path.join(buffer_dir, "manifest.json")) as f:
        manifest = json.load(f)
    d_model = manifest["d_model"]
    train, val = [], []
    min_len = episode_len + 2
    for info in manifest["assets"]:
        pair, path = info["pair"], info["path"]
        if not os.path.exists(path):
            path = os.path.join(buffer_dir, Path(path).name)
        if not os.path.exists(path):
            continue
        data = np.load(path)
        h, v, r = data["h_last"], data["vol"], data["returns"]
        bif = data["bifurcation_index"] if "bifurcation_index" in data else np.zeros(h.shape[0], dtype=np.float32)
        if h.shape[0] < min_len:
            continue
        bucket = int(hashlib.md5(pair.encode()).hexdigest(), 16) % 1000
        (val if bucket < int(val_ratio * 1000) else train).append((pair, h, v, r, bif))
    log.info("Buffer: %d train, %d val, d_model=%d", len(train), len(val), d_model)
    return train, val, d_model


def sample_portfolio_windows(
    assets: list,
    episode_len: int,
    rng: np.random.Generator,
    n_portfolios: int,
    k_assets: int,
    use_per: bool = False,
    per_alpha: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample N portfolios of K assets each, pre-sliced to windows.

    Returns:
        h: (N, K, win_len, d_model)
        v: (N, K, win_len)
        r: (N, K, win_len)
    """
    win_len = episode_len + 1
    d_model = assets[0][1].shape[1]
    h = np.zeros((n_portfolios, k_assets, win_len, d_model), dtype=np.float32)
    v = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    r = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)

    for i in range(n_portfolios):
        # PER: weight asset selection by mean bifurcation_index per asset
        # High bifurcation = chaotic regime = more informative transitions → sample more
        if use_per and len(assets) > 1:
            asset_priorities = np.array(
                [(a[4].mean() + 1e-5) ** per_alpha for a in assets],
                dtype=np.float64,
            )
            asset_weights = asset_priorities / asset_priorities.sum()
            asset_indices = rng.choice(
                len(assets), size=k_assets,
                replace=k_assets > len(assets), p=asset_weights,
            )
        else:
            asset_indices = rng.choice(len(assets), size=k_assets, replace=k_assets > len(assets))
        for j, ai in enumerate(asset_indices):
            _, h_a, v_a, r_a, bif_a = assets[ai]
            max_start = max(h_a.shape[0] - episode_len - 1, 1)
            si = rng.integers(max_start)
            h[i, j] = h_a[si:si + win_len]
            v[i, j] = v_a[si:si + win_len]
            r[i, j] = r_a[si:si + win_len]

    return h, v, r


def main():
    parser = argparse.ArgumentParser(description="Train Cross-Sectional Market-Neutral DQN")
    parser.add_argument("--buffer_dir", type=str, default="data/rl_buffer_v2/")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/cross_sectional/")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--n_eval", type=int, default=None)
    parser.add_argument("--n_portfolios", type=int, default=None)
    parser.add_argument("--k_assets", type=int, default=None)
    parser.add_argument("--episode_len", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    # ── JAX init ──
    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from src.jax_v6.config import CrossSectionalConfig, load_config
    from src.jax_v6.strate_iv.env_cross_sectional import (
        obs_dim as get_obs_dim,
        make_portfolio_obs,
        compute_scores,
        noisy_rank_and_allocate,
        rank_and_allocate,
        score_weighted_allocate,
        noisy_score_weighted_allocate,
        reset_portfolio,
        step_portfolio,
    )
    from src.jax_v6.strate_iv.dqn_agent import (
        create_agent, update_step, update_target, ReplayBuffer,
    )

    # ── Config ──
    if args.config:
        full_cfg = load_config(args.config)
        cfg = full_cfg.cross_sectional or CrossSectionalConfig()
    else:
        cfg = CrossSectionalConfig()

    # CLI overrides
    from dataclasses import replace
    overrides = {}
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps
    if args.eval_freq is not None:
        overrides["eval_freq"] = args.eval_freq
    if args.n_eval is not None:
        overrides["n_eval"] = args.n_eval
    if args.n_portfolios is not None:
        overrides["n_portfolios"] = args.n_portfolios
    if args.k_assets is not None:
        overrides["k_assets"] = args.k_assets
    if args.episode_len is not None:
        overrides["episode_len"] = args.episode_len
    if overrides:
        cfg = replace(cfg, **overrides)

    K = cfg.k_assets
    N = cfg.n_portfolios
    episode_len = cfg.episode_len
    n_long = max(int(K * cfg.long_frac), 1)
    n_short = max(int(K * cfg.short_frac), 1)
    steps_per_collect = N * episode_len * K  # transitions per collect

    log.info("Config: K=%d, N=%d, L=%d, long=%d, short=%d, steps/collect=%d",
             K, N, episode_len, n_long, n_short, steps_per_collect)

    # ── Load buffer ──
    train_assets, val_assets, d_model = load_buffer(args.buffer_dir, episode_len)
    if not train_assets:
        log.error("No training assets")
        return
    if len(train_assets) < K:
        log.warning("Only %d train assets < K=%d, sampling with replacement", len(train_assets), K)

    o_dim = get_obs_dim(d_model)
    win_len = episode_len + 1
    log.info("obs_dim=%d, d_model=%d, win_len=%d", o_dim, d_model, win_len)

    # ── Init agent (same Q-net as single-asset) ──
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    q_net, agent_state, tx = create_agent(
        obs_dim=o_dim, hidden_dim=cfg.hidden_dim,
        n_actions=3, n_quantiles=cfg.n_quantiles,
        lr=cfg.lr, key=init_key,
    )
    log.info("QNetwork: %d params", sum(x.size for x in jax.tree.leaves(agent_state.q_params)))

    # ── Replay buffer (per-asset transitions) ──
    replay = ReplayBuffer(capacity=cfg.buffer_capacity, obs_dim=o_dim)

    # ── Define single-portfolio functions ──

    def collect_single(q_params, h_K, vol_K, ret_K, key, noise_scale):
        """Run one portfolio episode. Returns per-asset transitions.

        V2: per-asset reward (z-scored alpha), not portfolio mean.

        Args:
            q_params: Q-network params (shared)
            h_K: (K, win_len, d_model)
            vol_K: (K, win_len)
            ret_K: (K, win_len)
            key: PRNG key
            noise_scale: exploration noise on scores

        Returns:
            Tuple of (obs, action, per_asset_reward, next_obs, done)
            each (episode_len, K, ...)
        """
        obs, state = reset_portfolio(h_K, vol_K, K)

        def body(carry, _):
            obs, state, key = carry
            key, score_key, noise_key = jax.random.split(key, 3)

            # Score K assets
            scores = compute_scores(q_net, q_params, obs, cfg.cvar_alpha)

            # Allocate: soft (conviction-weighted) or hard (equal-weight ranking)
            if cfg.soft_alloc:
                weights = noisy_score_weighted_allocate(
                    scores, noise_key, noise_scale, n_long, n_short)
            else:
                positions = noisy_rank_and_allocate(
                    scores, noise_key, noise_scale, n_long, n_short)
                weights = positions.astype(jnp.float32)

            # Step portfolio (V3: soft weights + quadratic slippage; v6.1: risk_parity)
            next_obs, next_state, per_asset_rew, portfolio_rew, done = step_portfolio(
                state, weights, h_K, vol_K, ret_K,
                episode_len, cfg.fee_rate, cfg.slippage_factor,
                risk_parity=cfg.risk_parity,
            )

            # Per-asset action = sign(weight) mapped to {0,1,2} for Q-net replay
            actions = (jnp.sign(weights) + 1).astype(jnp.int32)  # {-1,0,+1} → {0,1,2}

            # Store per-asset rewards (K,) — not portfolio mean
            done_K = jnp.broadcast_to(done.astype(jnp.float32), (K,))

            return (next_obs, next_state, key), (obs, actions, per_asset_rew, next_obs, done_K)

        _, transitions = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        # transitions: (obs(L,K,D), actions(L,K), reward(L,K), next_obs(L,K,D), done(L,K))
        return transitions

    def eval_single(q_params, h_K, vol_K, ret_K, key):
        """Greedy portfolio episode, returns total portfolio reward."""
        obs, state = reset_portfolio(h_K, vol_K, K)

        def body(carry, _):
            obs, state, key = carry
            key, _ = jax.random.split(key)

            scores = compute_scores(q_net, q_params, obs, cfg.cvar_alpha)
            if cfg.soft_alloc:
                weights = score_weighted_allocate(scores, n_long, n_short)
            else:
                positions = rank_and_allocate(scores, n_long, n_short)
                weights = positions.astype(jnp.float32)

            next_obs, next_state, per_asset_rew, portfolio_rew, done = step_portfolio(
                state, weights, h_K, vol_K, ret_K,
                episode_len, cfg.fee_rate, cfg.slippage_factor,
                risk_parity=cfg.risk_parity,
            )
            return (next_obs, next_state, key), portfolio_rew

        _, rewards = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return rewards.sum()

    # ── vmap + jit ──
    # q_params: shared | h/v/r/key/noise: batched per portfolio
    batched_collect = jax.jit(jax.vmap(
        collect_single, in_axes=(None, 0, 0, 0, 0, 0)))
    batched_eval = jax.jit(jax.vmap(
        eval_single, in_axes=(None, 0, 0, 0, 0)))

    # ── Q-network update + target (reused from dqn_agent) ──
    @jax.jit
    def jit_update(agent_state, batch):
        return update_step(q_net, agent_state, tx, batch,
                           gamma=cfg.gamma, cvar_alpha=cfg.cvar_alpha,
                           n_quantiles=cfg.n_quantiles)

    @jax.jit
    def jit_update_target(agent_state):
        return update_target(agent_state, tau=cfg.ema_tau)

    np_rng = np.random.default_rng(42)

    # ── JIT warmup ──
    log.info("JIT compiling (vmap collect %d portfolios × %d assets + eval + update)...", N, K)
    t_jit = time.time()

    # Warmup collect
    dh = jnp.zeros((N, K, win_len, d_model))
    dv = jnp.zeros((N, K, win_len))
    dr = jnp.zeros((N, K, win_len))
    key, *wks = jax.random.split(key, N + 1)
    # Ape-X: per-portfolio noise scale (log-spaced)
    noise_vector = jnp.logspace(jnp.log10(0.01), jnp.log10(0.5), N)
    log.info("Ape-X noise: min=%.3f, median=%.3f, max=%.3f",
             float(noise_vector[0]), float(noise_vector[N // 2]), float(noise_vector[-1]))
    _ = batched_collect(agent_state.q_params, dh, dv, dr, jnp.stack(wks), noise_vector)

    # Warmup eval
    n_eval = cfg.n_eval
    dh_e = jnp.zeros((n_eval, K, win_len, d_model))
    dv_e = jnp.zeros((n_eval, K, win_len))
    dr_e = jnp.zeros((n_eval, K, win_len))
    key, *wks_e = jax.random.split(key, n_eval + 1)
    _ = batched_eval(agent_state.q_params, dh_e, dv_e, dr_e, jnp.stack(wks_e))

    # Warmup update
    dummy_batch = {
        "obs": jnp.zeros((cfg.batch_size, o_dim)),
        "action": jnp.zeros(cfg.batch_size, dtype=jnp.int32),
        "reward": jnp.zeros(cfg.batch_size),
        "next_obs": jnp.zeros((cfg.batch_size, o_dim)),
        "done": jnp.zeros(cfg.batch_size),
    }
    agent_state, _ = jit_update(agent_state, dummy_batch)

    # Re-init agent (discard warmup weights)
    key, ik = jax.random.split(key)
    _, agent_state, tx = create_agent(
        obs_dim=o_dim, hidden_dim=cfg.hidden_dim,
        n_actions=3, n_quantiles=cfg.n_quantiles,
        lr=cfg.lr, key=ik,
    )
    log.info("JIT done in %.1fs", time.time() - t_jit)

    # ── Training loop ──
    os.makedirs(args.output_dir, exist_ok=True)
    best_sharpe = -float("inf")
    step_count = 0
    collect_count = 0
    update_count = 0
    last_loss = 0.0
    next_eval_at = cfg.eval_freq
    # 1 update per 16 transitions (reduced from //4 to prevent overfitting
    # with 65K transitions/collect — V1 had 16K updates/collect which was too aggressive)
    updates_per_collect = steps_per_collect // 16
    target_update_every = 25

    t_train = time.time()

    while step_count < cfg.total_steps:
        # ── Sample N×K windows (numpy) ──
        h_np, v_np, r_np = sample_portfolio_windows(
            train_assets, episode_len, np_rng, N, K,
            use_per=cfg.use_per, per_alpha=cfg.per_alpha)

        # ── Collect N portfolios in parallel (Ape-X noise) ──
        key, *ep_keys = jax.random.split(key, N + 1)
        transitions = batched_collect(
            agent_state.q_params,
            jnp.array(h_np), jnp.array(v_np), jnp.array(r_np),
            jnp.stack(ep_keys), noise_vector,
        )
        # transitions: (obs(N,L,K,D), actions(N,L,K), reward(N,L,K), next_obs(N,L,K,D), done(N,L,K))
        obs_t, act_t, rew_t, nobs_t, done_t = transitions

        # ── Flatten per-asset transitions → replay buffer ──
        # V2: reward is already per-asset (N,L,K), no broadcast needed
        obs_flat = np.asarray(obs_t).reshape(-1, o_dim)
        nobs_flat = np.asarray(nobs_t).reshape(-1, o_dim)
        act_flat = np.asarray(act_t).reshape(-1)
        rew_flat = np.asarray(rew_t).reshape(-1)
        done_flat = np.asarray(done_t).reshape(-1)

        n_new = obs_flat.shape[0]  # N * L * K
        idx = replay.pos % replay.capacity
        if idx + n_new <= replay.capacity:
            s = slice(idx, idx + n_new)
            replay.obs[s] = obs_flat
            replay.action[s] = act_flat
            replay.reward[s] = rew_flat
            replay.next_obs[s] = nobs_flat
            replay.done[s] = done_flat
        else:
            first = replay.capacity - idx
            replay.obs[idx:] = obs_flat[:first]
            replay.obs[:n_new - first] = obs_flat[first:]
            replay.action[idx:] = act_flat[:first]
            replay.action[:n_new - first] = act_flat[first:]
            replay.reward[idx:] = rew_flat[:first]
            replay.reward[:n_new - first] = rew_flat[first:]
            replay.next_obs[idx:] = nobs_flat[:first]
            replay.next_obs[:n_new - first] = nobs_flat[first:]
            replay.done[idx:] = done_flat[:first]
            replay.done[:n_new - first] = done_flat[first:]
        replay.pos += n_new
        replay.size = min(replay.size + n_new, replay.capacity)

        step_count += n_new
        collect_count += 1

        # ── Gradient updates ──
        if replay.size >= cfg.batch_size:
            for _ in range(updates_per_collect):
                indices = np_rng.integers(replay.size, size=cfg.batch_size)
                batch = {
                    "obs": jnp.array(replay.obs[indices]),
                    "action": jnp.array(replay.action[indices]),
                    "reward": jnp.array(replay.reward[indices]),
                    "next_obs": jnp.array(replay.next_obs[indices]),
                    "done": jnp.array(replay.done[indices]),
                }
                agent_state, metrics = jit_update(agent_state, batch)
                update_count += 1
                last_loss = float(metrics["loss"])

                if update_count % target_update_every == 0:
                    agent_state = jit_update_target(agent_state)

        # ── Log ──
        elapsed = time.time() - t_train
        sps = step_count / max(elapsed, 0.01)
        log.info("step=%d/%d  noise=[%.2f-%.2f]  buf=%d  upd=%d  %.0f sps  loss=%.4f  "
                 "[collect #%d: %d portfolios × %d assets × %d steps]",
                 step_count, cfg.total_steps,
                 float(noise_vector[0]), float(noise_vector[-1]),
                 replay.size, update_count, sps, last_loss,
                 collect_count, N, K, episode_len)

        # ── Eval (vmap'd portfolio Sharpe) ──
        if step_count >= next_eval_at:
            next_eval_at += cfg.eval_freq
            t_ev = time.time()

            h_ev, v_ev, r_ev = sample_portfolio_windows(
                val_assets, episode_len, np_rng, n_eval, K,
                use_per=False)
            key, *ev_keys = jax.random.split(key, n_eval + 1)
            rewards = batched_eval(
                agent_state.q_params,
                jnp.array(h_ev), jnp.array(v_ev), jnp.array(r_ev),
                jnp.stack(ev_keys),
            )
            ep_rets = np.asarray(rewards)
            mean_ret = float(ep_rets.mean())
            std_ret = float(ep_rets.std()) + 1e-8
            sharpe = mean_ret / std_ret

            log.info("EVAL step=%d: sharpe=%.4f  mean=%.4f  std=%.4f  n=%d  (%.1fs)",
                     step_count, sharpe, mean_ret, float(ep_rets.std()),
                     n_eval, time.time() - t_ev)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                save_path = os.path.join(args.output_dir, "best_cs_dqn.npz")
                flat = {str(k): np.asarray(v)
                        for k, v in jax.tree.leaves_with_path(agent_state.q_params)}
                np.savez(save_path, **flat)
                log.info("New best Sharpe=%.4f → %s", best_sharpe, save_path)
                with open(os.path.join(args.output_dir, "best_eval.json"), "w") as f:
                    json.dump({
                        "step": step_count,
                        "sharpe": sharpe,
                        "mean_return": mean_ret,
                        "std_return": float(ep_rets.std()),
                        "k_assets": K,
                        "n_long": n_long,
                        "n_short": n_short,
                        "n_portfolios": N,
                    }, f, indent=2)

    # ── Final save ──
    flat = {str(k): np.asarray(v)
            for k, v in jax.tree.leaves_with_path(agent_state.q_params)}
    np.savez(os.path.join(args.output_dir, "final_cs_dqn.npz"), **flat)

    elapsed = time.time() - t0
    log.info("=== DONE: %d steps, %d collects (%d portfolios × %d assets each), "
             "%d updates, best_sharpe=%.4f, %.1fs (%.1f min) ===",
             step_count, collect_count, N, K, update_count, best_sharpe,
             elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
