"""All-Weather DQN — V4 vmap multi-env.

Key change: jax.vmap runs N_ENVS episodes in parallel per collection step.
Each collection produces N_ENVS × 64 = 16K+ transitions from different
assets/timestamps simultaneously. Massive decorrelation.

Usage:
    PYTHONPATH=. python scripts/train_strate_iv_allweather.py \
        --buffer_dir data/rl_buffer_v2/ --total_steps 1000000 --n_envs 256

    # Smoke test
    PYTHONPATH=. python scripts/train_strate_iv_allweather.py \
        --buffer_dir data/rl_buffer_v2/ --total_steps 500 --n_envs 4
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_aw")


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
        if h.shape[0] < min_len:
            continue
        bucket = int(hashlib.md5(pair.encode()).hexdigest(), 16) % 1000
        (val if bucket < int(val_ratio * 1000) else train).append((pair, h, v, r))
    log.info("Buffer: %d train, %d val, d_model=%d", len(train), len(val), d_model)
    return train, val, d_model


def sample_windows(assets, episode_len, rng, n):
    """Sample n pre-sliced windows (numpy). Returns (n, win_len, ...) arrays."""
    win_len = episode_len + 1
    d_model = assets[0][1].shape[1]
    h_out = np.zeros((n, win_len, d_model), dtype=np.float32)
    v_out = np.zeros((n, win_len), dtype=np.float32)
    r_out = np.zeros((n, win_len), dtype=np.float32)
    for i in range(n):
        ai = rng.integers(len(assets))
        _, h, v, r = assets[ai]
        ms = max(h.shape[0] - episode_len - 1, 1)
        si = rng.integers(ms)
        h_out[i] = h[si:si + win_len]
        v_out[i] = v[si:si + win_len]
        r_out[i] = r[si:si + win_len]
    return h_out, v_out, r_out


def main():
    parser = argparse.ArgumentParser(description="Train All-Weather DQN (V4 vmap)")
    parser.add_argument("--buffer_dir", type=str, default="data/rl_buffer/")
    parser.add_argument("--config", type=str, default="configs/scaling/v6e_multi.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/allweather/")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--eval_freq", type=int, default=50000)
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--n_envs", type=int, default=256)
    parser.add_argument("--eps_decay", type=int, default=400000)
    args = parser.parse_args()

    t0 = time.time()

    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from src.jax_v6.config import load_config, AllWeatherConfig
    from src.jax_v6.strate_iv.env_allweather import reset, step, obs_dim as get_obs_dim
    from src.jax_v6.strate_iv.dqn_agent import (
        create_agent, compute_epsilon,
        update_step, update_target, ReplayBuffer,
    )
    from src.jax_v6.strate_iv.critic import cvar_from_quantiles

    config = load_config(args.config)
    aw_cfg = config.allweather or AllWeatherConfig()
    total_steps = args.total_steps or aw_cfg.total_steps
    episode_len = aw_cfg.episode_len
    N = args.n_envs
    steps_per_collect = N * episode_len  # e.g. 256 × 64 = 16384

    log.info("Config: n_envs=%d, episode_len=%d, steps_per_collect=%d, "
             "eps_decay=%d, total_steps=%d",
             N, episode_len, steps_per_collect, args.eps_decay, total_steps)

    # ── Load buffer ──
    train_assets, val_assets, d_model = load_buffer(args.buffer_dir, episode_len)
    if not train_assets:
        log.error("No training assets")
        return

    obs_dim = get_obs_dim(d_model)
    win_len = episode_len + 1
    log.info("obs_dim=%d, d_model=%d", obs_dim, d_model)

    # ── Init agent ──
    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    q_net, agent_state, tx = create_agent(
        obs_dim=obs_dim, hidden_dim=aw_cfg.hidden_dim,
        n_actions=3, n_quantiles=aw_cfg.n_quantiles,
        lr=aw_cfg.lr, key=init_key,
    )
    log.info("QNetwork: %d params", sum(x.size for x in jax.tree.leaves(agent_state.q_params)))

    # ── Replay buffer ──
    replay = ReplayBuffer(capacity=aw_cfg.buffer_capacity, obs_dim=obs_dim)

    # ── Define single-env functions (no @jax.jit — will be vmapped) ──

    def collect_single(q_params, h_win, vol_win, ret_win, key, epsilon):
        """One episode, returns transitions."""
        obs, state = reset(h_win, vol_win, 0)

        def body(carry, _):
            obs, state, key = carry
            key, ak, ek = jax.random.split(key, 3)
            q_vals = q_net.apply({"params": q_params}, obs[None])
            cvar = cvar_from_quantiles(q_vals[0], alpha=aw_cfg.cvar_alpha)
            greedy = jnp.argmax(cvar).astype(jnp.int32)
            rand = jax.random.randint(ak, (), 0, 3)
            action = jnp.where(jax.random.uniform(ek) < epsilon, rand, greedy).astype(jnp.int32)
            nobs, ns, reward, done = step(state, action, h_win, vol_win, ret_win,
                                          0, episode_len, aw_cfg.fee_rate, aw_cfg.risk_lambda)
            return (nobs, ns, key), (obs, action, reward, nobs, done.astype(jnp.float32))

        _, transitions = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return transitions

    def eval_single(q_params, h_win, vol_win, ret_win, key):
        """Greedy episode, returns total reward."""
        obs, state = reset(h_win, vol_win, 0)

        def body(carry, _):
            obs, state, key = carry
            key, _ = jax.random.split(key)
            q_vals = q_net.apply({"params": q_params}, obs[None])
            cvar = cvar_from_quantiles(q_vals[0], alpha=aw_cfg.cvar_alpha)
            action = jnp.argmax(cvar).astype(jnp.int32)
            nobs, ns, reward, done = step(state, action, h_win, vol_win, ret_win,
                                          0, episode_len, aw_cfg.fee_rate, aw_cfg.risk_lambda)
            return (nobs, ns, key), reward

        _, rewards = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return rewards.sum()

    # ── vmap + jit ──
    #   q_params: shared | h/v/r/key/epsilon: batched (Ape-X style)
    batched_collect = jax.jit(jax.vmap(
        collect_single, in_axes=(None, 0, 0, 0, 0, 0)))
    batched_eval = jax.jit(jax.vmap(
        eval_single, in_axes=(None, 0, 0, 0, 0)))

    # ── Q-network update + target ──
    @jax.jit
    def jit_update(agent_state, batch):
        return update_step(q_net, agent_state, tx, batch,
                           gamma=aw_cfg.gamma, cvar_alpha=aw_cfg.cvar_alpha,
                           n_quantiles=aw_cfg.n_quantiles)

    @jax.jit
    def jit_update_target(agent_state):
        return update_target(agent_state, tau=aw_cfg.ema_tau)

    np_rng = np.random.default_rng(42)

    # ── JIT warmup ──
    log.info("JIT compiling (vmap collect %d envs + eval + update)...", N)
    t_jit = time.time()

    # Warmup collect (N envs)
    dh = jnp.zeros((N, win_len, d_model))
    dv = jnp.zeros((N, win_len))
    dr = jnp.zeros((N, win_len))
    key, *wks = jax.random.split(key, N + 1)
    # Ape-X: fixed per-env epsilons (log-spaced from sniper=0.01 to explorer=0.5)
    eps_vector = jnp.logspace(jnp.log10(0.01), jnp.log10(0.5), N)
    log.info("Ape-X epsilons: min=%.3f, median=%.3f, max=%.3f",
             float(eps_vector[0]), float(eps_vector[N // 2]), float(eps_vector[-1]))
    _ = batched_collect(agent_state.q_params, dh, dv, dr, jnp.stack(wks), eps_vector)

    # Warmup eval (n_eval envs)
    dh_e = jnp.zeros((args.n_eval, win_len, d_model))
    dv_e = jnp.zeros((args.n_eval, win_len))
    dr_e = jnp.zeros((args.n_eval, win_len))
    key, *wks_e = jax.random.split(key, args.n_eval + 1)
    _ = batched_eval(agent_state.q_params, dh_e, dv_e, dr_e, jnp.stack(wks_e))

    # Warmup update
    dummy_batch = {
        "obs": jnp.zeros((aw_cfg.batch_size, obs_dim)),
        "action": jnp.zeros(aw_cfg.batch_size, dtype=jnp.int32),
        "reward": jnp.zeros(aw_cfg.batch_size),
        "next_obs": jnp.zeros((aw_cfg.batch_size, obs_dim)),
        "done": jnp.zeros(aw_cfg.batch_size),
    }
    agent_state, _ = jit_update(agent_state, dummy_batch)

    # Re-init agent (discard warmup weights)
    key, ik = jax.random.split(key)
    _, agent_state, tx = create_agent(
        obs_dim=obs_dim, hidden_dim=aw_cfg.hidden_dim,
        n_actions=3, n_quantiles=aw_cfg.n_quantiles,
        lr=aw_cfg.lr, key=ik,
    )
    log.info("JIT done in %.1fs", time.time() - t_jit)

    # ── Training loop ──
    os.makedirs(args.output_dir, exist_ok=True)
    best_sharpe = -float("inf")
    step_count = 0
    collect_count = 0
    update_count = 0
    last_loss = 0.0
    next_eval_at = args.eval_freq
    # Keep same replay ratio as V3: 1 update per 4 transitions
    updates_per_collect = steps_per_collect // 4
    target_update_every = 25  # gradient steps between EMA updates

    t_train = time.time()

    while step_count < total_steps:
        # ── Sample N windows (numpy, ~1ms) ──
        h_np, v_np, r_np = sample_windows(train_assets, episode_len, np_rng, N)

        # ── Collect N episodes in parallel (Ape-X: per-env fixed epsilon) ──
        key, *ep_keys = jax.random.split(key, N + 1)
        transitions = batched_collect(
            agent_state.q_params,
            jnp.array(h_np), jnp.array(v_np), jnp.array(r_np),
            jnp.stack(ep_keys), eps_vector,
        )
        # transitions: (obs, act, rew, nobs, done) each (N, episode_len, ...)

        # ── Flatten + add to replay ──
        obs_f, act_f, rew_f, nobs_f, done_f = [
            np.asarray(x).reshape(-1, *x.shape[2:]) for x in transitions
        ]
        n_new = obs_f.shape[0]  # N * episode_len
        idx = replay.pos % replay.capacity
        if idx + n_new <= replay.capacity:
            s = slice(idx, idx + n_new)
            replay.obs[s] = obs_f
            replay.action[s] = act_f.ravel()
            replay.reward[s] = rew_f.ravel()
            replay.next_obs[s] = nobs_f
            replay.done[s] = done_f.ravel()
        else:
            first = replay.capacity - idx
            replay.obs[idx:] = obs_f[:first]
            replay.obs[:n_new - first] = obs_f[first:]
            replay.action[idx:] = act_f.ravel()[:first]
            replay.action[:n_new - first] = act_f.ravel()[first:]
            replay.reward[idx:] = rew_f.ravel()[:first]
            replay.reward[:n_new - first] = rew_f.ravel()[first:]
            replay.next_obs[idx:] = nobs_f[:first]
            replay.next_obs[:n_new - first] = nobs_f[first:]
            replay.done[idx:] = done_f.ravel()[:first]
            replay.done[:n_new - first] = done_f.ravel()[first:]
        replay.pos += n_new
        replay.size = min(replay.size + n_new, replay.capacity)

        step_count += n_new
        collect_count += 1

        # ── Gradient updates ──
        if replay.size >= aw_cfg.batch_size:
            for u in range(updates_per_collect):
                indices = np_rng.integers(replay.size, size=aw_cfg.batch_size)
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

        # ── Log (every collection) ──
        elapsed = time.time() - t_train
        sps = step_count / max(elapsed, 0.01)
        log.info("step=%d/%d  eps=[%.2f-%.2f]  buf=%d  upd=%d  %.0f sps  loss=%.4f  "
                 "[collect #%d: %d envs × %d steps]",
                 step_count, total_steps,
                 float(eps_vector[0]), float(eps_vector[-1]),
                 replay.size, update_count, sps, last_loss,
                 collect_count, N, episode_len)

        # ── Eval (vmap'd — all episodes in 1 call) ──
        if step_count >= next_eval_at:
            next_eval_at += args.eval_freq
            t_ev = time.time()

            h_ev, v_ev, r_ev = sample_windows(val_assets, episode_len, np_rng, args.n_eval)
            key, *ev_keys = jax.random.split(key, args.n_eval + 1)
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
                     args.n_eval, time.time() - t_ev)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                save_path = os.path.join(args.output_dir, "best_dqn.npz")
                flat = {str(k): np.asarray(v)
                        for k, v in jax.tree.leaves_with_path(agent_state.q_params)}
                np.savez(save_path, **flat)
                log.info("New best Sharpe=%.4f → %s", best_sharpe, save_path)
                with open(os.path.join(args.output_dir, "best_eval.json"), "w") as f:
                    json.dump({"step": step_count, "sharpe": sharpe,
                               "mean_return": mean_ret,
                               "std_return": float(ep_rets.std()),
                               "eps_min": float(eps_vector[0]),
                               "eps_max": float(eps_vector[-1])}, f, indent=2)

    # ── Final save ──
    flat = {str(k): np.asarray(v)
            for k, v in jax.tree.leaves_with_path(agent_state.q_params)}
    np.savez(os.path.join(args.output_dir, "final_dqn.npz"), **flat)

    elapsed = time.time() - t0
    log.info("=== DONE: %d steps, %d collects (%d envs each), %d updates, "
             "best_sharpe=%.4f, %.1fs (%.1f min) ===",
             step_count, collect_count, N, update_count, best_sharpe,
             elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
