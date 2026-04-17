"""TD-MPC2 continuous agent on All-Weather RL buffer (JIT-optimized).

Replaces the discrete 3-action DQN with a continuous position agent:
  - World model: encode obs → latent, predict next latent + reward
  - Actor: latent → continuous action ∈ [-1, 1] (short↔long)
  - Distributional critic: quantile regression + CVaR
  - CQL regularization (offline RL)

Episode collection and eval use lax.scan (same optimization as DQN V3).
Exploration: actor + Gaussian noise with linear decay.

Usage:
    PYTHONPATH=. python scripts/train_tdmpc2_allweather.py \
        --buffer_dir data/rl_buffer_v2/ --total_steps 100000

    # Smoke test
    PYTHONPATH=. python scripts/train_tdmpc2_allweather.py \
        --buffer_dir data/rl_buffer_v2/ --total_steps 200
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_tdmpc2")


def load_buffer(buffer_dir, episode_len, val_ratio=0.2):
    """Load RL buffer, split train/val by asset hash."""
    import hashlib
    with open(os.path.join(buffer_dir, "manifest.json")) as f:
        manifest = json.load(f)
    d_model = manifest["d_model"]
    train_assets, val_assets = [], []
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
        (val_assets if bucket < int(val_ratio * 1000) else train_assets).append((pair, h, v, r))
    log.info("Buffer: %d train, %d val, d_model=%d", len(train_assets), len(val_assets), d_model)
    return train_assets, val_assets, d_model


def save_params(agent, path):
    """Save world model + actor + critic params to npz."""
    import jax
    flat = {}
    for prefix, params in [("wm", agent.wm_params), ("actor", agent.actor_params),
                           ("critic", agent.critic_params)]:
        for k, v in jax.tree.leaves_with_path(params):
            flat[f"{prefix}/{k}"] = np.asarray(v)
    np.savez(path, **flat)


def main():
    parser = argparse.ArgumentParser(description="Train TD-MPC2 All-Weather")
    parser.add_argument("--buffer_dir", default="data/rl_buffer_v2/")
    parser.add_argument("--output_dir", default="checkpoints/tdmpc2_aw/")
    parser.add_argument("--total_steps", type=int, default=100_000)
    parser.add_argument("--eval_freq", type=int, default=5000)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--episode_len", type=int, default=64)
    parser.add_argument("--noise_start", type=float, default=0.3)
    parser.add_argument("--noise_end", type=float, default=0.05)
    parser.add_argument("--noise_decay_steps", type=int, default=50_000)
    args = parser.parse_args()

    t0 = time.time()

    # ── JAX init ──
    import jax
    import jax.numpy as jnp
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from dataclasses import replace
    from src.jax_v6.config import StrateIVJAXConfig
    from src.jax_v6.strate_iv.tdmpc2 import TDMPC2Agent
    from src.jax_v6.strate_iv.world_model import WorldModel

    # ── Load buffer ──
    episode_len = args.episode_len
    train_assets, val_assets, d_model = load_buffer(args.buffer_dir, episode_len)
    if not train_assets:
        log.error("No training assets")
        return

    obs_dim = 2 * d_model + 3   # h_t + h_prev + vol + position + cpnl
    action_dim = 1
    win_len = episode_len + 1

    # ── Config (override buffer capacity) ──
    cfg = replace(StrateIVJAXConfig(), buffer_capacity=500_000, use_planning=False, cql_alpha=0.0)
    log.info("Config: obs=%d, latent=%d, hidden=%d, n_layers=%d, lr=%.1e, gamma=%.2f, cql=%.1f",
             obs_dim, cfg.latent_dim, cfg.hidden_dim, cfg.n_layers, cfg.lr, cfg.gamma, cfg.cql_alpha)

    # ── Init agent ──
    key = jax.random.PRNGKey(42)
    agent = TDMPC2Agent(cfg, obs_dim=obs_dim, action_dim=action_dim, rng_key=key)
    log.info("TD-MPC2 params: %s", agent.param_count())

    # ── Continuous env (inline, JAX-pure) ──
    class ContState(NamedTuple):
        step: jnp.ndarray       # () int32
        position: jnp.ndarray   # () float32 ∈ [-1, 1]
        cpnl: jnp.ndarray       # () float32

    fee_rate = cfg.tc_rate    # 0.0008
    risk_lambda = 0.5

    def _make_obs(h_t, h_prev, vol_t, pos, cpnl):
        return jnp.concatenate([h_t, h_prev, jnp.array([vol_t, pos, cpnl])])

    def _reset(h_win, vol_win):
        state = ContState(jnp.int32(0), jnp.float32(0.0), jnp.float32(0.0))
        obs = _make_obs(h_win[0], h_win[0], vol_win[0], 0.0, 0.0)
        return obs, state

    def _step(state, action_scalar, h_win, vol_win, ret_win):
        t = state.step
        pos = jnp.clip(action_scalar, -1.0, 1.0)
        raw_pnl = pos * ret_win[t + 1]
        fee = fee_rate * jnp.abs(pos - state.position)
        risk = jnp.abs(pos) * risk_lambda * jnp.abs(vol_win[t])
        reward = raw_pnl - fee - risk
        new_cpnl = state.cpnl + reward
        new_step = t + 1
        done = new_step >= episode_len
        new_state = ContState(new_step, pos, new_cpnl)
        obs = _make_obs(h_win[new_step], h_win[new_step - 1],
                        vol_win[new_step], pos, new_cpnl)
        return obs, new_state, reward, done

    # ── JIT'd episode collection (actor + noise) ──
    wm_module = agent.world_model
    actor_module = agent.actor_module

    @jax.jit
    def collect_episode(wm_params, actor_params, h_win, vol_win, ret_win, key, noise_std):
        obs, state = _reset(h_win, vol_win)

        def body(carry, _):
            obs, state, key = carry
            key, nk = jax.random.split(key)
            z = wm_module.apply(wm_params, obs[None],
                                method=WorldModel.encode)[0]       # (latent,)
            a = actor_module.apply(actor_params, z[None])[0]  # (1,)
            a = jnp.clip(a + noise_std * jax.random.normal(nk, a.shape), -1.0, 1.0)
            next_obs, ns, reward, done = _step(state, a[0], h_win, vol_win, ret_win)
            return (next_obs, ns, key), (obs, a, reward, next_obs, done.astype(jnp.float32))

        (_, _, _), transitions = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return transitions  # obs(L,D), action(L,1), reward(L,), next_obs(L,D), done(L,)

    @jax.jit
    def eval_episode(wm_params, actor_params, h_win, vol_win, ret_win, key):
        obs, state = _reset(h_win, vol_win)

        def body(carry, _):
            obs, state, key = carry
            key, _ = jax.random.split(key)
            z = wm_module.apply(wm_params, obs[None],
                                method=WorldModel.encode)[0]
            a = actor_module.apply(actor_params, z[None])[0]
            next_obs, ns, reward, done = _step(state, a[0], h_win, vol_win, ret_win)
            return (next_obs, ns, key), reward

        (_, _, _), rewards = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return rewards.sum()

    # ── JIT warmup ──
    log.info("JIT warmup...")
    t_jit = time.time()
    dh = jnp.zeros((win_len, d_model))
    dv = jnp.zeros((win_len,))
    dr = jnp.zeros((win_len,))
    key, wk1, wk2 = jax.random.split(key, 3)
    _ = collect_episode(agent.wm_params, agent.actor_params, dh, dv, dr, wk1, jnp.float32(0.3))
    _ = eval_episode(agent.wm_params, agent.actor_params, dh, dv, dr, wk2)
    log.info("JIT done in %.1fs", time.time() - t_jit)

    # ── Replay buffer (numpy, continuous actions) ──
    cap = cfg.buffer_capacity
    buf_obs = np.zeros((cap, obs_dim), dtype=np.float32)
    buf_act = np.zeros((cap, action_dim), dtype=np.float32)
    buf_rew = np.zeros(cap, dtype=np.float32)
    buf_nobs = np.zeros((cap, obs_dim), dtype=np.float32)
    buf_done = np.zeros(cap, dtype=np.float32)
    buf_pos = 0
    buf_size = 0

    np_rng = np.random.default_rng(42)

    # ── Training loop ──
    os.makedirs(args.output_dir, exist_ok=True)
    best_sharpe = -float("inf")
    step_count = 0
    episode_count = 0
    update_count = 0
    last_metrics = {}
    next_eval_at = args.eval_freq
    updates_per_ep = episode_len // 4

    t_train = time.time()

    while step_count < args.total_steps:
        # ── Sample window ──
        ai = np_rng.integers(len(train_assets))
        _, h_last, vol, returns = train_assets[ai]
        ms = max(h_last.shape[0] - episode_len - 1, 1)
        si = np_rng.integers(ms)

        h_win = jnp.array(h_last[si:si + win_len])
        vol_win = jnp.array(vol[si:si + win_len])
        ret_win = jnp.array(returns[si:si + win_len])

        # ── Noise schedule (linear decay) ──
        frac = min(step_count / max(args.noise_decay_steps, 1), 1.0)
        noise_std = args.noise_start + (args.noise_end - args.noise_start) * frac

        # ── Collect episode (JIT'd lax.scan) ──
        key, ek = jax.random.split(key)
        transitions = collect_episode(
            agent.wm_params, agent.actor_params,
            h_win, vol_win, ret_win, ek, jnp.float32(noise_std))

        # ── Add to replay ──
        obs_np, act_np, rew_np, nobs_np, done_np = [np.asarray(x) for x in transitions]
        n = episode_len
        idx = buf_pos % cap
        if idx + n <= cap:
            s = slice(idx, idx + n)
            buf_obs[s] = obs_np
            buf_act[s] = act_np
            buf_rew[s] = rew_np
            buf_nobs[s] = nobs_np
            buf_done[s] = done_np
        else:
            for i in range(n):
                j = (buf_pos + i) % cap
                buf_obs[j] = obs_np[i]
                buf_act[j] = act_np[i]
                buf_rew[j] = rew_np[i]
                buf_nobs[j] = nobs_np[i]
                buf_done[j] = done_np[i]
        buf_pos += n
        buf_size = min(buf_size + n, cap)

        step_count += episode_len
        episode_count += 1

        # ── Updates (world model + critic + actor) ──
        if buf_size >= cfg.batch_size:
            for _ in range(updates_per_ep):
                indices = np_rng.integers(buf_size, size=cfg.batch_size)
                batch = {
                    "obs": jnp.array(buf_obs[indices]),
                    "action": jnp.array(buf_act[indices]),
                    "reward": jnp.array(buf_rew[indices]),
                    "next_obs": jnp.array(buf_nobs[indices]),
                    "done": jnp.array(buf_done[indices]),
                }
                last_metrics = agent.update(batch)
                update_count += 1

        # ── Log ──
        if episode_count % 10 == 0 and last_metrics:
            elapsed = time.time() - t_train
            sps = step_count / max(elapsed, 0.01)
            losses = "  ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in sorted(last_metrics.items()))
            log.info("step=%d/%d  noise=%.3f  buf=%d  upd=%d  %.0f sps  %s",
                     step_count, args.total_steps, noise_std, buf_size,
                     update_count, sps, losses)

        # ── Eval ──
        if step_count >= next_eval_at:
            next_eval_at += args.eval_freq
            t_ev = time.time()
            ep_rets = []
            for _ in range(args.n_eval):
                vi = np_rng.integers(len(val_assets))
                _, vh, vv, vr = val_assets[vi]
                vms = max(vh.shape[0] - episode_len - 1, 1)
                vsi = np_rng.integers(vms)
                key, evk = jax.random.split(key)
                ret = eval_episode(agent.wm_params, agent.actor_params,
                                   jnp.array(vh[vsi:vsi + win_len]),
                                   jnp.array(vv[vsi:vsi + win_len]),
                                   jnp.array(vr[vsi:vsi + win_len]), evk)
                ep_rets.append(float(ret))

            ep_rets = np.array(ep_rets)
            mean_ret = float(ep_rets.mean())
            std_ret = float(ep_rets.std()) + 1e-8
            sharpe = mean_ret / std_ret

            log.info("EVAL step=%d: sharpe=%.4f  mean=%.4f  std=%.4f  (%.1fs)",
                     step_count, sharpe, mean_ret, float(ep_rets.std()), time.time() - t_ev)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                save_params(agent, os.path.join(args.output_dir, "best_tdmpc2.npz"))
                log.info("New best Sharpe=%.4f → %s", best_sharpe, args.output_dir)
                with open(os.path.join(args.output_dir, "best_eval.json"), "w") as f:
                    json.dump({"step": step_count, "sharpe": sharpe,
                               "mean_return": mean_ret, "std_return": float(ep_rets.std()),
                               "noise_std": noise_std}, f, indent=2)

    # ── Final save ──
    save_params(agent, os.path.join(args.output_dir, "final_tdmpc2.npz"))
    elapsed = time.time() - t0
    log.info("=== DONE: %d steps, %d episodes, %d updates, best_sharpe=%.4f, %.1fs (%.1f min) ===",
             step_count, episode_count, update_count, best_sharpe, elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
