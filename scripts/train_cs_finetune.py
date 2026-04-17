"""Cross-Sectional DQN — RL fine-tuning with Embedding Adapter.

Identical to train_cross_sectional.py except:
  - Q-net = AdaptedQNet (residual adapter on JEPA embedding dims + QR-DQN head)
  - Optionally warm-starts from a pretrained DQN checkpoint
  - Differential LR: adapter=1e-5 (adapter_lr), DQN head=1e-3 (dqn_lr)
  - Adapter is zero-initialized → identity at training start (no regression)

The adapter learns a task-specific linear/nonlinear projection that maps
the self-supervised JEPA representation into a trading-signal-rich subspace,
effectively "fine-tuning" the last layer of the encoder via RL gradient.

Usage:
    # Fine-tune from best v6.1 checkpoint (recommended)
    PYTHONPATH=. python scripts/train_cs_finetune.py \\
        --buffer_dir data/rl_buffer_v4/ \\
        --init_checkpoint checkpoints/cs_v4_lr1e3/best_cs_dqn.npz \\
        --output_dir checkpoints/cs_finetune/ \\
        --total_steps 1000000

    # Train adapter from scratch (no DQN warm-start)
    PYTHONPATH=. python scripts/train_cs_finetune.py \\
        --buffer_dir data/rl_buffer_v4/ --total_steps 1000000

    # Smoke test
    PYTHONPATH=. python scripts/train_cs_finetune.py \\
        --buffer_dir data/rl_buffer_v4/ --total_steps 500 \\
        --n_portfolios 2 --k_assets 4
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_cs_ft")


# ── Checkpoint I/O ─────────────────────────────────────────────────────────────

def load_dqn_checkpoint(path: str) -> dict:
    """Load saved QNetwork params from .npz → nested dict.

    The file was saved as:
        flat = {str(k): np.asarray(v) for k, v in jax.tree.leaves_with_path(q_params)}
    Keys look like "(DictKey(key='Dense_0'), DictKey(key='kernel'))".
    We parse back to {'Dense_0': {'kernel': ..., 'bias': ...}, ...}.
    """
    data = np.load(path, allow_pickle=False)
    result = {}
    for key in data.files:
        parts = re.findall(r"key='([^']+)'", key)
        if not parts:
            parts = key.replace("\\", "/").split("/")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = data[key]
    return result


def inject_dqn_weights(adapted_params: dict, ckpt: dict) -> dict:
    """Copy pretrained QNetwork weights into AdaptedQNet's DQN head.

    Mapping: Dense_0 → fc1, Dense_1 → fc2, Dense_2 → fc_out
    The weight shapes are compatible because the adapter output has the
    same dimension as the original QNetwork input (2*d_model + 3).
    """
    mapping = {"Dense_0": "fc1", "Dense_1": "fc2", "Dense_2": "fc_out"}
    new_params = dict(adapted_params)
    loaded = 0
    for old, new in mapping.items():
        if old in ckpt and new in adapted_params:
            ckpt_kernel = ckpt[old]["kernel"]
            init_kernel = adapted_params[new]["kernel"]
            if ckpt_kernel.shape == init_kernel.shape:
                new_params[new] = {k: np.array(v) for k, v in ckpt[old].items()}
                loaded += 1
                log.info("  %s → %s  kernel=%s", old, new, ckpt_kernel.shape)
            else:
                log.warning("  Shape mismatch %s: ckpt=%s init=%s — skipping",
                            old, ckpt_kernel.shape, init_kernel.shape)
    log.info("Loaded %d / %d DQN head layers from checkpoint", loaded, len(mapping))
    return new_params


# ── Buffer loading (identical to train_cross_sectional.py) ─────────────────────

def load_buffer(buffer_dir: str, episode_len: int, val_ratio: float = 0.2):
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
        bucket = int(__import__("hashlib").md5(pair.encode()).hexdigest(), 16) % 1000
        (val if bucket < int(val_ratio * 1000) else train).append((pair, h, v, r, bif))
    log.info("Buffer: %d train, %d val, d_model=%d", len(train), len(val), d_model)
    return train, val, d_model


def sample_portfolio_windows(assets, episode_len, rng, n_portfolios, k_assets,
                              use_per=False, per_alpha=0.6):
    win_len = episode_len + 1
    d_model = assets[0][1].shape[1]
    h = np.zeros((n_portfolios, k_assets, win_len, d_model), dtype=np.float32)
    v = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    r = np.zeros((n_portfolios, k_assets, win_len), dtype=np.float32)
    for i in range(n_portfolios):
        if use_per and len(assets) > 1:
            prio = np.array([(a[4].mean() + 1e-5) ** per_alpha for a in assets], dtype=np.float64)
            p = prio / prio.sum()
            idxs = rng.choice(len(assets), size=k_assets, replace=k_assets > len(assets), p=p)
        else:
            idxs = rng.choice(len(assets), size=k_assets, replace=k_assets > len(assets))
        for j, ai in enumerate(idxs):
            _, h_a, v_a, r_a, _ = assets[ai]
            si = rng.integers(max(h_a.shape[0] - episode_len - 1, 1))
            h[i, j] = h_a[si:si + win_len]
            v[i, j] = v_a[si:si + win_len]
            r[i, j] = r_a[si:si + win_len]
    return h, v, r


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_dir",       type=str, default="data/rl_buffer_v4/")
    parser.add_argument("--init_checkpoint",  type=str, default=None,
                        help="Path to pretrained best_cs_dqn.npz for DQN warm-start")
    parser.add_argument("--output_dir",       type=str, default="checkpoints/cs_finetune/")
    parser.add_argument("--total_steps",      type=int, default=1_000_000)
    parser.add_argument("--dqn_lr",           type=float, default=1e-3)
    parser.add_argument("--adapter_lr",       type=float, default=1e-5)
    parser.add_argument("--adapter_dim",      type=int, default=256)
    parser.add_argument("--eval_freq",        type=int, default=None)
    parser.add_argument("--n_eval",           type=int, default=None)
    parser.add_argument("--n_portfolios",     type=int, default=None)
    parser.add_argument("--k_assets",         type=int, default=None)
    parser.add_argument("--episode_len",      type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    import jax
    import jax.numpy as jnp
    import optax
    log.info("JAX: %d devices (%s)", jax.device_count(), jax.devices()[0].device_kind)

    from src.jax_v6.config import CrossSectionalConfig
    from src.jax_v6.strate_iv.env_cross_sectional import (
        obs_dim as get_obs_dim,
        compute_scores, reset_portfolio, step_portfolio,
        noisy_score_weighted_allocate, score_weighted_allocate,
        noisy_rank_and_allocate, rank_and_allocate,
    )
    from src.jax_v6.strate_iv.dqn_agent import DQNState, ReplayBuffer, update_target
    from src.jax_v6.strate_iv.critic import cvar_from_quantiles, quantile_huber_loss
    from src.jax_v6.strate_iv.adapted_qnet import AdaptedQNet

    cfg = CrossSectionalConfig()
    from dataclasses import replace
    overrides = {}
    for field in ("total_steps", "eval_freq", "n_eval", "n_portfolios", "k_assets", "episode_len"):
        v = getattr(args, field, None)
        if v is not None:
            overrides[field] = v
    if overrides:
        cfg = replace(cfg, **overrides)

    K = cfg.k_assets
    N = cfg.n_portfolios
    episode_len = cfg.episode_len
    n_long  = max(int(K * cfg.long_frac), 1)
    n_short = max(int(K * cfg.short_frac), 1)
    steps_per_collect = N * episode_len * K

    log.info("Config: K=%d N=%d L=%d long=%d short=%d steps/collect=%d",
             K, N, episode_len, n_long, n_short, steps_per_collect)
    log.info("LR: adapter=%.2e  DQN=%.2e  adapter_dim=%d",
             args.adapter_lr, args.dqn_lr, args.adapter_dim)

    # ── Load buffer ──
    train_assets, val_assets, d_model = load_buffer(args.buffer_dir, episode_len)
    if not train_assets:
        log.error("No training assets found in %s", args.buffer_dir)
        return

    o_dim = get_obs_dim(d_model)
    win_len = episode_len + 1
    log.info("obs_dim=%d  d_model=%d  win_len=%d", o_dim, d_model, win_len)

    # ── Build AdaptedQNet ──
    q_net = AdaptedQNet(
        d_model=d_model,
        hidden_dim=cfg.hidden_dim,
        n_actions=3,
        n_quantiles=cfg.n_quantiles,
        adapter_dim=args.adapter_dim,
    )

    key = jax.random.PRNGKey(42)
    key, init_key = jax.random.split(key)
    dummy_obs = jnp.zeros((1, o_dim))
    q_params = q_net.init(init_key, dummy_obs)["params"]

    # ── Optionally warm-start DQN head from checkpoint ──
    if args.init_checkpoint:
        log.info("Loading DQN checkpoint: %s", args.init_checkpoint)
        ckpt = load_dqn_checkpoint(args.init_checkpoint)
        q_params = inject_dqn_weights(q_params, ckpt)
    else:
        log.info("No checkpoint provided — training adapter + DQN head from scratch")

    total_params = sum(x.size for x in jax.tree.leaves(q_params))
    adapter_params = sum(
        q_params[k][s].size
        for k in ("adapt_fc", "adapt_out")
        for s in ("kernel", "bias")
        if k in q_params
    )
    log.info("AdaptedQNet: %d total params  (%d adapter / %d DQN head)",
             total_params, adapter_params, total_params - adapter_params)

    # ── Differential LR optimizer (optax.multi_transform) ──
    def make_param_labels(params):
        labels = {}
        for k, v in params.items():
            label = "adapter" if k.startswith("adapt") else "dqn"
            labels[k] = jax.tree.map(lambda _: label, v)
        return labels

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.multi_transform(
            {"adapter": optax.adam(args.adapter_lr),
             "dqn":     optax.adam(args.dqn_lr)},
            make_param_labels,
        ),
    )
    opt_state = tx.init(q_params)
    target_params = jax.tree.map(lambda x: x.copy(), q_params)

    agent_state = DQNState(
        q_params=q_params,
        target_params=target_params,
        opt_state=opt_state,
        step=jnp.int32(0),
    )

    # ── Replay buffer ──
    replay = ReplayBuffer(capacity=cfg.buffer_capacity, obs_dim=o_dim)

    # ── Custom update_step with multi_transform tx ──
    @jax.jit
    def jit_update(agent_state, batch):
        taus = (jnp.arange(cfg.n_quantiles, dtype=jnp.float32) + 0.5) / cfg.n_quantiles

        def loss_fn(q_params):
            q_all  = q_net.apply({"params": q_params}, batch["obs"])
            B = batch["obs"].shape[0]
            q_pred = q_all[jnp.arange(B), batch["action"].astype(jnp.int32)]

            q_next_all = q_net.apply({"params": agent_state.target_params}, batch["next_obs"])
            cvar_next  = jax.vmap(lambda q: cvar_from_quantiles(q, alpha=cfg.cvar_alpha))(q_next_all)
            best_a     = jnp.argmax(cvar_next, axis=-1)
            q_next     = q_next_all[jnp.arange(B), best_a]

            targets = batch["reward"][:, None] + cfg.gamma * (1.0 - batch["done"][:, None]) * q_next
            loss = quantile_huber_loss(q_pred, targets, taus)
            return loss, q_pred.mean()

        (loss, mean_q), grads = jax.value_and_grad(loss_fn, has_aux=True)(agent_state.q_params)
        updates, new_opt = tx.update(grads, agent_state.opt_state, agent_state.q_params)
        new_params = optax.apply_updates(agent_state.q_params, updates)

        new_state = DQNState(
            q_params=new_params,
            target_params=agent_state.target_params,
            opt_state=new_opt,
            step=agent_state.step + 1,
        )
        return new_state, {"loss": loss, "mean_q": mean_q}

    @jax.jit
    def jit_update_target(agent_state):
        return update_target(agent_state, tau=cfg.ema_tau)

    # ── Collect / eval (same as train_cross_sectional.py) ──
    def collect_single(q_params, h_K, vol_K, ret_K, key, noise_scale):
        obs, state = reset_portfolio(h_K, vol_K, K)

        def body(carry, _):
            obs, state, key = carry
            key, noise_key = jax.random.split(key)
            scores = compute_scores(q_net, q_params, obs, cfg.cvar_alpha)
            if cfg.soft_alloc:
                weights = noisy_score_weighted_allocate(scores, noise_key, noise_scale, n_long, n_short)
            else:
                positions = noisy_rank_and_allocate(scores, noise_key, noise_scale, n_long, n_short)
                weights = positions.astype(jnp.float32)
            next_obs, next_state, per_asset_rew, _, done = step_portfolio(
                state, weights, h_K, vol_K, ret_K,
                episode_len, cfg.fee_rate, cfg.slippage_factor,
                risk_parity=cfg.risk_parity,
            )
            actions  = (jnp.sign(weights) + 1).astype(jnp.int32)
            done_K   = jnp.broadcast_to(done.astype(jnp.float32), (K,))
            return (next_obs, next_state, key), (obs, actions, per_asset_rew, next_obs, done_K)

        _, transitions = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return transitions

    def eval_single(q_params, h_K, vol_K, ret_K, key):
        obs, state = reset_portfolio(h_K, vol_K, K)

        def body(carry, _):
            obs, state, key = carry
            key, _ = jax.random.split(key)
            scores = compute_scores(q_net, q_params, obs, cfg.cvar_alpha)
            if cfg.soft_alloc:
                weights = score_weighted_allocate(scores, n_long, n_short)
            else:
                weights = rank_and_allocate(scores, n_long, n_short).astype(jnp.float32)
            next_obs, next_state, _, portfolio_rew, done = step_portfolio(
                state, weights, h_K, vol_K, ret_K,
                episode_len, cfg.fee_rate, cfg.slippage_factor,
                risk_parity=cfg.risk_parity,
            )
            return (next_obs, next_state, key), portfolio_rew

        _, rewards = jax.lax.scan(body, (obs, state, key), None, length=episode_len)
        return rewards.sum()

    batched_collect = jax.jit(jax.vmap(collect_single, in_axes=(None, 0, 0, 0, 0, 0)))
    batched_eval    = jax.jit(jax.vmap(eval_single,    in_axes=(None, 0, 0, 0, 0)))

    # ── JIT warmup ──
    log.info("JIT compiling (N=%d portfolios × K=%d assets)...", N, K)
    t_jit = time.time()
    noise_vector = jnp.logspace(jnp.log10(0.01), jnp.log10(0.5), N)

    dh = jnp.zeros((N, K, win_len, d_model))
    dv = jnp.zeros((N, K, win_len))
    dr = jnp.zeros((N, K, win_len))
    key, *wks = jax.random.split(key, N + 1)
    _ = batched_collect(agent_state.q_params, dh, dv, dr, jnp.stack(wks), noise_vector)

    n_eval = cfg.n_eval
    key, *wks_e = jax.random.split(key, n_eval + 1)
    dh_e = jnp.zeros((n_eval, K, win_len, d_model))
    dv_e = jnp.zeros((n_eval, K, win_len))
    dr_e = jnp.zeros((n_eval, K, win_len))
    _ = batched_eval(agent_state.q_params, dh_e, dv_e, dr_e, jnp.stack(wks_e))

    dummy_batch = {
        "obs":      jnp.zeros((cfg.batch_size, o_dim)),
        "action":   jnp.zeros(cfg.batch_size, dtype=jnp.int32),
        "reward":   jnp.zeros(cfg.batch_size),
        "next_obs": jnp.zeros((cfg.batch_size, o_dim)),
        "done":     jnp.zeros(cfg.batch_size),
    }
    agent_state, _ = jit_update(agent_state, dummy_batch)

    # Restore non-dummy state (discard warmup grads, keep loaded weights)
    agent_state = DQNState(
        q_params=q_params,
        target_params=target_params,
        opt_state=tx.init(q_params),
        step=jnp.int32(0),
    )
    log.info("JIT done in %.1fs", time.time() - t_jit)

    # ── Training loop ──
    os.makedirs(args.output_dir, exist_ok=True)
    best_sharpe = -float("inf")
    step_count  = 0
    collect_count = 0
    update_count  = 0
    last_loss     = 0.0
    next_eval_at  = cfg.eval_freq
    updates_per_collect = steps_per_collect // 16
    target_update_every = 25
    np_rng = np.random.default_rng(42)
    t_train = time.time()

    while step_count < cfg.total_steps:
        h_np, v_np, r_np = sample_portfolio_windows(
            train_assets, episode_len, np_rng, N, K,
            use_per=getattr(cfg, "use_per", False),
            per_alpha=getattr(cfg, "per_alpha", 0.6))

        key, *ep_keys = jax.random.split(key, N + 1)
        transitions = batched_collect(
            agent_state.q_params,
            jnp.array(h_np), jnp.array(v_np), jnp.array(r_np),
            jnp.stack(ep_keys), noise_vector,
        )
        obs_t, act_t, rew_t, nobs_t, done_t = transitions

        obs_flat  = np.asarray(obs_t).reshape(-1, o_dim)
        nobs_flat = np.asarray(nobs_t).reshape(-1, o_dim)
        act_flat  = np.asarray(act_t).reshape(-1)
        rew_flat  = np.asarray(rew_t).reshape(-1)
        done_flat = np.asarray(done_t).reshape(-1)

        n_new = obs_flat.shape[0]
        idx = replay.pos % replay.capacity
        if idx + n_new <= replay.capacity:
            s = slice(idx, idx + n_new)
            replay.obs[s]      = obs_flat
            replay.action[s]   = act_flat
            replay.reward[s]   = rew_flat
            replay.next_obs[s] = nobs_flat
            replay.done[s]     = done_flat
        else:
            first = replay.capacity - idx
            for arr, src in [(replay.obs, obs_flat), (replay.next_obs, nobs_flat),
                             (replay.action, act_flat), (replay.reward, rew_flat),
                             (replay.done, done_flat)]:
                arr[idx:]       = src[:first]
                arr[:n_new-first] = src[first:]
        replay.pos  += n_new
        replay.size  = min(replay.size + n_new, replay.capacity)

        step_count    += n_new
        collect_count += 1

        if replay.size >= cfg.batch_size:
            for _ in range(updates_per_collect):
                indices = np_rng.integers(replay.size, size=cfg.batch_size)
                batch = {
                    "obs":      jnp.array(replay.obs[indices]),
                    "action":   jnp.array(replay.action[indices]),
                    "reward":   jnp.array(replay.reward[indices]),
                    "next_obs": jnp.array(replay.next_obs[indices]),
                    "done":     jnp.array(replay.done[indices]),
                }
                agent_state, metrics = jit_update(agent_state, batch)
                update_count += 1
                last_loss = float(metrics["loss"])
                if update_count % target_update_every == 0:
                    agent_state = jit_update_target(agent_state)

        elapsed = time.time() - t_train
        sps = step_count / max(elapsed, 0.01)
        log.info("step=%d/%d  buf=%d  upd=%d  %.0f sps  loss=%.4f",
                 step_count, cfg.total_steps, replay.size, update_count, sps, last_loss)

        if step_count >= next_eval_at:
            next_eval_at += cfg.eval_freq
            h_ev, v_ev, r_ev = sample_portfolio_windows(
                val_assets, episode_len, np_rng, n_eval, K)
            key, *ev_keys = jax.random.split(key, n_eval + 1)
            rewards = batched_eval(
                agent_state.q_params,
                jnp.array(h_ev), jnp.array(v_ev), jnp.array(r_ev),
                jnp.stack(ev_keys),
            )
            ep_rets = np.asarray(rewards)
            sharpe  = ep_rets.mean() / (ep_rets.std() + 1e-8)
            log.info("EVAL step=%d: sharpe=%.4f  mean=%.4f  std=%.4f  n=%d",
                     step_count, sharpe, ep_rets.mean(), ep_rets.std(), n_eval)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                save_path = os.path.join(args.output_dir, "best_cs_ft.npz")
                flat = {str(k): np.asarray(v)
                        for k, v in jax.tree.leaves_with_path(agent_state.q_params)}
                np.savez(save_path, **flat)
                log.info("New best Sharpe=%.4f → %s", best_sharpe, save_path)
                with open(os.path.join(args.output_dir, "best_eval.json"), "w") as f:
                    json.dump({
                        "step": int(step_count), "sharpe": float(sharpe),
                        "mean_return": float(ep_rets.mean()),
                        "std_return": float(ep_rets.std()),
                        "adapter_lr": args.adapter_lr,
                        "dqn_lr": args.dqn_lr,
                        "adapter_dim": args.adapter_dim,
                        "init_checkpoint": args.init_checkpoint,
                    }, f, indent=2)

    flat = {str(k): np.asarray(v)
            for k, v in jax.tree.leaves_with_path(agent_state.q_params)}
    np.savez(os.path.join(args.output_dir, "final_cs_ft.npz"), **flat)

    elapsed = time.time() - t0
    log.info("=== DONE: %d steps, %d collects, %d updates, best_sharpe=%.4f, %.1fs (%.1f min) ===",
             step_count, collect_count, update_count, best_sharpe, elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
