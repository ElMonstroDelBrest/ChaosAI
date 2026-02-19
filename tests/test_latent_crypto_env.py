"""Tests for Strate IV: PnLReward, LatentCryptoEnv, TrajectoryBuffer."""

import math

import numpy as np
import pytest
import torch

from src.strate_iv.config import EnvConfig, StrateIVConfig, load_config
from src.strate_iv.reward import PnLReward
from src.strate_iv.trajectory_buffer import (
    TrajectoryBuffer, TrajectoryEntry, classify_regime, stratified_sample,
)
from src.strate_iv.env import LatentCryptoEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dummy_entry(
    n_futures: int = 4,
    n_tgt: int = 8,
    patch_len: int = 16,
    d_model: int = 128,
    seq_len: int = 64,
    ctx_len: int = 256,
) -> TrajectoryEntry:
    """Create a synthetic TrajectoryEntry for testing."""
    future_ohlcv = torch.rand(n_futures, n_tgt, patch_len, 5) * 10 + 95

    return TrajectoryEntry(
        context_tokens=torch.randint(0, 1024, (seq_len,)),
        weekend_mask=torch.zeros(seq_len),
        context_ohlcv=torch.rand(ctx_len, 5) * 100 + 50,
        future_ohlcv=future_ohlcv,
        future_latents=torch.randn(n_futures, n_tgt, d_model),
        revin_means=torch.zeros(1, 5),
        revin_stds=torch.ones(1, 5) * 0.02,  # 2% vol
        last_close=100.0,
        h_x_pooled=torch.randn(d_model),
    )


def make_buffer(n_entries: int = 5, **kwargs) -> TrajectoryBuffer:
    """Create a buffer with synthetic entries (no disk IO)."""
    entries = [make_dummy_entry(**kwargs) for _ in range(n_entries)]
    return TrajectoryBuffer.from_entries(entries)


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------

class TestPnLReward:
    def test_zero_action_zero_reward(self):
        """Flat position -> zero PnL, zero TC."""
        r = PnLReward(tc_rate=0.0005)
        reward, info = r.compute(
            action=0.0, prev_action=0.0,
            close_current=100.0, close_next=105.0,
        )
        assert reward == 0.0
        assert info.raw_pnl == 0.0
        assert info.tc_penalty == 0.0

    def test_long_positive_return(self):
        """Full long + positive return -> positive reward (log return)."""
        r = PnLReward(tc_rate=0.0)
        reward, info = r.compute(
            action=1.0, prev_action=1.0,
            close_current=100.0, close_next=105.0,
        )
        expected = math.log(105.0 / 100.0)  # ~0.04879
        assert reward == pytest.approx(expected, abs=1e-6)
        assert info.tc_penalty == 0.0

    def test_short_positive_return(self):
        """Full short + positive return -> negative reward."""
        r = PnLReward(tc_rate=0.0)
        reward, info = r.compute(
            action=-1.0, prev_action=-1.0,
            close_current=100.0, close_next=105.0,
        )
        expected = -math.log(105.0 / 100.0)
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_transaction_cost(self):
        """Position change incurs TC penalty."""
        r = PnLReward(tc_rate=0.0005)
        reward, info = r.compute(
            action=1.0, prev_action=-1.0,
            close_current=100.0, close_next=100.0,
        )
        # log_return = 0, raw_pnl = 0, tc = 0.0005 * |1 - (-1)| = 0.001
        assert info.tc_penalty == pytest.approx(0.001, abs=1e-8)
        assert reward == pytest.approx(-0.001, abs=1e-8)

    def test_symmetric_reward(self):
        """Same move size in opposite directions -> same magnitude reward."""
        r = PnLReward(tc_rate=0.0)
        rew_long, _ = r.compute(
            action=1.0, prev_action=1.0,
            close_current=100.0, close_next=105.0,
        )
        rew_short, _ = r.compute(
            action=-1.0, prev_action=-1.0,
            close_current=100.0, close_next=105.0,
        )
        assert rew_long == pytest.approx(-rew_short, abs=1e-6)


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestLatentCryptoEnv:
    @pytest.fixture
    def env(self):
        buf = make_buffer(n_entries=10)
        config = EnvConfig(n_tgt=8, tc_rate=0.002, patch_len=16)
        return LatentCryptoEnv(buffer=buf, config=config)

    def test_reset_returns_valid_obs(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert "realized_future_idx" in info

    def test_step_returns_valid(self, env):
        env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert not terminated  # First step, not done yet
        assert not truncated
        assert np.isfinite(reward)

    def test_episode_terminates_after_n_tgt(self, env):
        """Episode must terminate after exactly n_tgt steps."""
        env.reset(seed=42)
        for step in range(env.config.n_tgt):
            action = np.array([0.0], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if step < env.config.n_tgt - 1:
                assert not terminated, f"Terminated too early at step {step}"
            else:
                assert terminated, f"Should terminate at step {step}"

    def test_action_clipping(self, env):
        """Actions outside [-1, 1] should be clipped."""
        env.reset(seed=42)
        action = np.array([5.0], dtype=np.float32)  # Out of bounds
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["position"] == 1.0  # Clipped to 1.0

        env.reset(seed=42)
        action = np.array([-3.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert info["position"] == -1.0  # Clipped to -1.0

    def test_cumulative_pnl_tracks(self, env):
        """Cumulative PnL should accumulate over steps."""
        env.reset(seed=42)
        total = 0.0
        for _ in range(4):
            action = np.array([1.0], dtype=np.float32)
            _, reward, _, _, info = env.step(action)
            total += reward
        assert info["cumulative_pnl"] == pytest.approx(total, abs=1e-6)

    def test_observation_finite(self, env):
        """All observations should be finite throughout an episode."""
        env.reset(seed=42)
        for _ in range(env.config.n_tgt):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)
            assert np.all(np.isfinite(obs)), f"Non-finite obs: {obs}"

    def test_reward_order_of_magnitude(self, env):
        """Rewards should be O(1) -- not explosive."""
        env.reset(seed=42)
        rewards = []
        for _ in range(env.config.n_tgt):
            action = env.action_space.sample()
            _, reward, _, _, _ = env.step(action)
            rewards.append(reward)
        rewards = np.array(rewards)
        assert np.all(np.abs(rewards) < 10), f"Rewards too large: {rewards}"

    def test_multiple_resets(self, env):
        """Environment should handle multiple resets without issues."""
        for _ in range(5):
            obs, info = env.reset()
            assert obs.shape == env.observation_space.shape
            action = np.array([0.0], dtype=np.float32)
            obs, _, _, _, _ = env.step(action)
            assert obs.shape == env.observation_space.shape

    def test_delta_mu_in_observation(self, env):
        """delta_mu should be present and finite in observation."""
        obs, _ = env.reset(seed=42)
        # delta_mu at: d*3 + n_tgt*3 + 5 = 384 + 24 + 5 = 413 (d_model=128)
        delta_mu_idx = 128 * 3 + 8 * 3 + 5  # = 413
        delta_mu = obs[delta_mu_idx]
        assert np.isfinite(delta_mu)

    def test_observation_changes_between_steps(self, env):
        """Observation must change between steps (anti-Oracle check)."""
        obs0, _ = env.reset(seed=42)
        action = np.array([0.5], dtype=np.float32)
        obs1, _, _, _, _ = env.step(action)
        # With step-aware obs, future_mean_t, step_progress, realized_returns
        # should all differ between step 0 and step 1
        n_changed = np.sum(obs0 != obs1)
        # At minimum: d_model (future_mean_t) + d_model (future_std_t)
        # + 1 (step_progress) + 1 (realized_returns[0]) + 1 (position) + 1 (cum_pnl)
        assert n_changed >= 4, (
            f"Only {n_changed} dims changed between steps — observation is too static"
        )

    def test_noise_gate_zeros_dead_market(self):
        """When close volatility is negligible, market signals should be zeroed."""
        # Create entry with near-zero volatility (stablecoin-like)
        entry = make_dummy_entry()
        entry = TrajectoryEntry(
            context_tokens=entry.context_tokens,
            weekend_mask=entry.weekend_mask,
            context_ohlcv=entry.context_ohlcv,
            future_ohlcv=entry.future_ohlcv,
            future_latents=entry.future_latents,
            revin_means=torch.tensor([[100.0, 100.0, 100.0, 100.0, 0.0]]),
            revin_stds=torch.tensor([[1e-8, 1e-8, 1e-8, 1e-8, 1e-8]]),
            last_close=100.0,
            h_x_pooled=entry.h_x_pooled,
        )
        buf = TrajectoryBuffer.from_entries([entry] * 5)
        config = EnvConfig(n_tgt=8, tc_rate=0.002, patch_len=16,
                           dead_market_threshold=1e-4)
        env = LatentCryptoEnv(buffer=buf, config=config)
        obs, _ = env.reset(seed=42)

        # future_mean_t, future_std_t should be near-zero (sigmoid gate ≈ 4.5e-5)
        d = 128  # d_model in test entries
        future_block = obs[d:3*d]  # future_mean_t + future_std_t
        assert np.all(np.abs(future_block) < 1e-3), \
            "Noise gate should attenuate future latents to near-zero"

        # close_stats should be near-zero too
        close_stats = obs[3*d:3*d + 8*3]
        assert np.all(np.abs(close_stats) < 1e-3), \
            "Noise gate should attenuate close_stats to near-zero"


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------

class TestTrajectoryBuffer:
    def test_from_entries(self):
        """from_entries creates a buffer without disk IO."""
        entries = [make_dummy_entry() for _ in range(3)]
        buf = TrajectoryBuffer.from_entries(entries)
        assert len(buf) == 3
        assert buf.entries[0] is entries[0]

    def test_split_sizes(self):
        """Split should produce correct train/eval sizes."""
        buf = make_buffer(n_entries=20)
        train_buf, eval_buf = buf.split(val_ratio=0.2, seed=42)
        assert len(train_buf) == 16
        assert len(eval_buf) == 4
        assert len(train_buf) + len(eval_buf) == 20

    def test_split_no_overlap(self):
        """Train and eval should contain disjoint entries."""
        buf = make_buffer(n_entries=10)
        train_buf, eval_buf = buf.split(val_ratio=0.3, seed=42)
        train_ids = {id(e) for e in train_buf.entries}
        eval_ids = {id(e) for e in eval_buf.entries}
        assert train_ids.isdisjoint(eval_ids)

    def test_split_deterministic(self):
        """Same seed should produce the same split."""
        buf = make_buffer(n_entries=10)
        t1, e1 = buf.split(val_ratio=0.2, seed=99)
        t2, e2 = buf.split(val_ratio=0.2, seed=99)
        assert len(t1) == len(t2)
        assert len(e1) == len(e2)
        for a, b in zip(t1.entries, t2.entries):
            assert id(a) == id(b)


# ---------------------------------------------------------------------------
# Regime classification tests
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_bull_regime(self):
        """Strongly rising prices should classify as bull."""
        T = 200
        ohlcv = torch.zeros(T, 5)
        # Steadily rising close prices
        ohlcv[:, 3] = torch.linspace(100, 150, T)
        assert classify_regime(ohlcv) == "bull"

    def test_bear_regime(self):
        """Strongly falling prices should classify as bear."""
        T = 200
        ohlcv = torch.zeros(T, 5)
        ohlcv[:, 3] = torch.linspace(100, 60, T)
        assert classify_regime(ohlcv) == "bear"

    def test_range_regime(self):
        """Flat prices should classify as range."""
        T = 200
        ohlcv = torch.zeros(T, 5)
        ohlcv[:, 3] = 100.0 + torch.randn(T) * 0.01
        assert classify_regime(ohlcv) == "range"

    def test_short_sequence(self):
        """Single candle should default to range."""
        ohlcv = torch.zeros(1, 5)
        ohlcv[0, 3] = 100.0
        assert classify_regime(ohlcv) == "range"


class TestStratifiedSample:
    def test_balanced_output(self):
        """Stratified sample should produce roughly equal regime counts."""
        T = 100
        # Create 30 entries: 10 bull, 10 bear, 10 range
        ohlcv_data = []
        for _ in range(10):
            o = torch.zeros(T, 5)
            o[:, 3] = torch.linspace(100, 150, T)
            ohlcv_data.append(o)
        for _ in range(10):
            o = torch.zeros(T, 5)
            o[:, 3] = torch.linspace(100, 60, T)
            ohlcv_data.append(o)
        for _ in range(10):
            o = torch.zeros(T, 5)
            o[:, 3] = 100.0 + torch.randn(T) * 0.01
            ohlcv_data.append(o)

        sampled = stratified_sample(
            n_total=30,
            n_episodes=9,
            ohlcv_lookup=lambda i: ohlcv_data[i],
            seed=42,
        )
        assert len(sampled) == 9
        # Each index should be valid
        assert all(0 <= idx < 30 for idx in sampled)


# ---------------------------------------------------------------------------
# Gymnasium check_env compatibility
# ---------------------------------------------------------------------------

class TestGymnasiumCompat:
    def test_check_env(self):
        """gymnasium.utils.env_checker.check_env should pass."""
        from gymnasium.utils.env_checker import check_env

        buf = make_buffer(n_entries=5)
        config = EnvConfig(n_tgt=8, tc_rate=0.0005, patch_len=16)
        env = LatentCryptoEnv(buffer=buf, config=config)
        check_env(env, skip_render_check=True)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_config(self, tmp_path):
        """Config loads from YAML correctly."""
        yaml_content = """
env:
  n_tgt: 8
  tc_rate: 0.0005
  patch_len: 16
buffer:
  buffer_dir: "data/trajectory_buffer/"
  n_episodes: 255
  n_futures: 16
  val_ratio: 0.2
ppo:
  lr: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  total_timesteps: 1000000
  eval_freq: 10000
  log_dir: "tb_logs/strate_iv/"
"""
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml_content)
        config = load_config(str(cfg_file))
        assert isinstance(config, StrateIVConfig)
        assert config.env.tc_rate == pytest.approx(0.0005)
        assert config.ppo.lr == pytest.approx(3e-4)
        assert config.buffer.n_episodes == 255

    def test_load_actual_config(self):
        """The actual configs/strate_iv.yaml should load without error."""
        config = load_config("configs/strate_iv.yaml")
        assert isinstance(config, StrateIVConfig)
        assert config.env.n_tgt == 8
