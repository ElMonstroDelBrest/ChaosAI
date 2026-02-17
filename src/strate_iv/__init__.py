"""Strate IV — Latent Regime RL.

Uses pre-computed JEPA latent representations as observations for a PPO agent
that learns continuous portfolio positioning (long/short/flat) conditioned on
the detected market regime.

Architecture: TrajectoryBuffer → LatentCryptoEnv → PPO Agent
"""

EPS = 1e-8
