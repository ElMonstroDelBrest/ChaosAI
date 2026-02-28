"""Macro context loader for training.

Loads macro_context.npz (produced by build_macro_graph.py) and provides
per-batch macro context vectors aligned to sequence positions.

The macro context is injected into the Mamba2Encoder via gated additive
fusion (same pattern as GNN embeddings).

Usage in training loop:
    macro_ctx = MacroContext("data/macro/macro_context.npz", macro_dim=48)
    # For each batch, get (B, S, macro_dim) context:
    macro_features = macro_ctx.get_batch_context(batch_size, seq_len)
"""

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class MacroContext:
    """Provides macro context vectors for training batches.

    Since macro signals are global (not per-asset), the same context vector
    is broadcast across all samples in a batch. The context is a sliding
    window over the macro signal matrix.

    For now, macro context is position-invariant within a sequence (same
    vector repeated S times). This is because we don't have per-candle
    timestamp alignment in ArrayRecord. Future: add timestamp metadata
    to ArrayRecord for exact temporal alignment.

    Args:
        macro_path: Path to macro_context.npz.
        macro_dim: Number of macro signals to use (top-k by variance).
            If 0 or None, all signals are used.
        seed: Random seed for reproducible signal selection.
    """

    def __init__(self, macro_path: str, macro_dim: int = 48, seed: int = 42):
        data = np.load(macro_path, allow_pickle=True)
        signals = data["signals"]  # (T, N_signals)
        signal_names = list(data["signal_names"])

        self.n_timesteps, self.n_total_signals = signals.shape
        log.info("Loaded macro context: %d timesteps × %d signals from %s",
                 self.n_timesteps, self.n_total_signals, macro_path)

        # Select top-k signals by variance (most informative)
        if macro_dim > 0 and macro_dim < self.n_total_signals:
            variances = np.var(signals, axis=0)
            top_k_idx = np.argsort(variances)[-macro_dim:]
            top_k_idx.sort()
            self.signals = signals[:, top_k_idx].astype(np.float32)
            self.signal_names = [signal_names[i] for i in top_k_idx]
            log.info("Selected top-%d signals by variance: %s...",
                     macro_dim, self.signal_names[:5])
        else:
            self.signals = signals.astype(np.float32)
            self.signal_names = signal_names

        self.macro_dim = self.signals.shape[1]
        self._step = 0
        self._rng = np.random.default_rng(seed)

    def get_batch_context(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Get macro context for one batch.

        Returns a (B, S, macro_dim) array. Each sample in the batch gets
        a contiguous window of macro signals.

        The window position advances with each call (simulates temporal
        progression through the dataset).
        """
        # Wrap around if we've exhausted the macro timeline
        max_start = max(self.n_timesteps - seq_len, 1)
        start = self._step % max_start
        self._step += seq_len

        # Extract a (seq_len, macro_dim) window
        end = min(start + seq_len, self.n_timesteps)
        window = self.signals[start:end]

        # Pad if window is shorter than seq_len
        if window.shape[0] < seq_len:
            pad = seq_len - window.shape[0]
            window = np.pad(window, ((0, pad), (0, 0)), constant_values=0.0)

        # Broadcast to (B, S, macro_dim)
        return np.broadcast_to(window[None, :, :], (batch_size, seq_len, self.macro_dim)).copy()

    def get_random_batch_context(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Get macro context from random positions (for shuffled training).

        Each sample in the batch gets a different random window.
        Vectorized: no Python for-loop over batch_size.
        """
        max_start = max(self.n_timesteps - seq_len, 1)
        starts = self._rng.integers(0, max_start, size=batch_size)

        # Build (batch_size, seq_len) index array via broadcasting
        offsets = np.arange(seq_len)[None, :]  # (1, seq_len)
        indices = starts[:, None] + offsets     # (batch_size, seq_len)
        # Clamp to valid range
        indices = np.clip(indices, 0, self.n_timesteps - 1)
        # Vectorized gather: (batch_size, seq_len, macro_dim)
        return self.signals[indices]
