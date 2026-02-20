"""Training metrics for TPU performance reporting.

Computes MFU (Model FLOPs Utilization) and throughput metrics
needed for TRC (TPU Research Cloud) reporting.

Peak TFLOPS reference (bf16):
  - TPU v5p: 459 TFLOPS/chip
  - TPU v6e: 920 TFLOPS/chip (Trillium)
  - TPU v4:  275 TFLOPS/chip
"""


def compute_mfu(
    n_params: int,
    batch_size: int,
    seq_len: int,
    step_time: float,
    n_chips: int,
    peak_tflops: float = 459.0,
) -> float:
    """Compute Model FLOPs Utilization (MFU).

    Uses the standard 6*N approximation for transformer/SSM FLOPs per token
    (2N forward + 2N backward + 2N activation recompute ≈ 6N).

    Args:
        n_params: Total model parameters.
        batch_size: Global batch size.
        seq_len: Sequence length.
        step_time: Wall-clock seconds per training step.
        n_chips: Number of TPU chips.
        peak_tflops: Peak bf16 TFLOPS per chip (459 for v5p).

    Returns:
        MFU as a fraction (0.0 to 1.0).
    """
    if step_time <= 0:
        return 0.0
    # FLOPs per step: 6 * N_params * tokens_per_step
    tokens_per_step = batch_size * seq_len
    flops_per_step = 6.0 * n_params * tokens_per_step
    # Achieved TFLOPS across all chips
    achieved_tflops = flops_per_step / step_time / 1e12
    # Peak TFLOPS for all chips
    total_peak_tflops = peak_tflops * n_chips
    return achieved_tflops / total_peak_tflops


def compute_tokens_per_second(
    batch_size: int,
    seq_len: int,
    step_time: float,
) -> float:
    """Compute training throughput in tokens/second.

    Args:
        batch_size: Global batch size.
        seq_len: Sequence length.
        step_time: Wall-clock seconds per training step.

    Returns:
        Tokens processed per second.
    """
    if step_time <= 0:
        return 0.0
    return (batch_size * seq_len) / step_time
