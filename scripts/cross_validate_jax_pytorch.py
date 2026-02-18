"""Cross-validate JAX Fin-JEPA against PyTorch ground truth.

1. Load PyTorch checkpoint
2. Map weight names: PyTorch -> Flax conventions
3. Transpose weights: PyTorch (out, in) -> Flax (in, out)
4. Forward pass with identical inputs (same seed)
5. Assert max |y_jax - y_pytorch| < 1e-4 (bf16 tolerance)
"""

import argparse
import re
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jax_v6.config import load_config as load_jax_config
from src.jax_v6.encoders.mamba2_encoder import Mamba2Encoder as JaxEncoder
from src.strate_ii.encoder import Mamba2Encoder as PytorchEncoder


# --------------------------------------------------------------------------
# Weight name mapping: PyTorch -> Flax
# --------------------------------------------------------------------------

def pytorch_to_flax_name(pt_name: str) -> str:
    """Convert a PyTorch parameter name to Flax nested key path.

    Examples:
        layers.0.in_proj.weight -> layers_0/in_proj/kernel
        layers.0.norm.weight -> layers_0/LayerNorm_0/scale
        layers.0.norm.bias -> layers_0/LayerNorm_0/bias
        layers.0.A_log -> layers_0/A_log
        layers.0.conv.conv.weight -> layers_0/conv/Conv_0/kernel
        layers.0.conv.conv.bias -> layers_0/conv/Conv_0/bias
        layers.0.vol_proj.weight -> layers_0/vol_proj/kernel
        layers.0.vol_proj.bias -> layers_0/vol_proj/bias
        layers.0.exo_proj.weight -> layers_0/exo_proj/kernel
        layers.0.exo_proj.bias -> layers_0/exo_proj/bias
        layers.0.out_proj.weight -> layers_0/out_proj/kernel
        codebook_embed.weight -> codebook_embed/embedding
        input_proj.weight -> input_proj/kernel
        input_proj.bias -> input_proj/bias
        mask_embed -> mask_embed
        norm.weight -> norm/scale
        norm.bias -> norm/bias
    """
    name = pt_name

    # layers.{i} -> layers_{i}
    name = re.sub(r"layers\.(\d+)", r"layers_\1", name)

    # .conv.conv. -> .conv.Conv_0.
    name = name.replace(".conv.conv.", ".conv.Conv_0.")

    # .norm. at block level -> .LayerNorm_0.
    name = re.sub(r"(layers_\d+)\.norm\.", r"\1.LayerNorm_0.", name)

    # Final norm
    if name.startswith("norm."):
        name = name.replace("norm.", "norm.")

    # .weight -> /kernel or /scale or /embedding
    if name.endswith(".weight"):
        base = name[:-7]
        # LayerNorm and final norm -> scale
        if "LayerNorm" in base or base == "norm":
            name = base + ".scale"
        # Embedding -> embedding
        elif "codebook_embed" in base:
            name = base + ".embedding"
        else:
            name = base + ".kernel"
    elif name.endswith(".bias"):
        pass  # Keep .bias as is

    # Convert dots to slashes for Flax path
    return name.replace(".", "/")


def should_transpose(flax_name: str) -> bool:
    """Check if a weight needs transposing (Dense kernels).

    PyTorch Linear: (out_features, in_features)
    Flax Dense: (in_features, out_features)
    Conv1d: PyTorch (out, in/groups, K) -> Flax (K, in/groups, out) but
    for depthwise conv (groups=d_inner): (d, 1, K) -> (K, 1, d)
    """
    if flax_name.endswith("/kernel"):
        if "Conv_0" in flax_name:
            return True  # Conv needs special handling
        return True  # Dense kernel
    return False


def convert_weight(pt_tensor: torch.Tensor, flax_name: str) -> np.ndarray:
    """Convert a PyTorch weight tensor to Flax format."""
    w = pt_tensor.detach().cpu().float().numpy()

    if "Conv_0/kernel" in flax_name:
        # PyTorch depthwise Conv1d: (D, 1, K) -> Flax Conv: (K, 1, D)
        # Actually Flax Conv with feature_group_count=D expects (K, 1, D)
        if w.ndim == 3:
            w = np.transpose(w, (2, 1, 0))  # (D, 1, K) -> (K, 1, D)
        return w

    if flax_name.endswith("/kernel") and w.ndim == 2:
        # PyTorch Linear: (out, in) -> Flax Dense: (in, out)
        return w.T

    return w


def load_pytorch_weights(ckpt_path: str) -> dict:
    """Load PyTorch checkpoint and extract encoder state dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Extract context_encoder weights (strip prefix if present)
    encoder_sd = {}
    prefixes = ["context_encoder.", "model.context_encoder.", ""]
    for key, val in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix) and prefix:
                encoder_sd[key[len(prefix):]] = val
                break
        else:
            # No prefix matched, keep as is (might be encoder-only checkpoint)
            encoder_sd[key] = val

    return encoder_sd


def build_flax_params(pytorch_sd: dict) -> dict:
    """Convert full PyTorch state dict to Flax nested params dict."""
    flax_flat = {}
    for pt_name, pt_tensor in pytorch_sd.items():
        flax_name = pytorch_to_flax_name(pt_name)
        flax_flat[flax_name] = convert_weight(pt_tensor, flax_name)

    # Convert flat path dict to nested dict
    params = {}
    for path, value in flax_flat.items():
        parts = path.split("/")
        d = params
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = jnp.array(value)

    return params


# --------------------------------------------------------------------------
# Forward pass comparison
# --------------------------------------------------------------------------

def compare_forward(
    pytorch_encoder: PytorchEncoder,
    jax_params: dict,
    jax_config: dict,
    seq_len: int = 128,
    batch_size: int = 4,
    seed: int = 42,
    tolerance: float = 1e-4,
):
    """Run identical forward passes and compare outputs.

    Args:
        pytorch_encoder: Initialized PyTorch encoder.
        jax_params: Flax params dict.
        jax_config: Config for Mamba2Encoder.
        seq_len: Sequence length.
        batch_size: Batch size.
        seed: Random seed for input generation.
        tolerance: Max absolute error threshold.
    """
    # Generate identical inputs
    rng = np.random.RandomState(seed)
    token_indices_np = rng.randint(0, 1024, size=(batch_size, seq_len)).astype(np.int64)
    weekend_mask_np = (rng.random((batch_size, seq_len)) > 0.9).astype(np.float32)
    exo_clock_np = rng.randn(batch_size, seq_len, 2).astype(np.float32)

    # PyTorch forward
    with torch.no_grad():
        pt_tokens = torch.from_numpy(token_indices_np)
        pt_weekend = torch.from_numpy(weekend_mask_np)
        pt_exo = torch.from_numpy(exo_clock_np)
        pt_out = pytorch_encoder(pt_tokens, weekend_mask=pt_weekend, exo_clock=pt_exo)
        pt_out_np = pt_out.cpu().numpy()

    # JAX forward
    jax_encoder = JaxEncoder(**jax_config)
    jax_tokens = jnp.array(token_indices_np)
    jax_weekend = jnp.array(weekend_mask_np)
    jax_exo = jnp.array(exo_clock_np)

    jax_out = jax_encoder.apply(
        {"params": jax_params},
        jax_tokens,
        weekend_mask=jax_weekend,
        exo_clock=jax_exo,
    )
    jax_out_np = np.array(jax_out)

    # Compare
    max_err = np.max(np.abs(pt_out_np - jax_out_np))
    mean_err = np.mean(np.abs(pt_out_np - jax_out_np))
    rel_err = max_err / (np.max(np.abs(pt_out_np)) + 1e-8)

    print(f"\n{'='*60}")
    print(f"Cross-validation results:")
    print(f"  Max absolute error:  {max_err:.6e}")
    print(f"  Mean absolute error: {mean_err:.6e}")
    print(f"  Relative error:      {rel_err:.6e}")
    print(f"  PyTorch output range: [{pt_out_np.min():.4f}, {pt_out_np.max():.4f}]")
    print(f"  JAX output range:     [{jax_out_np.min():.4f}, {jax_out_np.max():.4f}]")
    print(f"  Tolerance:           {tolerance:.1e}")
    print(f"  PASS: {max_err < tolerance}")
    print(f"{'='*60}")

    if max_err >= tolerance:
        # Detailed per-layer diagnosis
        print("\nDiagnosis: checking per-position errors...")
        pos_errors = np.max(np.abs(pt_out_np - jax_out_np), axis=(0, 2))
        worst_pos = np.argmax(pos_errors)
        print(f"  Worst position: {worst_pos} (error={pos_errors[worst_pos]:.6e})")

        raise AssertionError(
            f"Cross-validation FAILED: max error {max_err:.6e} >= {tolerance:.1e}"
        )

    return max_err


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate JAX Fin-JEPA encoder against PyTorch"
    )
    parser.add_argument(
        "--pytorch_ckpt", required=True,
        help="Path to PyTorch Strate II checkpoint"
    )
    parser.add_argument(
        "--config", default="configs/strate_ii.yaml",
        help="Path to Strate II YAML config"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-4,
        help="Max absolute error tolerance (default: 1e-4 for bf16)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for comparison"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Loading PyTorch checkpoint: {args.pytorch_ckpt}")
    pytorch_sd = load_pytorch_weights(args.pytorch_ckpt)
    print(f"  {len(pytorch_sd)} parameters loaded")

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    mamba_cfg = config.get("mamba2", {})
    embed_cfg = config.get("embedding", {})

    # Build PyTorch encoder
    pt_encoder = PytorchEncoder(
        num_codes=embed_cfg.get("num_codes", 1024),
        codebook_dim=embed_cfg.get("codebook_dim", 64),
        d_model=mamba_cfg.get("d_model", 128),
        d_state=mamba_cfg.get("d_state", 16),
        n_layers=mamba_cfg.get("n_layers", 6),
        n_heads=mamba_cfg.get("n_heads", 2),
        expand_factor=mamba_cfg.get("expand_factor", 2),
        conv_kernel=mamba_cfg.get("conv_kernel", 4),
        seq_len=embed_cfg.get("seq_len", 128),
    )
    pt_encoder.load_state_dict(pytorch_sd, strict=False)
    pt_encoder.eval()

    # Build Flax params
    print("Converting weights PyTorch -> Flax...")
    flax_params = build_flax_params(pytorch_sd)

    jax_config = {
        "num_codes": embed_cfg.get("num_codes", 1024),
        "codebook_dim": embed_cfg.get("codebook_dim", 64),
        "d_model": mamba_cfg.get("d_model", 128),
        "d_state": mamba_cfg.get("d_state", 16),
        "n_layers": mamba_cfg.get("n_layers", 6),
        "n_heads": mamba_cfg.get("n_heads", 2),
        "expand_factor": mamba_cfg.get("expand_factor", 2),
        "conv_kernel": mamba_cfg.get("conv_kernel", 4),
        "seq_len": embed_cfg.get("seq_len", 128),
        "chunk_size": mamba_cfg.get("chunk_size", 128),
    }

    # Compare
    seq_len = embed_cfg.get("seq_len", 128)
    max_err = compare_forward(
        pt_encoder, flax_params, jax_config,
        seq_len=seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        tolerance=args.tolerance,
    )

    print(f"\nCross-validation PASSED (max error: {max_err:.6e})")


if __name__ == "__main__":
    main()
