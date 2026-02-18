"""Pre-compute trajectory buffer for Strate IV RL training.

Usage:
    # Smoke test with random weights (no checkpoints needed):
    PYTHONPATH=. python scripts/precompute_trajectories.py --smoke_test

    # From trained checkpoints:
    PYTHONPATH=. python scripts/precompute_trajectories.py \
        --strate_ii_checkpoint checkpoints/strate_ii/best.ckpt \
        --strate_i_checkpoint checkpoints/strate_i_best.ckpt \
        --token_dir data/tokens/ \
        --ohlcv_dir data/binance_1h_subset/ \
        --output_dir data/trajectory_buffer/
"""

import argparse
from pathlib import Path

import torch


def smoke_test(
    n_episodes: int = 10,
    n_futures: int = 4,
    n_tgt: int = 8,
    output_dir: str = "data/trajectory_buffer/",
):
    """Smoke test: generate buffer entries with random weights."""
    from src.strate_i.decoder import Decoder
    from src.strate_i.revin import RevIN
    from src.strate_ii.jepa import FinJEPA
    from src.strate_ii.multiverse import MultiverseGenerator
    from src.strate_iv.trajectory_buffer import TrajectoryPrecomputer, TrajectoryEntry

    print("=== Smoke test: pre-computing trajectory buffer with random weights ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build models with random weights
    jepa = FinJEPA(
        num_codes=1024, codebook_dim=64, d_model=128, d_state=16,
        n_layers=2, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=64, pred_hidden_dim=256, pred_n_layers=2,
        pred_z_dim=32,
    ).to(device).eval()

    decoder = Decoder(
        latent_dim=64, hidden_channels=128, out_channels=5,
        patch_length=16, n_layers=4, kernel_size=3,
    ).to(device).eval()

    revin = RevIN(n_channels=5).to(device)
    generator = MultiverseGenerator(jepa, decoder, revin)

    precomputer = TrajectoryPrecomputer(
        generator=generator, jepa=jepa,
        n_futures=n_futures, n_tgt=n_tgt,
    )

    # Generate synthetic episodes
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    S = 48  # context sequence length
    for i in range(n_episodes):
        tokens = torch.randint(0, 1024, (S,), device=device)
        weekend_mask = torch.zeros(S, device=device)
        # Mark weekends every 7th position
        weekend_mask[5::7] = 1.0
        weekend_mask[6::7] = 1.0

        context_ohlcv = torch.rand(S * 16, 5, device=device) * 100 + 30000

        entry = precomputer.compute_entry(tokens, weekend_mask, context_ohlcv)

        save_dict = {
            "context_tokens": entry.context_tokens,
            "weekend_mask": entry.weekend_mask,
            "context_ohlcv": entry.context_ohlcv,
            "future_ohlcv": entry.future_ohlcv,
            "future_latents": entry.future_latents,
            "revin_means": entry.revin_means,
            "revin_stds": entry.revin_stds,
            "last_close": entry.last_close,
            "h_x_pooled": entry.h_x_pooled,
        }
        torch.save(save_dict, out_path / f"episode_{i:05d}.pt")

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{n_episodes}] episodes saved")

    print(f"\n  Buffer saved to {out_path} ({n_episodes} episodes)")
    print(f"  Each entry: {n_futures} futures, {n_tgt} targets")

    # Quick validation
    from src.strate_iv.trajectory_buffer import TrajectoryBuffer
    buf = TrajectoryBuffer(str(out_path))
    print(f"  Loaded back: {len(buf)} entries")
    entry = buf.sample()
    print(f"  Sample entry shapes:")
    print(f"    h_x_pooled:      {entry.h_x_pooled.shape}")
    print(f"    future_ohlcv:    {entry.future_ohlcv.shape}")
    print(f"    future_latents:  {entry.future_latents.shape}")
    print(f"    revin_stds:      {entry.revin_stds.shape}")

    print("=== Smoke test passed ===")


def precompute_from_checkpoints(args):
    """Pre-compute buffer from trained checkpoints.

    Token files are named {PAIR}_seq{NNNN}.pt (e.g. BTCUSDT_seq0003.pt).
    OHLCV files are {PAIR}.pt (e.g. BTCUSDT.pt) in ohlcv_dir.
    Each token sequence i covers OHLCV[i*seq_len*patch_len : (i+1)*seq_len*patch_len].
    """
    from src.strate_i.config import load_config as load_strate_i_config
    from src.strate_i.lightning_module import StrateILightningModule
    from src.strate_i.revin import RevIN
    from src.strate_ii.config import load_config as load_strate_ii_config
    from src.strate_ii.lightning_module import StrateIILightningModule
    from src.strate_ii.multiverse import MultiverseGenerator
    from src.strate_ii.data.token_dataset import TokenSequenceDataset
    from src.strate_iv.config import load_config as load_strate_iv_config
    from src.strate_iv.trajectory_buffer import TrajectoryPrecomputer, stratified_sample

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load configs
    strate_iv_config = load_strate_iv_config(args.config)

    # Load Strate II
    # Detect z_dim from checkpoint to handle pre-Strate III checkpoints (z_dim=0)
    strate_ii_config = load_strate_ii_config(args.strate_ii_config)
    ckpt_state = torch.load(args.strate_ii_checkpoint, weights_only=False, map_location="cpu")
    # Handle torch.compile prefix (_orig_mod.) in checkpoint keys
    sd = ckpt_state["state_dict"]
    prefix = "_orig_mod." if any(k.startswith("_orig_mod.") for k in sd) else ""

    # CFM-aware detection: Phase D checkpoints have `flow_predictor.vf.0.weight`
    # instead of (or alongside) `predictor.mlp.0.weight`
    has_cfm = f"{prefix}jepa.flow_predictor.vf.0.weight" in sd
    if has_cfm:
        print("  CFM flow_predictor detected in checkpoint (Phase D)")

    # Detect z_dim from checkpoint to handle pre-Strate III (z_dim=0) checkpoints
    pred_key = f"{prefix}jepa.predictor.mlp.0.weight"
    if pred_key in sd:
        pred_in_dim = sd[pred_key].shape[1]
        ckpt_z_dim = pred_in_dim - strate_ii_config.mamba2.d_model * 2
        if ckpt_z_dim != strate_ii_config.predictor.z_dim:
            print(f"  Checkpoint z_dim={ckpt_z_dim} (config has {strate_ii_config.predictor.z_dim}), overriding")
            from dataclasses import replace
            pred_cfg = replace(strate_ii_config.predictor, z_dim=ckpt_z_dim)
            strate_ii_config = replace(strate_ii_config, predictor=pred_cfg)
    else:
        # CFM-only checkpoint: predictor MLP absent, z_dim irrelevant
        print("  predictor.mlp not found in checkpoint — assuming CFM-only model")
    del ckpt_state

    strate_ii = StrateIILightningModule.load_from_checkpoint(
        args.strate_ii_checkpoint, config=strate_ii_config,
        strict=False, weights_only=False,
    )
    jepa = strate_ii.jepa.to(device).eval()

    # Load Strate I for decoder
    strate_i_config = load_strate_i_config(args.strate_i_config)
    strate_i = StrateILightningModule.load_from_checkpoint(
        args.strate_i_checkpoint, config=strate_i_config,
        weights_only=False,
    )
    decoder = strate_i.tokenizer.vqvae.decoder.to(device).eval()
    revin = RevIN(n_channels=5).to(device)

    # Load codebook
    codebook_weights = strate_i.tokenizer.vqvae.codebook.embeddings.clone()
    jepa.load_codebook(codebook_weights.to(device))

    generator = MultiverseGenerator(jepa, decoder, revin)

    seq_len = strate_ii_config.embedding.seq_len
    patch_len = strate_i_config.patch.patch_length  # 16

    precomputer = TrajectoryPrecomputer(
        generator=generator,
        jepa=jepa,
        n_futures=strate_iv_config.buffer.n_futures,
        n_tgt=strate_iv_config.env.n_tgt,
    )

    # Load token dataset
    dataset = TokenSequenceDataset(
        token_dir=args.token_dir,
        seq_len=seq_len,
    )

    # Build OHLCV lookup: token file name → (pair, seq_idx) → OHLCV slice
    ohlcv_dir = Path(args.ohlcv_dir)
    token_dir = Path(args.token_dir)
    token_files = sorted(token_dir.glob("*.pt"))

    # Cache loaded OHLCV tensors per pair
    ohlcv_cache: dict[str, torch.Tensor] = {}

    def ohlcv_lookup(idx):
        """Load the OHLCV slice corresponding to token sequence idx."""
        fname = token_files[idx].stem  # e.g. "BTCUSDT_seq0003"
        pair, seq_part = fname.rsplit("_seq", 1)
        seq_idx = int(seq_part)

        # Load and cache full OHLCV for this pair
        if pair not in ohlcv_cache:
            ohlcv_path = ohlcv_dir / f"{pair}.pt"
            if not ohlcv_path.exists():
                raise FileNotFoundError(
                    f"[INTEGRITY FATAL] OHLCV missing for {pair}: {ohlcv_path}\n"
                    f"Using dummy stats would corrupt RevIN distribution and poison the "
                    f"trajectory buffer. Re-run convert_parquet_to_pt.py to fix."
                )
            ohlcv_cache[pair] = torch.load(ohlcv_path, weights_only=True)

        full_ohlcv = ohlcv_cache[pair]  # (T, 5)

        # Each token sequence covers seq_len patches of patch_len candles
        candles_per_seq = seq_len * patch_len  # 64 * 16 = 1024
        start = seq_idx * candles_per_seq
        end = start + candles_per_seq

        # Clamp to available data
        end = min(end, full_ohlcv.shape[0])
        start = min(start, max(0, end - candles_per_seq))

        return full_ohlcv[start:end]

    n_episodes = args.n_episodes or strate_iv_config.buffer.n_episodes
    output_dir = args.output_dir or strate_iv_config.buffer.buffer_dir

    print(f"Token files: {len(token_files)}")
    print(f"OHLCV dir: {ohlcv_dir}")
    print(f"Episodes to generate: {n_episodes}")
    print(f"Futures per episode: {strate_iv_config.buffer.n_futures}")
    print(f"Targets per episode: {strate_iv_config.env.n_tgt}")
    print(f"Stratified: {getattr(args, 'stratified', False)}")
    print()

    # Stratified sampling: balanced bull/bear/range representation
    sampled_indices = None
    if getattr(args, "stratified", False):
        print("Classifying regimes for stratified sampling...")
        sampled_indices = stratified_sample(
            n_total=len(dataset),
            n_episodes=n_episodes,
            ohlcv_lookup=ohlcv_lookup,
            threshold=getattr(args, "regime_threshold", 1.0),
        )

    precomputer.run(
        dataset=dataset,
        ohlcv_lookup=ohlcv_lookup,
        output_dir=output_dir,
        n_episodes=n_episodes,
        device=str(device),
        indices=sampled_indices,
    )

    print("Pre-computation complete.")


def precompute_historical(args):
    """Pre-compute buffer using REAL historical futures (no MultiverseGenerator).

    For each token sequence i covering OHLCV[i*S*P : (i+1)*S*P]:
      - Context: the token sequence itself + its OHLCV
      - Future: the NEXT n_tgt*patch_len candles of actual OHLCV
      - h_x_pooled: JEPA context encoder on the tokens
      - future_latents: JEPA target encoder on tokenized future patches
      - future_ohlcv: real OHLCV (N=1)

    This avoids MultiverseGenerator bias entirely.
    """
    from src.strate_i.config import load_config as load_strate_i_config
    from src.strate_i.lightning_module import StrateILightningModule
    from src.strate_ii.config import load_config as load_strate_ii_config
    from src.strate_ii.lightning_module import StrateIILightningModule
    from src.strate_ii.data.token_dataset import TokenSequenceDataset
    from src.strate_iv.config import load_config as load_strate_iv_config
    from src.strate_iv.trajectory_buffer import stratified_sample
    from src.strate_i.data.transforms import compute_log_returns, extract_patches

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    strate_iv_config = load_strate_iv_config(args.config)

    # Load Strate II (JEPA encoder only — no generator needed)
    strate_ii_config = load_strate_ii_config(args.strate_ii_config)
    ckpt_state = torch.load(args.strate_ii_checkpoint, weights_only=False, map_location="cpu")
    sd = ckpt_state["state_dict"]
    prefix = "_orig_mod." if any(k.startswith("_orig_mod.") for k in sd) else ""
    pred_key = f"{prefix}jepa.predictor.mlp.0.weight"
    if pred_key in sd:
        pred_in_dim = sd[pred_key].shape[1]
        ckpt_z_dim = pred_in_dim - strate_ii_config.mamba2.d_model * 2
        if ckpt_z_dim != strate_ii_config.predictor.z_dim:
            from dataclasses import replace
            pred_cfg = replace(strate_ii_config.predictor, z_dim=ckpt_z_dim)
            strate_ii_config = replace(strate_ii_config, predictor=pred_cfg)
    del ckpt_state

    strate_ii = StrateIILightningModule.load_from_checkpoint(
        args.strate_ii_checkpoint, config=strate_ii_config,
        strict=False, weights_only=False,
    )
    jepa = strate_ii.jepa.to(device).eval()

    # Load Strate I (tokenizer for future patches)
    strate_i_config = load_strate_i_config(args.strate_i_config)
    strate_i = StrateILightningModule.load_from_checkpoint(
        args.strate_i_checkpoint, config=strate_i_config,
        weights_only=False,
    )
    tokenizer = strate_i.tokenizer.to(device).eval()

    # Load codebook into JEPA
    codebook_weights = strate_i.tokenizer.vqvae.codebook.embeddings.clone()
    jepa.load_codebook(codebook_weights.to(device))

    seq_len = strate_ii_config.embedding.seq_len    # 128
    patch_len = strate_i_config.patch.patch_length   # 16
    n_tgt = strate_iv_config.env.n_tgt               # 8
    d_model = strate_ii_config.mamba2.d_model         # 512

    candles_per_seq = seq_len * patch_len             # 128 * 16 = 2048
    future_candles = n_tgt * patch_len                # 8 * 16 = 128

    # Load token dataset
    dataset = TokenSequenceDataset(token_dir=args.token_dir, seq_len=seq_len)

    # Build OHLCV file mapping
    ohlcv_dir = Path(args.ohlcv_dir)
    token_dir = Path(args.token_dir)
    token_files = sorted(token_dir.glob("*.pt"))

    ohlcv_cache: dict[str, torch.Tensor] = {}

    def get_pair_ohlcv(pair: str) -> torch.Tensor:
        if pair not in ohlcv_cache:
            ohlcv_path = ohlcv_dir / f"{pair}.pt"
            if ohlcv_path.exists():
                ohlcv_cache[pair] = torch.load(ohlcv_path, weights_only=True)
            else:
                ohlcv_cache[pair] = torch.empty(0, 5)
        return ohlcv_cache[pair]

    # Filter: only keep sequences that have enough future data
    valid_indices = []
    for idx in range(len(dataset)):
        fname = token_files[idx].stem
        pair, seq_part = fname.rsplit("_seq", 1)
        seq_idx = int(seq_part)
        full_ohlcv = get_pair_ohlcv(pair)
        future_start = (seq_idx + 1) * candles_per_seq
        future_end = future_start + future_candles
        if future_end <= full_ohlcv.shape[0]:
            valid_indices.append(idx)

    print(f"Token files: {len(token_files)}")
    print(f"Valid (have future data): {len(valid_indices)}")

    n_episodes = args.n_episodes or strate_iv_config.buffer.n_episodes
    output_dir = args.output_dir or strate_iv_config.buffer.buffer_dir

    # Stratified or random sampling from valid indices
    def ohlcv_lookup_for_stratified(idx):
        fname = token_files[idx].stem
        pair, seq_part = fname.rsplit("_seq", 1)
        seq_idx = int(seq_part)
        full = get_pair_ohlcv(pair)
        start = seq_idx * candles_per_seq
        end = start + candles_per_seq
        end = min(end, full.shape[0])
        start = min(start, max(0, end - candles_per_seq))
        return full[start:end]

    if getattr(args, "stratified", False):
        print("Classifying regimes for stratified sampling...")
        # Create a restricted lookup that only uses valid indices
        valid_set = set(valid_indices)
        sampled = stratified_sample(
            n_total=len(dataset),
            n_episodes=n_episodes,
            ohlcv_lookup=ohlcv_lookup_for_stratified,
            threshold=getattr(args, "regime_threshold", 1.0),
        )
        # Filter to valid only
        sampled = [i for i in sampled if i in valid_set]
        if len(sampled) < n_episodes:
            import random as rng_mod
            extra = rng_mod.choices(
                [i for i in sampled], k=n_episodes - len(sampled)
            )
            sampled.extend(extra)
        indices = sampled[:n_episodes]
    else:
        import random as rng_mod
        if len(valid_indices) >= n_episodes:
            indices = rng_mod.sample(valid_indices, n_episodes)
        else:
            indices = valid_indices * (n_episodes // len(valid_indices) + 1)
            indices = indices[:n_episodes]

    print(f"Episodes to generate: {len(indices)}")
    print(f"Mode: HISTORICAL (real futures)")
    print()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for i, idx in enumerate(indices):
            fname = token_files[idx].stem
            pair, seq_part = fname.rsplit("_seq", 1)
            seq_idx = int(seq_part)
            full_ohlcv = get_pair_ohlcv(pair)

            # Context OHLCV
            ctx_start = seq_idx * candles_per_seq
            ctx_end = ctx_start + candles_per_seq
            context_ohlcv = full_ohlcv[ctx_start:ctx_end]  # (T, 5)

            # Future OHLCV (real)
            fut_start = ctx_end
            fut_end = fut_start + future_candles
            future_ohlcv_flat = full_ohlcv[fut_start:fut_end]  # (n_tgt*P, 5)
            future_ohlcv = future_ohlcv_flat.reshape(
                n_tgt, patch_len, 5
            ).unsqueeze(0)  # (1, n_tgt, patch_len, 5)

            # Load tokens
            sample = dataset[idx]
            token_indices = sample["token_indices"].to(device)
            weekend_mask = sample.get("weekend_mask")
            if weekend_mask is not None:
                weekend_mask = weekend_mask.to(device)

            # JEPA context encoding → h_x_pooled
            tokens_b = token_indices.unsqueeze(0)  # (1, S)
            wm_b = weekend_mask.unsqueeze(0) if weekend_mask is not None else None
            h_x = jepa.context_encoder(tokens_b, weekend_mask=wm_b)  # (1, S, d_model)
            h_x_pooled = h_x[:, -1, :].squeeze(0)  # (d_model,)

            # Tokenize future patches → encode through JEPA target encoder
            # Include last context candle as anchor for the first log return
            anchor_and_future = full_ohlcv[ctx_end - 1:fut_end]  # (n_tgt*P + 1, 5)
            future_log_ret = compute_log_returns(anchor_and_future)  # (n_tgt*P, 5)
            future_patches = extract_patches(future_log_ret, patch_len, patch_len)
            if future_patches.shape[0] >= n_tgt:
                future_tokens = tokenizer.tokenize(
                    future_patches[:n_tgt].to(device)
                )  # (n_tgt,)
                fut_tokens_b = future_tokens.unsqueeze(0)  # (1, n_tgt)
                h_fut = jepa.target_encoder(fut_tokens_b)  # (1, n_tgt, d_model)
                future_latents = h_fut.squeeze(0).unsqueeze(0)  # (1, n_tgt, d_model)
            else:
                future_latents = torch.zeros(1, n_tgt, d_model)

            # RevIN stats from context
            ctx_t = context_ohlcv.unsqueeze(0)  # (1, T, 5)
            revin_means = ctx_t.mean(dim=1, keepdim=True).squeeze(0)  # (1, 5)
            revin_stds = ctx_t.std(dim=1, keepdim=True).squeeze(0) + 1e-8  # (1, 5)

            last_close = context_ohlcv[-1, 3].item()

            save_dict = {
                "context_tokens": token_indices.cpu(),
                "weekend_mask": weekend_mask.cpu() if weekend_mask is not None else None,
                "context_ohlcv": context_ohlcv.cpu(),
                "future_ohlcv": future_ohlcv.cpu(),
                "future_latents": future_latents.cpu(),
                "revin_means": revin_means.cpu(),
                "revin_stds": revin_stds.cpu(),
                "last_close": last_close,
                "h_x_pooled": h_x_pooled.cpu(),
            }
            torch.save(save_dict, out_path / f"episode_{saved:05d}.pt")
            saved += 1

            if saved % 50 == 0:
                print(f"  [{saved}/{len(indices)}] episodes saved")

    print(f"\nHistorical buffer complete: {saved} episodes in {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Strate IV trajectory buffer")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run smoke test with random weights")
    parser.add_argument("--config", type=str, default="configs/strate_iv.yaml")
    parser.add_argument("--strate_ii_checkpoint", type=str, default=None)
    parser.add_argument("--strate_ii_config", type=str, default="configs/strate_ii.yaml")
    parser.add_argument("--strate_i_checkpoint", type=str, default=None)
    parser.add_argument("--strate_i_config", type=str, default="configs/strate_i_binance.yaml")
    parser.add_argument("--token_dir", type=str, default="data/tokens/")
    parser.add_argument("--ohlcv_dir", type=str, default="data/binance_1h_subset/")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output dir from config")
    parser.add_argument("--n_episodes", type=int, default=None,
                        help="Override n_episodes from config")
    parser.add_argument("--stratified", action="store_true",
                        help="Use stratified sampling (balanced bull/bear/range)")
    parser.add_argument("--regime_threshold", type=float, default=1.0,
                        help="Z-score threshold for regime classification (default: 1.0)")
    parser.add_argument("--historical", action="store_true",
                        help="Use real historical futures instead of MultiverseGenerator")

    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        if args.strate_ii_checkpoint is None or args.strate_i_checkpoint is None:
            parser.error("--strate_ii_checkpoint and --strate_i_checkpoint required "
                         "(or use --smoke_test)")
        if args.historical:
            precompute_historical(args)
        else:
            precompute_from_checkpoints(args)


if __name__ == "__main__":
    main()
