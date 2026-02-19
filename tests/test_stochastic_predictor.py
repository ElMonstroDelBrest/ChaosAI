"""Tests for Strate III: stochastic predictor, generate_futures, output_proj."""

import torch
import pytest

from src.strate_ii.predictor import Predictor
from src.strate_ii.jepa import FinJEPA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def predictor():
    """Small stochastic predictor for testing."""
    return Predictor(
        d_model=32, hidden_dim=64, n_layers=2, seq_len=16,
        dropout=0.0, z_dim=8,
    )


@pytest.fixture
def jepa():
    """Small JEPA with stochastic predictor for testing."""
    return FinJEPA(
        num_codes=64, codebook_dim=16, d_model=32, d_state=4,
        n_layers=2, n_heads=2, expand_factor=2, conv_kernel=4,
        seq_len=16, pred_hidden_dim=64, pred_n_layers=2,
        pred_dropout=0.0, pred_z_dim=8,
        mask_ratio=0.5, block_size_min=2, block_size_max=4, tau=0.99,
    )


# ---------------------------------------------------------------------------
# Predictor tests
# ---------------------------------------------------------------------------

class TestPredictor:
    def test_shape_z_none(self, predictor):
        """z=None → same output shape as deterministic predictor."""
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        out = predictor(h_x, target_pos, z=None)
        assert out.shape == (4, 6, 32)

    def test_shape_z_provided(self, predictor):
        """z provided → same output shape."""
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        z = torch.randn(4, 6, 8)
        out = predictor(h_x, target_pos, z=z)
        assert out.shape == (4, 6, 32)

    def test_diversity(self, predictor):
        """Two different z values → different outputs."""
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        z1 = torch.randn(4, 6, 8)
        z2 = torch.randn(4, 6, 8)
        out1 = predictor(h_x, target_pos, z=z1)
        out2 = predictor(h_x, target_pos, z=z2)
        assert not torch.allclose(out1, out2, atol=1e-6), \
            "Different z should produce different outputs"

    def test_determinism_z_zero(self, predictor):
        """z=0 (explicit zeros) → output identical across calls."""
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        z = torch.zeros(4, 6, 8)
        out1 = predictor(h_x, target_pos, z=z)
        out2 = predictor(h_x, target_pos, z=z)
        assert torch.allclose(out1, out2, atol=1e-7), \
            "Same z=0 should produce identical outputs"

    def test_determinism_z_none(self, predictor):
        """z=None → deterministic (zeros internally), output identical."""
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        out1 = predictor(h_x, target_pos, z=None)
        out2 = predictor(h_x, target_pos, z=None)
        assert torch.allclose(out1, out2, atol=1e-7)

    def test_z_none_equals_z_zero(self, predictor):
        """z=None and z=zeros should give identical results."""
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        out_none = predictor(h_x, target_pos, z=None)
        out_zero = predictor(h_x, target_pos, z=torch.zeros(4, 6, 8))
        assert torch.allclose(out_none, out_zero, atol=1e-7)

    def test_z_dim_zero(self):
        """z_dim=0 → fully deterministic predictor, z ignored."""
        pred = Predictor(d_model=32, hidden_dim=64, n_layers=2, seq_len=16,
                         dropout=0.0, z_dim=0)
        h_x = torch.randn(4, 16, 32)
        target_pos = torch.randint(0, 16, (4, 6))
        out = pred(h_x, target_pos, z=None)
        assert out.shape == (4, 6, 32)


# ---------------------------------------------------------------------------
# FinJEPA Strate III tests
# ---------------------------------------------------------------------------

class TestFinJEPAStrateIII:
    def test_generate_futures_shape(self, jepa):
        """generate_futures → (N, B, N_tgt, d_model)."""
        tokens = torch.randint(0, 64, (4, 16))
        target_pos = torch.arange(8, 14).unsqueeze(0).expand(4, -1)  # (4, 6)
        futures = jepa.generate_futures(tokens, None, target_pos, n_samples=8)
        assert futures.shape == (8, 4, 6, 32)

    def test_generate_futures_diversity(self, jepa):
        """Different samples should be different (stochastic)."""
        tokens = torch.randint(0, 64, (2, 16))
        target_pos = torch.arange(8, 14).unsqueeze(0).expand(2, -1)
        futures = jepa.generate_futures(tokens, None, target_pos, n_samples=4)
        # Compare first two samples
        assert not torch.allclose(futures[0], futures[1], atol=1e-5), \
            "Different samples should produce different trajectories"

    def test_output_proj_shape(self, jepa):
        """output_proj: (B, d_model) → (B, codebook_dim)."""
        h = torch.randn(4, 32)  # d_model=32
        z = jepa.project_to_codebook_space(h)
        assert z.shape == (4, 16)  # codebook_dim=16

    def test_output_proj_batched(self, jepa):
        """output_proj works with arbitrary leading dims."""
        h = torch.randn(8, 4, 6, 32)  # (N, B, N_tgt, d_model)
        z = jepa.project_to_codebook_space(h)
        assert z.shape == (8, 4, 6, 16)

    def test_forward_backward_compat(self, jepa):
        """Training forward() still works with stochastic predictor."""
        tokens = torch.randint(0, 64, (4, 16))
        out = jepa(tokens)
        expected_keys = {"loss", "invariance", "variance", "covariance",
                         "cfm_loss", "mask_ratio", "n_targets"}
        assert set(out.keys()) == expected_keys
        for k, v in out.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_forward_backward_pass(self, jepa):
        """Backward pass works with stochastic predictor."""
        tokens = torch.randint(0, 64, (4, 16))
        out = jepa(tokens)
        out["loss"].backward()
        # Predictor should have gradients
        has_grad = any(
            p.grad is not None
            for p in jepa.predictor.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients in predictor after backward"

    def test_forward_with_weekend(self, jepa):
        """Forward with weekend mask still works."""
        tokens = torch.randint(0, 64, (4, 16))
        weekend = torch.zeros(4, 16)
        weekend[:, 5:7] = 1.0
        out = jepa(tokens, weekend_mask=weekend)
        assert torch.isfinite(out["loss"])

    def test_loss_finite_multiple_steps(self):
        """Loss should be finite over multiple training steps."""
        jepa = FinJEPA(
            num_codes=32, codebook_dim=8, d_model=16, d_state=4,
            n_layers=1, n_heads=2, expand_factor=2, conv_kernel=2,
            seq_len=8, pred_hidden_dim=32, pred_n_layers=1,
            pred_z_dim=4,
            mask_ratio=0.5, block_size_min=2, block_size_max=3,
        )
        optimizer = torch.optim.Adam(
            [p for p in jepa.parameters() if p.requires_grad], lr=1e-3
        )
        tokens = torch.randint(0, 32, (8, 8))
        for _ in range(5):
            out = jepa(tokens)
            loss = out["loss"]
            assert torch.isfinite(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            jepa.update_target_encoder()


# ---------------------------------------------------------------------------
# MultiverseGenerator tests
# ---------------------------------------------------------------------------

class TestMultiverseGenerator:
    def test_full_pipeline(self, jepa):
        """Full pipeline: JEPA → decode → denorm → OHLCV."""
        from src.strate_i.decoder import Decoder
        from src.strate_i.revin import RevIN
        from src.strate_ii.multiverse import MultiverseGenerator

        decoder = Decoder(latent_dim=16, hidden_channels=32, out_channels=5,
                          patch_length=16, n_layers=2, kernel_size=3)
        revin = RevIN(n_channels=5)

        generator = MultiverseGenerator(jepa, decoder, revin)

        B, S, N_tgt, T = 2, 16, 4, 64
        tokens = torch.randint(0, 64, (B, S))
        target_pos = torch.arange(8, 8 + N_tgt).unsqueeze(0).expand(B, -1)
        context_ohlcv = torch.rand(B, T, 5) * 100 + 1  # Positive OHLCV

        result = generator.generate(
            tokens, None, target_pos, context_ohlcv, n_samples=4,
        )

        assert result["latents"].shape == (4, B, N_tgt, 32)
        assert result["codebook_z"].shape == (4, B, N_tgt, 16)
        assert result["patches_norm"].shape == (4, B, N_tgt, 16, 5)
        assert result["patches_real"].shape == (4, B, N_tgt, 16, 5)
        assert result["ohlcv"].shape == (4, B, N_tgt, 16, 5)

    def test_ohlcv_positive_prices(self, jepa):
        """Reconstructed OHLCV prices should be positive."""
        from src.strate_i.decoder import Decoder
        from src.strate_i.revin import RevIN
        from src.strate_ii.multiverse import MultiverseGenerator

        decoder = Decoder(latent_dim=16, hidden_channels=32, out_channels=5,
                          patch_length=16, n_layers=2, kernel_size=3)
        revin = RevIN(n_channels=5)
        generator = MultiverseGenerator(jepa, decoder, revin)

        B = 2
        tokens = torch.randint(0, 64, (B, 16))
        target_pos = torch.arange(8, 12).unsqueeze(0).expand(B, -1)
        context_ohlcv = torch.rand(B, 64, 5) * 100 + 10

        result = generator.generate(tokens, None, target_pos, context_ohlcv, n_samples=4)
        ohlc_prices = result["ohlcv"][..., :4]
        assert (ohlc_prices > 0).all(), "All OHLC prices should be positive (exp guarantees this)"

    def test_multiverse_diversity(self, jepa):
        """Different samples should produce different OHLCV candles."""
        from src.strate_i.decoder import Decoder
        from src.strate_i.revin import RevIN
        from src.strate_ii.multiverse import MultiverseGenerator

        decoder = Decoder(latent_dim=16, hidden_channels=32, out_channels=5,
                          patch_length=16, n_layers=2, kernel_size=3)
        revin = RevIN(n_channels=5)
        generator = MultiverseGenerator(jepa, decoder, revin)

        B = 2
        tokens = torch.randint(0, 64, (B, 16))
        target_pos = torch.arange(8, 12).unsqueeze(0).expand(B, -1)
        context_ohlcv = torch.rand(B, 64, 5) * 100 + 10

        result = generator.generate(tokens, None, target_pos, context_ohlcv, n_samples=4)
        # Different samples should differ
        assert not torch.allclose(result["ohlcv"][0], result["ohlcv"][1], atol=1e-5)
