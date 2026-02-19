"""Tests for Fin-JEPA: E2E forward, EMA, no-grad on E_y."""

import torch
import pytest

from src.strate_ii.jepa import FinJEPA


@pytest.fixture
def jepa():
    """Small JEPA for testing."""
    return FinJEPA(
        num_codes=64,
        codebook_dim=16,
        d_model=32,
        d_state=4,
        n_layers=2,
        n_heads=2,
        expand_factor=2,
        conv_kernel=4,
        seq_len=16,
        pred_hidden_dim=64,
        pred_n_layers=2,
        pred_dropout=0.0,
        mask_ratio=0.5,
        block_size_min=2,
        block_size_max=4,
        tau=0.99,
    )


def test_forward_e2e(jepa):
    """Forward pass should produce finite loss and correct keys."""
    tokens = torch.randint(0, 64, (4, 16))
    out = jepa(tokens)

    expected_keys = {"loss", "invariance", "variance", "covariance", "cfm_loss", "mask_ratio", "n_targets"}
    assert set(out.keys()) == expected_keys

    for k, v in out.items():
        assert torch.isfinite(v), f"{k} is not finite: {v}"


def test_forward_with_weekend(jepa):
    """Forward pass works with weekend mask."""
    tokens = torch.randint(0, 64, (4, 16))
    weekend = torch.zeros(4, 16)
    weekend[:, 5:7] = 1.0
    weekend[:, 12:14] = 1.0

    out = jepa(tokens, weekend_mask=weekend)
    assert torch.isfinite(out["loss"])


def test_no_grad_target_encoder(jepa):
    """Target encoder parameters must have requires_grad=False."""
    for name, param in jepa.target_encoder.named_parameters():
        assert not param.requires_grad, \
            f"Target encoder param '{name}' has requires_grad=True"


def test_no_grad_after_backward(jepa):
    """After backward, target encoder params should have grad=None."""
    tokens = torch.randint(0, 64, (4, 16))
    out = jepa(tokens)
    out["loss"].backward()

    for name, param in jepa.target_encoder.named_parameters():
        assert param.grad is None, \
            f"Target encoder param '{name}' has gradient after backward"


def test_context_encoder_has_grad(jepa):
    """Context encoder params should have gradients after backward."""
    tokens = torch.randint(0, 64, (4, 16))
    out = jepa(tokens)
    out["loss"].backward()

    has_grad = False
    for name, param in jepa.context_encoder.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break

    assert has_grad, "No gradients in context encoder after backward"


def test_predictor_has_grad(jepa):
    """Predictor params should have gradients after backward."""
    tokens = torch.randint(0, 64, (4, 16))
    out = jepa(tokens)
    out["loss"].backward()

    has_grad = False
    for name, param in jepa.predictor.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break

    assert has_grad, "No gradients in predictor after backward"


def test_ema_update(jepa):
    """EMA update should correctly blend target and context encoder params."""
    tau = 0.99
    jepa.set_tau(tau)

    # Record old target params
    old_params = {
        name: param.data.clone()
        for name, param in jepa.target_encoder.named_parameters()
    }

    # Modify context encoder (simulate training step)
    with torch.no_grad():
        for param in jepa.context_encoder.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    # Perform EMA update
    jepa.update_target_encoder()

    # Verify: param_y_new ≈ τ*param_y_old + (1-τ)*param_x
    for name, p_y in jepa.target_encoder.named_parameters():
        p_y_old = old_params[name]
        # Find corresponding context encoder param
        p_x = dict(jepa.context_encoder.named_parameters())[name]
        expected = tau * p_y_old + (1.0 - tau) * p_x.data
        assert torch.allclose(p_y.data, expected, atol=1e-6), \
            f"EMA incorrect for '{name}'"


def test_ema_initial_match(jepa):
    """Initially, target encoder should be a copy of context encoder."""
    for (name_x, p_x), (name_y, p_y) in zip(
        jepa.context_encoder.named_parameters(),
        jepa.target_encoder.named_parameters(),
    ):
        assert name_x == name_y
        assert torch.allclose(p_x.data, p_y.data), \
            f"Initial mismatch: {name_x}"


def test_codebook_frozen(jepa):
    """Codebook embeddings should be frozen (no grad)."""
    assert not jepa.context_encoder.codebook_embed.weight.requires_grad
    assert not jepa.target_encoder.codebook_embed.weight.requires_grad


def test_load_codebook(jepa):
    """load_codebook should update both encoders."""
    weights = torch.randn(64, 16)
    jepa.load_codebook(weights)

    assert torch.allclose(jepa.context_encoder.codebook_embed.weight.data, weights)
    assert torch.allclose(jepa.target_encoder.codebook_embed.weight.data, weights)


def test_loss_decreases_dummy():
    """Basic sanity: loss should be computable and finite for multiple steps."""
    jepa = FinJEPA(
        num_codes=32, codebook_dim=8, d_model=16, d_state=4,
        n_layers=1, n_heads=2, expand_factor=2, conv_kernel=2,
        seq_len=8, pred_hidden_dim=32, pred_n_layers=1,
        mask_ratio=0.5, block_size_min=2, block_size_max=3,
    )
    optimizer = torch.optim.Adam(
        [p for p in jepa.parameters() if p.requires_grad], lr=1e-3
    )

    tokens = torch.randint(0, 32, (8, 8))
    losses = []

    for _ in range(5):
        out = jepa(tokens)
        loss = out["loss"]
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()

    # All losses should be finite
    assert all(torch.isfinite(torch.tensor(l)) for l in losses)
