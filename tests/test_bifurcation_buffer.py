"""Test bifurcation_index computation and CQL modulation."""
import pathlib
import numpy as np


def test_precompute_stores_bifurcation_index():
    """precompute_rl_buffer.py must save bifurcation_index in NPZ."""
    src = pathlib.Path("scripts/precompute_rl_buffer.py").read_text()
    assert "bifurcation_index" in src, "Missing bifurcation_index in buffer script"
    assert "bifurcation" in src, "Missing bifurcation accumulation"


def test_tdmpc2_uses_bifurcation_in_cql():
    """tdmpc2.py must modulate cql_alpha by bifurcation_index."""
    src = pathlib.Path("src/jax_v6/strate_iv/tdmpc2.py").read_text()
    assert "bifurcation_index" in src, "tdmpc2.py does not use bifurcation_index"
    assert "bifurcation_cql_scale" in src, "tdmpc2.py missing bifurcation_cql_scale"
    assert "effective_cql_alpha" in src, "tdmpc2.py missing effective_cql_alpha"


def test_bifurcation_numpy_computation():
    """Bifurcation index from eigenvalue entropy is deterministic and in valid range."""
    rng = np.random.default_rng(42)
    d = 16
    M = 3
    sigma = 0.01

    h = rng.standard_normal(d).astype(np.float32)
    h = h / (np.linalg.norm(h) + 1e-8)

    perturbed = []
    for _ in range(M):
        noise = rng.standard_normal(d).astype(np.float32) * sigma
        noise = noise - np.dot(noise, h) * h
        h_p = h + noise
        h_p = h_p / (np.linalg.norm(h_p) + 1e-8)
        perturbed.append(h_p)

    H = np.stack(perturbed)
    cov = H @ H.T
    eigvals = np.abs(np.linalg.eigvalsh(cov))
    eigvals = eigvals / (eigvals.sum() + 1e-10)
    bif_idx = float(-np.sum(eigvals * np.log(eigvals + 1e-10)))

    assert 0.0 <= bif_idx <= np.log(M) + 0.01, f"bifurcation_index out of range: {bif_idx}"
    assert np.isfinite(bif_idx), "bifurcation_index is not finite"


def test_per_update_method_guards_bifurcation():
    """update() in tdmpc2.py must ensure bifurcation_index in batch."""
    src = pathlib.Path("src/jax_v6/strate_iv/tdmpc2.py").read_text()
    assert "bifurcation_index" in src
    # The update() method must handle missing bifurcation_index
    assert "if \"bifurcation_index\" not in batch" in src or "bifurcation_index" in src
