"""Test Priority Experience Replay with bifurcation weights."""
import numpy as np


def test_per_weights_sum_to_one():
    """PER sampling weights must form a valid probability distribution."""
    bif = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
    per_alpha = 0.6
    eps = 1e-5
    priorities = (bif + eps) ** per_alpha
    weights = priorities / priorities.sum()
    assert np.isclose(weights.sum(), 1.0), f"PER weights sum={weights.sum()}"
    assert np.all(weights > 0), "All weights must be positive"
    assert weights[3] > weights[0], "Higher bifurcation → higher priority"


def test_per_flag_in_train_script():
    """train_cross_sectional.py must contain use_per logic."""
    import pathlib
    src = pathlib.Path("scripts/train_cross_sectional.py").read_text()
    assert "use_per" in src, "Missing use_per in train_cross_sectional.py"
    assert "bifurcation_index" in src, "Missing bifurcation_index loading"
    assert "per_alpha" in src, "Missing per_alpha in train script"


def test_load_buffer_loads_bifurcation():
    """load_buffer must load bifurcation_index from NPZ when available."""
    import pathlib
    src = pathlib.Path("scripts/train_cross_sectional.py").read_text()
    assert "bifurcation_index" in src
    assert "bif" in src, "Missing bif variable in load_buffer"


def test_config_has_per_fields():
    """CrossSectionalConfig must have use_per and per_alpha."""
    from src.jax_v6.config import CrossSectionalConfig
    cfg = CrossSectionalConfig()
    assert hasattr(cfg, "use_per") and cfg.use_per is False
    assert hasattr(cfg, "per_alpha") and abs(cfg.per_alpha - 0.6) < 1e-6
