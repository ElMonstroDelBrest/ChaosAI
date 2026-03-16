def test_config_ret_weight_default_zero():
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.config import Mamba2Config
    cfg = Mamba2Config()
    assert cfg.ret_weight == 0.0


def test_config_ret_weight_from_yaml():
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.config import load_config
    cfg = load_config("configs/scaling/v6e_38m.yaml")
    assert cfg.mamba2.ret_weight == 0.01
