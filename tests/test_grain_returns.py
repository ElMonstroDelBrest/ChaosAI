"""Test that grain_loader parses returns field correctly."""
import numpy as np
import tensorflow as tf


def _make_serialized_example(seq_len=4, with_returns=True):
    """Create a minimal serialized tf.train.Example with/without returns."""
    feature = {
        "token_indices": tf.train.Feature(int64_list=tf.train.Int64List(value=[0]*seq_len)),
        "weekend_mask": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0]*seq_len)),
        "exo_clock": tf.train.Feature(float_list=tf.train.FloatList(value=[0.0]*(seq_len*2))),
        "exo_clock_dim": tf.train.Feature(int64_list=tf.train.Int64List(value=[2])),
        "pair_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[b"test__BTC"])),
        "original_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len])),
        "source_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
        "scale_id": tf.train.Feature(int64_list=tf.train.Int64List(value=[0])),
    }
    if with_returns:
        feature["returns"] = tf.train.Feature(
            float_list=tf.train.FloatList(value=[0.01, -0.02, 0.03, 0.0]))
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def test_parse_example_with_returns():
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.data.grain_loader import _parse_example
    serialized = _make_serialized_example(seq_len=4, with_returns=True)
    result = _parse_example(serialized, seq_len=4)
    assert "returns" in result
    np.testing.assert_allclose(result["returns"], [0.01, -0.02, 0.03, 0.0], atol=1e-6)


def test_parse_example_without_returns_fallback():
    """Legacy shards without returns field must fall back to zeros."""
    import sys; sys.path.insert(0, ".")
    from src.jax_v6.data.grain_loader import _parse_example
    serialized = _make_serialized_example(seq_len=4, with_returns=False)
    result = _parse_example(serialized, seq_len=4)
    assert "returns" in result
    np.testing.assert_allclose(result["returns"], [0.0]*4, atol=1e-6)
