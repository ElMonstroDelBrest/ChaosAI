"""Test that pretokenize_tpu stores returns in ArrayRecord."""
import numpy as np
import pytest
import tempfile, os


def test_write_shard_stores_returns():
    """_write_shard must include a 'returns' field in each record."""
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordWriter, ArrayRecordReader

    import sys
    sys.path.insert(0, ".")
    from scripts.pretokenize_tpu import _write_shard

    seq_len = 4
    n_seqs = 2
    pair_name = "test__BTCUSDT"
    ti = np.zeros((n_seqs, seq_len), dtype=np.int64)
    ec = np.zeros((n_seqs, seq_len, 2), dtype=np.float32)
    am = np.ones((n_seqs, seq_len), dtype=np.float32)
    ret = np.array([[0.01, -0.02, 0.03, 0.0], [-0.01, 0.0, 0.02, -0.03]], dtype=np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        _write_shard((pair_name, ti, ec, am, ret, 0, n_seqs, tmp, seq_len, 2, None))
        shard_path = os.path.join(tmp, f"{pair_name}.arrayrecord")
        assert os.path.exists(shard_path)

        reader = ArrayRecordReader(shard_path)
        record = tf.train.Example()
        record.ParseFromString(reader.read(0))
        features = record.features.feature
        assert "returns" in features, "returns field missing from ArrayRecord"
        returns_stored = list(features["returns"].float_list.value)
        assert len(returns_stored) == seq_len
        np.testing.assert_allclose(returns_stored, ret[0], atol=1e-6)
