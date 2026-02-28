"""Data I/O helpers — lazy-imports TensorFlow (not always available)."""

import numpy as np


def read_arrayrecord_tokens(shard_path):
    """Read all token_indices from an ArrayRecord shard.

    Args:
        shard_path: Path to a .arrayrecord file.

    Returns:
        np.ndarray of shape (N, seq_len) dtype int64.
    """
    # Lazy imports — tf and array_record are only on TPU VMs
    import tensorflow as tf
    from array_record.python.array_record_module import ArrayRecordReader

    reader = ArrayRecordReader(str(shard_path))
    n = reader.num_records()
    tokens_list = []
    for i in range(n):
        reader.seek(i)
        raw = reader.read()
        example = tf.train.Example.FromString(raw)
        tids = list(example.features.feature["token_indices"].int64_list.value)
        tokens_list.append(tids)
    reader.close()
    return np.array(tokens_list, dtype=np.int64)
