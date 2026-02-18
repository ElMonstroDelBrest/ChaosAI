"""Convert .pt token files to ArrayRecord format for GCS + Grain pipeline.

Input:  data/tokens_v5/{pair_name}_seq{XXXX}.pt
        Each file contains {token_indices: int64(S,), weekend_mask: float32(S,),
        exo_clock: float32(S, 2)}

Output: data/arrayrecord/{pair_name}.arrayrecord  (1 file per pair)
        + data/arrayrecord/manifest.json (shard list + total count)

Records are zero-padded to seq_len=128.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# tensorflow only for tf.train.Example serialization
import tensorflow as tf
from array_record.python.array_record_module import ArrayRecordWriter


def pt_to_example(
    data: dict,
    pair_name: str,
    seq_id: int,
    target_seq_len: int = 128,
) -> bytes:
    """Convert a .pt dict to a serialized tf.train.Example protobuf.

    Args:
        data: Dict from torch.load with token_indices, weekend_mask, exo_clock.
        pair_name: Trading pair name (e.g. "BTCUSDT").
        seq_id: Sequential index.
        target_seq_len: Pad/truncate to this length.

    Returns:
        Serialized protobuf bytes.
    """
    token_indices = data["token_indices"].numpy().astype(np.int64)
    weekend_mask = data["weekend_mask"].numpy().astype(np.float32)

    # exo_clock may be absent in older files
    if "exo_clock" in data and data["exo_clock"] is not None:
        exo_clock = data["exo_clock"].numpy().astype(np.float32)
    else:
        exo_clock = np.zeros((len(token_indices), 2), dtype=np.float32)

    original_len = len(token_indices)

    # Pad to target_seq_len
    if original_len < target_seq_len:
        pad_len = target_seq_len - original_len
        token_indices = np.pad(token_indices, (0, pad_len), constant_values=0)
        weekend_mask = np.pad(weekend_mask, (0, pad_len), constant_values=0.0)
        exo_clock = np.pad(exo_clock, ((0, pad_len), (0, 0)), constant_values=0.0)
    elif original_len > target_seq_len:
        token_indices = token_indices[:target_seq_len]
        weekend_mask = weekend_mask[:target_seq_len]
        exo_clock = exo_clock[:target_seq_len]
        original_len = target_seq_len

    feature = {
        "token_indices": tf.train.Feature(
            int64_list=tf.train.Int64List(value=token_indices.tolist())
        ),
        "weekend_mask": tf.train.Feature(
            float_list=tf.train.FloatList(value=weekend_mask.tolist())
        ),
        "exo_clock": tf.train.Feature(
            float_list=tf.train.FloatList(value=exo_clock.flatten().tolist())
        ),
        "pair_name": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[pair_name.encode("utf-8")])
        ),
        "seq_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[seq_id])
        ),
        "original_len": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[original_len])
        ),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def convert(
    input_dir: str,
    output_dir: str,
    target_seq_len: int = 128,
):
    """Convert all .pt files to ArrayRecord shards (1 per pair).

    Args:
        input_dir: Directory containing .pt files.
        output_dir: Output directory for .arrayrecord files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Group files by pair name
    pt_files = sorted(input_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    pair_files = defaultdict(list)
    for f in pt_files:
        # Extract pair name: {pair_name}_seq{XXXX}.pt
        match = re.match(r"(.+?)_seq(\d+)\.pt$", f.name)
        if match:
            pair_name = match.group(1)
            seq_id = int(match.group(2))
            pair_files[pair_name].append((seq_id, f))
        else:
            print(f"Warning: skipping {f.name} (unexpected naming)")

    manifest = {"shards": [], "total_records": 0, "seq_len": target_seq_len}
    total = 0

    for pair_name in sorted(pair_files):
        files = sorted(pair_files[pair_name], key=lambda x: x[0])
        shard_path = output_path / f"{pair_name}.arrayrecord"

        writer = ArrayRecordWriter(str(shard_path), "group_size:1")
        count = 0

        for seq_id, pt_file in files:
            data = torch.load(pt_file, map_location="cpu", weights_only=True)
            record = pt_to_example(data, pair_name, seq_id, target_seq_len)
            writer.write(record)
            count += 1

        writer.close()
        manifest["shards"].append({"pair": pair_name, "path": str(shard_path), "count": count})
        total += count
        print(f"  {pair_name}: {count} records -> {shard_path}")

    manifest["total_records"] = total

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {total} records across {len(pair_files)} shards.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .pt token files to ArrayRecord for Grain pipeline"
    )
    parser.add_argument("--input", default="data/tokens_v5/", help="Input .pt directory")
    parser.add_argument("--output", default="data/arrayrecord/", help="Output ArrayRecord directory")
    parser.add_argument("--seq_len", type=int, default=128, help="Target sequence length (with padding)")
    args = parser.parse_args()

    convert(args.input, args.output, args.seq_len)
