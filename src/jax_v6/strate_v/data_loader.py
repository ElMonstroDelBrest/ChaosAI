"""Graph data loading: PyG .pt files -> jraph.GraphsTuple."""

import numpy as np
import jax
import jax.numpy as jnp
import jraph
import torch
from pathlib import Path


def load_pyg_as_jraph(pt_path: str | Path) -> jraph.GraphsTuple:
    """Load a PyG .pt file and convert to jraph.GraphsTuple."""
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)

    nodes = data.x.numpy().astype(np.float32)
    senders = data.edge_index[0].numpy().astype(np.int32)
    receivers = data.edge_index[1].numpy().astype(np.int32)
    edges = (
        data.edge_attr.numpy().astype(np.float32)
        if data.edge_attr is not None
        else np.zeros((senders.shape[0], 2), dtype=np.float32)
    )

    flow = (
        data.exchange_net_flow.item()
        if hasattr(data, "exchange_net_flow") and data.exchange_net_flow is not None
        else 0.0
    )

    return jraph.GraphsTuple(
        nodes=jnp.array(nodes),
        edges=jnp.array(edges),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array([nodes.shape[0]]),
        n_edge=jnp.array([senders.shape[0]]),
        globals=jnp.array([[flow]]),
    )


class GraphPairDataset:
    """Dataset of consecutive graph pairs for temporal contrastive learning."""

    def __init__(self, graph_dirs: list[str]):
        self.pairs = []
        for gdir in graph_dirs:
            pt_files = sorted(Path(gdir).glob("*.pt"))
            for i in range(len(pt_files) - 1):
                self.pairs.append((pt_files[i], pt_files[i + 1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f_t, f_tp1 = self.pairs[idx]
        return (load_pyg_as_jraph(f_t), load_pyg_as_jraph(f_tp1))


def negative_sampling_jax(key, senders, receivers, n_nodes, n_neg):
    """Random negative edge sampling (JAX-compatible).

    Samples random (src, dst) pairs that are unlikely to be real edges.
    For large sparse graphs, collision with real edges is negligible.
    """
    k1, k2 = jax.random.split(key)
    neg_src = jax.random.randint(k1, (n_neg,), 0, n_nodes)
    neg_dst = jax.random.randint(k2, (n_neg,), 0, n_nodes)
    return neg_src, neg_dst
