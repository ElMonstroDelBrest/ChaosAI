"""Tests for JAX GNN data loading pipeline."""
import pytest
import numpy as np
from pathlib import Path

jax = pytest.importorskip("jax")
jraph = pytest.importorskip("jraph")
torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")


@pytest.fixture
def tmp_graph_dir(tmp_path):
    """Create temporary .pt graph files mimicking real data."""
    from torch_geometric.data import Data

    for i in range(5):
        n = 50 + i * 10
        e = 100 + i * 20
        data = Data(
            x=torch.randn(n, 8),
            edge_index=torch.randint(0, n, (2, e)),
            edge_attr=torch.randn(e, 2),
        )
        data.exchange_net_flow = torch.tensor(float(i) * 0.1)
        torch.save(data, tmp_path / f"2024-01-01_{i:02d}.pt")
    return tmp_path


class TestGraphLoader:
    def test_load_single_graph(self, tmp_graph_dir):
        from src.jax_v6.strate_v.data_loader import load_pyg_as_jraph

        graph = load_pyg_as_jraph(tmp_graph_dir / "2024-01-01_00.pt")
        assert graph.nodes.shape[1] == 8
        assert graph.edges.shape[1] == 2
        assert graph.n_node.shape == (1,)
        assert graph.globals.shape == (1, 1)  # exchange_net_flow

    def test_load_graph_pair(self, tmp_graph_dir):
        from src.jax_v6.strate_v.data_loader import GraphPairDataset

        ds = GraphPairDataset([str(tmp_graph_dir)])
        assert len(ds) > 0
        pair = ds[0]
        assert len(pair) == 2  # (graph_t, graph_t+1)

    def test_batch_graphs(self, tmp_graph_dir):
        from src.jax_v6.strate_v.data_loader import load_pyg_as_jraph

        graphs = []
        for f in sorted(tmp_graph_dir.glob("*.pt"))[:3]:
            graphs.append(load_pyg_as_jraph(f))
        batched = jraph.batch(graphs)
        assert batched.n_node.shape[0] == 3  # 3 graphs
        assert batched.nodes.shape[0] == sum(g.nodes.shape[0] for g in graphs)

    def test_negative_sampling(self, tmp_graph_dir):
        from src.jax_v6.strate_v.data_loader import negative_sampling_jax

        senders = np.array([0, 1, 2, 3])
        receivers = np.array([1, 2, 3, 0])
        key = jax.random.PRNGKey(0)
        neg_src, neg_dst = negative_sampling_jax(
            key, senders, receivers, n_nodes=5, n_neg=4
        )
        assert neg_src.shape == (4,)
        assert neg_dst.shape == (4,)
