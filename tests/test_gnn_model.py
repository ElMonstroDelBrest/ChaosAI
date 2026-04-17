"""Unit tests for OnChainGNN model."""

import pytest
import torch

torch_geometric = pytest.importorskip("torch_geometric")
from torch_geometric.data import Data, Batch


# Lazy import to allow test collection even without torch_geometric
@pytest.fixture
def gnn():
    from src.strate_v.gnn_model import OnChainGNN
    return OnChainGNN(node_features=8, gnn_dim=256)


@pytest.fixture
def sample_data():
    """Single graph with 100 nodes, 500 edges."""
    x = torch.randn(100, 8)
    edge_index = torch.randint(0, 100, (2, 500))
    edge_attr = torch.randn(500, 2)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.exchange_net_flow = torch.tensor(0.5)
    return data


@pytest.fixture
def batched_data(sample_data):
    """Batch of 4 graphs."""
    graphs = []
    for _ in range(4):
        x = torch.randn(50, 8)
        edge_index = torch.randint(0, 50, (2, 200))
        edge_attr = torch.randn(200, 2)
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        d.exchange_net_flow = torch.tensor(0.3)
        graphs.append(d)
    return Batch.from_data_list(graphs)


class TestOnChainGNN:
    def test_forward_single_graph(self, gnn, sample_data):
        """Single graph -> (gnn_dim,) embedding."""
        emb = gnn(sample_data)
        assert emb.shape == (256,), f"Expected (256,), got {emb.shape}"

    def test_forward_batched(self, gnn, batched_data):
        """Batched graphs -> (B, gnn_dim) embeddings."""
        emb = gnn(batched_data)
        assert emb.shape == (4, 256), f"Expected (4, 256), got {emb.shape}"

    def test_mxu_alignment(self, gnn):
        """gnn_dim must be a multiple of 128."""
        assert gnn.gnn_dim % 128 == 0

    def test_encode_returns_node_embeddings(self, gnn, sample_data):
        """encode() -> (N, gnn_dim) node-level embeddings."""
        node_emb = gnn.encode(sample_data)
        assert node_emb.shape == (100, 256), f"Expected (100, 256), got {node_emb.shape}"

    def test_link_prediction_loss(self, gnn, sample_data):
        """Link prediction loss is a scalar."""
        node_emb = gnn.encode(sample_data)
        pos_edges = sample_data.edge_index[:, :50].T  # (50, 2)
        neg_edges = torch.randint(0, 100, (50, 2))
        loss = gnn.link_prediction_loss(node_emb, pos_edges, neg_edges)
        assert loss.dim() == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_contrastive_loss(self, gnn):
        """InfoNCE contrastive loss is a scalar."""
        emb_t = torch.randn(8, 256)
        emb_tp1 = torch.randn(8, 256)
        loss = gnn.contrastive_loss(emb_t, emb_tp1)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_flow_prediction_loss(self, gnn):
        """Flow prediction loss is a scalar."""
        emb = torch.randn(4, 256)
        target = torch.randn(4)
        loss = gnn.flow_prediction_loss(emb, target)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_gradient_flow(self, gnn, sample_data):
        """Gradients flow through the entire model."""
        emb = gnn(sample_data)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        loss = emb.sum()
        loss.backward()
        # Check at least one param has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gnn.parameters())
        assert has_grad, "No gradients found in model parameters"

    def test_empty_graph(self, gnn):
        """Model handles a graph with 1 node and 0 edges."""
        data = Data(
            x=torch.randn(1, 8),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 2),
        )
        emb = gnn(data)
        assert emb.shape == (256,), f"Expected (256,), got {emb.shape}"

    def test_param_count(self, gnn):
        """Model should be around ~450K params (ballpark check)."""
        n_params = sum(p.numel() for p in gnn.parameters())
        assert 100_000 < n_params < 2_000_000, f"Unexpected param count: {n_params:,}"

    def test_mxu_alignment_assertion(self):
        """gnn_dim not a multiple of 128 should raise AssertionError."""
        from src.strate_v.gnn_model import OnChainGNN
        with pytest.raises(AssertionError):
            OnChainGNN(gnn_dim=200)
