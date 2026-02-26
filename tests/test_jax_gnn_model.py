"""Tests for JAX GNN model (Strate V port)."""
import pytest
import numpy as np

jax = pytest.importorskip("jax")
jraph = pytest.importorskip("jraph")
flax = pytest.importorskip("flax")

import jax.numpy as jnp
from jax import random


def make_graph(n_nodes=100, n_edges=500, node_feat=8, edge_feat=2, rng_seed=0):
    """Create a single jraph.GraphsTuple for testing."""
    rng = np.random.RandomState(rng_seed)
    nodes = rng.randn(n_nodes, node_feat).astype(np.float32)
    edges = rng.randn(n_edges, edge_feat).astype(np.float32)
    senders = rng.randint(0, n_nodes, (n_edges,)).astype(np.int32)
    receivers = rng.randint(0, n_nodes, (n_edges,)).astype(np.int32)
    return jraph.GraphsTuple(
        nodes=jnp.array(nodes),
        edges=jnp.array(edges),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([n_edges]),
        globals=jnp.array([[0.5]]),
    )


def make_batched_graphs(n_graphs=4, rng_seed=0):
    """Create a batched jraph.GraphsTuple (multiple graphs)."""
    graphs = []
    for i in range(n_graphs):
        n = 50 + i * 10
        e = 200 + i * 50
        graphs.append(make_graph(n_nodes=n, n_edges=e, rng_seed=rng_seed + i))
    return jraph.batch(graphs)


@pytest.fixture
def gnn():
    from src.jax_v6.strate_v.gnn_model import OnChainGNN
    return OnChainGNN(
        node_features=8, edge_features=2, hidden_dim=128,
        gnn_dim=256, n_layers=3, gat_heads=4, dropout=0.0,
    )


@pytest.fixture
def gnn_params(gnn):
    graph = make_graph()
    key = random.PRNGKey(42)
    params = gnn.init(key, graph, deterministic=True)
    return params


class TestOnChainGNNJAX:
    def test_init_and_forward(self, gnn, gnn_params):
        """Forward pass returns (gnn_dim,) for single graph."""
        graph = make_graph()
        emb = gnn.apply(gnn_params, graph, deterministic=True)
        assert emb.shape == (256,), f"Expected (256,), got {emb.shape}"

    def test_batched_forward(self, gnn, gnn_params):
        """Batched forward returns (B, gnn_dim)."""
        graphs = make_batched_graphs(n_graphs=4)
        emb = gnn.apply(gnn_params, graphs, deterministic=True)
        assert emb.shape == (4, 256), f"Expected (4, 256), got {emb.shape}"

    def test_encode_returns_node_embeddings(self, gnn, gnn_params):
        """encode() returns (N, gnn_dim) node embeddings."""
        graph = make_graph(n_nodes=100)
        node_emb = gnn.apply(gnn_params, graph, deterministic=True, method=gnn.encode)
        assert node_emb.shape == (100, 256)

    def test_mxu_alignment(self, gnn):
        assert gnn.gnn_dim % 128 == 0

    def test_param_count(self, gnn, gnn_params):
        """~450K params (same ballpark as PyTorch version)."""
        n_params = sum(x.size for x in jax.tree.leaves(gnn_params))
        assert 100_000 < n_params < 2_000_000, f"Unexpected: {n_params:,}"

    def test_gradient_flow(self, gnn, gnn_params):
        """Gradients flow through the model."""
        graph = make_graph()
        def loss_fn(params):
            emb = gnn.apply(params, graph, deterministic=True)
            return emb.sum()
        grads = jax.grad(loss_fn)(gnn_params)
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(jnp.abs(g).sum() > 0 for g in flat_grads)
        assert has_nonzero, "No nonzero gradients"

    def test_empty_graph(self, gnn, gnn_params):
        """Handle graph with 1 node, 0 edges."""
        graph = jraph.GraphsTuple(
            nodes=jnp.zeros((1, 8)),
            edges=jnp.zeros((0, 2)),
            senders=jnp.array([], dtype=jnp.int32),
            receivers=jnp.array([], dtype=jnp.int32),
            n_node=jnp.array([1]),
            n_edge=jnp.array([0]),
            globals=jnp.array([[0.0]]),
        )
        graph = jraph.pad_with_graphs(graph, n_node=2, n_edge=1, n_graph=2)
        emb = gnn.apply(gnn_params, graph, deterministic=True)
        assert emb.shape[-1] == 256

    def test_link_prediction_loss(self, gnn, gnn_params):
        """Link prediction loss is a scalar, non-NaN."""
        graph = make_graph()
        node_emb = gnn.apply(gnn_params, graph, deterministic=True, method=gnn.encode)
        pos_edges = jnp.stack([graph.senders[:50], graph.receivers[:50]], axis=1)
        neg_src = jax.random.randint(random.PRNGKey(0), (50,), 0, 100)
        neg_dst = jax.random.randint(random.PRNGKey(1), (50,), 0, 100)
        neg_edges = jnp.stack([neg_src, neg_dst], axis=1)
        loss = gnn.apply(gnn_params, node_emb, pos_edges, neg_edges,
                         method=gnn.link_prediction_loss)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"
        assert jnp.isfinite(loss)

    def test_contrastive_loss(self, gnn, gnn_params):
        """InfoNCE contrastive loss is a scalar."""
        emb_t = jax.random.normal(random.PRNGKey(0), (8, 256))
        emb_tp1 = jax.random.normal(random.PRNGKey(1), (8, 256))
        loss = gnn.apply(gnn_params, emb_t, emb_tp1,
                         method=gnn.contrastive_loss)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_flow_prediction_loss(self, gnn, gnn_params):
        """Flow prediction loss is a scalar."""
        emb = jax.random.normal(random.PRNGKey(0), (4, 256))
        target = jax.random.normal(random.PRNGKey(1), (4,))
        loss = gnn.apply(gnn_params, emb, target,
                         method=gnn.flow_prediction_loss)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
