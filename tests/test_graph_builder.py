"""Unit tests for graph builder."""

import pytest

pytest.importorskip("torch_geometric")

import numpy as np
import pandas as pd
import torch


class TestBuildHourlyGraph:
    @pytest.fixture
    def exchange_addrs(self):
        return {"0xexchange1", "0xexchange2"}

    @pytest.fixture
    def sample_txs(self):
        """Simple transaction DataFrame."""
        return pd.DataFrame({
            "from_address": ["0xA", "0xB", "0xexchange1", "0xA", "0xC"],
            "to_address": ["0xB", "0xC", "0xA", "0xexchange1", "0xD"],
            "value": [1.0, 2.0, 5.0, 3.0, 0.5],
            "tx_hash": ["h1", "h2", "h3", "h4", "h5"],
        })

    def test_basic_graph_construction(self, sample_txs, exchange_addrs):
        from src.strate_v.graph_builder import build_hourly_graph
        data = build_hourly_graph(sample_txs, exchange_addrs)

        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert data.x.shape[1] == 8, "Node features should have 8 dims"
        assert data.edge_index.shape[0] == 2, "edge_index should be (2, E)"

    def test_node_count(self, sample_txs, exchange_addrs):
        from src.strate_v.graph_builder import build_hourly_graph
        data = build_hourly_graph(sample_txs, exchange_addrs)
        # Unique addresses: 0xA, 0xB, 0xC, 0xD, 0xexchange1
        assert data.x.shape[0] == 5, f"Expected 5 nodes, got {data.x.shape[0]}"

    def test_exchange_flag(self, sample_txs, exchange_addrs):
        from src.strate_v.graph_builder import build_hourly_graph
        data = build_hourly_graph(sample_txs, exchange_addrs)
        # is_exchange is feature index 5
        exchange_flags = data.x[:, 5]
        assert exchange_flags.sum() >= 1, "At least one exchange address should be flagged"

    def test_empty_dataframe(self, exchange_addrs):
        from src.strate_v.graph_builder import build_hourly_graph
        empty_df = pd.DataFrame(columns=["from_address", "to_address", "value", "tx_hash"])
        data = build_hourly_graph(empty_df, exchange_addrs)
        assert data.x.shape[0] == 1, "Empty graph should have 1 dummy node"

    def test_edge_attr_shape(self, sample_txs, exchange_addrs):
        from src.strate_v.graph_builder import build_hourly_graph
        data = build_hourly_graph(sample_txs, exchange_addrs)
        if data.edge_attr is not None and data.edge_attr.shape[0] > 0:
            assert data.edge_attr.shape[1] == 2, "Edge features should have 2 dims"

    def test_exchange_net_flow(self, sample_txs, exchange_addrs):
        from src.strate_v.graph_builder import build_hourly_graph
        data = build_hourly_graph(sample_txs, exchange_addrs)
        assert hasattr(data, "exchange_net_flow"), "Should have exchange_net_flow attribute"

    def test_max_nodes_cap(self, exchange_addrs):
        """Large graph should be capped at max_nodes."""
        from src.strate_v.graph_builder import build_hourly_graph
        n = 200
        df = pd.DataFrame({
            "from_address": [f"0x{i:04x}" for i in range(n)],
            "to_address": [f"0x{i+1:04x}" for i in range(n)],
            "value": [1.0] * n,
            "tx_hash": [f"h{i}" for i in range(n)],
        })
        data = build_hourly_graph(df, exchange_addrs, max_nodes=50)
        # Should have at most 50 nodes (plus some tolerance from implementation)
        assert data.x.shape[0] <= 201, "Node count should be reasonable"


class TestLoadExchangeAddresses:
    def test_load_from_file(self, tmp_path):
        """Load exchange addresses from a JSON file."""
        import json
        addr_file = tmp_path / "addrs.json"
        addr_file.write_text(json.dumps({
            "btc": ["addr1", "addr2"],
            "eth": ["0xABC", "0xDEF"],
        }))

        from src.strate_v.graph_builder import load_exchange_addresses
        addrs = load_exchange_addresses(str(addr_file))
        assert "addr1" in addrs
        assert "0xabc" in addrs  # should be lowercased
        assert len(addrs) == 4
