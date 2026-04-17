"""Graph builder: construct PyG Data objects from on-chain transaction data.

Builds hourly graph snapshots from parquet transaction data.
Nodes = wallet addresses, edges = transactions.

Node features (8 dims):
  in_value_log, out_value_log, in_degree, out_degree,
  net_flow, is_exchange, tx_count, unique_counterparties

Edge features (2 dims):
  value_log, tx_count
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def load_exchange_addresses(path: str = "data/onchain/exchange_addresses.json") -> set[str]:
    """Load known exchange addresses from JSON file.

    Args:
        path: Path to exchange addresses JSON.

    Returns:
        Set of lowercase hex addresses.
    """
    with open(path) as f:
        data = json.load(f)
    addresses = set()
    for chain_addrs in data.values():
        for addr in chain_addrs:
            addresses.add(addr.lower())
    return addresses


def build_hourly_graph(
    tx_df: pd.DataFrame,
    exchange_addresses: set[str],
    max_nodes: int = 50_000,
    k_hop: int = 2,
) -> Data:
    """Build a PyG graph from hourly transaction data.

    Args:
        tx_df: DataFrame with columns: from_address, to_address, value, tx_hash.
        exchange_addresses: Set of known exchange addresses (lowercase).
        max_nodes: Maximum nodes per snapshot (subgraph cap).
        k_hop: Subgraph sampling depth around exchange addresses.

    Returns:
        PyG Data object with node features, edge index, edge attributes.
    """
    if tx_df.empty:
        return Data(
            x=torch.zeros(1, 8, dtype=torch.float32),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            edge_attr=torch.zeros(0, 2, dtype=torch.float32),
        )

    # Normalize addresses
    tx_df = tx_df.copy()
    tx_df["from_address"] = tx_df["from_address"].str.lower()
    tx_df["to_address"] = tx_df["to_address"].str.lower()

    # Collect all unique addresses
    all_addresses = set(tx_df["from_address"].unique()) | set(tx_df["to_address"].unique())

    # K-hop subgraph around exchange addresses
    exchange_in_graph = all_addresses & exchange_addresses
    if exchange_in_graph and len(all_addresses) > max_nodes:
        # BFS from exchange addresses
        selected = set(exchange_in_graph)
        frontier = set(exchange_in_graph)
        for _ in range(k_hop):
            new_frontier = set()
            for addr in frontier:
                neighbors_out = set(tx_df[tx_df["from_address"] == addr]["to_address"])
                neighbors_in = set(tx_df[tx_df["to_address"] == addr]["from_address"])
                new_frontier |= neighbors_out | neighbors_in
            frontier = new_frontier - selected
            selected |= frontier
            if len(selected) >= max_nodes:
                break
        # Cap
        all_addresses = set(list(selected)[:max_nodes])
        # Filter transactions
        tx_df = tx_df[
            tx_df["from_address"].isin(all_addresses) & tx_df["to_address"].isin(all_addresses)
        ]

    # Build address -> index mapping
    addr_list = sorted(all_addresses)
    addr_to_idx = {addr: i for i, addr in enumerate(addr_list)}
    N = len(addr_list)

    # Aggregate edge statistics
    edge_stats = (
        tx_df.groupby(["from_address", "to_address"])
        .agg(total_value=("value", "sum"), tx_count=("tx_hash", "count"))
        .reset_index()
    )

    # Build edge_index and edge_attr
    src_indices = []
    dst_indices = []
    edge_attrs = []
    for _, row in edge_stats.iterrows():
        src = addr_to_idx.get(row["from_address"])
        dst = addr_to_idx.get(row["to_address"])
        if src is not None and dst is not None:
            src_indices.append(src)
            dst_indices.append(dst)
            val_log = math.log1p(float(row["total_value"]))
            edge_attrs.append([val_log, float(row["tx_count"])])

    if not src_indices:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 2, dtype=torch.float32)
    else:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    # Build node features (8 dims)
    node_features = np.zeros((N, 8), dtype=np.float32)

    # Per-node aggregation
    for _, row in tx_df.iterrows():
        src = addr_to_idx.get(row["from_address"])
        dst = addr_to_idx.get(row["to_address"])
        val = float(row.get("value", 0))
        if src is not None:
            node_features[src, 1] += val      # out_value
            node_features[src, 3] += 1        # out_degree
            node_features[src, 6] += 1        # tx_count
        if dst is not None:
            node_features[dst, 0] += val      # in_value
            node_features[dst, 2] += 1        # in_degree
            node_features[dst, 6] += 1        # tx_count

    # Unique counterparties
    for i, addr in enumerate(addr_list):
        counterparties = set()
        mask_from = tx_df["from_address"] == addr
        mask_to = tx_df["to_address"] == addr
        counterparties |= set(tx_df.loc[mask_from, "to_address"])
        counterparties |= set(tx_df.loc[mask_to, "from_address"])
        node_features[i, 7] = len(counterparties)

    # Log-transform values
    node_features[:, 0] = np.log1p(node_features[:, 0])  # in_value_log
    node_features[:, 1] = np.log1p(node_features[:, 1])  # out_value_log

    # Net flow
    node_features[:, 4] = node_features[:, 0] - node_features[:, 1]  # net_flow (log scale)

    # Is exchange
    for i, addr in enumerate(addr_list):
        if addr in exchange_addresses:
            node_features[i, 5] = 1.0

    x = torch.from_numpy(node_features)

    # Store exchange net flow as graph attribute (for flow prediction loss)
    exchange_mask = x[:, 5] > 0.5
    if exchange_mask.any():
        exchange_net_flow = x[exchange_mask, 4].mean().item()
    else:
        exchange_net_flow = 0.0

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.exchange_net_flow = torch.tensor(exchange_net_flow, dtype=torch.float32)
    data.num_nodes_actual = N

    return data
