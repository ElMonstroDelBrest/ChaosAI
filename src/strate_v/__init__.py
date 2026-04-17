"""Strate V — GNN On-Chain: Graph Neural Network for blockchain transaction analysis.

Pre-computes on-chain graph embeddings (256 dims) from Bitcoin + Ethereum
transaction graphs (BigQuery public datasets). Embeddings are stored in
ArrayRecords and consumed by the JEPA pipeline as additional features.

GNN architecture: GraphSAGE + GATv2 (3 message-passing layers, ~450K params).
Training: self-supervised (link prediction + temporal contrastive + exchange flow).
"""
