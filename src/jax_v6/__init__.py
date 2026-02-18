"""JAX/TPU v6 implementation of Fin-JEPA (Strate II).

Chunked SSD (State Space Duality) replaces associative_scan for MXU utilization.
GSPMD (NamedSharding + Mesh) replaces jax.pmap for multi-host TPU v4-32.
"""
