"""Strate V — Execution Layer (nervous system for order routing).

Components:
  - SynapticVector: signal from brain (Strate IV) to execution
  - CortexNode:     decision publisher (ZMQ PUB, dry-run if pyzmq absent)
  - CerebellumNode: smooth execution (TWAP) with task cancellation
  - ReflexArc:      flash-crash circuit breaker (rolling price, 5% threshold)
"""

from .synapse import SynapticVector
from .cortex_node import CortexNode
from .cerebellum_node import CerebellumNode, TWAPState
from .reflex_arc import ReflexArc
