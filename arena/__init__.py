"""PROTOCOL-ARENA — An OpenEnv RL gym where agents learn to speak MCP and A2A
protocol frames under schema drift.

Trained policy deploys directly into any MCP/A2A-compliant client.
"""

from .models import (
    OrchestratorAction,
    OrchestratorObservation,
    OrchestratorState,
    MCPCall,
    A2ACall,
    DAGDelta,
    KGOp,
    ProtocolResult,
    BudgetBreakdown,
    DiscoverySnapshot,
    DAGSnapshot,
    KGFact,
)
from .client import ProtocolArenaEnv

__all__ = [
    "OrchestratorAction",
    "OrchestratorObservation",
    "OrchestratorState",
    "MCPCall",
    "A2ACall",
    "DAGDelta",
    "KGOp",
    "ProtocolResult",
    "BudgetBreakdown",
    "DiscoverySnapshot",
    "DAGSnapshot",
    "KGFact",
    "ProtocolArenaEnv",
]
