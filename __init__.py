"""PROTOCOL-ARENA root package.

Re-exports the primary types from arena.* so `from protocol_arena import ...`
and `from arena import ...` both work.
"""

from arena import (
    ProtocolArenaEnv,
    OrchestratorAction,
    OrchestratorObservation,
    OrchestratorState,
    MCPCall,
    A2ACall,
    DAGDelta,
    KGOp,
)

__all__ = [
    "ProtocolArenaEnv",
    "OrchestratorAction",
    "OrchestratorObservation",
    "OrchestratorState",
    "MCPCall",
    "A2ACall",
    "DAGDelta",
    "KGOp",
]
