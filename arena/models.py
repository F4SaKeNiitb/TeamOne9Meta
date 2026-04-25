"""PROTOCOL-ARENA — Typed action, observation, state schemas.

Shapes deliberately narrow so GRPO does not waste probability mass on malformed
protocol frames. Every observation carries its reward decomposition for
transparent training diagnostics.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


ActionKind = Literal["mcp", "a2a", "plan", "memory", "submit", "rewind"]


FailureClass = Literal[
    "protocol_malformed", "tool_not_found", "schema_drift_rename",
    "schema_drift_additive", "schema_drift_tightening", "rate_limited",
    "auth_required", "peer_offline", "policy_violation",
    "budget_exhausted", "injection_attempt", "honeypot_triggered",
    "unknown",
]


class FailureDiagnostic(BaseModel):
    """Structured failure classification — §18.12 of the plan.

    Emitted on every non-ok ProtocolResult so the env can report a
    top-N failure-mode table without free-text string mining.
    """
    failure_class: FailureClass = "unknown"
    drift_event: Optional[str] = None
    recovery_attempted: bool = False
    reason: str = ""


class MCPCall(BaseModel):
    server_id: str = Field(..., description="ID of the discovered MCP server")
    tool: str = Field(..., description="Tool name on that server")
    args: Dict[str, Any] = Field(default_factory=dict)


class A2ACall(BaseModel):
    agent_card_id: str = Field(..., description="AgentCard ID from discover_agents")
    task_spec: str = Field(..., description="Free-form task spec for the peer")
    stream: bool = Field(default=False, description="Use SSE streaming")


class DAGDelta(BaseModel):
    add_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    remove_nodes: List[str] = Field(default_factory=list)
    add_edges: List[List[str]] = Field(default_factory=list)
    remove_edges: List[List[str]] = Field(default_factory=list)


class KGOp(BaseModel):
    op: Literal["query", "write"] = "query"
    pattern: Optional[str] = None
    fact: Optional[Dict[str, Any]] = None
    top_k: int = 5


class OrchestratorAction(Action):
    kind: ActionKind
    rationale: str = Field(..., min_length=20)
    mcp_call: Optional[MCPCall] = None
    a2a_call: Optional[A2ACall] = None
    dag_delta: Optional[DAGDelta] = None
    kg_op: Optional[KGOp] = None
    final: Optional[str] = None
    rewind_n: Optional[int] = Field(default=None, ge=1, le=5)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0,
                              description="Agent's self-reported P(success). "
                                          "Used only for Brier calibration — "
                                          "not in the reward.")


class ProtocolResult(BaseModel):
    ok: bool
    status_code: int = 200
    body: Any = None
    error: Optional[str] = None
    drift_hint: Optional[str] = None
    retry_after: Optional[float] = None
    schema_diff: Optional[Dict[str, Any]] = None
    diagnostic: Optional[FailureDiagnostic] = None


class BudgetBreakdown(BaseModel):
    tokens_remaining: int = 0
    calls_remaining: int = 0
    latency_remaining_ms: int = 0
    initial_tokens: int = 0
    initial_calls: int = 0
    initial_latency_ms: int = 0


class ToolSpec(BaseModel):
    server_id: str
    name: str
    description: str
    json_schema: Dict[str, Any] = Field(default_factory=dict, alias="schema")
    version: str = "1"

    model_config = {"populate_by_name": True}


class AgentCard(BaseModel):
    id: str
    capabilities: List[str]
    persona: Literal["cooperative", "cranky", "stale"] = "cooperative"
    online: bool = True


class DiscoverySnapshot(BaseModel):
    tools: List[ToolSpec] = Field(default_factory=list)
    peers: List[AgentCard] = Field(default_factory=list)


class DAGNode(BaseModel):
    id: str
    op: str
    status: Literal["pending", "running", "ok", "error"] = "pending"
    result_preview: Optional[str] = None


class DAGSnapshot(BaseModel):
    nodes: List[DAGNode] = Field(default_factory=list)
    edges: List[List[str]] = Field(default_factory=list)
    checkpoints: List[str] = Field(default_factory=list)


class KGFact(BaseModel):
    subject: str
    predicate: str
    obj: str
    confidence: float = 1.0
    provenance: Optional[str] = None


class OrchestratorObservation(Observation):
    task_spec: str = ""
    task_id: str = ""
    turn: int = 0
    max_turns: int = 12
    budget: BudgetBreakdown = Field(default_factory=BudgetBreakdown)
    last_result: Optional[ProtocolResult] = None
    discovered: DiscoverySnapshot = Field(default_factory=DiscoverySnapshot)
    dag_state: DAGSnapshot = Field(default_factory=DAGSnapshot)
    memory_context: List[KGFact] = Field(default_factory=list)
    otel_trace_id: str = ""
    feedback: str = ""
    reward_signals: Dict[str, float] = Field(default_factory=dict)


class OrchestratorState(State):
    task_id: str = ""
    max_turns: int = 12
    current_score: float = 0.0
    drift_armed: bool = False
    drift_fired: bool = False
    honeypot_triggered: bool = False
    injection_followed: bool = False
    running_hit_score: float = 0.0
    reported_confidence: float = 0.5
    failure_diagnostics: List[Dict[str, Any]] = Field(default_factory=list)
