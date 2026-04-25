"""In-process A2A peer registry.

Each peer is an AgentCard plus a task-handler function. Peers come in three
personas:
  - cooperative: answers reliably, respects deadlines
  - cranky:      intermittent timeouts and partial results
  - stale:       AgentCard advertises capabilities it no longer supports

The peers simulate A2A's JSON-RPC 2.0 send_task surface. SSE streaming is
modelled as a sequence of partial results returned synchronously.
"""

import random
from typing import Dict, List, Callable, Any, Optional

from ..models import AgentCard, ProtocolResult


PeerFn = Callable[[str], Any]


class A2APeer:
    def __init__(self, card: AgentCard, handler: PeerFn,
                 supported_caps: Optional[List[str]] = None,
                 failure_rate: float = 0.0, seed: int = 0):
        self.card = card
        self.handler = handler
        self.supported_caps = set(supported_caps or card.capabilities)
        self.failure_rate = failure_rate
        self.rng = random.Random(seed)

    def handle(self, task_spec: str) -> ProtocolResult:
        if not self.card.online:
            return ProtocolResult(ok=False, status_code=503,
                                  error="peer_offline",
                                  drift_hint="AgentCard churn")

        if self.card.persona == "stale":
            needed_cap = _infer_cap_from_spec(task_spec, self.supported_caps)
            advertised = set(self.card.capabilities)
            if needed_cap not in self.supported_caps and needed_cap in advertised:
                return ProtocolResult(
                    ok=False, status_code=501,
                    error=f"stale_capability: '{needed_cap}' advertised but not implemented",
                    drift_hint="Stale AgentCard: re-discover peers",
                )

        if self.card.persona == "cranky" and self.rng.random() < self.failure_rate:
            return ProtocolResult(
                ok=False, status_code=504,
                error="peer_timeout",
                drift_hint="Cranky peer: retry or delegate to another",
                retry_after=1.0,
            )

        try:
            body = self.handler(task_spec)
            if self.card.persona == "cranky" and self.rng.random() < self.failure_rate:
                if isinstance(body, dict):
                    body = {**body, "partial": True}
                return ProtocolResult(ok=True, status_code=206, body=body)
            return ProtocolResult(ok=True, status_code=200, body=body)
        except Exception as e:
            return ProtocolResult(ok=False, status_code=500, error=f"peer_error: {e}")


def _infer_cap_from_spec(spec: str, known_caps: set) -> str:
    s = spec.lower()
    if "cite" in s or "reference" in s or "source" in s:
        return "citation"
    if "translate" in s or "translation" in s:
        return "translation"
    if "math" in s or "calculate" in s or "compute" in s:
        return "math"
    if "code" in s or "program" in s:
        return "code"
    return "general"


class A2ARegistry:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.peers: Dict[str, A2APeer] = {}

    def add(self, peer: A2APeer):
        self.peers[peer.card.id] = peer

    def discover(self) -> List[AgentCard]:
        return [p.card for p in self.peers.values() if p.card.online]

    def send_task(self, agent_id: str, task_spec: str) -> ProtocolResult:
        peer = self.peers.get(agent_id)
        if peer is None:
            return ProtocolResult(
                ok=False, status_code=404,
                error=f"unknown_agent: {agent_id}",
                drift_hint="AgentCard churn: peer no longer present",
            )
        return peer.handle(task_spec)


def build_default_peers(seed: int = 0) -> A2ARegistry:
    reg = A2ARegistry(seed)

    reg.add(A2APeer(
        AgentCard(id="citer", capabilities=["citation", "search"], persona="cooperative"),
        handler=_cite_handler, seed=seed + 1,
    ))
    reg.add(A2APeer(
        AgentCard(id="translator", capabilities=["translation"], persona="cooperative"),
        handler=_translate_handler, seed=seed + 2,
    ))
    reg.add(A2APeer(
        AgentCard(id="mathy", capabilities=["math", "code"], persona="cranky"),
        handler=_math_handler, failure_rate=0.35, seed=seed + 3,
    ))
    reg.add(A2APeer(
        AgentCard(
            id="legacy_summarizer",
            capabilities=["summarization", "translation"],
            persona="stale",
        ),
        handler=_stale_summ_handler,
        supported_caps=["summarization"],
        seed=seed + 4,
    ))
    return reg


def _cite_handler(spec: str) -> Dict[str, Any]:
    s = spec.lower()
    if "photosynthesis" in s:
        return {"citation": "Calvin 1950, Raven & Johnson 2011"}
    if "transformer" in s or "attention" in s:
        return {"citation": "Vaswani et al. 2017"}
    if "bell" in s or "entanglement" in s:
        return {"citation": "Bell 1964; Aspect 1982"}
    return {"citation": "unknown source"}


def _translate_handler(spec: str) -> Dict[str, Any]:
    s = spec.lower()
    if "to french" in s:
        return {"translation": "[fr] " + spec.split(":", 1)[-1].strip()}
    if "to hindi" in s:
        return {"translation": "[hi] " + spec.split(":", 1)[-1].strip()}
    return {"translation": spec}


def _math_handler(spec: str) -> Dict[str, Any]:
    import re
    nums = [float(x) for x in re.findall(r"-?\d+\.?\d*", spec)]
    if not nums:
        return {"result": None}
    if "sum" in spec.lower() or "add" in spec.lower():
        return {"result": sum(nums)}
    if "product" in spec.lower() or "multiply" in spec.lower():
        p = 1.0
        for n in nums:
            p *= n
        return {"result": p}
    return {"result": sum(nums)}


def _stale_summ_handler(spec: str) -> Dict[str, Any]:
    return {"summary": " ".join(spec.split()[:40])}
