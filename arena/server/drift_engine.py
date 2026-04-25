"""Schema Drift Engine.

Applies one of seven drift classes to a live MCPRegistry / A2ARegistry at a
scheduled turn of an episode. Each drift class maps to a documented MCP/A2A
failure pattern so tasks can cite source issues.

Drift classes:
  - additive         : a new required field appears on an existing tool
  - renaming         : tool is renamed; old name returns 410 for N turns, then 404
  - tightening       : a field gains a regex / type constraint
  - rate_limit       : rpm is tightened (e.g., 60 -> 4)
  - agentcard_churn  : a peer disappears or goes offline
  - policy           : a field gains PII-redaction requirement
  - auth             : a tool gains a required scope

Drift is deterministic from the episode seed + schedule so results replay.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ..protocols.mcp_harness import MCPRegistry
from ..protocols.a2a_harness import A2ARegistry


DRIFT_CLASSES = (
    "additive",
    "renaming",
    "tightening",
    "rate_limit",
    "agentcard_churn",
    "policy",
    "auth",
)


@dataclass
class DriftEvent:
    turn: int
    klass: str
    target_server: Optional[str] = None
    target_tool: Optional[str] = None
    target_peer: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftSchedule:
    events: List[DriftEvent] = field(default_factory=list)
    # When non-None: reactively drift the most-used tool on turn >= reactive_after.
    reactive_after: Optional[int] = None


class DriftEngine:
    def __init__(self, schedule: DriftSchedule, seed: int = 0):
        self.schedule = schedule
        self.rng = random.Random(seed)
        self.fired: List[DriftEvent] = []
        self.use_histogram: Dict[str, int] = {}

    def note_use(self, server_id: str, tool: str):
        key = f"{server_id}.{tool}"
        self.use_histogram[key] = self.use_histogram.get(key, 0) + 1

    def maybe_fire(self, turn: int,
                   mcp: MCPRegistry, a2a: A2ARegistry) -> List[DriftEvent]:
        fired_now: List[DriftEvent] = []
        for ev in list(self.schedule.events):
            if ev.turn == turn:
                self._apply(ev, mcp, a2a)
                self.fired.append(ev)
                fired_now.append(ev)
                self.schedule.events.remove(ev)

        if self.schedule.reactive_after is not None and turn >= self.schedule.reactive_after:
            if self.use_histogram:
                hot = max(self.use_histogram.items(), key=lambda kv: kv[1])[0]
                server_id, tool = hot.split(".", 1)
                if server_id in mcp.servers and tool in mcp.servers[server_id].tools:
                    klass = self.rng.choice(["additive", "tightening", "rate_limit"])
                    ev = DriftEvent(turn=turn, klass=klass,
                                    target_server=server_id, target_tool=tool,
                                    detail={"reactive": True})
                    self._apply(ev, mcp, a2a)
                    self.fired.append(ev)
                    fired_now.append(ev)
                    self.use_histogram = {}

        return fired_now

    def _apply(self, ev: DriftEvent, mcp: MCPRegistry, a2a: A2ARegistry):
        k = ev.klass
        if k == "additive":
            srv = mcp.servers.get(ev.target_server)
            if srv and ev.target_tool in srv.tools:
                field = ev.detail.get("field", "schema_version")
                srv.required_fields.setdefault(ev.target_tool, []).append(field)
                srv.tools[ev.target_tool].version = _bump(srv.tools[ev.target_tool].version)

        elif k == "renaming":
            srv = mcp.servers.get(ev.target_server)
            if srv and ev.target_tool in srv.tools:
                new_name = ev.detail.get("new_name", f"{ev.target_tool}_v2")
                old_spec = srv.tools[ev.target_tool]
                srv.tools[new_name] = old_spec.model_copy(
                    update={"name": new_name, "version": _bump(old_spec.version)}
                )
                srv.impls[new_name] = srv.impls[ev.target_tool]
                srv.required_fields[new_name] = list(srv.required_fields.get(ev.target_tool, []))
                srv.deprecated[ev.target_tool] = new_name

        elif k == "tightening":
            srv = mcp.servers.get(ev.target_server)
            if srv and ev.target_tool in srv.tools:
                field = ev.detail.get("field", "query")
                pattern = ev.detail.get("pattern", r"[A-Za-z0-9_ ]{1,200}")
                srv.tightened_fields.setdefault(ev.target_tool, {})[field] = pattern

        elif k == "rate_limit":
            srv = mcp.servers.get(ev.target_server)
            if srv:
                srv.rate_limit_rpm = ev.detail.get("rpm", 4)

        elif k == "agentcard_churn":
            peer = a2a.peers.get(ev.target_peer)
            if peer:
                peer.card.online = False

        elif k == "policy":
            srv = mcp.servers.get(ev.target_server)
            if srv and ev.target_tool in srv.tools:
                field = ev.detail.get("field", "query")
                srv.policy_redacted_fields.setdefault(ev.target_tool, []).append(field)

        elif k == "auth":
            srv = mcp.servers.get(ev.target_server)
            if srv and ev.target_tool in srv.tools:
                scope = ev.detail.get("scope", "elevated")
                srv.auth_scopes.setdefault(ev.target_tool, []).append(scope)


def _bump(version: str) -> str:
    try:
        return str(int(version) + 1)
    except Exception:
        return version + "'"
