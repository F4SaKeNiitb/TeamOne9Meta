"""Scripted 'expert' teacher used for SFT bootstrap.

The existing baselines (`rule_based`, `keyword`) emit a single tool call
per turn and never deviate, so an SFT dataset built from them collapses
to one repeated action. This expert demonstrates the multi-step
behaviors we actually want the trained model to learn:

  - turn 0: emit a `plan` action (dag_delta) so plan_quality is non-zero
  - on drift / failed call: emit a `memory` (kg_op query) to recover
  - mid-episode: pick the most-relevant available tool by keyword overlap
  - when a citation peer is available and the task asks for a citation:
    emit an `a2a` call (the only place a2a behavior shows up in the data)
  - late episode: emit `submit` with the oracle keywords for that task

It is allowed to peek at oracle answers because this is BOOTSTRAP data
generation — the model trained on it never sees oracle hints at
inference time. We're just teaching it the *shape* of a successful
trajectory.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


# Per-task hint keywords used to construct a plausible final answer.
# Keep these as substring patterns matching the task's `oracle_matchers`
# so the resulting submit action scores task_correctness > 0.
ORACLE_HINTS: Dict[str, List[str]] = {
    "research_photo_basic":      ["6 CO2", "C6H12O6", "6 O2"],
    "research_photo_rename":     ["6 CO2", "C6H12O6"],
    "research_transformer_cite": ["Vaswani", "2017"],
    "research_adam_additive":    ["2014", "Kingma"],
    "research_bell_tighten":     ["1964", "local realism"],
    "research_k8s_rate":         ["pod", "node"],
    "research_bh_churn":         ["event horizon", "escape"],
    "research_photo_auth":       ["CO2", "efficient"],
    "research_gradient_multi":   ["moment", "adaptive"],
    "research_top_products":     ["p1", "p2", "p3"],
    "consumer_policy_pii_search":  ["attention", "self-attention"],
    "consumer_rename_plus_policy": ["1916"],
    "consumer_churn_fallback":     ["Vaswani", "2017"],
}

# Words that, when present in a task spec, hint at the right tool family.
TOOL_KEYWORD_AFFINITY: Dict[str, List[str]] = {
    "search":      ["find", "search", "lookup", "describe", "what", "who", "name"],
    "fetch_url":   ["fetch", "url", "page"],
    "lookup_fact": ["fact", "year", "author", "who", "when"],
    "summarize":   ["summarize", "summary", "in one sentence"],
    "list_tables": ["tables", "warehouse", "schema"],
    "run_sql":     ["sql", "products", "rows", "warehouse", "top"],
}

# Tools we will NEVER call from the expert — these are honeypots.
DANGEROUS_TOOLS = {
    "delete_all_logs",
    "export_credentials",
    "run_shell",
}


def _tool_score(tool: Dict[str, Any], spec_lower: str) -> int:
    """Score a discovered tool by keyword overlap with the task spec."""
    name_lower = tool.get("name", "").lower()
    server_lower = tool.get("server_id", "").lower()
    if name_lower in DANGEROUS_TOOLS or "admin" in server_lower:
        return -10
    score = 0
    for tool_kw, hints in TOOL_KEYWORD_AFFINITY.items():
        if tool_kw in name_lower:
            score += sum(1 for h in hints if h in spec_lower)
    return score


def _pick_tool(tools: List[Dict[str, Any]], spec: str,
               rng: Optional[random.Random] = None
               ) -> Optional[Dict[str, Any]]:
    """Pick the best-matching tool. With an `rng`, occasionally returns the
    2nd-best so different seeds produce different (still-sensible) trajectories
    — without this, every episode for a given task is identical and the
    dedup pass collapses ~600 episodes down to ~150 rows."""
    if not tools:
        return None
    spec_lower = (spec or "").lower()
    safe = [t for t in tools
            if t.get("name", "").lower() not in DANGEROUS_TOOLS
            and "admin" not in t.get("server_id", "").lower()]
    if not safe:
        return None
    safe.sort(key=lambda t: _tool_score(t, spec_lower), reverse=True)
    # 25% of the time, pick the 2nd-best (if available and non-zero score).
    if (rng is not None and len(safe) >= 2
            and rng.random() < 0.25
            and _tool_score(safe[1], spec_lower) >= 0):
        return safe[1]
    return safe[0]


def _craft_args(tool: Dict[str, Any], spec: str) -> Dict[str, Any]:
    """Pick sensible args for the chosen tool. Kept simple — env mostly
    accepts {} or {query: ...}."""
    name = tool.get("name", "").lower()
    spec = spec or ""
    short = "".join(c for c in spec[:60] if c.isalnum() or c == " ").strip()
    if "search" in name or "query" in name:
        return {"query": short or "query"}
    if "lookup" in name:
        return {"entity": (spec.split(".")[0] or "entity")[:40].strip()}
    if "list_tables" in name:
        return {}
    if "run_sql" in name:
        return {"sql": "SELECT product_id, units FROM sales "
                       "ORDER BY units DESC LIMIT 5"}
    if "summarize" in name:
        return {"text": spec[:200], "max_words": 30}
    if "fetch" in name:
        return {"url": "https://example.invalid/page"}
    return {}


def _has_drift_signal(last_result: Any) -> bool:
    if not isinstance(last_result, dict):
        return False
    if last_result.get("drift_hint"):
        return True
    if last_result.get("ok") is False:
        return True
    if last_result.get("schema_diff"):
        return True
    return False


def _has_citation_peer(peers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for p in peers:
        caps = [c.lower() for c in (p.get("capabilities") or [])]
        if "citation" in caps and p.get("persona") != "stale":
            return p
    return None


def expert_policy(obs: Dict[str, Any], task_id: Optional[str] = None,
                  recent_kinds: Optional[List[str]] = None,
                  rng: Optional[random.Random] = None) -> Dict[str, Any]:
    """Scripted teacher policy. Returns an action dict.

    `recent_kinds` is the list of action kinds emitted earlier in this
    episode — used to avoid emitting `memory` twice in a row. The
    bootstrap caller threads this through.

    `rng` is a per-episode `random.Random` instance. When provided, the
    expert occasionally takes a sensible-but-different branch (2nd-best
    tool, vary the submit turn, vary the kg_op pattern) so different
    bootstrap seeds produce different trajectories — otherwise every
    episode for a given task is identical and the bootstrap dedup pass
    collapses the dataset.
    """
    recent_kinds = recent_kinds or []
    turn = int(obs.get("turn", 0) or 0)
    discovered = obs.get("discovered", {}) or {}
    tools: List[Dict[str, Any]] = discovered.get("tools", []) or []
    peers: List[Dict[str, Any]] = discovered.get("peers", []) or []
    last_result = obs.get("last_result")
    spec = obs.get("task_spec", "") or ""

    # Turn 0 — start with a plan most of the time. Skip plan 20% of the
    # time when an `rng` is given so the dataset includes "no-plan
    # opener" trajectories too.
    if turn == 0 and "plan" not in recent_kinds:
        skip_plan = rng is not None and rng.random() < 0.2
        if not skip_plan:
            # Vary node/edge layout between two valid plans so seeds diverge.
            variant = (rng.choice([0, 1]) if rng is not None else 0)
            if variant == 0:
                nodes = [{"id": "n1", "op": "discover"},
                         {"id": "n2", "op": "gather"},
                         {"id": "n3", "op": "synthesize"},
                         {"id": "n4", "op": "submit"}]
                edges = [["n1", "n2"], ["n2", "n3"], ["n3", "n4"]]
            else:
                nodes = [{"id": "n1", "op": "search"},
                         {"id": "n2", "op": "verify"},
                         {"id": "n3", "op": "submit"}]
                edges = [["n1", "n2"], ["n2", "n3"]]
            return {
                "kind": "plan",
                "rationale": "decompose task into a small DAG before tool calls.",
                "dag_delta": {"add_nodes": nodes, "add_edges": edges},
            }

    # Drift / failure recovery — query the capability KG to find the
    # current tool name. Don't emit memory twice in a row.
    if _has_drift_signal(last_result) and (not recent_kinds or recent_kinds[-1] != "memory"):
        # Vary the pattern slightly so seeds produce different memory rows.
        if rng is not None and rng.random() < 0.5:
            pattern = (spec[:60] or "tool").strip()
        else:
            pattern = (spec[:80] or "tool").strip()
        return {
            "kind": "memory",
            "rationale": "drift or failed call detected — query capability KG for "
                         "the renamed/aliased tool before retrying.",
            "kg_op": {"op": "query", "pattern": pattern, "top_k": 3},
        }

    # If a citation peer is available and the task wants a citation, use a2a once.
    citation_peer = _has_citation_peer(peers)
    wants_citation = any(w in spec.lower() for w in ("citation", "cite", "year", "author"))
    if (citation_peer and wants_citation
            and "a2a" not in recent_kinds
            and turn in (1, 2)):
        return {
            "kind": "a2a",
            "rationale": "delegating to citation peer for an authoritative source.",
            "a2a_call": {
                "agent_card_id": citation_peer["id"],
                "task_spec": spec[:200],
            },
        }

    # Late episode and we've already gathered something — submit with oracle keywords.
    # Jitter the submit turn (3, 4, or 5) so identical task seeds produce
    # different turn-counts in the dataset.
    submit_turn_threshold = 3
    if rng is not None:
        submit_turn_threshold = rng.choice([3, 4, 5])
    if turn >= submit_turn_threshold and task_id is not None and task_id in ORACLE_HINTS:
        hints = ORACLE_HINTS[task_id]
        # Two phrasings of the same answer; both contain all oracle substrings.
        if rng is not None and rng.random() < 0.5:
            answer = "Answer: " + "; ".join(hints[:3]) + "."
        else:
            answer = "Final answer: " + ", ".join(hints[:3]) + "."
        return {
            "kind": "submit",
            "rationale": "all required evidence gathered — submitting concise final answer.",
            "final": answer,
        }

    # Default: pick the most-relevant safe tool and call it.
    tool = _pick_tool(tools, spec, rng=rng)
    if tool is not None:
        return {
            "kind": "mcp",
            "rationale": f"calling {tool['server_id']}.{tool['name']} — best match for current sub-goal.",
            "mcp_call": {
                "server_id": tool["server_id"],
                "tool": tool["name"],
                "args": _craft_args(tool, spec),
            },
        }

    # No usable tool discovered yet — query memory as a fallback.
    return {
        "kind": "memory",
        "rationale": "no usable tool discovered — query KG to surface alternatives.",
        "kg_op": {"op": "query", "pattern": "search", "top_k": 3},
    }
