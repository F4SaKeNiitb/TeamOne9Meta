"""Deterministic baseline policies for ablation + comparison.

  - random_policy            : uniformly picks an action kind
  - keyword_policy           : always searches/fetches based on task_spec tokens
  - rule_based_policy        : simple heuristic that tries MCP first, KG query
                                on drift hints, falls back to peers.

Use these to generate the "before training" baseline row on the results slide.
Frontier API baselines (GPT-4o, Claude, Gemini) are driven via inference.py
with MODEL_NAME set appropriately.
"""

import random
from typing import Any, Dict


def random_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    kind = random.choice(["mcp", "a2a", "memory", "plan", "submit"])
    d = {"kind": kind, "rationale": "random baseline — uniformly sampled action."}
    if kind == "mcp":
        tools = (obs.get("discovered") or {}).get("tools") or []
        if tools:
            t = random.choice(tools)
            d["mcp_call"] = {"server_id": t["server_id"], "tool": t["name"],
                              "args": {"query": "anything"}}
        else:
            d["kind"] = "memory"
            d["kg_op"] = {"op": "query", "pattern": "any", "top_k": 3}
    elif kind == "a2a":
        peers = (obs.get("discovered") or {}).get("peers") or []
        if peers:
            p = random.choice(peers)
            d["a2a_call"] = {"agent_card_id": p["id"], "task_spec": "help please"}
        else:
            d["kind"] = "memory"
            d["kg_op"] = {"op": "query", "pattern": "any", "top_k": 3}
    elif kind == "memory":
        d["kg_op"] = {"op": "query", "pattern": obs.get("task_spec", "")[:30], "top_k": 3}
    elif kind == "plan":
        d["dag_delta"] = {"add_nodes": [{"id": f"r{random.randint(0,99)}", "op": "noop"}]}
    elif kind == "submit":
        d["final"] = "I don't know yet."
    return d


def keyword_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Always uses web.search with the first few tokens of the spec."""
    spec = obs.get("task_spec", "")
    if obs.get("turn", 0) >= 3:
        return {"kind": "submit",
                "rationale": "keyword baseline: time to submit what I have",
                "final": spec}
    return {"kind": "mcp",
            "rationale": "keyword baseline: direct search on task tokens",
            "mcp_call": {"server_id": "web", "tool": "search",
                          "args": {"query": " ".join(spec.split()[:6])}}}


def rule_based_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    turn = obs.get("turn", 0)
    last = obs.get("last_result") or {}
    hint = (last.get("drift_hint") if last else "") or ""

    if hint and "rename" in hint.lower():
        return {"kind": "memory",
                "rationale": "rule-based: rename drift hint -> check KG for new name",
                "kg_op": {"op": "query", "pattern": "renamed_to", "top_k": 3}}

    if turn == 0:
        return {"kind": "mcp",
                "rationale": "rule-based: start with a broad search",
                "mcp_call": {"server_id": "web", "tool": "search",
                              "args": {"query": " ".join(obs.get("task_spec", "").split()[:5])}}}

    if turn >= 4:
        return {"kind": "submit",
                "rationale": "rule-based: submit accumulated best guess",
                "final": obs.get("task_spec", "")}

    return {"kind": "mcp",
            "rationale": "rule-based: follow up with knowledge base lookup",
            "mcp_call": {"server_id": "kb", "tool": "lookup_fact",
                          "args": {"entity": obs.get("task_spec", "")[:20]}}}
