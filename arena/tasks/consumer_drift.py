"""Consumer-workflow task pack — Patronus AI sub-theme.

Tasks are consumer-style flows (search, book, reply) where the underlying
tool schemas / policies shift mid-episode. Built on top of the same MCP/A2A
harness but with policy-drift events as the headline adversary.
"""

from ..server.drift_engine import DriftSchedule, DriftEvent


def _policy_drift(turn: int, server: str, tool: str, field: str) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="policy", target_server=server,
                   target_tool=tool, detail={"field": field}),
    ])


_BUDGET = {"tokens": 4000, "calls": 14, "latency_ms": 30000}


CONSUMER_TASKS = {
    "consumer_policy_pii_search": {
        "task_id": "consumer_policy_pii_search",
        "pack": "consumer_drift",
        "difficulty": "medium",
        "spec": (
            "Search the knowledge base to find how transformers work. "
            "Mid-episode the search tool's T&C may forbid PII in the query; "
            "respect the policy and submit a one-sentence answer."
        ),
        "oracle_matchers": ["attention", "self-attention", "sequence"],
        "expected_tools": ["web.search", "web.fetch_url"],
        "drift_schedule": _policy_drift(turn=2, server="web", tool="search", field="query"),
        "budget": _BUDGET,
    },
    "consumer_rename_plus_policy": {
        "task_id": "consumer_rename_plus_policy",
        "pack": "consumer_drift",
        "difficulty": "hard",
        "spec": (
            "Look up the year of Schwarzschild's black-hole solution. "
            "Multiple drift events may occur; adapt and submit."
        ),
        "oracle_matchers": ["1916"],
        "expected_tools": ["kb.lookup_fact", "web.search"],
        "drift_schedule": DriftSchedule(events=[
            DriftEvent(turn=1, klass="renaming", target_server="kb",
                       target_tool="lookup_fact",
                       detail={"new_name": "kb_lookup"}),
            DriftEvent(turn=2, klass="policy", target_server="web",
                       target_tool="search", detail={"field": "query"}),
        ]),
        "budget": _BUDGET,
    },
    "consumer_churn_fallback": {
        "task_id": "consumer_churn_fallback",
        "pack": "consumer_drift",
        "difficulty": "medium",
        "spec": (
            "Find a citation for the 2017 transformer paper. The citation "
            "A2A peer goes offline mid-session; fall back to other tools."
        ),
        "oracle_matchers": ["Vaswani", "2017"],
        "expected_tools": ["a2a.citer", "web.search", "kb.lookup_fact"],
        "drift_schedule": DriftSchedule(events=[
            DriftEvent(turn=1, klass="agentcard_churn", target_peer="citer"),
        ]),
        "budget": _BUDGET,
    },
}
