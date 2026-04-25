"""Research-synthesis task pack.

Each task is a dict with:
  - task_id, pack, difficulty
  - spec           : natural-language description handed to the agent
  - oracle         : canonical correct answer (string pattern or set)
  - oracle_matchers: list of substring patterns; ANY match counts as correct
  - expected_tools : hint set (informational; not a hard constraint)
  - drift_schedule : DriftSchedule applied to this episode
  - budget         : token / call / latency caps
"""

from ..server.drift_engine import DriftSchedule, DriftEvent


def _no_drift() -> DriftSchedule:
    return DriftSchedule(events=[])


def _rename_drift(turn: int, server: str, tool: str, new_name: str) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="renaming", target_server=server,
                   target_tool=tool, detail={"new_name": new_name}),
    ])


def _additive_drift(turn: int, server: str, tool: str, field: str) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="additive", target_server=server,
                   target_tool=tool, detail={"field": field}),
    ])


def _tighten_drift(turn: int, server: str, tool: str, field: str, pattern: str) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="tightening", target_server=server,
                   target_tool=tool, detail={"field": field, "pattern": pattern}),
    ])


def _rate_drift(turn: int, server: str, rpm: int) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="rate_limit",
                   target_server=server, detail={"rpm": rpm}),
    ])


def _churn_drift(turn: int, peer: str) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="agentcard_churn", target_peer=peer),
    ])


def _auth_drift(turn: int, server: str, tool: str, scope: str) -> DriftSchedule:
    return DriftSchedule(events=[
        DriftEvent(turn=turn, klass="auth", target_server=server,
                   target_tool=tool, detail={"scope": scope}),
    ])


_BUDGET_DEFAULT = {"tokens": 4000, "calls": 12, "latency_ms": 30000}


RESEARCH_TASKS = {
    "research_photo_basic": {
        "task_id": "research_photo_basic",
        "pack": "research",
        "difficulty": "easy",
        "spec": (
            "Find the primary chemical equation of photosynthesis. "
            "Use the available tools; submit a single-line answer containing the equation."
        ),
        "oracle_matchers": ["6 CO2", "C6H12O6", "6 O2"],
        "expected_tools": ["web.search", "web.fetch_url", "kb.lookup_fact"],
        "drift_schedule": _no_drift(),
        "budget": _BUDGET_DEFAULT,
    },
    "research_photo_rename": {
        "task_id": "research_photo_rename",
        "pack": "research",
        "difficulty": "medium",
        "spec": (
            "Find the equation of photosynthesis. The web search tool may change name "
            "mid-session; adapt and submit the equation regardless."
        ),
        "oracle_matchers": ["6 CO2", "C6H12O6"],
        "expected_tools": ["web.search", "web.fetch_url"],
        "drift_schedule": _rename_drift(turn=2, server="web", tool="search", new_name="query_web"),
        "budget": _BUDGET_DEFAULT,
    },
    "research_transformer_cite": {
        "task_id": "research_transformer_cite",
        "pack": "research",
        "difficulty": "medium",
        "spec": (
            "Who introduced the original transformer architecture and in what year? "
            "Use any combination of search, knowledge-base lookup, or an A2A citation peer."
        ),
        "oracle_matchers": ["Vaswani", "2017"],
        "expected_tools": ["web.search", "kb.lookup_fact", "a2a.citer"],
        "drift_schedule": _no_drift(),
        "budget": _BUDGET_DEFAULT,
    },
    "research_adam_additive": {
        "task_id": "research_adam_additive",
        "pack": "research",
        "difficulty": "hard",
        "spec": (
            "Name the year Adam was published and its first author. The knowledge-base "
            "lookup tool may gain a new required field mid-session; adapt."
        ),
        "oracle_matchers": ["2014", "Kingma"],
        "expected_tools": ["kb.lookup_fact", "web.search"],
        "drift_schedule": _additive_drift(turn=1, server="kb", tool="lookup_fact",
                                          field="schema_version"),
        "budget": _BUDGET_DEFAULT,
    },
    "research_bell_tighten": {
        "task_id": "research_bell_tighten",
        "pack": "research",
        "difficulty": "hard",
        "spec": (
            "Find the year Bell's theorem was published and what it shows. "
            "Note: query strings may become regex-constrained mid-session."
        ),
        "oracle_matchers": ["1964", "local realism"],
        "expected_tools": ["web.search", "kb.lookup_fact"],
        "drift_schedule": _tighten_drift(turn=2, server="web", tool="search",
                                         field="query", pattern=r"[A-Za-z0-9 ]{1,80}"),
        "budget": _BUDGET_DEFAULT,
    },
    "research_k8s_rate": {
        "task_id": "research_k8s_rate",
        "pack": "research",
        "difficulty": "medium",
        "spec": (
            "Describe the role of the kube-scheduler in Kubernetes. Rate limits on "
            "the web server may tighten during the episode."
        ),
        "oracle_matchers": ["pod", "node"],
        "expected_tools": ["web.search", "kb.lookup_fact"],
        "drift_schedule": _rate_drift(turn=2, server="web", rpm=3),
        "budget": _BUDGET_DEFAULT,
    },
    "research_bh_churn": {
        "task_id": "research_bh_churn",
        "pack": "research",
        "difficulty": "hard",
        "spec": (
            "Describe the event horizon of a black hole in one sentence. "
            "The citation A2A peer may go offline; use alternative sources if so."
        ),
        "oracle_matchers": ["event horizon", "escape"],
        "expected_tools": ["web.search", "a2a.citer"],
        "drift_schedule": _churn_drift(turn=2, peer="citer"),
        "budget": _BUDGET_DEFAULT,
    },
    "research_photo_auth": {
        "task_id": "research_photo_auth",
        "pack": "research",
        "difficulty": "hard",
        "spec": (
            "Summarize in one sentence how C4 plants differ from C3 plants. "
            "The fetch_url tool may gain a required OAuth scope during the episode."
        ),
        "oracle_matchers": ["CO2", "efficient"],
        "expected_tools": ["web.search", "web.fetch_url"],
        "drift_schedule": _auth_drift(turn=2, server="web", tool="fetch_url",
                                      scope="elevated"),
        "budget": _BUDGET_DEFAULT,
    },
    "research_gradient_multi": {
        "task_id": "research_gradient_multi",
        "pack": "research",
        "difficulty": "medium",
        "spec": (
            "Briefly explain how the Adam optimizer differs from vanilla SGD."
        ),
        "oracle_matchers": ["moment", "adaptive"],
        "expected_tools": ["web.search", "web.fetch_url"],
        "drift_schedule": _no_drift(),
        "budget": _BUDGET_DEFAULT,
    },
    "research_top_products": {
        "task_id": "research_top_products",
        "pack": "research",
        "difficulty": "easy",
        "spec": (
            "Using the data warehouse, return the top 5 products by units sold."
        ),
        "oracle_matchers": ["p1", "p2", "p3"],
        "expected_tools": ["data.list_tables", "data.run_sql"],
        "drift_schedule": _no_drift(),
        "budget": _BUDGET_DEFAULT,
    },
}
