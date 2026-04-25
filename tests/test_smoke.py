"""Smoke tests for PROTOCOL-ARENA.

Validates that the environment can run a full episode end-to-end with the
rule-based baseline policy, across at least one task per pack, with drift on.
"""

import pytest

from arena.server.arena_env import ProtocolArenaEnvironment
from arena.models import OrchestratorAction
from arena.eval.baselines import rule_based_policy, keyword_policy
from arena.tasks import ALL_TASKS


@pytest.mark.parametrize("task_id", [
    "research_photo_basic",
    "research_photo_rename",
    "consumer_churn_fallback",
])
def test_full_episode_runs(task_id):
    env = ProtocolArenaEnvironment()
    obs = env.reset(task_id=task_id, seed=7)
    steps = 0
    while not obs.done and steps < 12:
        decision = rule_based_policy(obs.model_dump())
        decision.setdefault("rationale", "smoke test action — padded to length.")
        if len(decision["rationale"]) < 20:
            decision["rationale"] = (decision["rationale"] + " " * 25)[:40]
        decision = {k: v for k, v in decision.items()
                    if k in {"kind", "rationale", "mcp_call", "a2a_call",
                             "dag_delta", "kg_op", "final"}}
        action = OrchestratorAction(**decision)
        obs = env.step(action)
        steps += 1
    assert obs.done
    assert 0.0 <= obs.reward <= 1.0


def test_reward_signals_populated_at_end():
    env = ProtocolArenaEnvironment()
    obs = env.reset(task_id="research_photo_basic", seed=0)
    while not obs.done:
        d = keyword_policy(obs.model_dump())
        d.setdefault("rationale", "smoke test keyword policy trigger.")
        if len(d["rationale"]) < 20:
            d["rationale"] = (d["rationale"] + " " * 25)[:40]
        d = {k: v for k, v in d.items()
             if k in {"kind", "rationale", "mcp_call", "a2a_call",
                      "dag_delta", "kg_op", "final"}}
        obs = env.step(OrchestratorAction(**d))
    assert "task_correctness" in obs.reward_signals
    assert "plan_quality" in obs.reward_signals


def test_drift_fires_and_is_recorded():
    env = ProtocolArenaEnvironment()
    obs = env.reset(task_id="research_photo_rename", seed=0)
    for _ in range(6):
        if obs.done:
            break
        d = rule_based_policy(obs.model_dump())
        d.setdefault("rationale", "drift test action to drive env.")
        if len(d["rationale"]) < 20:
            d["rationale"] = (d["rationale"] + " " * 25)[:40]
        d = {k: v for k, v in d.items()
             if k in {"kind", "rationale", "mcp_call", "a2a_call",
                      "dag_delta", "kg_op", "final"}}
        obs = env.step(OrchestratorAction(**d))
    assert env.state.drift_fired is True


def test_registry_has_tasks():
    assert len(ALL_TASKS) >= 10
    for t in ALL_TASKS.values():
        assert t["oracle_matchers"]
        assert t["budget"]["calls"] > 0
