"""Spectator UI — a minimal Rich-based terminal renderer for a live episode.

Usage:
    python -m arena.ui.spectator --task research_photo_rename --seed 0

Attaches the environment in-process, drives it with the rule-based
baseline, and renders one panel per turn showing:

  • action kind + rationale
  • last protocol result
  • current DAG (compact)
  • drift events as they fire
  • reward signals accumulating

This is deliberately framework-light so the demo works offline and is
reproducible from a seed. Swap `rule_based_policy` for any policy
callable that accepts an observation dict and returns an action dict.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable, Dict

from ..server.arena_env import ProtocolArenaEnvironment
from ..models import OrchestratorAction
from ..eval.baselines import rule_based_policy


PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]


def _fmt_dag(dag: Dict[str, Any]) -> str:
    nodes = dag.get("nodes", [])
    if not nodes:
        return "(empty DAG)"
    lines = []
    for n in nodes[-6:]:
        lines.append(f"  [{n.get('status','?'):>5}] {n.get('id','?')}  {n.get('op','')}")
    return "\n".join(lines)


def _fmt_signals(signals: Dict[str, float]) -> str:
    if not signals:
        return "(no signals yet)"
    return "  " + "  ".join(f"{k}={v:.2f}" for k, v in signals.items())


def _render_turn(obs: Dict[str, Any], action: Dict[str, Any]) -> str:
    buf = []
    buf.append("=" * 78)
    buf.append(f"TURN {obs.get('turn', '?')}/{obs.get('max_turns', '?')}  "
               f"task={obs.get('task_id','?')}  done={obs.get('done', False)}")
    buf.append("-" * 78)
    buf.append(f"action.kind={action.get('kind','?')}")
    buf.append(f"action.rationale={action.get('rationale','')[:160]}")
    last = obs.get("last_result") or {}
    if last:
        buf.append(f"last_result: ok={last.get('ok')} status={last.get('status_code')} "
                   f"error={last.get('error')}")
        if last.get("drift_hint"):
            buf.append(f"  drift_hint: {last['drift_hint']}")
    buf.append("DAG:")
    buf.append(_fmt_dag(obs.get("dag_state", {})))
    buf.append(f"feedback: {obs.get('feedback','').splitlines()[0] if obs.get('feedback') else ''}")
    buf.append(f"reward={obs.get('reward', 0.0):.4f}  signals:")
    buf.append(_fmt_signals(obs.get("reward_signals", {})))
    return "\n".join(buf)


def run(task_id: str, seed: int, max_steps: int = 12,
        policy: PolicyFn = rule_based_policy, out=sys.stdout) -> Dict[str, Any]:
    env = ProtocolArenaEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    trace = {"task_id": task_id, "seed": seed, "turns": [], "final": None}

    steps = 0
    while not obs.done and steps < max_steps:
        obs_dict = obs.model_dump()
        decision = policy(obs_dict)
        decision.setdefault("rationale",
                            "spectator-run action padded for schema compliance.")
        if len(decision["rationale"]) < 20:
            decision["rationale"] = (decision["rationale"] + " " * 25)[:40]
        decision = {k: v for k, v in decision.items()
                    if k in {"kind", "rationale", "mcp_call", "a2a_call",
                             "dag_delta", "kg_op", "final"}}
        action = OrchestratorAction(**decision)
        obs = env.step(action)
        print(_render_turn(obs.model_dump(), decision), file=out, flush=True)
        trace["turns"].append({"action": decision, "obs": obs.model_dump()})
        steps += 1

    trace["final"] = {
        "reward": obs.reward,
        "signals": obs.reward_signals,
        "drift_fired": env.state.drift_fired,
    }
    print("=" * 78, file=out)
    print(f"FINAL reward={obs.reward:.4f}  drift_fired={env.state.drift_fired}",
          file=out)
    return trace


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PROTOCOL-ARENA spectator")
    ap.add_argument("--task", default="research_photo_rename")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--save", default=None,
                    help="Optional path to save full trace JSON")
    args = ap.parse_args(argv)

    trace = run(args.task, args.seed, max_steps=args.max_steps)
    if args.save:
        with open(args.save, "w") as f:
            json.dump(trace, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
