"""Find the most demo-worthy (task_id, seed) pair.

A "killer seed" for the live demo has these properties (in order of
weight):
  1. Drift fires *mid-episode*, not at turn 0 or the last turn.
  2. The DAG grows to ≥4 nodes (visually compelling).
  3. The rule-based baseline scores in the 0.3–0.7 range — high enough
     to look like the agent is doing something, low enough that the
     trained agent has visible room to improve.
  4. At least one safe `memory` query happens (proves the recovery path
     is reachable from this state).

Usage:
    python scripts/find_killer_seed.py            # scans all tasks × seeds 0..4
    python scripts/find_killer_seed.py --top 5    # top-5 candidates

The output is a ranked list with the metric breakdown so the team can
pick the one whose narrative best matches the pitch.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.server.arena_env import ProtocolArenaEnvironment
from arena.models import OrchestratorAction
from arena.eval.baselines import rule_based_policy
from arena.tasks import ALL_TASKS


def _score_episode(env: ProtocolArenaEnvironment, task_id: str,
                   seed: int, max_turns: int = 12) -> Dict[str, Any]:
    obs = env.reset(task_id=task_id, seed=seed)
    n_turns = 0
    saw_memory = False
    drift_turn = None
    prior_drift_fired = bool(env.state.drift_fired)
    n_dag_nodes_max = 0

    for _ in range(max_turns):
        if obs.done:
            break
        d = rule_based_policy(obs.model_dump())
        d.setdefault("rationale", "killer-seed scan baseline action.")
        if len(d["rationale"]) < 20:
            d["rationale"] = (d["rationale"] + " " * 25)[:40]
        d = {k: v for k, v in d.items()
             if k in {"kind", "rationale", "mcp_call", "a2a_call",
                      "dag_delta", "kg_op", "final"}}
        if d.get("kind") == "memory":
            saw_memory = True
        obs = env.step(OrchestratorAction(**d))
        n_turns += 1

        try:
            now_fired = bool(env.state.drift_fired)
            if drift_turn is None and now_fired and not prior_drift_fired:
                drift_turn = obs.turn
            prior_drift_fired = now_fired
        except Exception:
            pass
        try:
            n_dag_nodes_max = max(n_dag_nodes_max, len(obs.dag_state.nodes))
        except Exception:
            pass

    final_reward = float(obs.reward)
    return {
        "task_id": task_id, "seed": seed,
        "final_reward": round(final_reward, 3),
        "turns": n_turns, "max_turns": max_turns,
        "drift_turn": drift_turn, "drift_fired": drift_turn is not None,
        "dag_nodes": n_dag_nodes_max,
        "saw_memory_query": saw_memory,
        "honeypot": bool(env.state.honeypot_triggered),
        "injection": bool(env.state.injection_followed),
    }


def _demo_score(ep: Dict[str, Any]) -> float:
    """Higher = better demo candidate."""
    if ep["honeypot"] or ep["injection"]:
        return -1.0  # safety breach mid-demo is bad optics
    s = 0.0
    # mid-episode drift is most dramatic
    dt = ep.get("drift_turn")
    if dt is not None:
        mid = ep["max_turns"] / 2
        s += 1.0 - abs(dt - mid) / max(1.0, mid)        # peaks at mid-episode
    # dag-density bonus
    s += min(1.0, ep["dag_nodes"] / 6.0) * 0.4
    # reward in the dramatic range 0.3–0.7
    fr = ep["final_reward"]
    s += (1.0 - min(1.0, abs(fr - 0.5) / 0.5)) * 0.4
    if ep["saw_memory_query"]:
        s += 0.2
    return round(s, 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args(argv)

    tasks = args.tasks or list(ALL_TASKS.keys())
    env = ProtocolArenaEnvironment()
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for tid in tasks:
        for seed in args.seeds:
            try:
                ep = _score_episode(env, tid, seed)
            except Exception as e:
                print(f"[warn] {tid}/{seed} crashed: {e}", file=sys.stderr)
                continue
            rows.append((_demo_score(ep), ep))

    rows.sort(key=lambda r: r[0], reverse=True)

    print(f"\n{'rank':>4}  {'demo':>5}  {'task':<32} {'seed':>4}  "
          f"{'reward':>6}  {'drift@':>6}  {'dag':>3}  notes")
    print("-" * 92)
    for i, (score, ep) in enumerate(rows[:args.top], 1):
        notes = []
        if ep["saw_memory_query"]: notes.append("KG-recovery")
        if ep["drift_fired"]:      notes.append("drift")
        if ep["honeypot"]:         notes.append("HP!")
        if ep["injection"]:        notes.append("INJ!")
        dt = str(ep["drift_turn"]) if ep["drift_turn"] is not None else "—"
        print(f"{i:>4}  {score:>5.2f}  {ep['task_id']:<32} {ep['seed']:>4}  "
              f"{ep['final_reward']:>6.3f}  {dt:>6}  {ep['dag_nodes']:>3}  "
              f"{','.join(notes)}")

    if rows:
        best = rows[0][1]
        print(f"\n[killer-seed] PICK:  --task {best['task_id']} --seed {best['seed']}")
        print(f"[killer-seed] open  http://localhost:7861/?task={best['task_id']}&seed={best['seed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
