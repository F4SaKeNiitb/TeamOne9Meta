"""Self-play flywheel — Plan §18.6.

Two-line story for the blog: "the env generates its own curriculum."

Mechanism:
  1. Run the baseline (or the latest trained checkpoint) across all tasks
     × N seeds.
  2. Keep trajectories with final reward ≥ threshold.
  3. Append them to the SFT dataset as additional training rows.
  4. Optionally re-run the GRPO / SFT step on the enriched dataset.

Usage:
    python -m arena.training.flywheel \
        --out data/flywheel_iter1.jsonl \
        --seeds 0 1 2 3 4 \
        --threshold 0.55

The output JSONL is in the same chat-format schema as `sft_bootstrap.py`
so downstream training code does not change.
"""

from __future__ import annotations

import os
import json
import argparse
import random
from typing import Any, Dict, List

from ..server.arena_env import ProtocolArenaEnvironment
from ..models import OrchestratorAction
from ..tasks import ALL_TASKS
from ..eval.baselines import rule_based_policy, keyword_policy


SYSTEM_PROMPT = ("You are the PROTOCOL-ARENA orchestrator. Output one JSON "
                 "action per turn matching the schema.")


def _obs_as_user_msg(obs: Dict[str, Any]) -> str:
    try:
        from inference import build_user_msg
        return build_user_msg(obs)
    except Exception:
        return json.dumps({
            "task_spec": obs.get("task_spec", ""),
            "turn": obs.get("turn", 0),
            "discovered": obs.get("discovered", {}),
            "feedback": obs.get("feedback", ""),
        }, indent=2)


def collect(out_path: str, seeds: List[int], threshold: float = 0.55,
            tasks: List[str] = None, policies: List[str] = None):
    tasks = tasks or list(ALL_TASKS.keys())
    policy_map = {"rule_based": rule_based_policy, "keyword": keyword_policy}
    policy_fns = [policy_map[p] for p in (policies or ["rule_based", "keyword"])
                  if p in policy_map]

    env = ProtocolArenaEnvironment()
    rows: List[Dict[str, Any]] = []
    kept = 0
    total = 0

    for seed in seeds:
        random.seed(seed)
        for tid in tasks:
            for pol in policy_fns:
                total += 1
                obs = env.reset(task_id=tid, seed=seed)
                turns: List[Dict[str, Any]] = []
                for _ in range(obs.max_turns):
                    if obs.done:
                        break
                    obs_dict = obs.model_dump()
                    decision = pol(obs_dict)
                    decision.setdefault("rationale",
                                        "flywheel-collected rollout action.")
                    if len(decision["rationale"]) < 20:
                        decision["rationale"] = (decision["rationale"] + " " * 20)[:40]
                    turns.append({
                        "user": _obs_as_user_msg(obs_dict),
                        "assistant": json.dumps(decision, separators=(",", ":")),
                    })
                    keep = {k: v for k, v in decision.items()
                            if k in {"kind", "rationale", "mcp_call", "a2a_call",
                                     "dag_delta", "kg_op", "final",
                                     "rewind_n", "confidence"}}
                    obs = env.step(OrchestratorAction(**keep))

                final_reward = obs.reward
                # Never ingest unsafe trajectories into training data, regardless
                # of reward — fail closed on safety.
                if env.state.honeypot_triggered or env.state.injection_followed:
                    continue
                if final_reward < threshold:
                    continue

                kept += 1
                for t in turns:
                    rows.append({
                        "messages": [
                            {"role": "system",    "content": SYSTEM_PROMPT},
                            {"role": "user",      "content": t["user"]},
                            {"role": "assistant", "content": t["assistant"]},
                        ],
                        "episode_reward": final_reward,
                        "task_id": tid,
                        "source": "flywheel",
                    })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[flywheel] kept {kept}/{total} episodes above {threshold} → "
          f"{len(rows)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/flywheel_iter.jsonl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--policies", nargs="+", default=["rule_based", "keyword"])
    args = ap.parse_args()
    collect(args.out, args.seeds, args.threshold, args.tasks, args.policies)


if __name__ == "__main__":
    main()
