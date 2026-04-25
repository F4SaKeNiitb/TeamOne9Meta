"""Phase A — SFT bootstrap.

Generate oracle rollouts by running `rule_based_policy` against every task,
filter by final reward >= threshold, serialize as a chat-format JSONL suitable
for SFT via Unsloth or HF TRL.

Usage:
    python -m arena.training.sft_bootstrap --out data/sft_rollouts.jsonl --episodes 400
"""

import os
import json
import argparse
import random
from typing import Any, Dict, List

from ..server.arena_env import ProtocolArenaEnvironment
from ..models import OrchestratorAction
from ..tasks import ALL_TASKS
from ..eval.baselines import rule_based_policy, keyword_policy


def _obs_to_user_msg(obs_dict: Dict[str, Any]) -> str:
    from inference import build_user_msg as _bu  # type: ignore
    return _bu(obs_dict)


def _fallback_user_msg(obs: Dict[str, Any]) -> str:
    return json.dumps({"task_spec": obs.get("task_spec", ""),
                       "turn": obs.get("turn", 0),
                       "discovered": obs.get("discovered", {}),
                       "feedback": obs.get("feedback", "")}, indent=2)


def generate(out_path: str, n_episodes: int = 400, min_reward: float = 0.4,
             seed: int = 0):
    random.seed(seed)
    env = ProtocolArenaEnvironment()
    rows: List[Dict[str, Any]] = []
    tasks = list(ALL_TASKS.keys())

    system_prompt = open(os.path.join(
        os.path.dirname(__file__), "_system_prompt.txt"), "r"
    ).read() if os.path.exists(os.path.join(
        os.path.dirname(__file__), "_system_prompt.txt")) else "You are the PROTOCOL-ARENA orchestrator."

    for ep in range(n_episodes):
        task_id = random.choice(tasks)
        obs = env.reset(task_id=task_id, seed=seed + ep)
        policy = random.choice([rule_based_policy, keyword_policy])
        turns: List[Dict[str, Any]] = []
        for _ in range(obs.max_turns):
            if obs.done:
                break
            obs_dict = obs.model_dump()
            try:
                user_msg = _obs_to_user_msg(obs_dict)
            except Exception:
                user_msg = _fallback_user_msg(obs_dict)
            decision = policy(obs_dict)
            decision.setdefault("rationale", "oracle-bootstrap action for SFT.")
            if len(decision["rationale"]) < 20:
                decision["rationale"] = decision["rationale"] + " " * 20
            turns.append({
                "user": user_msg,
                "assistant": json.dumps(decision, separators=(",", ":")),
            })
            action = OrchestratorAction(**{k: v for k, v in decision.items()
                                           if k in {"kind", "rationale", "mcp_call",
                                                    "a2a_call", "dag_delta", "kg_op", "final"}})
            obs = env.step(action)
        final_reward = obs.reward
        if final_reward < min_reward:
            continue
        for t in turns:
            rows.append({
                "messages": [
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": t["user"]},
                    {"role": "assistant", "content": t["assistant"]},
                ],
                "episode_reward": final_reward,
                "task_id": task_id,
            })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} SFT rows to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_rollouts.jsonl")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--min-reward", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, args.episodes, args.min_reward, args.seed)


if __name__ == "__main__":
    main()
