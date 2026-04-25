"""Deterministic episode replay.

Runs a saved action-sequence against a fresh environment instance with the
original seed + task_id. Used by:
  - debugging (bisect which turn introduced the regression)
  - the demo video (play back the 'money-shot' episode any time)
  - judges who want to reproduce a reported result
"""

import json
import argparse
from typing import Any, Dict, List

from .arena_env import ProtocolArenaEnvironment
from ..models import OrchestratorAction


def replay(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    env = ProtocolArenaEnvironment()
    env.reset(seed=data["seed"], task_id=data["task_id"])
    rewards: List[float] = []
    for frame in data["actions"]:
        action = OrchestratorAction(**frame)
        obs = env.step(action)
        rewards.append(obs.reward)
        if obs.done:
            break
    return {
        "task_id": data["task_id"],
        "seed": data["seed"],
        "rewards": rewards,
        "final_reward": rewards[-1] if rewards else 0.0,
        "turns": len(rewards),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", help="Path to a saved episode JSON")
    args = ap.parse_args()
    out = replay(args.episode)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
