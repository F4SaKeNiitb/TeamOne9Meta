"""Phase A — SFT bootstrap (rewritten).

Build a high-quality SFT dataset from scripted-expert + baseline rollouts
across all 13 tasks. Compared to the original bootstrap, this version:

  - mixes a scripted EXPERT policy (50%), rule_based (25%), keyword
    (15%), and random (10%) so the dataset has all action kinds
    (mcp, plan, memory, a2a, submit), not just `mcp`-with-search;
  - filters episodes by `task_correctness > 0` *or*
    (frame_validity >= 0.6 AND final_reward >= 0.55), so the model
    only learns from rollouts that actually progressed the task;
  - stratifies episodes across tasks (one task per cycle) so no
    single task dominates the dataset;
  - deduplicates per-(user_msg, action_kind) so a 12-turn loop of
    identical actions doesn't produce 12 identical training rows;
  - injects a synthetic `submit` turn when a kept episode never
    submitted, so the model sees the final-turn behavior;
  - prints a diagnostics block so it's obvious what the dataset looks
    like before you ship 30 minutes of GPU time on it.

Usage:
    python -m arena.training.sft_bootstrap \\
        --out data/sft_rollouts.jsonl --episodes 600

Then upload `data/sft_rollouts.jsonl` to Colab as `sft.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from ..server.arena_env import ProtocolArenaEnvironment
from ..models import OrchestratorAction
from ..tasks import ALL_TASKS
from ..eval.baselines import rule_based_policy, keyword_policy, random_policy
from ..rewards.signals import score_task_correctness
from .expert_policy import expert_policy, ORACLE_HINTS


SYSTEM_PROMPT = "You are the PROTOCOL-ARENA orchestrator."

# Action keys that survive into the assistant message. Anything else
# emitted by a policy is dropped before the env sees it.
VALID_ACTION_KEYS = {"kind", "rationale", "mcp_call", "a2a_call",
                     "dag_delta", "kg_op", "final", "rewind_n", "confidence"}


def _build_user_msg(obs_dict: Dict[str, Any]) -> str:
    """Reuse inference.build_user_msg so SFT and inference see the SAME
    prompt format. Falls back to a compact JSON dump if the import
    fails (e.g., when bootstrap is run from inside a notebook)."""
    try:
        from inference import build_user_msg as _bu  # type: ignore
        return _bu(obs_dict)
    except Exception:
        return json.dumps({
            "task_spec": obs_dict.get("task_spec", ""),
            "turn": obs_dict.get("turn", 0),
            "discovered": obs_dict.get("discovered", {}),
            "feedback": obs_dict.get("feedback", ""),
            "last_result": obs_dict.get("last_result"),
        }, indent=2)


def _ensure_rationale(d: Dict[str, Any], label: str) -> None:
    d.setdefault("rationale", f"{label} bootstrap action — at least twenty chars.")
    if len(d["rationale"]) < 20:
        d["rationale"] = (d["rationale"] + " " * 24)[:40]


def _make_policies() -> List[Tuple[str, Callable, float]]:
    """Return (label, policy_fn, weight) tuples. policy_fn signature:
    (obs_dict, task_id, recent_kinds, rng) → action dict.

    `rng` is a per-episode `random.Random`. The expert uses it to vary
    branch choices; baselines ignore it."""
    def _expert(obs, tid, kinds, rng):
        return expert_policy(obs, task_id=tid, recent_kinds=kinds, rng=rng)

    def _rule(obs, _tid, _kinds, _rng):
        return rule_based_policy(obs)

    def _kw(obs, _tid, _kinds, _rng):
        return keyword_policy(obs)

    def _rand(obs, _tid, _kinds, _rng):
        return random_policy(obs)

    return [
        ("expert",     _expert, 0.50),
        ("rule_based", _rule,   0.25),
        ("keyword",    _kw,     0.15),
        ("random",     _rand,   0.10),
    ]


def _rollout_episode(env: ProtocolArenaEnvironment, policy_fn: Callable,
                     policy_name: str, task_id: str, ep_seed: int,
                     max_turns: int) -> Dict[str, Any]:
    """Run one episode. Returns a dict with the recorded turns and
    summary stats (task_correctness, frame_validity, final_reward)."""
    obs = env.reset(task_id=task_id, seed=ep_seed)
    turns: List[Dict[str, Any]] = []
    recent_kinds: List[str] = []
    frames_total = 0
    frames_valid = 0
    # Per-episode rng so deterministic policies (the expert) can branch
    # differently per seed without affecting the global numpy RNG.
    ep_rng = random.Random(ep_seed)

    for _ in range(max_turns):
        if obs.done:
            break
        obs_dict = obs.model_dump()
        try:
            action_dict = policy_fn(obs_dict, task_id, recent_kinds, ep_rng)
        except Exception:
            # A misbehaving policy on one obs shouldn't kill the episode.
            break
        if not isinstance(action_dict, dict) or "kind" not in action_dict:
            break
        _ensure_rationale(action_dict, policy_name)

        cleaned = {k: v for k, v in action_dict.items() if k in VALID_ACTION_KEYS}
        try:
            action = OrchestratorAction(**cleaned)
        except Exception:
            # Skip invalid action; continue with next turn (env state unchanged).
            break

        user_msg = _build_user_msg(obs_dict)
        turns.append({
            "user": user_msg,
            "assistant": json.dumps(cleaned, separators=(",", ":")),
            "kind": cleaned.get("kind", "?"),
        })
        recent_kinds.append(cleaned.get("kind", "?"))
        obs = env.step(action)
        frames_total += 1
        if obs.last_result and getattr(obs.last_result, "ok", False):
            frames_valid += 1

    oracle = ALL_TASKS[task_id].get("oracle_matchers", [])
    final_answer = getattr(env, "_final_answer", "") or ""
    tc = score_task_correctness(final_answer, oracle)
    fv = frames_valid / max(1, frames_total)
    fr = float(obs.reward)

    return {
        "turns": turns,
        "task_correctness": tc,
        "frame_validity": fv,
        "final_reward": fr,
        "submitted": any(t["kind"] == "submit" for t in turns),
    }


def generate(out_path: str, n_episodes: int, seed: int,
             min_task_correctness: float, min_final_reward: float,
             min_frame_validity: float, max_per_task: int,
             max_turns: int = 12) -> None:
    rng = random.Random(seed)
    env = ProtocolArenaEnvironment()
    tasks = sorted(ALL_TASKS.keys())
    policies = _make_policies()
    pol_names   = [p[0] for p in policies]
    pol_fns     = {p[0]: p[1] for p in policies}
    pol_weights = [p[2] for p in policies]

    per_task_rows: Dict[str, List[Dict[str, Any]]] = {t: [] for t in tasks}
    seen_pairs: set = set()  # (user_hash, action_kind) for dedup

    diag = {
        "attempted":       0,
        "filtered_low_tc": 0,
        "filtered_quota":  0,
        "kept_episodes":   0,
        "kept_rows":       0,
        "by_policy":       Counter(),
        "by_task":         Counter(),
        "by_kind":         Counter(),
        "tc_distribution": Counter(),  # bucketed task_correctness per kept episode
        "submit_injected": 0,
        "scrubbed_submits": 0,
    }

    print(f"[bootstrap] generating {n_episodes} episodes across "
          f"{len(tasks)} tasks; policies={dict(zip(pol_names, pol_weights))}")

    for ep_idx in range(n_episodes):
        task_id = tasks[ep_idx % len(tasks)]   # stratified: cycle tasks
        policy_name = rng.choices(pol_names, weights=pol_weights, k=1)[0]
        ep = _rollout_episode(env, pol_fns[policy_name], policy_name,
                              task_id, seed + ep_idx, max_turns)
        diag["attempted"] += 1

        # Filter — keep iff task progressed enough.
        keep = (ep["task_correctness"] >= min_task_correctness > 0) or (
            ep["final_reward"]   >= min_final_reward and
            ep["frame_validity"] >= min_frame_validity)
        if not keep:
            diag["filtered_low_tc"] += 1
            continue

        if len(per_task_rows[task_id]) >= max_per_task:
            diag["filtered_quota"] += 1
            continue

        # If the episode submitted but the answer was wrong (task_correctness=0),
        # the baselines tend to paste the task spec as the answer. Training on
        # those teaches the model to submit the prompt back. Strip those submits
        # so the synthetic-submit injection below replaces them with an
        # oracle-keyed answer.
        if ep["task_correctness"] == 0.0:
            before = len(ep["turns"])
            ep["turns"] = [t for t in ep["turns"] if t["kind"] != "submit"]
            if before != len(ep["turns"]):
                ep["submitted"] = False
                diag["scrubbed_submits"] += (before - len(ep["turns"]))

        # Inject a synthetic submit turn if the kept episode never submitted
        # (or had its bad submit scrubbed above).
        if not ep["submitted"] and task_id in ORACLE_HINTS and ep["turns"]:
            hints = ORACLE_HINTS[task_id][:3]
            submit_action = {
                "kind": "submit",
                "rationale": "all required evidence gathered — submitting concise final answer.",
                "final": "Answer: " + "; ".join(hints) + ".",
            }
            ep["turns"].append({
                "user":      ep["turns"][-1]["user"],
                "assistant": json.dumps(submit_action, separators=(",", ":")),
                "kind":      "submit",
            })
            diag["submit_injected"] += 1

        diag["kept_episodes"] += 1
        diag["by_policy"][policy_name] += 1
        diag["by_task"][task_id] += 1
        # Bucket task_correctness into 0.0/0.33/0.67/1.0 for distribution view.
        bucket = round(ep["task_correctness"] * 3) / 3
        diag["tc_distribution"][f"{bucket:.2f}"] += 1

        for t in ep["turns"]:
            key = (hash(t["user"]) % (2**63), t["kind"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            per_task_rows[task_id].append({
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": t["user"]},
                    {"role": "assistant", "content": t["assistant"]},
                ],
                "task_id": task_id,
                "policy": policy_name,
                "task_correctness": ep["task_correctness"],
                "frame_validity":   ep["frame_validity"],
                "final_reward":     ep["final_reward"],
            })
            diag["kept_rows"] += 1
            diag["by_kind"][t["kind"]] += 1
            if len(per_task_rows[task_id]) >= max_per_task:
                break

    # Flatten and shuffle so the dataloader sees a mix, not 80 photo rows in a row.
    rows: List[Dict[str, Any]] = [r for task in per_task_rows.values() for r in task]
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n[bootstrap] === diagnostics ===")
    print(f"  episodes attempted   : {diag['attempted']}")
    print(f"  episodes filtered    : {diag['filtered_low_tc']} (low task progress) "
          f"+ {diag['filtered_quota']} (per-task quota)")
    print(f"  episodes kept        : {diag['kept_episodes']}")
    print(f"  rows after dedup     : {diag['kept_rows']}")
    print(f"  synthetic submits    : {diag['submit_injected']}")
    print(f"  scrubbed bad submits : {diag['scrubbed_submits']}  "
          f"(replaced with oracle-keyed answers)")
    print(f"  by policy            : {dict(diag['by_policy'])}")
    print(f"  by action kind       : {dict(diag['by_kind'])}")
    print(f"  tc distribution      : {dict(diag['tc_distribution'])}")
    print(f"  per-task row counts  :")
    for t in tasks:
        print(f"    {t:<32} {len(per_task_rows[t])}")
    print(f"  output               : {out_path}")

    # Hard sanity warnings — surface BEFORE the user spends GPU time.
    issues: List[str] = []
    if diag["kept_rows"] < 200:
        issues.append(f"only {diag['kept_rows']} rows kept — "
                      "consider raising --episodes")
    if diag["by_kind"].get("mcp", 0) > 0.85 * diag["kept_rows"]:
        issues.append(f"{diag['by_kind'].get('mcp',0)}/{diag['kept_rows']} rows "
                      "are mcp — dataset is unbalanced")
    if diag["by_kind"].get("submit", 0) < 5:
        issues.append(f"only {diag['by_kind'].get('submit',0)} submit rows — "
                      "model won't learn to finish")
    missing_tasks = [t for t in tasks if not per_task_rows[t]]
    if missing_tasks:
        issues.append(f"tasks with ZERO rows: {missing_tasks}")
    if issues:
        print(f"\n[bootstrap] ⚠️  warnings:")
        for w in issues:
            print(f"    - {w}")
        print(f"  consider re-running with adjusted --min-* thresholds.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft_rollouts.jsonl")
    ap.add_argument("--episodes", type=int, default=600,
                    help="Total rollout attempts. Stratified across tasks.")
    ap.add_argument("--min-task-correctness", type=float, default=0.34,
                    help="Threshold for the 'task actually progressed' "
                         "filter (set to 0 to fall back on the reward gate).")
    ap.add_argument("--min-final-reward", type=float, default=0.55)
    ap.add_argument("--min-frame-validity", type=float, default=0.6)
    ap.add_argument("--max-per-task", type=int, default=80,
                    help="Cap rows per task. Stops any one task from "
                         "dominating the dataset.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args(argv)

    generate(
        out_path=args.out,
        n_episodes=args.episodes,
        seed=args.seed,
        min_task_correctness=args.min_task_correctness,
        min_final_reward=args.min_final_reward,
        min_frame_validity=args.min_frame_validity,
        max_per_task=args.max_per_task,
        max_turns=args.max_turns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
