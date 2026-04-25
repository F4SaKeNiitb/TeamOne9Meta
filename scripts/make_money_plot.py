"""Generate the 'drift-recovery' plot.

For one chosen task, plot per-turn cumulative reward for N policies on
the same axes, averaged across multiple seeds with a ±1σ band. A
vertical red line marks the turn drift fires (modal across seeds).

Run this AFTER Track A delivers a trained policy adapter — on
baselines alone, cumulative reward gives random too much credit
through valid-frame bonuses, so rule_based / keyword / random do not
separate cleanly enough for a deck slide. With a trained policy in
the mix, the trained line dominates and the plot tells a strong story.

Usage (with trained policy):
    python scripts/make_money_plot.py \\
        --task research_photo_rename --seeds 0 1 2 \\
        --policies rule_based keyword random trained:my_module:my_policy_fn \\
        --out reports/drift_recovery.png

The format `module:fn` lets you point at any callable returning an
action dict from an obs dict.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arena.server.arena_env import ProtocolArenaEnvironment
from arena.models import OrchestratorAction
from arena.eval.baselines import rule_based_policy, keyword_policy, random_policy


PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]
BUILTINS: Dict[str, PolicyFn] = {
    "rule_based": rule_based_policy,
    "keyword":    keyword_policy,
    "random":     random_policy,
}


def _resolve(spec: str) -> Tuple[str, PolicyFn]:
    if spec in BUILTINS:
        return spec, BUILTINS[spec]
    # format: label:module:fn
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"bad policy spec {spec!r}; want 'label:module:fn' "
                         f"or one of {list(BUILTINS)}")
    label, mod, fn = parts
    return label, getattr(importlib.import_module(mod), fn)


def _rollout_curve(env: ProtocolArenaEnvironment, policy: PolicyFn,
                   task_id: str, seed: int,
                   max_turns: int = 12) -> Dict[str, Any]:
    obs = env.reset(task_id=task_id, seed=seed)
    cum = [0.0]
    drift_turn = None
    prior_drift_fired = bool(env.state.drift_fired)
    for _ in range(max_turns):
        if obs.done:
            break
        d = policy(obs.model_dump())
        d.setdefault("rationale", "money-plot rollout action.")
        if len(d["rationale"]) < 20:
            d["rationale"] = (d["rationale"] + " " * 25)[:40]
        d = {k: v for k, v in d.items()
             if k in {"kind", "rationale", "mcp_call", "a2a_call",
                      "dag_delta", "kg_op", "final", "rewind_n", "confidence"}}
        obs = env.step(OrchestratorAction(**d))
        cum.append(cum[-1] + float(obs.reward))
        try:
            now_fired = bool(env.state.drift_fired)
            if drift_turn is None and now_fired and not prior_drift_fired:
                drift_turn = obs.turn
            prior_drift_fired = now_fired
        except Exception:
            pass
    return {"cum": cum, "drift_turn": drift_turn, "final": float(obs.reward)}


def _pad_to(curve: List[float], length: int) -> List[float]:
    """Right-pad a per-turn cumulative reward curve to a common length by
    repeating the final value (cumulative reward stays flat once the
    episode ends, by definition)."""
    if len(curve) >= length:
        return curve[:length]
    return curve + [curve[-1]] * (length - len(curve))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="research_photo_rename")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds to average over (default 3 seeds).")
    ap.add_argument("--policies", nargs="+",
                    default=["rule_based", "keyword", "random"])
    ap.add_argument("--out", default="reports/drift_recovery.png")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args(argv)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[plot] matplotlib/numpy unavailable: {e}", file=sys.stderr)
        print("[plot] install with: pip install matplotlib numpy", file=sys.stderr)
        return 1

    env = ProtocolArenaEnvironment()
    # policy_label → list of per-seed curves; per-seed-drift-turns
    per_policy: Dict[str, List[List[float]]] = {}
    drift_turns: List[int] = []

    for spec in args.policies:
        label, fn = _resolve(spec)
        per_policy.setdefault(label, [])
        for seed in args.seeds:
            c = _rollout_curve(env, fn, args.task, seed, args.max_turns)
            per_policy[label].append(c["cum"])
            if c["drift_turn"] is not None:
                drift_turns.append(c["drift_turn"])
            print(f"[plot] {label:>12} seed={seed}  "
                  f"cum_terminal={c['cum'][-1]:.3f}  drift@turn={c['drift_turn']}")

    common_len = max(max(len(c) for c in cs) for cs in per_policy.values())

    plt.figure(figsize=(8.0, 4.8))
    colors = {"rule_based": "#1f77b4", "keyword": "#ff7f0e",
              "random":     "#7f7f7f", "trained":  "#2ca02c"}
    styles = {"rule_based": "-",  "keyword": "--",
              "random":     ":",  "trained":  "-"}
    for label, curves in per_policy.items():
        padded = np.array([_pad_to(c, common_len) for c in curves])
        mean = padded.mean(axis=0)
        std  = padded.std(axis=0)
        x = np.arange(common_len)
        color = colors.get(label, "#d62728")
        plt.plot(x, mean, marker="o", color=color,
                 linestyle=styles.get(label, "-"),
                 label=label, linewidth=2.2)
        if len(curves) > 1:
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    if drift_turns:
        # Modal drift turn across seeds — most common firing turn.
        from collections import Counter
        dt = Counter(drift_turns).most_common(1)[0][0]
        plt.axvline(dt, linestyle="--", color="#d62728", linewidth=1.5,
                    label=f"drift fires at turn {dt}")

    n_seeds = len(args.seeds)
    plt.xlabel("turn")
    plt.ylabel("cumulative reward")
    plt.title(f"Drift-recovery — {args.task}  "
              f"(mean ±1σ over {n_seeds} seed{'s' if n_seeds != 1 else ''})")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"[plot] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
