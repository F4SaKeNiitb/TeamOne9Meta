"""Generate the headline 'drift-recovery' plot.

For one chosen task, plot per-turn cumulative reward for N policies
(rule_based, keyword, optionally a trained policy) on the same axes.
A vertical red line marks the turn drift fires. The figure is the
single most-effective image in the README — it shows training paid
off, on a real drift event, in one glance.

Usage:
    python scripts/make_money_plot.py \\
        --task research_photo_rename --seed 0 \\
        --policies rule_based keyword \\
        --out reports/drift_recovery.png

Adding a trained policy:
    --policies rule_based keyword trained:my_module:my_policy_fn

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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="research_photo_rename")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", nargs="+",
                    default=["rule_based", "keyword"])
    ap.add_argument("--out", default="reports/drift_recovery.png")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args(argv)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}", file=sys.stderr)
        print("[plot] install with: pip install matplotlib", file=sys.stderr)
        return 1

    env = ProtocolArenaEnvironment()
    curves: List[Tuple[str, Dict[str, Any]]] = []
    drift_turns: List[int] = []
    for spec in args.policies:
        label, fn = _resolve(spec)
        c = _rollout_curve(env, fn, args.task, args.seed, args.max_turns)
        curves.append((label, c))
        if c["drift_turn"] is not None:
            drift_turns.append(c["drift_turn"])
        print(f"[plot] {label:>12}  final={c['final']:.3f}  "
              f"cum_terminal={c['cum'][-1]:.3f}  drift@turn={c['drift_turn']}")

    plt.figure(figsize=(7.5, 4.5))
    colors = {"rule_based": "#1f77b4", "keyword": "#ff7f0e",
              "random":     "#7f7f7f", "trained":  "#2ca02c"}
    styles = {"rule_based": "-",  "keyword": "--",
              "random":     ":",  "trained":  "-"}
    for label, c in curves:
        x = list(range(len(c["cum"])))
        plt.plot(x, c["cum"], marker="o",
                 color=colors.get(label, "#d62728"),
                 linestyle=styles.get(label, "-"),
                 label=label, linewidth=2.2)
    if drift_turns:
        dt = min(drift_turns)
        plt.axvline(dt, linestyle="--", color="#d62728", linewidth=1.5,
                    label=f"drift fires at turn {dt}")
    plt.xlabel("turn")
    plt.ylabel("cumulative reward")
    plt.title(f"Drift-recovery — {args.task} (seed {args.seed})")
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
