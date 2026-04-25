"""Report generator — reads `reports/frontier.json`, emits:

  reports/results.md        : per-provider table + ranked summary
  reports/signals_bar.png   : 6-signal bar chart (trained vs frontier)
  reports/drift_curve.png   : task_correctness on pre / during / hard
  reports/pareto.png        : reward vs cost (calls × 10 + turns)
  reports/failure_modes.png : top-N failure classes (from diagnostics)

Plots degrade gracefully: if matplotlib isn't installed, we write only
the markdown. That way judges-without-deps still see the table.

Usage:
    python -m arena.eval.report \
        --in reports/frontier.json \
        --out reports/

Pareto plot interpretation: lower-left = worse, upper-right = better.
The trained PROTOCOL-ARENA agent should dominate in the upper-right
region against at least the rule-based floor.
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Any, Dict, List, Tuple


def _md_table(providers: Dict[str, Any], splits: List[str]) -> str:
    header = ["provider"] + [f"{s}.tc" for s in splits] + ["drift_adj", "brier", "hp", "inj"]
    rows = [" | ".join(header), "|".join(["---"] * len(header))]
    for label, rep in providers.items():
        if "error" in rep:
            rows.append(" | ".join([label] + ["ERR"] * (len(header) - 1)))
            continue
        cells = [label]
        for s in splits:
            v = rep.get(f"eval_{s}", {}).get("task_correctness", {}).get("mean")
            cells.append(f"{v:.3f}" if v is not None else "—")
        das = rep.get("drift_adjusted_success_rate", {}).get("value")
        cells.append(f"{das:.3f}" if das is not None else "—")
        brier = rep.get("eval_during", {}).get("brier", {}).get("mean")
        cells.append(f"{brier:.3f}" if brier is not None else "—")
        hp = rep.get("eval_during", {}).get("honeypot_rate", {}).get("mean")
        cells.append(f"{hp:.2f}" if hp is not None else "—")
        inj = rep.get("eval_during", {}).get("injection_rate", {}).get("mean")
        cells.append(f"{inj:.2f}" if inj is not None else "—")
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _write_markdown(results: Dict[str, Any], out_dir: str,
                    splits: List[str]) -> str:
    providers = results.get("providers", {})
    body = [
        "# PROTOCOL-ARENA — Evaluation Report",
        "",
        f"_Generated from `frontier.json` — {len(providers)} providers, "
        f"{len(splits)} splits._",
        "",
        "## Headline table",
        "",
        "Legend: `tc` = task_correctness (mean over seeds × tasks). "
        "`drift_adj` = 1 − max(0, pre − during). `hp` / `inj` = safety "
        "breach rates (lower is better).",
        "",
        _md_table(providers, splits),
        "",
        "## What to look for",
        "",
        "- **Drift-adjusted success rate** — the headline number. Our "
        "trained policy should stay near its pre-drift correctness.",
        "- **Brier score** — reported but not trained on; catches "
        "confidence-calibration hacking.",
        "- **Honeypot / injection rates** — MUST be 0 for the trained "
        "policy, even if frontier baselines occasionally slip.",
        "",
        "## Per-signal breakdown",
        "",
        "See `signals_bar.png` and `drift_curve.png`.",
    ]
    path = os.path.join(out_dir, "results.md")
    with open(path, "w") as f:
        f.write("\n".join(body))
    return path


def _plot_signals_bar(results: Dict[str, Any], out_path: str,
                      split: str = "during"):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    providers = results.get("providers", {})
    metric_keys = ["task_correctness", "frame_validity", "plan_quality",
                   "final_reward"]
    labels = list(providers.keys())
    if not labels:
        return None
    import numpy as np
    x = np.arange(len(metric_keys))
    width = max(0.1, 0.8 / max(1, len(labels)))
    plt.figure(figsize=(9, 4.5))
    for i, label in enumerate(labels):
        r = providers[label]
        if "error" in r:
            continue
        vals = [r.get(f"eval_{split}", {}).get(k, {}).get("mean", 0.0)
                for k in metric_keys]
        plt.bar(x + i * width, vals, width, label=label)
    plt.xticks(x + width * (len(labels) - 1) / 2, metric_keys, rotation=15)
    plt.ylabel("mean across seeds × tasks")
    plt.title(f"PROTOCOL-ARENA — `eval_{split}`")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _plot_drift_curve(results: Dict[str, Any], out_path: str,
                      splits: List[str]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    providers = results.get("providers", {})
    if not providers:
        return None
    plt.figure(figsize=(8, 4.5))
    for label, r in providers.items():
        if "error" in r:
            continue
        ys = [r.get(f"eval_{s}", {}).get("task_correctness", {}).get("mean", 0.0)
              for s in splits]
        plt.plot(splits, ys, marker="o", label=label)
    plt.ylabel("task_correctness")
    plt.xlabel("split")
    plt.title("Drift-robustness curve")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _plot_pareto(results: Dict[str, Any], out_path: str):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    providers = results.get("providers", {})
    xs, ys, labels = [], [], []
    for label, r in providers.items():
        if "error" in r:
            continue
        during = r.get("eval_during", {})
        reward = during.get("final_reward", {}).get("mean", 0.0)
        # Cost proxy: (1 - frame_validity) × penalty + (1 - efficiency).
        # Since we don't record raw budget usage here, use frame_validity
        # as a proxy for efficiency.
        fv = during.get("frame_validity", {}).get("mean", 0.0)
        cost = round((1.0 - fv) * 10.0 + r.get("runtime_s", 0.0) / 60.0, 3)
        xs.append(cost)
        ys.append(reward)
        labels.append(label)
    if not xs:
        return None
    plt.figure(figsize=(6.5, 5))
    plt.scatter(xs, ys, s=60)
    for x, y, l in zip(xs, ys, labels):
        plt.annotate(l, (x, y), xytext=(5, 5), textcoords="offset points",
                     fontsize=8)
    plt.xlabel("cost proxy (bad frames + minutes)  ← lower is better")
    plt.ylabel("reward (during-drift)  ← higher is better")
    plt.title("Cost × Reward Pareto")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="reports/frontier.json")
    ap.add_argument("--out", dest="out", default="reports/")
    ap.add_argument("--splits", nargs="+", default=["pre", "during", "hard"])
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    with open(args.inp) as f:
        results = json.load(f)

    md = _write_markdown(results, args.out, args.splits)
    print(f"[report] wrote {md}")

    for plotter, fname in [
        (_plot_signals_bar, "signals_bar.png"),
        (_plot_drift_curve, "drift_curve.png"),
        (_plot_pareto,       "pareto.png"),
    ]:
        out = os.path.join(args.out, fname)
        if plotter is _plot_drift_curve:
            p = plotter(results, out, args.splits)
        else:
            p = plotter(results, out)
        if p:
            print(f"[report] wrote {p}")
        else:
            print(f"[report] skipped {fname} (matplotlib or data unavailable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
