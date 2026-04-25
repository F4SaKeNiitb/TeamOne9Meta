"""Merge Colab-side eval results into reports/frontier.json.

Track A runs Qwen2.5-1.5B-Instruct (base AND trained-with-LoRA) inside
Google Colab where the GPU is already paid for. This script takes the
JSON file Colab produces and inserts it into the local frontier.json
under the right provider label, so scoring and plotting see all
columns side by side.

Expected Colab-side JSON shape — produced by `arena.eval.harness.run_eval`:

    {
      "eval_pre":    {"task_correctness": {"mean":..., ...}, ...},
      "eval_during": {...},
      "eval_hard":   {...},
      "drift_adjusted_success_rate": {"value": 0.123},
      "runtime_s": 42.0
    }

Usage:
    # In Colab, after Cell 7 (final eval), each block looks like:
    #   results_base    = run_eval(base_policy,    ...)
    #   results_trained = run_eval(trained_policy, ...)
    #   json.dump(results_base,    open("qwen_1_5b_base.json","w"), default=str)
    #   json.dump(results_trained, open("qwen_1_5b_trained.json","w"), default=str)
    #   # download both files via Colab's Files panel.

    python scripts/merge_colab_eval.py \\
        --frontier reports/frontier.json \\
        --label qwen-1.5b-base    --json qwen_1_5b_base.json
    python scripts/merge_colab_eval.py \\
        --frontier reports/frontier.json \\
        --label trained           --json qwen_1_5b_trained.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default="reports/frontier.json",
                    help="Local frontier.json to merge into.")
    ap.add_argument("--label", required=True,
                    help="Provider label, e.g. 'qwen-1.5b-base' or 'trained'.")
    ap.add_argument("--json", required=True,
                    help="Path to the JSON file downloaded from Colab.")
    args = ap.parse_args(argv)

    if not os.path.exists(args.json):
        print(f"[merge] missing colab JSON: {args.json}", file=sys.stderr)
        return 2

    new_block = json.load(open(args.json))

    if not isinstance(new_block, dict) or "eval_during" not in new_block:
        print(f"[merge] {args.json} doesn't look like a run_eval() output "
              f"(missing 'eval_during' key)", file=sys.stderr)
        return 3

    if os.path.exists(args.frontier):
        fr = json.load(open(args.frontier))
    else:
        fr = {"providers": {}}
    fr.setdefault("providers", {})

    fr["providers"][args.label] = new_block

    os.makedirs(os.path.dirname(args.frontier) or ".", exist_ok=True)
    with open(args.frontier, "w") as f:
        json.dump(fr, f, indent=2, default=str)

    tc = new_block.get("eval_during", {}).get("task_correctness", {}).get("mean")
    print(f"[merge] {args.label} merged into {args.frontier}  "
          f"during.task_correctness={tc}")
    print(f"[merge] providers now: {list(fr['providers'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
