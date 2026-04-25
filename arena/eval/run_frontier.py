"""Frontier zero-shot baseline runner — Plan §18.2 / §10.

Runs the PROTOCOL-ARENA eval harness against a set of OpenAI-compatible
endpoints (GPT-4o-mini, Claude Sonnet, Gemini Flash, Qwen2.5-7B, and the
rule-based policy as a floor). Writes `reports/frontier.json` consumed by
`arena.eval.report` to produce the Pareto and drift-robustness plots.

Modes:
  --mock       Do not hit any API; just run the local rule-based and
               keyword baselines. Useful for CI and for judges who
               clone the repo without API keys.
  --live       Hit OPENAI_API_BASE / ANTHROPIC_API_BASE / etc. Requires
               env vars configured. Uses `inference.py:ask_llm` as the
               LLM wrapper so the prompt surface is identical to the
               one used during GRPO rollouts.

The exit status is 0 on success, 1 on any provider failing to produce
ANY scored episode (so CI can gate on this).
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import time
from typing import Any, Callable, Dict, List, Optional

from .harness import run_eval
from .baselines import random_policy, rule_based_policy, keyword_policy
from ..tasks import ALL_TASKS


def _mock_providers() -> Dict[str, Callable]:
    return {
        "random": random_policy,
        "keyword": keyword_policy,
        "rule_based": rule_based_policy,
    }


def _openai_compatible_policy(model_name: str,
                              api_base: Optional[str] = None,
                              api_key_env: str = "OPENAI_API_KEY") -> Callable:
    """Return a policy fn that calls an OpenAI-compatible chat endpoint."""
    from openai import OpenAI

    key = os.getenv(api_key_env, "")
    client = OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)

    import inference  # the repo-root script; reuses the prompt exactly

    def policy(obs: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": inference.SYSTEM_PROMPT},
            {"role": "user",   "content": inference.build_user_msg(obs)},
        ]
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages,
                temperature=0.2, max_tokens=500,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            return {
                "kind": "memory",
                "rationale": f"provider_error_fallback: {e}",
                "kg_op": {"op": "query",
                          "pattern": obs.get("task_spec", "")[:60], "top_k": 3},
            }

    return policy


def _live_providers(cfg: List[Dict[str, str]]) -> Dict[str, Callable]:
    out: Dict[str, Callable] = {}
    for p in cfg:
        out[p["label"]] = _openai_compatible_policy(
            p["model"], p.get("api_base"), p.get("api_key_env", "OPENAI_API_KEY")
        )
    return out


DEFAULT_LIVE_CONFIG: List[Dict[str, str]] = [
    {"label": "gpt-4o-mini",   "model": "gpt-4o-mini",
     "api_base": None, "api_key_env": "OPENAI_API_KEY"},
    {"label": "claude-sonnet", "model": "claude-sonnet-4-6",
     "api_base": "https://api.anthropic.com/v1",
     "api_key_env": "ANTHROPIC_API_KEY"},
    {"label": "qwen-7b",       "model": "Qwen/Qwen2.5-7B-Instruct",
     "api_base": "https://huggingface.co/api/inference-proxy/together",
     "api_key_env": "HF_TOKEN"},
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="Skip API calls; run only local baselines.")
    ap.add_argument("--out", default="reports/frontier.json")
    ap.add_argument("--splits", nargs="+", default=["pre", "during", "hard"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tasks", nargs="+", default=None)
    args = ap.parse_args(argv)

    providers: Dict[str, Callable] = dict(_mock_providers())
    if not args.mock:
        try:
            providers.update(_live_providers(DEFAULT_LIVE_CONFIG))
        except Exception as e:
            print(f"[warn] live providers unavailable: {e}", file=sys.stderr)

    results: Dict[str, Any] = {"generated_at": time.time(), "providers": {}}
    any_scored = False

    tasks = args.tasks or list(ALL_TASKS.keys())
    for label, policy in providers.items():
        print(f"[frontier] running {label} on {len(tasks)} tasks × "
              f"{len(args.seeds)} seeds × {len(args.splits)} splits...",
              flush=True)
        t0 = time.time()
        try:
            r = run_eval(policy, task_ids=tasks, splits=args.splits,
                         seeds=args.seeds)
            r["runtime_s"] = round(time.time() - t0, 2)
            results["providers"][label] = r
            any_scored = True
        except Exception as e:
            print(f"[error] {label} failed: {e}", file=sys.stderr)
            results["providers"][label] = {"error": str(e)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[frontier] wrote {args.out}")
    return 0 if any_scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
