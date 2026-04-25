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


def _mask(key: str) -> str:
    if not key:
        return "<empty>"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}…{key[-4:]} (len={len(key)})"


def _openai_compatible_policy(model_name: str,
                              api_base: Optional[str] = None,
                              api_key_env: str = "OPENAI_API_KEY",
                              label: str = "?") -> Callable:
    """Return a policy fn that calls an OpenAI-compatible chat endpoint.

    Prints visible diagnostics so silent failures are impossible:
      - preflight: confirms the API key env var is set
      - per-call: counts api_ok / json_parse_fail / api_error / fallback
      - prints the FIRST error verbatim so you see what's wrong
    """
    from openai import OpenAI

    key = os.getenv(api_key_env, "")
    _assert_remote(api_base, label)
    print(f"[preflight] {label:<16} model={model_name}  "
          f"base={api_base or 'openai-default'}  "
          f"key_env={api_key_env}={_mask(key)}", flush=True)
    if not key:
        print(f"[preflight] {label}: ⚠️  {api_key_env} is EMPTY — every call will fall back",
              file=sys.stderr, flush=True)

    client = OpenAI(api_key=key, base_url=api_base) if api_base else OpenAI(api_key=key)

    import inference  # the repo-root script; reuses the prompt exactly

    counters = {"api_ok": 0, "api_err": 0, "json_err": 0,
                "first_err": None, "first_raw_sample": None,
                "tokens_in": 0, "tokens_out": 0}

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
        except Exception as e:
            counters["api_err"] += 1
            if counters["first_err"] is None:
                counters["first_err"] = f"{type(e).__name__}: {e}"
                print(f"[error] {label}: first API error → {counters['first_err']}",
                      file=sys.stderr, flush=True)
            return {
                "kind": "memory",
                "rationale": f"provider_error_fallback: {type(e).__name__}",
                "kg_op": {"op": "query",
                          "pattern": obs.get("task_spec", "")[:60], "top_k": 3},
            }

        try:
            usage = getattr(resp, "usage", None)
            if usage is not None:
                counters["tokens_in"]  += getattr(usage, "prompt_tokens", 0) or 0
                counters["tokens_out"] += getattr(usage, "completion_tokens", 0) or 0
        except Exception:
            pass

        raw = (resp.choices[0].message.content or "").strip()
        if counters["first_raw_sample"] is None:
            counters["first_raw_sample"] = raw[:200]
            print(f"[debug] {label}: first response sample → {raw[:120]!r}",
                  flush=True)

        try:
            if raw.startswith("```"):
                raw_inner = raw.split("```")[1]
                if raw_inner.startswith("json"):
                    raw_inner = raw_inner[4:]
                parsed = json.loads(raw_inner.strip())
            else:
                parsed = json.loads(raw)
            counters["api_ok"] += 1
            return parsed
        except Exception as e:
            counters["json_err"] += 1
            if counters["first_err"] is None:
                counters["first_err"] = f"JSONParseError: {e} | raw={raw[:200]!r}"
                print(f"[error] {label}: first JSON parse error → "
                      f"{counters['first_err']}", file=sys.stderr, flush=True)
            return {
                "kind": "memory",
                "rationale": "json_parse_fallback",
                "kg_op": {"op": "query",
                          "pattern": obs.get("task_spec", "")[:60], "top_k": 3},
            }

    policy._counters = counters    # type: ignore[attr-defined]
    policy._label = label          # type: ignore[attr-defined]
    return policy


def _live_providers(cfg: List[Dict[str, str]]) -> Dict[str, Callable]:
    out: Dict[str, Callable] = {}
    for p in cfg:
        out[p["label"]] = _openai_compatible_policy(
            p["model"], p.get("api_base"),
            p.get("api_key_env", "OPENAI_API_KEY"),
            label=p["label"],
        )
    return out


DEFAULT_LIVE_CONFIG: List[Dict[str, str]] = [
    {"label": "gpt-4o-mini",     "model": "gpt-4o-mini",
     "api_base": None, "api_key_env": "OPENAI_API_KEY"},
    {"label": "claude-haiku-4-5", "model": "claude-haiku-4-5-20251001",
     "api_base": "https://api.anthropic.com/v1",
     "api_key_env": "ANTHROPIC_API_KEY"},
]

# Qwen2.5-1.5B-Instruct is too small for HF's free router or Together to
# bother hosting. Evaluate it INSIDE Colab (Track A) where the GPU is
# already loaded, then merge results via scripts/merge_colab_eval.py.
# That keeps the apples-to-apples base-vs-trained comparison while
# guaranteeing the laptop never loads weights.


def _assert_remote(api_base: Optional[str], label: str) -> None:
    """Raise loudly if a config would cause a LOCAL model load.

    The OpenAI client itself never loads weights, but if someone swaps in
    a transformers-based wrapper, this guard fails fast instead of silently
    pulling 3 GB of safetensors onto a laptop.
    """
    if api_base is None:
        # Implicit OpenAI cloud endpoint — fine.
        return
    if api_base.startswith(("http://", "https://")):
        return
    raise RuntimeError(
        f"[guard] {label}: api_base={api_base!r} is not a remote URL. "
        "Refusing to run — every provider in DEFAULT_LIVE_CONFIG must be "
        "served over HTTPS to avoid local model loads on the laptop."
    )


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
        print(f"\n[frontier] === {label} === "
              f"{len(tasks)} tasks × {len(args.seeds)} seeds × {len(args.splits)} splits",
              flush=True)
        t0 = time.time()
        try:
            r = run_eval(policy, task_ids=tasks, splits=args.splits,
                         seeds=args.seeds)
            r["runtime_s"] = round(time.time() - t0, 2)

            # If this was a live provider, attach call counters & flag
            # the case where every "successful" episode was actually a
            # silent fallback.
            counters = getattr(policy, "_counters", None)
            if counters is not None:
                total = counters["api_ok"] + counters["api_err"] + counters["json_err"]
                hit_rate = counters["api_ok"] / total if total else 0.0
                r["api_calls"] = {
                    "ok": counters["api_ok"],
                    "api_error": counters["api_err"],
                    "json_error": counters["json_err"],
                    "total": total,
                    "hit_rate": round(hit_rate, 3),
                    "tokens_in": counters["tokens_in"],
                    "tokens_out": counters["tokens_out"],
                    "first_error": counters["first_err"],
                }
                print(f"[summary] {label}: api_ok={counters['api_ok']}  "
                      f"api_err={counters['api_err']}  json_err={counters['json_err']}  "
                      f"hit_rate={hit_rate:.1%}  "
                      f"tokens={counters['tokens_in']}→{counters['tokens_out']}  "
                      f"runtime={r['runtime_s']}s", flush=True)
                if total == 0:
                    print(f"[summary] {label}: ⚠️  ZERO API calls were made — "
                          f"check key/network", file=sys.stderr, flush=True)
                elif hit_rate < 0.5:
                    print(f"[summary] {label}: ⚠️  hit_rate < 50% — "
                          f"results are mostly fallbacks, NOT a real frontier baseline",
                          file=sys.stderr, flush=True)
            else:
                print(f"[summary] {label}: local baseline  runtime={r['runtime_s']}s",
                      flush=True)

            results["providers"][label] = r
            any_scored = True
        except Exception as e:
            print(f"[error] {label} crashed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            results["providers"][label] = {"error": f"{type(e).__name__}: {e}"}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Final cross-provider summary so a one-glance check tells you
    # which providers actually ran.
    print(f"\n[frontier] wrote {args.out}")
    print(f"[frontier] === final summary ===")
    for label, r in results["providers"].items():
        if "error" in r:
            print(f"  {label:<14} CRASH: {r['error'][:80]}")
        elif "api_calls" in r:
            ac = r["api_calls"]
            tag = "OK" if ac["hit_rate"] >= 0.5 else "FALLBACK"
            print(f"  {label:<14} {tag:<8} hit_rate={ac['hit_rate']:.0%}  "
                  f"calls={ac['total']}  tokens={ac['tokens_in']}+{ac['tokens_out']}")
        else:
            print(f"  {label:<14} LOCAL    runtime={r.get('runtime_s','?')}s")
    return 0 if any_scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
