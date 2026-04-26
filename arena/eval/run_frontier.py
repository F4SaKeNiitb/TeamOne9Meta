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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .harness import run_eval
from .baselines import random_policy, rule_based_policy, keyword_policy
from ..tasks import ALL_TASKS

# Single lock to keep log lines from interleaving across worker threads.
_PRINT_LOCK = threading.Lock()
# Separate lock around the JSON checkpoint file so concurrent providers
# can't tear each other's writes.
_SAVE_LOCK = threading.Lock()


def _log(msg: str, err: bool = False) -> None:
    with _PRINT_LOCK:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _save_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    """Atomic JSON write: write to .tmp then os.replace. Safe under
    concurrent callers via _SAVE_LOCK."""
    with _SAVE_LOCK:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp, path)


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
    _log(f"[preflight] {label:<16} model={model_name}  "
         f"base={api_base or 'openai-default'}  "
         f"key_env={api_key_env}={_mask(key)}")
    if not key:
        _log(f"[preflight] {label}: ⚠️  {api_key_env} is EMPTY — every call will fall back",
             err=True)

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
                _log(f"[error] {label}: first API error → {counters['first_err']}",
                     err=True)
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
            _log(f"[debug] {label}: first response sample → {raw[:120]!r}")

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


def _attach_api_calls(block: Dict[str, Any], policy: Callable,
                      label: str) -> None:
    """Mutate `block` to include the api_calls counter snapshot for this
    provider, plus log a one-line summary."""
    counters = getattr(policy, "_counters", None)
    if counters is None:
        _log(f"[summary] {label}: local baseline  runtime={block.get('runtime_s','?')}s")
        return
    total = counters["api_ok"] + counters["api_err"] + counters["json_err"]
    hit_rate = counters["api_ok"] / total if total else 0.0
    block["api_calls"] = {
        "ok": counters["api_ok"],
        "api_error": counters["api_err"],
        "json_error": counters["json_err"],
        "total": total,
        "hit_rate": round(hit_rate, 3),
        "tokens_in": counters["tokens_in"],
        "tokens_out": counters["tokens_out"],
        "first_error": counters["first_err"],
    }
    _log(f"[summary] {label}: api_ok={counters['api_ok']}  "
         f"api_err={counters['api_err']}  json_err={counters['json_err']}  "
         f"hit_rate={hit_rate:.1%}  "
         f"tokens={counters['tokens_in']}→{counters['tokens_out']}  "
         f"runtime={block.get('runtime_s','?')}s")
    if total == 0:
        _log(f"[summary] {label}: ⚠️  ZERO API calls — check key/network",
             err=True)
    elif hit_rate < 0.5:
        _log(f"[summary] {label}: ⚠️  hit_rate < 50% — mostly fallbacks, "
             f"NOT a real frontier baseline", err=True)


def _run_one_provider(label: str, policy: Callable, tasks: List[str],
                      splits: List[str], seeds: List[int],
                      cached: Optional[Dict[str, Any]] = None,
                      save_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None
                      ) -> Dict[str, Any]:
    """Run one provider's full sweep, looping per-split so that a crash
    mid-sweep keeps any already-finished split's results.

    Args:
      cached: previously-saved block for this provider; any
        `eval_<split>` keys present here cause that split to be skipped.
      save_cb(label, block): invoked after each split (and on error)
        with the latest block snapshot, giving the caller a chance to
        persist intermediate results.

    Safe to call from a worker thread — `run_eval` allocates its own
    ProtocolArenaEnvironment per call, the OpenAI client is thread-safe,
    and the policy's `_counters` dict is mutated only by this worker.
    """
    cached = cached or {}
    block: Dict[str, Any] = {k: v for k, v in cached.items() if k != "error"}

    cached_splits = [s for s in splits if f"eval_{s}" in cached]
    todo_splits   = [s for s in splits if f"eval_{s}" not in cached]
    if cached_splits:
        _log(f"[frontier] === {label} === resuming "
             f"({len(cached_splits)} cached: {cached_splits} | "
             f"{len(todo_splits)} todo: {todo_splits})")
    else:
        _log(f"[frontier] === {label} === "
             f"{len(tasks)} tasks × {len(seeds)} seeds × {len(splits)} splits  (started)")

    t0 = time.time()
    for split in todo_splits:
        t_split = time.time()
        try:
            r = run_eval(policy, task_ids=tasks, splits=[split], seeds=seeds)
        except Exception as e:
            _log(f"[error] {label}/{split} crashed: {type(e).__name__}: {e}",
                 err=True)
            block["error"] = f"{type(e).__name__}: {e} (during split={split})"
            block["runtime_s"] = round(time.time() - t0, 2)
            _attach_api_calls(block, policy, label)
            if save_cb:
                save_cb(label, dict(block))
            return block
        block.update(r)
        block["runtime_s"] = round(time.time() - t0, 2)
        _log(f"[checkpoint] {label}: split={split} done in "
             f"{round(time.time()-t_split,2)}s — saving partial")
        if save_cb:
            save_cb(label, dict(block))

    # Compute drift_adjusted_success_rate AFTER all splits are merged.
    # `harness.run_eval` only computes DAS when both eval_pre and eval_during
    # are present in the same call's report — but our per-split loop above
    # calls run_eval one split at a time, so DAS would otherwise never get
    # populated. Recompute it here from the merged block.
    pre    = block.get("eval_pre",    {}).get("task_correctness", {}).get("mean")
    during = block.get("eval_during", {}).get("task_correctness", {}).get("mean")
    if pre is not None and during is not None:
        block["drift_adjusted_success_rate"] = {
            "value": round(max(0.0, 1.0 - max(0.0, pre - during)), 3),
            "pre_mean": pre, "during_mean": during,
        }

    _attach_api_calls(block, policy, label)
    if save_cb:
        save_cb(label, dict(block))
    return block


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="Skip API calls; run only local baselines.")
    ap.add_argument("--out", default="reports/frontier.json")
    ap.add_argument("--splits", nargs="+", default=["pre", "during", "hard"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent providers (default 4). Set 1 to run "
                         "serially. Local baselines always run sequentially "
                         "first since they're CPU-bound and fast.")
    ap.add_argument("--resume", action="store_true",
                    help="If --out exists, load it and skip any "
                         "(provider, split) combo already present. "
                         "Lets you recover from a crash mid-sweep without "
                         "re-spending API budget on finished work.")
    args = ap.parse_args(argv)

    local: Dict[str, Callable] = dict(_mock_providers())
    live: Dict[str, Callable] = {}
    if not args.mock:
        try:
            live = _live_providers(DEFAULT_LIVE_CONFIG)
        except Exception as e:
            _log(f"[warn] live providers unavailable: {e}", err=True)

    # Resume: load any existing partial results so we can skip finished splits.
    cached_providers: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.out):
        try:
            prior = json.load(open(args.out))
            cached_providers = prior.get("providers", {}) or {}
            cached_summary = {
                lbl: [s for s in args.splits if f"eval_{s}" in (blk or {})]
                for lbl, blk in cached_providers.items()
            }
            _log(f"[resume] loaded {args.out}: {cached_summary}")
        except Exception as e:
            _log(f"[resume] failed to load {args.out}: {e} — starting fresh",
                 err=True)

    results: Dict[str, Any] = {"generated_at": time.time(),
                               "providers": dict(cached_providers)}
    any_scored = False
    tasks = args.tasks or list(ALL_TASKS.keys())
    order = list(local.keys()) + list(live.keys())  # for stable summary order

    def _save_cb(label: str, block: Dict[str, Any]) -> None:
        """Persist one provider's latest block + emit stable-ordered JSON.
        Called after each split completes (or on crash)."""
        results["providers"][label] = block
        results["generated_at"] = time.time()
        ordered = {k: results["providers"][k]
                   for k in order if k in results["providers"]}
        _save_json_atomic(args.out, {
            "generated_at": results["generated_at"],
            "providers": ordered,
        })

    # 1. Locals run sequentially — fast, CPU-bound, no API limits to spread.
    for label, policy in local.items():
        r = _run_one_provider(label, policy, tasks, args.splits, args.seeds,
                              cached=cached_providers.get(label),
                              save_cb=_save_cb)
        results["providers"][label] = r
        if "error" not in r:
            any_scored = True

    # 2. Live providers run concurrently — different services, no shared
    # rate limit. Wall time = max(per-provider time) instead of sum.
    if live:
        max_workers = max(1, min(args.workers, len(live)))
        _log(f"[frontier] launching {len(live)} live providers across "
             f"{max_workers} threads (concurrent)")
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="provider") as ex:
            futures = {
                ex.submit(_run_one_provider, label, policy,
                          tasks, args.splits, args.seeds,
                          cached_providers.get(label), _save_cb): label
                for label, policy in live.items()
            }
            for fut in as_completed(futures):
                label = futures[fut]
                r = fut.result()
                results["providers"][label] = r
                if "error" not in r:
                    any_scored = True

    # Re-emit providers in the original order so downstream consumers see
    # a stable layout regardless of completion timing.
    results["providers"] = {k: results["providers"][k]
                            for k in order if k in results["providers"]}

    _save_json_atomic(args.out, results)

    # Final cross-provider summary so a one-glance check tells you
    # which providers actually ran.
    _log(f"\n[frontier] wrote {args.out}")
    _log(f"[frontier] === final summary ===")
    for label, r in results["providers"].items():
        if "error" in r:
            _log(f"  {label:<18} CRASH: {r['error'][:80]}")
        elif "api_calls" in r:
            ac = r["api_calls"]
            tag = "OK" if ac["hit_rate"] >= 0.5 else "FALLBACK"
            _log(f"  {label:<18} {tag:<8} hit_rate={ac['hit_rate']:.0%}  "
                 f"calls={ac['total']}  tokens={ac['tokens_in']}+{ac['tokens_out']}  "
                 f"runtime={r.get('runtime_s','?')}s")
        else:
            _log(f"  {label:<18} LOCAL    runtime={r.get('runtime_s','?')}s")
    return 0 if any_scored else 1


if __name__ == "__main__":
    raise SystemExit(main())
