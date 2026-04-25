"""PROTOCOL-ARENA — Baseline Inference Script.

Drives an OpenAI-compatible LLM against the PROTOCOL-ARENA environment and
emits the mandatory [START] / [STEP] / [END] structured logs.

Env vars:
  API_BASE_URL  - OpenAI-compatible endpoint (default HF inference proxy)
  MODEL_NAME    - e.g. Qwen/Qwen2.5-7B-Instruct
  HF_TOKEN      - auth token
  IMAGE_NAME    - docker image name when connecting via EnvClient.from_docker_image
  ENV_BASE_URL  - already-running server URL (skip docker path)

The agent's action space is real MCP/A2A frames. It talks to the LLM via the
OpenAI client, parses a single JSON action per turn, submits to the env.
"""

import os
import sys
import json
import asyncio
from typing import Any, Dict, List, Optional

from openai import OpenAI


API_BASE_URL = os.getenv("API_BASE_URL", "https://huggingface.co/api/inference-proxy/together")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "protocol_arena")

BENCHMARK = "protocol_arena"

TASK_IDS = [
    "research_photo_basic",
    "research_photo_rename",
    "research_transformer_cite",
    "research_adam_additive",
    "research_bell_tighten",
    "research_bh_churn",
    "consumer_policy_pii_search",
    "consumer_rename_plus_policy",
]

MAX_TURNS = 12
SUCCESS_THRESHOLD = 0.5

llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── Mandatory structured logs ────────────────────────────────────────────────

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    err = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={done}{err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rs = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={success} steps={steps} score={score:.4f} rewards=[{rs}]", flush=True)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the PROTOCOL-ARENA orchestrator. You solve research and consumer tasks by emitting one action per turn as a JSON object.

ACTION SCHEMA (emit ONE action per turn):
{
  "kind": "mcp" | "a2a" | "plan" | "memory" | "submit",
  "rationale": "brief reason, >= 20 chars",

  # when kind == "mcp":
  "mcp_call": {"server_id": "<id>", "tool": "<name>", "args": { ... }},

  # when kind == "a2a":
  "a2a_call": {"agent_card_id": "<id>", "task_spec": "<prompt>"},

  # when kind == "plan":
  "dag_delta": {"add_nodes": [{"id":"n1","op":"search"}], "add_edges":[["n0","n1"]]},

  # when kind == "memory":
  "kg_op": {"op":"query","pattern":"<text>","top_k":5},

  # when kind == "submit":
  "final": "<final answer string>"
}

RULES
1. Always discover first: list available MCP tools and A2A peers via the observation's `discovered`.
2. If a tool returns an error, read the `drift_hint` field and ADAPT your next call.
3. Use `memory` to look up prior observed tool renames or failure history.
4. When confident, emit `submit` with a concise final answer that contains the key facts.
5. Never output text outside the single JSON object. No markdown fences.
"""


def build_user_msg(obs: Dict[str, Any]) -> str:
    disc = obs.get("discovered", {}) or {}
    tools = disc.get("tools", []) or []
    peers = disc.get("peers", []) or []

    tools_brief = "\n".join(
        f"- {t['server_id']}.{t['name']} v{t.get('version','1')}: {t.get('description','')[:80]}"
        for t in tools
    ) or "(no tools)"
    peers_brief = "\n".join(
        f"- {p['id']} [{p.get('persona','?')}] caps={p.get('capabilities',[])}"
        for p in peers
    ) or "(no peers)"

    last = obs.get("last_result") or {}
    last_brief = "(n/a)" if not last else json.dumps(last)[:280]

    mem = obs.get("memory_context", []) or []
    mem_brief = "\n".join(f"- {m['subject']} {m['predicate']} {m['obj']}" for m in mem) or "(empty)"

    budget = obs.get("budget", {}) or {}

    return f"""TASK: {obs.get('task_spec','')}

TURN {obs.get('turn',0)}/{obs.get('max_turns',MAX_TURNS)}   BUDGET calls={budget.get('calls_remaining','?')} tokens={budget.get('tokens_remaining','?')}

AVAILABLE MCP TOOLS:
{tools_brief}

AVAILABLE A2A PEERS:
{peers_brief}

MEMORY CONTEXT (capability KG):
{mem_brief}

LAST RESULT: {last_brief}

FEEDBACK: {obs.get('feedback','')}

Emit ONE JSON action now.""".strip()


def ask_llm(obs: Dict[str, Any]) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_msg(obs)},
    ]
    resp = llm.chat.completions.create(
        model=MODEL_NAME, messages=messages,
        temperature=0.2, max_tokens=600,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return {
            "kind": "memory",
            "rationale": "LLM returned invalid JSON; falling back to a safe memory lookup.",
            "kg_op": {"op": "query", "pattern": obs.get("task_spec", "")[:60], "top_k": 3},
        }


# ── Main driver ──────────────────────────────────────────────────────────────

async def run_task(task_id: str) -> float:
    from arena import ProtocolArenaEnv, OrchestratorAction

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        env_url = os.getenv("ENV_BASE_URL", "")
        if env_url:
            env = ProtocolArenaEnv(base_url=env_url)
        else:
            env = await ProtocolArenaEnv.from_docker_image(IMAGE_NAME)

        async with env:
            result = await env.reset(task_id=task_id)
            obs = _obs_dict(result.observation)

            for step in range(1, MAX_TURNS + 1):
                if result.done:
                    break
                decision = ask_llm(obs)
                action = _coerce_action(decision, OrchestratorAction)
                result = await env.step(action)
                obs = _obs_dict(result.observation)

                steps_taken = step
                rewards.append(result.reward or 0.0)
                log_step(step=step, action=f"kind={decision.get('kind','?')}",
                         reward=rewards[-1], done=result.done)
                if result.done:
                    break

        score = max(0.0, min(1.0, rewards[-1] if rewards else 0.0))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def _obs_dict(obs) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return getattr(obs, "__dict__", {})


def _coerce_action(decision: Dict[str, Any], action_cls):
    d = dict(decision)
    d.setdefault("rationale", "auto: fallback rationale provided by client.")
    if len(d["rationale"]) < 20:
        d["rationale"] = (d["rationale"] + " " * 20)[:40]
    d.setdefault("kind", "memory")
    # prune fields not matching kind so Pydantic validation stays lenient
    valid_keys = {"kind", "rationale", "mcp_call", "a2a_call", "dag_delta", "kg_op", "final"}
    d = {k: v for k, v in d.items() if k in valid_keys}
    return action_cls(**d)


async def main():
    print(f"[INFO] PROTOCOL-ARENA Baseline", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Tasks: {TASK_IDS}", flush=True)

    scores: Dict[str, float] = {}
    for tid in TASK_IDS:
        scores[tid] = await run_task(tid)

    print("\n[SUMMARY] Per-task:", flush=True)
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}", flush=True)
    avg = sum(scores.values()) / max(1, len(scores))
    print(f"  average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
