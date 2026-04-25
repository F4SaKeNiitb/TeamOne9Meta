# PROTOCOL-ARENA

> The first OpenEnv RL gym where agents learn to **speak MCP and A2A protocol
> frames under mid-episode schema drift.**

A policy trained in PROTOCOL-ARENA deploys *directly* into Claude Desktop,
Cursor, mcp-cli, or any A2A-compliant client — no adapter layer, because the
action space is literally the wire protocol.

---

## Why this exists

MCP (Model Context Protocol) and A2A (Agent-to-Agent) are now the default
connective tissue between LLMs and the outside world, yet no public RL gym
trains agents to speak them correctly when the schema changes underneath.
Production servers routinely rename tools, tighten required fields, revoke
auth scopes, throttle, or take peers offline. Today's orchestrator LLMs
fail catastrophically on these events.

PROTOCOL-ARENA provides:

1. **Real protocol frames** as the action space (MCPCall, A2ACall, DAG delta,
   KG op, submit)
2. **Seven documented drift classes** firing mid-episode: `additive`,
   `renaming`, `tightening`, `rate_limit`, `agentcard_churn`, `policy`,
   `auth`
3. **A persistent capability knowledge graph** (SQLite+FTS5, union-find
   dedup, BFS retrieval) that lets the agent out-live its context window
4. **Six-signal reward with provenance**: `task_correctness` (0.30),
   `drift_robustness` (0.20), `plan_quality` (0.20), `protocol_hygiene`
   (0.15), `efficiency` (0.10), `memory_hit_bonus` (0.05). Brier
   calibration is reported as a sentinel (not in the reward) and a
   GraphSAGE-style scorer contributes 40% of the plan-quality term.
5. **Adversarial safety layer**: honeypot tools (`delete_all_logs`,
   `export_credentials`, `run_shell`) and prompt-injection snippets
   embedded in search results. Triggering either zeroes hygiene and
   applies a hard reward penalty, and the self-play flywheel refuses
   to ingest unsafe trajectories regardless of reward.
6. **Rewind primitive** — the action space includes `rewind_n` so the
   policy can step back N turns and try a different branch when drift
   bites.
7. **Deterministic replay** — every episode reconstructible from
   `(task_id, seed)`

---

## Task packs (13 tasks shipping)

### Research-synthesis (10 tasks)

| Task | Drift | What's tested |
|---|---|---|
| `research_photo_basic` | none | baseline MCP search+fetch |
| `research_photo_rename` | renaming @ turn 2 | tool gets renamed mid-episode |
| `research_transformer_cite` | additive | new required arg introduced |
| `research_adam_additive` | additive | KB schema gains required field |
| `research_bell_tighten` | tightening | query fields become stricter |
| `research_k8s_rate` | rate_limit | RPM limit tightens |
| `research_bh_churn` | agentcard_churn | citer peer goes offline |
| `research_photo_auth` | auth | `fetch_url` gains auth scope |
| `research_gradient_multi` | — | multi-agent synthesis |
| `research_top_products` | — | SQL-style data warehouse path |

### Consumer-drift (3 tasks, Patronus-aligned)

| Task | Drift | What's tested |
|---|---|---|
| `consumer_policy_pii_search` | policy | PII redaction mid-episode |
| `consumer_rename_plus_policy` | compound | rename + policy simultaneously |
| `consumer_churn_fallback` | agentcard_churn | peer dies → fallback tool |

---

## Action space

```python
OrchestratorAction(
    kind="mcp" | "a2a" | "plan" | "memory" | "submit",
    rationale="...",                          # min 20 chars
    mcp_call=MCPCall(server_id, tool, args),  # if kind == mcp
    a2a_call=A2ACall(agent_card_id, task_spec, stream),
    dag_delta=DAGDelta(add_nodes, remove_nodes, add_edges, remove_edges),
    kg_op=KGOp(op="query|write", pattern, fact, top_k),
    final="...",                              # if kind == submit
)
```

## Observation space

Each step returns `task_spec`, `turn`, `budget`, `last_result`,
`discovered` (tools + peers), `dag_state`, `memory_context` (top-k facts),
`otel_trace_id`, `feedback`, `reward_signals`, `reward`, `done`.

## Reward

Composed at episode end from six weighted signals, each decomposed and
exposed on the final observation for full training transparency. Dense
per-step shaping rewards valid frames, recovery after drift, and memory
reuse.

---

## Quickstart

```bash
pip install -e .
uvicorn arena.server.app:app --host 0.0.0.0 --port 7860

# Run baseline against the running server
ENV_BASE_URL=http://localhost:7860 \
  API_BASE_URL=https://api.openai.com/v1 \
  MODEL_NAME=gpt-4o-mini \
  OPENAI_API_KEY=sk-... \
  python inference.py
```

## Docker

```bash
docker build -t protocol-arena .
docker run -p 7860:7860 protocol-arena
```

## Replay

```bash
arena-replay --trace path/to/episode.json
```

## Live demo (browser spectator)

```bash
python -m arena.ui.spectator_web --port 7861
# then open http://localhost:7861/?task=research_photo_rename&seed=0
```

A self-contained HTML page (no CDN) renders the live DAG, the six
reward signals as bars, and flashes red when drift fires or a
honeypot/injection breach is detected. Reload to start a new episode.

## Frontier eval + leaderboard

```bash
python -m arena.eval.run_frontier --mock              # local floor only
python -m arena.eval.run_frontier                     # hits configured APIs
python -m arena.eval.report --in reports/frontier.json --out reports/
python scripts/score_submission.py reports/frontier.json
```

`SUBMISSION.md` documents the leaderboard formula and submission
schema; `scripts/score_submission.py --emit submission.json …` builds
a valid bundle from a `frontier.json`.

## Self-play flywheel

```bash
python -m arena.training.flywheel \
    --out data/flywheel_iter1.jsonl \
    --seeds 0 1 2 3 4 --threshold 0.55
```

Runs the rule-based and keyword baselines across all tasks × seeds,
filters trajectories with `final_reward ≥ threshold` (and **never**
keeps trajectories that triggered a honeypot or followed an
injection), and emits SFT-format JSONL.

## OpenTelemetry export

If `OTEL_EXPORTER_OTLP_ENDPOINT` is set (e.g. `http://localhost:4318`),
spans fan out to the configured backend (Jaeger / Grafana Tempo /
Honeycomb) in addition to the in-memory tracer used for replay. With
no endpoint configured the in-memory tracer still works — useful for
air-gapped demos.

## Train (Colab)

Open `notebooks/PROTOCOL_ARENA_Colab.ipynb` — SFT bootstrap on oracle
rollouts, then GRPO with TRL + Unsloth 4-bit LoRA on Qwen2.5-7B.

## Test

```bash
pytest tests/
```

---

## Project layout

```
protocol-arena/
├── arena/
│   ├── models.py              # typed action / obs / state
│   ├── client.py              # ProtocolArenaEnv (EnvClient)
│   ├── protocols/
│   │   ├── mcp_harness.py     # in-process MCP server stubs
│   │   ├── a2a_harness.py     # A2A peers with personas
│   │   └── sandbox.py         # AST-checked Python sandbox
│   ├── server/
│   │   ├── arena_env.py       # core state machine
│   │   ├── drift_engine.py    # 7 drift classes
│   │   ├── otel.py            # episode tracing
│   │   ├── replay.py          # deterministic replay CLI
│   │   └── app.py             # FastAPI factory (port 7860)
│   ├── tasks/                 # research + consumer packs
│   ├── memory/
│   │   └── capability_kg.py   # SQLite FTS5 + union-find KG
│   ├── rewards/
│   │   └── signals.py         # 5-signal + dense shaping
│   ├── eval/                  # harness + baselines
│   ├── training/              # SFT bootstrap + GRPO
│   └── ui/                    # spectator (stretch)
├── notebooks/                 # Colab training notebook
├── tests/                     # smoke tests
├── plan.md                    # full strategy doc
├── HYPOTHESIS.md              # testable claims
├── openenv.yaml               # OpenEnv manifest
├── Dockerfile
└── inference.py               # baseline runner
```

---

## Theme alignment

- **Theme 3.1 — World Modeling**: The capability KG *is* a world model of the
  protocol layer; drift mutates the world out from under the policy.
- **Theme 2 — Long-Horizon**: 12-turn episodes with compounding drift force
  persistent memory beyond context.
- **Theme 1 — Multi-Agent**: A2A peers with distinct personas
  (cooperative / cranky / stale).
- **Patronus AI**: Consumer-drift pack mirrors documented LLM agent incidents.
- **Halluminate**: Every reward signal is decomposed, weighted, and exposed.

See `tbd.md` for the full strategy, architecture, and judging rubric
alignment, and `SUBMISSION.md` for the leaderboard contract.

## Reality check on prizes

Public hackathon pages sometimes list partner-track or sponsor-bonus
prizes that are administered separately from the main competition.
This repo's leaderboard formula (in `SUBMISSION.md`) is independent of
those tracks: we score on drift-adjusted correctness, frame validity,
plan quality, and a hard safety penalty — nothing else. Bonus
eligibility for any specific event is decided by that event's
organizers, not by this scorer.
