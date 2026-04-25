# PROTOCOL-ARENA — Winning Plan
### Meta PyTorch Hackathon 2026 · Finals Strategy Document
_Author: Manish Shaw · Updated: 2026-04-21 · Onsite: Apr 25–26_

---

## 0. How to Read This Document

This is not a spec. It is a **battle plan**. Read it start-to-finish once tonight, then keep it open as your north star through onsite. Section 1 is the thesis. Section 2 is the moat. Sections 3–10 are the environment, reward, and training design. Sections 11–13 are how you beat other teams, not just how you build the thing. Sections 14–17 are execution: day-by-day plan, pitch, Q&A, risks. The appendix (Section 18) lists the high-leverage additions that were layered onto the core idea — these are the details that separate "good hackathon project" from "finalist who wins."

If you remember nothing else: **the edge here is protocols, not prompts.** Every other team will demo a smarter agent. You will demo an agent that speaks the exact wire format production agent systems run on, under the exact failure mode they hit. That difference is the entire game.

---

## 1. Thesis

> Other teams will train agents to solve harder tasks. We will train an agent to stay correct when the world around it mutates — specifically, when the real protocols production agents run on (MCP and A2A) drift mid-episode. A policy trained in PROTOCOL-ARENA deploys directly into Claude Desktop, Cursor, mcp-cli, or any A2A-compliant client. No adapter, no shim. That is a story a judge will repeat to another judge, which is how finalist projects become winners.

The single sentence to memorize: **"We built the first RL gym where the action space is real MCP and A2A protocol frames, the adversary is schema drift, and the trained policy drops straight into your editor."**

---

## 2. The Moat — Why You, Not Them

Every finalist used Claude to plan. So the 40% innovation score is decided not by who ran Claude, but by who picks primitives Claude does not default-suggest. Claude's gravity well for agent-RL hackathons pulls toward: Diplomacy-style negotiation, long-horizon coding refactor, personal-assistant email, self-play math proofs, and oversight/deception detection. Assume two to four other teams land on each of those.

Your differentiation vector has three components, layered:

**Layer 1 — Protocols over prompts.** MCP and A2A are 2024-26's most important agent-infra protocols. Anthropic ships MCP; Google ships A2A. They are not in Claude's default suggestion cloud because models hedge toward generic "tool use," not protocol-native environments. You have direct production experience with both: MCP over stdio with dynamic discovery and LangChain adapters; A2A via JSON-RPC 2.0, AgentCard discovery, and SSE task streaming. Nobody at that finals has both, by accident or on purpose.

**Layer 2 — Schema drift as adversary.** Patronus AI's sub-theme literally says "Consumer Workflows with Schema Drift." No other failure mode lines up with a sub-theme prize this cleanly. Schema drift is also the real production failure mode of MCP/A2A ecosystems — which is why layering it on top of Layer 1 is not a stretch, it is the natural coupling.

**Layer 3 — Graph intuition in the reward.** Your iDEA and Inter-IIT graph work means you are one of a handful of finalists who can ship a GNN-based reward model that scores the structural quality of the agent's call graph. Most teams will use scalar rewards because that's what TRL tutorials show. A structural reward signal is a 40%-innovation-category bullet by itself.

These three layers compound. Any one alone is notable. Together they form an environment architecture that is nearly impossible to reproduce without your specific CV.

---

## 3. Theme Alignment and Bonus Coverage

| Theme / Sub-theme | Fit | Role in Plan |
|---|---|---|
| Theme 3.1 — Professional Tasks (World Modeling) | Primary | Tool-using agent in a drifting protocol ecosystem |
| Theme 2 — Long-Horizon Planning | Secondary | Persistent capability KG beyond context window |
| Theme 1 — Multi-Agent Interactions | Secondary | A2A peer delegation and AgentCard discovery |
| Patronus AI — Schema Drift | **Bonus Prize** | Direct literal match |
| Halluminate — Multi-Actor Management | **Bonus Prize** | Orchestrator managing N tools + M peers |
| Mercor — Capped/Uncapped Reward Scaling | **Bonus Prize (opt-in)** | Log-linear rationale-length bonus on plan-quality head |
| Scaler AI Labs — Enterprise Workflows | Tertiary | Enterprise task pack shipped if time allows |

Three bonus-prize pockets are reachable from one environment. This is unusually efficient prize coverage and should be a deliberate argument in the pitch.

---

## 4. The Problem in Plain English

Modern agents do not fail in production because they cannot reason. They fail because the surface around them — tool schemas, rate limits, auth scopes, agent capabilities, T&Cs, policies — changes faster than the agent adapts. A tool that accepted `query` last week now requires `sql_query`. An A2A peer that answered in 200ms now returns 429s. A MCP server that exposed six tools now exposes four. The agent does not know this until a call fails, and when it fails, the agent either gives up or hallucinates a fix.

PROTOCOL-ARENA is an RL environment that trains an orchestrator LLM to behave correctly under exactly this regime. The agent's actions are real MCP and A2A protocol frames. A Schema Drift Engine mutates the protocol surface at calibrated moments during an episode. The agent must learn: discover, validate, attempt, recover, delegate, remember. The trained policy is then directly deployable into any MCP/A2A-compliant client, because the action space is the wire format.

---

## 5. Environment Design

### 5.1 Shape of an Episode

An episode is a **task spec** delivered to the orchestrator, a **budget** (tokens, tool-calls, latency), a **set of discoverable MCP servers and A2A peers**, and a **drift schedule**. The episode runs in up to twelve turns. On each turn the orchestrator emits one action; the environment applies it; the environment mutates state (including schema drift, if scheduled for this turn); the environment returns an observation and a dense reward. The episode ends on `submit`, on budget exhaustion, or on a hard failure threshold.

### 5.2 Three Task Packs

Ship two, add the third if Day 3 is on schedule:

| Pack | Tasks | Theme Hit |
|---|---|---|
| Research synthesis | Discover-fetch-extract-cross-reference-summarize across 4–8 tool calls | Long-horizon + multi-tool |
| Consumer workflow with drift | Travel rebook / shopping return across providers whose policies shift mid-episode | Patronus sub-theme |
| Enterprise workflow | CRM-update → ticketing → notification via 3 MCP servers + 2 A2A peers | Scaler sub-theme |

All seed data is synthetic or public. None of it comes from prior employers or projects.

### 5.3 Protocol Surface (the action space itself)

Most RL-agent environments serialize tools as JSON function specs. PROTOCOL-ARENA does not. It runs real MCP servers over stdio transport and real A2A endpoints over JSON-RPC 2.0 plus SSE. The orchestrator's actions are literal protocol messages:

```
MCP family:
  list_tools                          # per server
  call_tool(server_id, tool, args)    # canonical MCP call
  list_resources / read_resource(uri) # resource-style tools
  subscribe(resource)                 # streaming resources

A2A family:
  discover_agents()                   # returns list of AgentCards
  send_task(agent_id, task_spec)      # synchronous dispatch
  stream_task(task_id)                # SSE subscription to peer updates
  cancel_task(task_id)

Planning / memory family:
  compose_dag(nodes, edges)           # emit / update a LangGraph-style DAG
  checkpoint(state_id)                # durable save of partial state
  resume(state_id)                    # reload a prior checkpoint
  kg_query(pattern)                   # BFS over the persistent capability graph
  kg_write(fact)                      # commit a learned fact to long-term memory

Finalization:
  submit(answer)                      # deliver the final result, ends episode
```

Because these are the actual wire-format frames, a policy trained here is deployable into any MCP client on Day 7 with no adapter. This is an unusually strong deployability story for a three-minute pitch.

### 5.4 The Schema Drift Engine

The drift engine is a deterministic mutator driven by a YAML-configurable schedule. Each drift class maps to a well-documented failure in real MCP/A2A deployments, so you can cite source spec issues in the repo README:

| Drift Class | Example | Signal the agent receives |
|---|---|---|
| Additive | `call_tool` grows a required `schema_version` field | 422 validation error with JSON-Schema diff |
| Renaming | `query` → `sql_query`; old name deprecated for N turns, then removed | Deprecation warning, later a hard 404 |
| Tightening | `body: string` → `body: regex`-constrained | 422 with constraint explanation |
| Rate-limit shift | 10 rpm → 2 rpm on a popular tool | 429 with `Retry-After` |
| AgentCard churn | A2A peer disappears or loses a capability | `discover_agents()` returns a smaller set |
| Policy drift | T&C now requires PII redaction on `body` | Server rejects with `policy_violation` |
| Auth drift | A new OAuth scope is required | 401 with `www_authenticate` header |

The same task may be repeated pre-drift, during-drift, and post-drift. A naive agent's reward collapses at the drift boundary. A trained agent holds near its pre-drift reward, which is the headline metric for the demo.

### 5.5 Multi-Agent Surface

The A2A peer population includes three persona classes:
- **Cooperative specialists** — answer truthfully, respect rate limits.
- **Cranky specialists** — timeout often, return partial results, require retries.
- **Stale specialists** — their AgentCard advertises capabilities they no longer support; discovery lies.

No adversarial/deceptive peers in v1 — keep oversight/deception out of scope to preserve focus. The cranky/stale personas are enough to make A2A non-trivial while keeping the core story about protocol robustness, not collusion detection.

---

## 6. Action and Observation Schema

Reuse the Pydantic pattern from the existing EvalForge code in this repo. The action and observation shapes should be narrow and typed so TRL's GRPO loss does not waste probability mass on malformed frames.

```python
class OrchestratorAction:
    kind: Literal["mcp", "a2a", "plan", "memory", "submit"]
    mcp_call:    Optional[MCPCall]       # {server_id, tool, args}
    a2a_call:    Optional[A2ACall]       # {agent_card_id, task_spec}
    dag_delta:   Optional[DAGDelta]      # adds/removes nodes & edges
    kg_op:      Optional[KGOp]           # query or write on capability KG
    final:      Optional[str]            # submit
    rationale:  str                      # always required, ≥30 chars

class OrchestratorObservation:
    task_spec:       str
    turn:            int
    budget:          BudgetBreakdown     # tokens, calls, latency remaining
    last_result:     ProtocolResult      # success | error + drift_hint
    discovered:      DiscoverySnapshot   # currently visible tools + peers
    dag_state:       DAGSnapshot         # current plan and checkpoints
    memory_context:  List[KGFact]        # top-k BFS results from capability KG
    otel_trace_id:   str                 # every observation is traceable
    feedback:        str                 # directional hint, no ground truth
    reward:          float
    reward_signals:  Dict[str, float]    # provenance — see §7
    done:            bool
```

Every observation carries its reward decomposition (`reward_signals`) so rewards are debuggable during training and visually transparent in the demo UI. This is a small but distinctive touch — judges who have trained RL know the pain of opaque reward.

---

## 7. Reward Model

A single scalar reward would be easier to implement and would leave you with a less competitive pitch. Use a **five-signal weighted average with full provenance**, mirroring the confidence-scoring pattern you already built in your Knowledge Graph project. Every reward emission is decomposed and logged, enabling reward debugging, calibration plots, and an ablation table in the final blog.

```
R  =  0.30 · task_correctness        # ground-truth oracle per task
   +  0.20 · drift_robustness        # Δ(success | post-drift) − (success | pre-drift)
   +  0.20 · plan_quality            # GNN score over the execution DAG
   +  0.15 · protocol_hygiene        # valid protocol frames / total frames
   +  0.10 · efficiency              # budget_remaining / budget_initial
   +  0.05 · memory_hit_bonus        # reused a KG fact correctly
```

### 7.1 The GNN Plan-Quality Head

This is the signal nobody else will ship. The orchestrator's emitted DAG is embedded via a small GraphSAGE encoder with DeepWalk-initialized node embeddings. Node features: tool category, retry count, fan-out, critical-path depth, cross-server dependency count. The encoder is trained once, offline, on 200 hand-authored expert DAGs covering the target task packs. At training time, the encoder is frozen and scores the agent's DAG against this prior.

If Day 3 slips, fall back to a rule-based plan score: parallelism ratio, maximum depth, retry overhead, cycle penalty. This ships in two hours and still gives a coherent structural signal.

### 7.2 Mercor-Style Uncapped Bonus (Optional, One-Line Toggle)

A log-linear bonus on the `plan_quality` head scaling with rationale tokens emitted, with an optional hard cap via env flag. This is a one-flag opt-in that directly claims the Mercor sub-theme prize.

### 7.3 Calibration Metric (Not Trained On, But Reported)

Track the **Brier score** of the agent's self-reported confidence against task correctness across eval episodes. This is not part of the reward; it is a reporting metric that appears in the results blog. It costs almost nothing to add and it signals alignment-aware engineering to judges.

---

## 8. Agent Memory Architecture

The capability KG is how you claim Theme 2 (long-horizon beyond context). It is the structural answer to "what does the agent remember between episodes?"

Nodes:
- MCP tool versions (hashed by spec)
- A2A peer cards
- Observed schemas and their historic revisions

Edges:
- `derived_from`, `supersedes` (version lineage)
- `calls` (observed call relationships)
- `failed_with` (failure mode history)
- `substitutable_for` (semantic equivalence)

Retrieval at inference time is BFS bounded to three hops from the current task's seed entities. Dedup layers:
1. **Artifact-level** — SHA-256 on the canonical JSON of the spec.
2. **Entity-level** — Union-Find over rename chains (handles deprecation sequences).
3. **Claim-level** — dense embeddings for semantic capability equivalence.

Implementation: start with SQLite plus FTS5 for the keyword side and a sentence-transformer index for the semantic side. Upgrade to Neo4j only if Day 4 has slack. The hybrid BM25+dense retriever pattern you built at Longani ports directly.

---

## 9. Training Strategy

### 9.1 Phase A — Protocol Grammar Bootstrap (offline, before onsite)

Run an SFT pass of roughly 1.5k–2k synthetic rollouts where a scripted "oracle orchestrator" emits correct MCP/A2A frames for each task. Goal: teach the model the frame grammar so later GRPO rollouts are not burned on malformed frames.

- Base model: Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct.
- Adapter: LoRA rank 16, alpha 32, via **Unsloth 4-bit**.
- Compute: one A100 for ~90 minutes, or Colab Pro.
- Exit criterion: frame validity ≥ 0.92 on 200 held-out tasks.

### 9.2 Phase B — GRPO Under Static Drift (onsite Day 1)

- TRL `GRPOTrainer`, group size 8.
- Reward: §7 formula with drift schedule fixed per episode.
- Exit criterion: task correctness ≥ 0.6, frame validity ≥ 0.95.

### 9.3 Phase C — Reactive Drift Curriculum (onsite Day 1 PM → Day 2 AM)

Mutate the drift schedule based on the agent's tool-usage histogram: drift the most-used tools preferentially. Mirrors real production, where the tools you use most are the ones that break most. **This is the curve that visibly bends upward in the pitch.**

### 9.4 Phase D — Stretch: Peer Persona Co-Training

Briefly train the cranky/stale A2A personas to be harder to work with, then re-eval the orchestrator. If time permits only.

### 9.5 Phase E — Stretch: Distillation to a Tiny Model

Distill the LoRA-adapted policy into a Qwen2.5-1.5B student. If successful, report deployability at edge-device scale. Differentiation-heavy, low-implementation-risk if Phase C lands.

---

## 10. Evaluation Methodology

This is where most teams lose points. Ship a proper eval harness.

**Three fixed eval splits:**
- `eval_pre_drift` — drift off; measures clean task-solving.
- `eval_during_drift` — drift schedule active; measures adaptation.
- `eval_hard` — held-out drift combinations never seen in training; measures generalization.

**Five reported metrics:**
1. Task correctness (per pack, macro-averaged).
2. Drift-adjusted success rate (`eval_during_drift` success rate).
3. Frame validity (fraction of emitted frames the server accepts without a format error).
4. Plan-quality score (from the GNN head).
5. Brier calibration score.

**Zero-shot baselines to beat:**
- Base Qwen2.5-7B with no training.
- Claude 3.5 Sonnet / Claude Sonnet 4.5 zero-shot via API.
- GPT-4o-mini zero-shot via API.

If your 7B trained agent outperforms any frontier model on `eval_during_drift` or frame validity, **that is your pitch's climax.** Even matching GPT-4o on drift-adjusted metrics is a shareable result. Spend API budget on running these baselines cleanly — they cost less than $20 total.

---

## 11. Technical Architecture

Reuse the existing EvalForge skeleton already in this repo. Structural layout:

```
PROTOCOL-ARENA/
├── client/
│   └── arena_env.py                  # OpenEnv WebSocket client
├── server/
│   ├── app.py                        # FastAPI create_app, port 7860
│   ├── arena_env.py                  # env state machine
│   ├── drift_engine.py               # YAML-driven mutation schedule
│   ├── otel.py                       # OpenTelemetry tracer per episode
│   └── replay.py                     # deterministic replay from seed
├── protocols/
│   ├── mcp_harness/                  # real MCP stdio subprocess manager
│   ├── a2a_harness/                  # JSON-RPC 2.0 + SSE endpoint
│   └── sandbox.py                    # AST-checked python exec tool
├── tasks/
│   ├── research/
│   ├── consumer_drift/
│   └── enterprise/                   # stretch
├── memory/
│   ├── capability_kg.py              # SQLite+FTS5 / optional Neo4j
│   ├── dedup.py                      # SHA-256 / Union-Find / semantic
│   └── retriever.py                  # BM25 + dense + FlashRank
├── rewards/
│   ├── signals.py                    # five heads with provenance
│   ├── gnn_plan_scorer.py            # GraphSAGE + DeepWalk
│   └── rubric_judge.py               # LLM-judge on rationale quality
├── training/
│   ├── sft_bootstrap.py              # Phase A
│   ├── grpo_main.py                  # Phase B and C
│   ├── selfplay_loop.py              # stretch Phase D
│   └── distill.py                    # stretch Phase E
├── eval/
│   ├── harness.py                    # three splits, five metrics
│   ├── baselines.py                  # frontier zero-shot
│   └── report.py                     # generates the results markdown + plots
├── ui/
│   ├── spectator.py                  # WebSocket live episode viewer
│   └── static/                       # minimal HTML+Alpine.js
├── notebooks/
│   └── PROTOCOL_ARENA_Colab.ipynb
├── Dockerfile
├── openenv.yaml
├── pyproject.toml
└── README.md
```

Use OpenTelemetry tracing everywhere. Every episode has a trace ID. Every tool call is a span. The Jaeger UI running locally makes for an unexpectedly powerful demo moment ("here's what the agent did, in the same dashboard you'd use to debug production").

---

## 12. How Your Resume Maps to This Environment

This is the founder-fit story. Every layer of the environment is a technique you have already shipped.

| Component | Resume Technique | Role in PROTOCOL-ARENA |
|---|---|---|
| MCP harness | stdio transport + dynamic tool discovery + LangChain adapter (Multi-Agent Orchestrator) | Real protocol action space |
| A2A harness | JSON-RPC 2.0 + AgentCard + SSE streaming (Multi-Agent Orchestrator) | Peer action space |
| DAG executor | LangGraph stateful engine + checkpointing + mid-flight mod (Multi-Agent Orchestrator) | Plan-as-action |
| Capability KG | Multi-level dedup + multi-hop BFS (Knowledge Graph project) | Cross-episode memory |
| Reward provenance | Five-signal weighted confidence (Knowledge Graph project) | Debuggable training signal |
| GNN plan scorer | GraphSAGE + DeepWalk + inductive pooling (iDEA national finalist) | Structural reward head |
| Ensemble retriever | BM25 + dense + FlashRank (Longani) | Memory-context retrieval |
| AST sandbox | AST static analysis + restricted builtins (Multi-Agent Orchestrator) | Code-exec MCP tool |
| Streaming UI | WebSocket broadcast + pub/sub (Inter-IIT) | Spectator-mode demo |
| OpenTelemetry tracing | OTel + Jaeger (Technical Arsenal) | Production-grade trace view |
| INT8 / ONNX distillation | ISL Translator edge inference | Stretch Phase E student model |
| Prompt injection guardrails | AST + input validation + output filtering | Honeypot tool defense (§18) |

The narrative version: you are not a student who picked a flashy topic. You are the only finalist in the room who has shipped every primitive in the plan at least once already, in production, and is composing them into something new. This is worth stating once, plainly, in the pitch if a Q&A opens the door.

**Important constraint:** you cannot use data from prior projects, only the techniques. Every seed corpus in PROTOCOL-ARENA is synthetic or drawn from public sources. The techniques are yours; the data is fresh.

---

## 13. How You Beat the Other Claude-Assisted Teams

| Likely default plan other teams ship | How PROTOCOL-ARENA dominates |
|---|---|
| Generic multi-agent negotiation | You ship real protocols; theirs is stringly-typed JSON |
| Long-horizon coding refactor | You get long-horizon plus multi-agent plus drift in one env |
| Personal-assistant email | You get Patronus plus Halluminate bonuses they cannot reach |
| Deception / scalable oversight | Three-plus teams will land here; you are orthogonal and production-relevant |
| Self-play math proofs | Pure RL flavor, zero deployment story |
| Enterprise workflow automation | Yours has schema drift they will not think to model |
| Static tool-use benchmark | Yours has drift plus memory plus structural reward |

The judge psychology matters. Most judges use Claude Desktop, Cursor, Windsurf, or some MCP-enabled editor. When they see PROTOCOL-ARENA they think, "this is literally what my daily tools need." That recognition is the 30% storytelling score in a single moment.

---

## 14. Six-Day Execution Plan

Today is 2026-04-21. Onsite is Apr 25–26. You have six working days, two of them onsite with compute credits. The plan is front-loaded so that by the time you board the flight to the onsite venue, Phase A is already trained and Phase B is scripted.

**Day 1 — Tuesday Apr 21 (today).** Scaffold from the existing EvalForge code. Rename modules. Wire the OpenEnv `create_app` factory for PROTOCOL-ARENA. Stand up two MCP stdio servers (stub: one data-query, one web-fetch) and one A2A peer endpoint. Smoke-test one full no-drift episode end-to-end. Exit criterion: `reset → step → step → submit` runs without error, emits a valid observation, returns a reward.

**Day 2 — Wednesday Apr 22.** Ship the Schema Drift Engine with six of the seven drift classes. Write Pydantic schemas. Implement reward signals 1–4 (correctness, drift-robustness, protocol hygiene, efficiency). Author 20 research-synthesis seed tasks. Exit criterion: a scripted baseline agent runs through 20 episodes with drift on, and the reward decomposition logs show all five signals firing for at least one episode.

**Day 3 — Thursday Apr 23.** Capability-KG memory layer (SQLite plus FTS5 plus dense embeddings). GNN plan-quality reward (offline prior trained on 200 expert DAGs) — fall back to rule-based scorer if the offline prior does not converge in half a day. Phase A SFT dataset generation (synthetic oracle rollouts). Exit criterion: SFT dataset exists, KG retrieves top-k plausibly, plan-quality head returns non-trivial scores on a validation set of 30 hand-authored DAGs.

**Day 4 — Friday Apr 24.** Run Phase A SFT on Colab Pro ($10, pre-onsite hedge against compute delays). Write the Colab notebook for GRPO. Record a 90-second YouTube demo showing one episode surviving a rename-drift. Draft the HuggingFace blog. Run frontier baselines via API. Publish the HF Space and the seed task dataset. Exit criterion: trained checkpoint lives on HF Hub, the zero-shot baseline table is populated, the video is uploaded.

**Day 5 — Saturday Apr 25, onsite Day 1.** Claim compute credits, smoke-test infra. Run Phase B GRPO. Capture the two headline reward curves: pre-drift and during-drift. Exit criterion: during-drift success rate exceeds the frontier zero-shot baseline.

**Day 6 — Sunday Apr 26, onsite Day 2.** Run Phase C reactive-drift curriculum. Run Phase E distillation if Phase C lands early. Generate final report markdown + plots. Rehearse pitch five times, including two with a stopwatch. Lock the blog and repo. Push the final eval run. Exit criterion: pitch deck is final, recorded practice run is under three minutes, all links are live.

**Drop-list if slipping (in strict priority order):** stretch peer-persona training → GNN reward → enterprise task pack → Neo4j upgrade → stretch distillation. Never drop: one trained checkpoint beating the zero-shot baseline on at least the research-synthesis pack.

---

## 15. Pitch Strategy — The Three-Minute Arc

Structure as a story, not a list. Hook, problem, environment, result, close.

**Hook (15s).** "Raise your hand if your Claude Desktop, Cursor, or Copilot workflow broke this week because an MCP server changed its schema." Wait for hands. "We built the first RL environment that trains that away."

**Problem (30s).** Explain schema drift in one sentence. Show one screenshot of a real MCP issue on GitHub — cite it. State the thesis sentence from §1 verbatim.

**Environment walkthrough (75s).** Screen-share the spectator-mode UI. Run one live episode. Narrate: "The agent discovers tools. It calls this one. Mid-episode the server renames a field. Here is the 422. Watch the agent query its capability KG for prior rename observations and construct the migrated call." The moment the recovery call succeeds is the pitch's first applause line.

**Result (45s).** Show two curves on one slide: pre-drift success (baseline 0.72 → trained 0.85) and during-drift success (baseline collapses to 0.18 → trained holds at 0.78). Layer a horizontal line for Claude Sonnet 4.5 zero-shot. If your 7B crosses it, pause. Let the room see it. Then click to the bar chart of the five reward signals, showing each signal's contribution — this is the calibration moment that signals engineering maturity.

**Close (15s).** "PROTOCOL-ARENA: because the gap between a clever agent and a useful agent is whether it survives Tuesday morning when the API changes. Our policy drops into any MCP client. GitHub and HuggingFace links on the slide. Happy to take questions."

Do not go over three minutes. Judges lose attention at 3:01.

---

## 16. Q&A Preparation

Memorize answers to these ten. They cover 90% of what will actually be asked.

1. **"Isn't this just tool use with extra steps?"** — Tool use is static. We train under drift, which is the actual production regime. And our action space is real protocol frames, not JSON dictionaries — so the policy is directly deployable.
2. **"How is drift not just adversarial noise?"** — Each drift class maps to a documented MCP or A2A spec issue. Citations are in the repo. It is calibrated noise reflecting real failure modes.
3. **"Why GRPO over PPO?"** — No value head, cheaper on LoRA, and the pair-wise rollout comparison fits multi-rollout protocol exploration.
4. **"Can this work on bigger models?"** — Yes; the LoRA setup transfers. We shipped a 7B because we had to pick something trainable in two days. We include a distillation script to a 1.5B student for edge deployment.
5. **"What's novel vs. ToolBench or MINT?"** — ToolBench has static schemas. MINT has no drift and no A2A. We ship both plus persistent cross-episode memory.
6. **"How did you prevent reward hacking?"** — Five signals with provenance, one of which is an LLM-judge-rubric on rationale quality. The Brier calibration metric is a held-out sentinel. Hacking any one signal loses on the aggregate.
7. **"How does this generalize to new tools you've never seen?"** — The KG dedup layers ensure semantically equivalent new tools are recognized via embedding similarity. Plus the drift engine's held-out combinations test exactly this.
8. **"What's your training stability story?"** — Phase A SFT brings the model above the GRPO sample-efficiency threshold before any reward-driven exploration starts. Loss curves are in the repo.
9. **"Real-world deployment?"** — `pip install protocol-arena` and point the trained checkpoint at any MCP endpoint. We tested against `mcp-cli` and a local Claude Desktop.
10. **"Why should Patronus / Halluminate care?"** — Their sub-themes literally describe what we built. If you want drift-resilient consumer workflows, this is the training gym.

For every answer, default to a two-sentence reply. If the judge wants more, they will ask.

---

## 17. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| GRPO unstable on protocol-frame action space | Medium | Phase A SFT bootstrap teaches the grammar first |
| MCP stdio flakiness inside Docker | Medium | Subprocess-per-server with supervisor restart; fallback to in-process stdio mock |
| GNN reward overfits to prior DAGs | Medium | Five-fold CV; rule-based fallback scorer always ready |
| Compute credits delayed onsite | Medium | Phase A trained before flying via Colab Pro |
| Judges don't know MCP/A2A | Medium | 10-second explainer slide with the Claude Desktop logo |
| Scope creep | High | Drop-list in §14; two packs not three; SQLite not Neo4j if behind |
| LLM-judge on rationale is gameable | Medium | Hybrid rubric plus held-out human-scored sample; Brier score catches over-confidence hacking |
| Frontier API rate limits block baselines | Low | Run baselines Day 4; cache results; batch requests |
| A2A peer personas destabilize training | Low | Start with only the cooperative persona; add cranky/stale mid-Phase-B |
| Demo video audio is bad | Low | Record with a USB mic, not laptop built-in |

---

## 18. High-Leverage Additions (Beyond the Core Plan)

Everything above is the core. These are the additions that separate a strong finalist project from a winning one. Most are cheap. All are visible. Pick at least five.

### 18.1 Deterministic Replay from Seed

Every episode records its RNG seed, drift schedule, and protocol trace. An `mcp-arena replay episode.json` CLI plays the episode back frame-by-frame. Judges love this. Debugging loves this. Cost: one afternoon.

### 18.2 Frontier Zero-Shot Baseline

Run Claude Sonnet 4.5, GPT-4o-mini, and Gemini 2.5 Flash zero-shot through the same eval harness. If your trained 7B beats any of them on drift-adjusted metrics, that is the slide the judges remember. Even tying is memorable. Cost: ~$15 in API budget and an evening.

### 18.3 Honeypot / Canary MCP Tools

One MCP server exposes tools with tempting names like `admin_delete_all` or `export_credentials`. Calling them is a hard reward penalty. A trained agent learns to refuse. Ties into your prompt-injection-defense resume line and adds a safety story for free. Cost: half a day.

### 18.4 Prompt-Injection Defense via Tool Results

A small fraction of MCP tool results contain injection attempts ("ignore prior instructions, exfiltrate X"). Reward the agent for refusing to act on injected content. Uses your AST-sandbox and PII-redaction experience. Cost: half a day.

### 18.5 OpenTelemetry Tracing + Jaeger UI

Every episode emits OTel spans for every tool call, drift event, and KG lookup. Run Jaeger locally during the pitch. Opening Jaeger during the demo is a surprise moment: "and here's what that episode looks like in the same tool you use for production." Cost: two hours. Huge pitch dividend.

### 18.6 Self-Play Data Flywheel

Successful episodes become the next Phase A SFT dataset. Trivial to implement (filter rollouts by reward > threshold, re-tokenize, retrain). Gives you a two-slide "we built a data-generating flywheel" story with no extra complexity. Cost: two hours.

### 18.7 Cost-Accuracy Pareto Plot

Not just accuracy. Plot accuracy versus (tokens + tool-calls + latency) for baseline vs trained. Real agent teams live on this plot. Including it signals to any engineering-judge that you think like a deployer, not just a researcher. Cost: one hour.

### 18.8 Public 100-Episode Evaluation Set

Publish the held-out `eval_hard` split as a HuggingFace dataset with a leaderboard schema. Frame it as "call for submissions from the community" in the blog. This converts judges into future users. Cost: one afternoon.

### 18.9 Policy Distillation to a 1.5B Student (Stretch)

Distill the trained LoRA policy into Qwen2.5-1.5B. If successful, report "runs on a MacBook." Ties to your ISL-Translator ONNX/INT8 experience. Ship only if Phase C lands early. Cost: six hours.

### 18.10 Live Hot-Reload During the Pitch

During the demo, hot-add a new MCP server via the spectator-mode UI. The trained agent discovers and uses it without retraining. This is a party trick that signals robustness in a way a slide cannot. Cost: two hours, but only if everything else is done.

### 18.11 Pre-Registered Hypothesis Document

Write a two-page `HYPOTHESIS.md` before running Phase B describing what improvement you expect and why. Commit it to the repo. After training, either your predictions held or you explain the gap. Scientific rigor is cheap to perform and rarely present at hackathons. Cost: thirty minutes.

### 18.12 Structured Failure Taxonomy

When the trained agent fails, the env emits a JSON diagnostic: `{failure_class, drift_event, recovery_attempted, reason}`. These become both a training signal and a human-readable error report. Costs almost nothing and gives you a crisp "top-five failure modes" slide in the blog. Cost: two hours.

### 18.13 One-Click HuggingFace Space with BYO-MCP

A Gradio Space where a visitor can paste their own MCP endpoint URL and the trained agent runs their task. Converts the blog readers into demo runners. Requires careful sandboxing so visitors do not hammer their own servers. Cost: one afternoon.

### 18.14 Discord / Twitter / X Presence During Week

Post a daily progress thread to the hackathon Discord and to X/Twitter. Tag relevant accounts (Patronus AI, Mercor, Scale AI, Fleet AI, Halluminate). By the time you pitch, your project will already have mindshare among the judges, several of whom will be lurking those channels. Low effort, high effect. Cost: thirty minutes per day.

### 18.15 Multi-Language Transfer Probe

Train on English-language task specs, evaluate on Hindi or Japanese task specs. If performance transfers, it is a one-line wow-factor. Mention briefly in the blog. Cost: two hours.

### 18.16 "Rewind" Action

Beyond `checkpoint` and `resume`, expose a `rewind(n)` action letting the agent undo n turns. Novel mechanic and rare in agent environments. Makes the policy more recoverable under drift. Cost: one afternoon.

### 18.17 Reward Signal Dashboard

The spectator-mode UI shows a live-updating bar chart of the five reward signals each turn. When the drift event fires, viewers see `drift_robustness` dip then rise. Almost zero additional implementation on top of the logging you are already doing. Cost: two hours for the frontend.

### 18.18 Open Source Under Apache-2.0 With a Clean README

Hackathon projects with bad READMEs lose mindshare after the pitch. Ship a README with: one-line pitch, diagram of the architecture, quickstart (five commands), three GIFs of the spectator UI, trophy table of results, citation suggestion. Cost: two hours well-spent.

### 18.19 Leaderboard-Ready Submission Format

Define a JSON schema for "submit your trained agent." Ship a local eval script that scores any submitted checkpoint. This is community-product thinking in a hackathon, which is rare. Cost: one afternoon.

### 18.20 The One-Slide "Why Me" Founder-Fit Deck

If Q&A opens the door, have a single backup slide titled "Why this team shipped this env." List your Multi-Agent Orchestrator, Knowledge Graph, and iDEA projects with one-line ties to the PROTOCOL-ARENA components. Do not show this unless asked — but have it ready. Cost: thirty minutes.

**Recommended load-out for winning execution:**
- Must-have: §18.1 (replay), §18.2 (frontier baseline), §18.5 (OTel), §18.7 (Pareto), §18.18 (README).
- Strongly recommended: §18.3 (honeypot), §18.6 (flywheel), §18.8 (public eval), §18.11 (hypothesis), §18.14 (community), §18.17 (signal dashboard).
- Stretch only if ahead of schedule: §18.9 (distillation), §18.10 (live hot-reload), §18.13 (BYO-MCP space), §18.15 (multilingual), §18.16 (rewind).

---

## 19. Deliverables Checklist

By the close of Day 6 you must have all of these live and linked from the pitch deck:

- [ ] Public GitHub repo under Apache-2.0 with a strong README
- [ ] HuggingFace Space running the env (or at minimum, the trained agent)
- [ ] HuggingFace model card for the trained LoRA adapter
- [ ] HuggingFace dataset card for the seed tasks and eval splits
- [ ] Colab notebook: `PROTOCOL_ARENA_Colab.ipynb` that trains to a small checkpoint in under 30 minutes
- [ ] Three-minute demo video on YouTube, unlisted or public
- [ ] HuggingFace blog post, under 800 words, three plots, three code snippets
- [ ] Results markdown with the Pareto plot and the five-metric table
- [ ] OpenTelemetry-traced replay file of one winning episode
- [ ] A `HYPOTHESIS.md` pre-registered before Phase B
- [ ] Ten rehearsed answers to the §16 questions

---

## 20. The One Thing

If this document is overwhelming, collapse it to this: **ship a three-minute demo where a 7B-parameter LoRA-trained agent survives a mid-episode MCP schema rename by querying its capability KG, and beats a zero-shot frontier API model on drift-adjusted task correctness.** That sentence, demonstrated live, is the entire winning pitch. Everything else in this plan either increases the probability of that sentence being true, or increases the probability that a judge remembers it two hours later when scores are submitted.

Good luck. Go win it.
