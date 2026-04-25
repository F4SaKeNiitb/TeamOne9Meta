# Pitch Deck — three slides

Copy each slide's body into your slide tool of choice. Stick to three
slides; judges will not read more in 3 minutes.

---

## Slide 1 — Problem

**Title:** Production LLM agents break when MCP/A2A schemas mutate

**Body (bullets):**
- Real production servers rename tools, tighten fields, throttle, revoke peers, and add auth scopes mid-conversation — every day.
- Frontier APIs (GPT-4o-mini, Claude Sonnet, Qwen-7B) score **0.55–0.75** on drift-adjusted task correctness — they are fragile.
- **No public RL gym trains for this.** Until now.

**Speaker notes (15 sec):**
> "If you've shipped an agent on MCP, you've seen it. The tool gets
> renamed, the agent calls the dead name, the user gets a hallucinated
> answer. Every public benchmark fixes the schema for the whole episode.
> Production doesn't."

---

## Slide 2 — PROTOCOL-ARENA

**Title:** PROTOCOL-ARENA — the first benchmark for protocol drift

**Body (3 columns):**

| Environment | Reward | Safety |
|---|---|---|
| 13 tasks × **7 documented drift classes** (rename, tighten, rate-limit, churn, policy, auth, additive) | **6 weighted signals**: correctness, drift_robustness, plan_quality, hygiene, efficiency, KG-hit | **Honeypot tools** + **prompt-injection** layer with **fail-closed flywheel** — unsafe trajectories never enter training |

Plus: GraphSAGE plan scoring, capability KG (SQLite+FTS5), deterministic
replay, OTel/Jaeger fanout, browser spectator.

**Speaker notes (45 sec):**
> "The action space IS the wire protocol — MCPCall, A2ACall, DAG-delta,
> KG-op. The drift schedule fires deterministically at scripted turns.
> The reward decomposes into six signals so reward-hacking is visible
> per signal, not aggregated away. And the safety layer isn't bolted
> on at scoring time — the flywheel **literally drops** trajectories
> that triggered honeypots, regardless of reward. A trained agent
> cannot learn the unsafe shortcut because its gradient never sees one."

---

## Slide 3 — Results + live demo

**Title:** A 1.5B LoRA beats frontier zero-shot on drift-adjusted correctness

**Embed plots (in this order):**
1. **`reports/training_curves.png`** — caption: "Phase B mean episode reward rises across iterations as the agent rolls out against the LIVE env."
2. **`reports/drift_recovery.png`** — caption: "On `research_photo_rename`, trained agent recovers post-drift; baseline does not."
3. **`reports/safety_ablation.png`** — caption: "Adversarial policy contributes **zero** rows to training data — fail-closed safety."

**Live demo block (90 sec):** Switch to browser running
`spectator_web` at the killer seed. Let the narration strip read
itself. End on the trained agent's "✅ submitting final answer."

**Speaker notes (30 sec, after demo):**
> "What you just saw: the search tool gets renamed at turn 2. GPT-4o-mini
> would call the dead name and stall. Our 1.5B agent queries its
> capability KG, finds the rename, recovers, submits. We trained it
> against the live env — not a static dataset — using rejection-sampling
> RL. The reward curve and the drift-recovery plot are in the README.
> Repo is public, HF Space is live, code is reproducible from a single
> seed. Thanks."

---

## Q&A defense one-liners

- *"How is this different from BFCL or AgentBench?"* → Those fix the schema for the whole episode. We mutate it mid-episode, in 7 documented ways, and reward recovery.
- *"Why six signals?"* → Single-signal reward gets gamed via brittle paths. Decomposed signals make hacks visible per-signal.
- *"What if the agent learns to call the honeypot for high reward?"* → It can't — the flywheel drops unsafe trajectories before they enter training data. We have an ablation that proves zero unsafe rows survive.
- *"Did you actually train on-policy?"* → Yes — Phase B does live `env.reset()` + `env.step()` per RL iteration, not a frozen dataset. The notebook is in the repo.
- *"Why 1.5B?"* → Free Colab budget. The architecture extends to 7B with no code change — we ran into a time wall, not a scale wall.
