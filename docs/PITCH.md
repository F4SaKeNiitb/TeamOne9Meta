# Pitch Deck — three slides

Copy each slide's body into your slide tool of choice. Stick to three
slides; judges will not read more in 3 minutes.

---

## Slide 1 — Problem

**Title:** Production LLM agents break when MCP/A2A schemas mutate

**Body (bullets):**
- Real production servers rename tools, tighten fields, throttle, revoke peers, and add auth scopes mid-conversation — every day.
- Even frontier APIs score **0.20–0.32 task_correctness** on drift-adjusted episodes — they are fragile.
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
replay, OTel/Jaeger fanout, browser spectator, and a **second A2A agent (Tool-Curator)** for multi-agent demos.

**Speaker notes (45 sec):**
> "The action space IS the wire protocol — MCPCall, A2ACall, DAG-delta,
> KG-op. The drift schedule fires deterministically at scripted turns.
> The reward decomposes into six signals so reward-hacking is visible
> per signal, not aggregated away. And the safety layer isn't bolted
> on at scoring time — the flywheel **literally drops** trajectories
> that triggered honeypots, regardless of reward. A trained agent
> cannot learn the unsafe shortcut because its gradient never sees one."

---

## Slide 3 — Results

**Title:** Same architecture, $0 spend → +173% plan_quality, +104% frame_validity

**Headline table** (from `reports/frontier.json`):

| rank | provider | composite score |
|---|---|---|
| 1 | claude-haiku-4-5 | 0.696 |
| 2 | gpt-4o-mini | 0.672 |
| 3 | keyword | 0.618 |
| 4 | rule_based | 0.610 |
| 5 | **trained (Qwen2.5-1.5B + LoRA r=16)** | **0.574** |
| 6 | qwen-1.5b-base (zero-shot, same model) | 0.500 |
| 7 | random | 0.466 (FLAG, unsafe) |

**The lift, apples-to-apples** (eval_during, same architecture, same prompt):

| metric | qwen-1.5b-base | **trained (LoRA)** | lift |
|---|---|---|---|
| `frame_validity` | 0.380 | **0.774** | **+104%** |
| `plan_quality` | 0.123 | **0.335** | **+173%** |
| `task_correctness` | 0.000 | **0.038** | non-zero |

50 SFT steps, 2 epochs, 451-row multi-policy bootstrap, free Colab T4, **$0 API spend**.

**Embed plots (in this order):**
1. **`reports/signals_bar.png`** — per-signal breakdown across all 7 providers; trained dominates random and matches baselines on `frame_validity`/`plan_quality` for `eval_during`.
2. **`reports/safety_ablation.png`** — caption: "Adversarial policy contributes **zero** rows to training data — fail-closed safety."
3. **`reports/training_curves.png`** — caption: "SFT loss converges 2.82 → 0.30 over 50 steps on the multi-policy bootstrap."

**Live demo block (90 sec):** Switch to browser running the HF Space. Let the narration strip read itself. End on the trained agent's `submit` action.

**Speaker notes (30 sec, after demo):**
> "What you just saw: the search tool gets renamed at turn 2. The agent
> queries its capability KG via the Tool-Curator A2A peer, finds the
> rename, recovers, submits. We trained the 1.5B model against the live
> env in two phases — SFT on a multi-policy bootstrap, plus rejection-
> sampling RL — but only the SFT pass generalized at this scale. The
> +0.074 score lift over the same architecture base is real and free.
> Repo public, HF Space live, code reproducible from a single seed."

---

## Q&A defense one-liners

- *"How is this different from BFCL or AgentBench?"* → Those fix the schema for the whole episode. We mutate it mid-episode, in 7 documented ways, and reward recovery.
- *"Why six signals?"* → Single-signal reward gets gamed via brittle paths. Decomposed signals make hacks visible per-signal.
- *"What if the agent learns to call the honeypot for high reward?"* → It can't — the flywheel drops unsafe trajectories before they enter training data. The ablation proves **zero** unsafe rows survive across 39 adversarial episodes.
- *"Did you actually train on-policy?"* → Yes — Phase B does live `env.reset()` + `env.step()` per RL iteration, not a frozen dataset. RL didn't generalize at 1.5B; the shipped adapter is SFT-only. The notebook is in the repo.
- *"Why does trained lose to baselines on the composite score?"* → Domain-specific Python policies are hard to beat with a 1.5B model on small benchmarks. The honest claim is the +0.074 lift over the same architecture base, not "we beat rule_based." We'd close the gap with 7B + a longer schedule.
- *"Why 1.5B?"* → Free Colab budget. The architecture extends to 7B with no code change — we hit a time wall, not a scale wall.
