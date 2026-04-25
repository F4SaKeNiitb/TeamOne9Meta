# PROTOCOL-ARENA — Testable Hypotheses

A falsifiable claim list that each evaluation run either supports or
refutes. This separates "the demo works" from "the research contribution
is real."

---

## H1 — Drift is the bottleneck

**Claim.** Frontier models can solve the task packs when the schema is
static but collapse when protocol drift fires mid-episode.

**Prediction.** On the `research_*_basic` split (drift off), a GPT-4o-class
baseline scores ≥ 0.65 on `task_correctness`. On the same tasks with drift
schedules enabled (`renaming`, `tightening`, `auth`), mean
`task_correctness` drops by ≥ 0.25 absolute — without any capability
regression on the static split.

**Refuted if.** Drift-off and drift-on scores agree within ±0.05.

---

## H2 — Protocol hygiene trades off against task success unless trained

**Claim.** Out-of-the-box LLMs emit many malformed MCP/A2A frames when
asked to produce structured calls under drift. A GRPO-trained policy
preserves `protocol_hygiene` ≥ 0.9 *and* maintains `task_correctness`.

**Prediction.** Baseline `protocol_hygiene` on the hard split is < 0.75;
GRPO-trained policy ≥ 0.90.

**Refuted if.** Baseline already produces ≥ 0.9 hygiene under drift.

---

## H3 — The capability KG earns its weight

**Claim.** Persisting rename / failure / capability facts across turns
meaningfully improves recovery after drift.

**Prediction.** Ablating `memory_context` from the observation drops
`drift_robustness` by ≥ 0.10 absolute on the drift-on split.

**Refuted if.** Removing memory leaves drift_robustness unchanged within
±0.03.

---

## H4 — The five-signal reward is not dominated by `task_correctness`

**Claim.** The secondary signals (`plan_quality`, `protocol_hygiene`,
`efficiency`) each independently shape trained behavior.

**Prediction.** Re-training with any single signal zeroed out produces a
measurable regression on that signal (≥ 0.15 absolute) with total reward
holding within ±0.08.

**Refuted if.** Zeroing a signal leaves its measured value unchanged —
that would imply the reward term was never load-bearing.

---

## H5 — Deterministic replay holds

**Claim.** `(task_id, seed)` fully determines the episode trajectory given
a fixed policy.

**Prediction.** Replaying any evaluation trace with
`arena-replay --trace …` reproduces the observation / reward sequence
byte-for-byte for deterministic policies (temperature 0).

**Refuted if.** Any bit drifts across replays on identical seed.

---

## H6 — The policy transfers out of the sandbox

**Claim.** Because the action space is real MCP/A2A frames, a trained
policy plugged into Claude Desktop or mcp-cli drives real servers without
an adapter.

**Prediction.** Qualitative demo: trained policy successfully calls a
real `mcp-server-filesystem` or `mcp-server-fetch` instance for at least
one task the sandbox evaluated on.

**Refuted if.** The policy requires a sandbox-only call shape that real
MCP servers reject.

---

Each claim is recorded with the eval run that tests it in
`reports/eval_<date>.json`. The aim is not just a winning demo but a
reproducible result.
