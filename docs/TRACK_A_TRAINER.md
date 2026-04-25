# Track A — Trainer

You own the **model and the numbers**. Your output is the trained adapter on HF Hub and a `reports/frontier.json` with 4 columns. This is the longest critical path; start hour 0.

**You will be the bottleneck. If you slip, the team slips. Own it.**

---

## Hour 0–0.5 · Setup

```bash
cd /Users/manish/Downloads/OpenEnv
git status                          # confirm clean working tree
python -m pytest -q                 # 6/6 must be green
python scripts/find_killer_seed.py  # confirm drift fires at turn 2
```

Open Google Colab → upload or open `notebooks/PROTOCOL_ARENA_Colab.ipynb`. Set runtime to **T4 GPU (free)** or L4 if available.

In Colab Cell 1, replace `<YOUR-GH-USER>` with the actual GitHub URL of the repo:

```python
!git clone https://github.com/<YOUR-GH-USER>/OpenEnv.git
```

---

## Hour 0.5–3 · A1 — SFT Phase A

The new bootstrap (`arena.training.sft_bootstrap`) uses a scripted
expert teacher + 3 baseline policies with stratified sampling, dedup,
and oracle-keyed submit injection. **It runs inside Colab** — just
execute Cell 2 and it produces the dataset in 3–5 min.

Run cells **1 → 2 → 3 → 4** in Colab.

**Expected timing:**
- Cell 1 (install + clone): ~3 min — change the `<YOUR-GH-USER>` placeholder to your fork first
- Cell 2 (SFT data gen): ~3–5 min, writes `data/sft.jsonl`
- Cell 3 (load Qwen2.5-1.5B 4-bit + LoRA): ~2 min
- Cell 4 (SFT train, 100 steps): **~25 min on T4**

After Cell 2, read the diagnostics block at the bottom of its output. Healthy:
- `episodes kept` ≥ 800
- `rows after dedup` ≥ 350
- `by action kind` shows mcp / submit / memory / a2a / plan all > 0
- `tc distribution` has > 60% at `1.00` (most kept episodes solved the task)
- `per-task row counts` — every task ≥ 20

If any are missing, re-run Cell 2 with `--episodes 2500` (edit inline) or check the diagnostics for warnings.

**Checkpoint at hour 3:**
```python
# In a Colab cell:
import pandas as pd
df = pd.read_csv("reports/sft_loss.csv")
print(df.tail())
print(f"loss start={df['loss'].iloc[0]:.3f}  end={df['loss'].iloc[-1]:.3f}")
```

✅ PASS: end loss in `[0.05, 0.5]` AND start loss > 1.5 → continue to A2.
   The new dataset is more diverse than the old one, so loss should
   plateau in `~0.1–0.3`, NOT bottom out near `0.02` (that was
   memorization of an unbalanced dataset).
❌ FAIL: loss is flat → STOP. Ping #team-trainer. Likely fixes:
- Check `sft.jsonl` has > 350 rows: `!wc -l sft.jsonl`
- Re-run the local bootstrap with a different `--seed`
- Lower learning rate to 1e-4 in Cell 4

---

## Hour 3–8 · A2 — RL Phase B + push adapter

Run cells **5 → 6 → 7 → 8**.

**Expected timing:**
- Cell 5 (RL, 4 iters × 16 eps): **~35 min on T4**
- Cell 6 (plot training_curves.png): instant
- Cell 7 (final eval trained vs rule_based): ~3 min
- Cell 8 (push adapter to HF Hub): ~2 min

**Before Cell 8**, set your HF token in Colab:
```python
from huggingface_hub import notebook_login
notebook_login()  # paste token
```

Then in Cell 8 set `HF_USER = "your-hf-username"` and run.

**Checkpoint at hour 8:**
```bash
# Confirm the adapter is reachable:
curl -sI https://huggingface.co/<your-hf-user>/protocol-arena-qwen-1.5b-lora-r16 | head -3
```

Hand the URL to **Track C** (they need it for `README_RESULTS_SECTION.md`).

❌ HARD RULE: **If `reports/training_curves.png` is not produced by hour 14**, kill Phase B and trigger the **SFT-only fallback** (see bottom).

---

## Hour 8–12 · A3 — Frontier baseline (the kill-shot)

This is what separates you from teams running only baselines. **Do not skip.**

Get API keys from Track C (they wired them at hour 0–2):
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

Run on a subset first to confirm wiring:
```bash
python -m arena.eval.run_frontier \
    --tasks research_photo_rename \
    --seeds 0 \
    --splits during \
    --out reports/frontier_smoke.json
cat reports/frontier_smoke.json | python -m json.tool | head -50
```

Then full run (all 13 tasks × 3 seeds × 3 splits, ~30 min):
```bash
python -m arena.eval.run_frontier --out reports/frontier.json
```

This auto-includes `gpt-4o-mini` and `claude-haiku-4-5` from `DEFAULT_LIVE_CONFIG`. If a provider errors, the runner records `"error": "..."` and continues — **don't panic if one fails, ship with the others**.

**Inject your trained-LoRA results into the same file** by running Cell 7 from the Colab and downloading the resulting JSON, or export from Colab:
```python
# In Colab, after Cell 7:
import json
with open("/content/OpenEnv/reports/trained_results.json","w") as f:
    json.dump(trained_eval, f, indent=2)
# Then download via Files panel.
```

Then merge locally:
```bash
python - <<'PY'
import json
fr = json.load(open("reports/frontier.json"))
tr = json.load(open("reports/trained_results.json"))
fr["providers"]["trained"] = tr
json.dump(fr, open("reports/frontier.json","w"), indent=2, default=str)
print("merged. providers:", list(fr["providers"].keys()))
PY
```

---

## Hour 12–14 · A4 — Score and lock numbers

```bash
python scripts/score_submission.py reports/frontier.json
```

You should see a ranked table with `trained` ideally in top 3. Note the score.

Emit the submission blob:
```bash
python scripts/score_submission.py reports/frontier.json \
    --label trained \
    --emit submission.json \
    --team-name "<YOUR-TEAM-NAME>" \
    --policy-kind lora \
    --model-id "Qwen/Qwen2.5-1.5B-Instruct" \
    --submitted-by "manish.shaw@iitb.ac.in" \
    --checkpoint-url "https://huggingface.co/<your-hf-user>/protocol-arena-qwen-1.5b-lora-r16" \
    --notes "1.5B LoRA r=16, SFT 100 steps + 4 iters rejection-sampling RL on live env"
```

Hand `reports/frontier.json` and `submission.json` to **Track C**. **NO MORE TRAINING after hour 14.**

---

## SFT-only fallback (trigger if hour-14 sync flags Phase B as failed)

```bash
# Reframe the story: small model close to frontier with just SFT.
# Edit README_RESULTS_SECTION.md to remove RL claims.
# Keep training_curves.png as SFT-loss-only by truncating the RL panel:
python - <<'PY'
import matplotlib.pyplot as plt, pandas as pd
df = pd.read_csv("reports/sft_loss.csv")
plt.figure(figsize=(7,4))
plt.plot(df["step"], df["loss"], color="#1f77b4", linewidth=2)
plt.xlabel("step"); plt.ylabel("SFT loss"); plt.grid(alpha=0.3)
plt.title("Phase A — SFT loss converges in 100 steps")
plt.tight_layout(); plt.savefig("reports/training_curves.png", dpi=150)
PY
```

The narrative becomes: *"1.5B LoRA after 100 SFT steps is within 80% of GPT-4o-mini on `eval_during.task_correctness`."* This is **stronger than a 0.02 RL improvement**.

---

## Done state at hour 14

- [ ] `reports/sft_loss.csv` — loss converged
- [ ] `reports/training_curves.png` — exists (or SFT-only fallback)
- [ ] `https://huggingface.co/<user>/protocol-arena-qwen-1.5b-lora-r16` — reachable
- [ ] `reports/frontier.json` — has providers: rule_based, gpt-4o-mini, claude-haiku-4-5, trained
- [ ] `submission.json` — generated, validated
- [ ] Numbers handed to Track C

After hour 14, you join Track B for demo dry-runs. **Do not touch the model.**
