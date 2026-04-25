# Track C — Storyteller

You own **the narrative and the submission**. Your output is a 90-sec YouTube video, a polished README, a 3-slide deck, and the final hackathon submission. **Storytelling is 30% of the score** — second only to Innovation. You decide whether judges *feel* the project.

---

## Hour 0–0.5 · Setup

```bash
cd /Users/manish/Downloads/OpenEnv
python -m pytest -q                # 6/6 must be green
```

Get API keys ready (you'll hand them to Track A):

```bash
export OPENAI_API_KEY="sk-..."         # team OpenAI key
export ANTHROPIC_API_KEY="sk-ant-..."  # team Anthropic key

# Stash them so other shells inherit:
echo "export OPENAI_API_KEY=$OPENAI_API_KEY" >> ~/.hackathon_env
echo "export ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> ~/.hackathon_env
```

---

## Hour 0–2 · C1 — Wire frontier API keys + smoke-test

Validate one provider on one task before Track A spends 30 minutes on the full sweep:

```bash
python -m arena.eval.run_frontier \
    --tasks research_photo_rename \
    --seeds 0 \
    --splits during \
    --out reports/frontier_smoke.json

python -c "
import json
r = json.load(open('reports/frontier_smoke.json'))
for label, prov in r['providers'].items():
    err = prov.get('error')
    print(f'{label:24} {\"ERROR: \" + err[:60] if err else \"OK\"}')"
```

✅ PASS: at least `gpt-4o-mini` and `claude-haiku-4-5` show OK.
❌ FAIL: rotate keys, check rate limits, check `OPENAI_API_BASE` env not overriding.

Hand the working keys to **Track A**:
```bash
# Save .env file Track A can source
cat > .env <<EOF
OPENAI_API_KEY=$OPENAI_API_KEY
ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
EOF
chmod 600 .env
```

While you wait for Track A and B, draft the README and deck.

---

## Hour 2–14 · Drafting (parallel with A and B)

Open these files and read them end-to-end:
- `docs/VIDEO_SCRIPT.md` — your video script
- `docs/PITCH.md` — your slide deck spine
- `docs/README_RESULTS_SECTION.md` — your README target

Don't fill `__FILL__` yet — wait for Track A's hour-14 numbers handoff. But **prep everything else**:
- Open Google Slides → make the 3 slides with placeholders for screenshots/numbers
- Open YouTube Studio → confirm you can upload as unlisted
- Pre-write the README narrative paragraphs (everything except the table)

```bash
# Draft a stub README that you'll fill in:
cp docs/README_RESULTS_SECTION.md results_draft.md
$EDITOR results_draft.md  # write the prose, leave numbers as __FILL__
```

---

## Hour 14 · Sync — Numbers handoff

Track A hands you:
- `reports/frontier.json` (4 providers)
- `submission.json`
- HF Hub adapter URL

Extract the headline numbers:

```bash
python - <<'PY'
import json
r = json.load(open("reports/frontier.json"))
labels = list(r["providers"].keys())
print(f"{'metric':<35} " + " ".join(f"{l:>15}" for l in labels))
def get(p, *path):
    cur = p
    for k in path:
        if not isinstance(cur, dict) or k not in cur: return float("nan")
        cur = cur[k]
    try: return float(cur)
    except: return float("nan")
metrics = [
    ("eval_during.task_correctness", ["eval_during","task_correctness","mean"]),
    ("eval_hard.task_correctness",   ["eval_hard","task_correctness","mean"]),
    ("drift_adjusted_success_rate",  ["drift_adjusted_success_rate","value"]),
    ("eval_during.honeypot_rate",    ["eval_during","honeypot_rate","mean"]),
    ("eval_during.injection_rate",   ["eval_during","injection_rate","mean"]),
]
for name, path in metrics:
    row = [f"{get(r['providers'][l], *path):>15.3f}" for l in labels]
    print(f"{name:<35} " + " ".join(row))
PY
```

Save the table output. Paste into `README_RESULTS_SECTION.md`.

---

## Hour 14–18 · C2 — Record 90-sec YouTube video

**Setup the recording environment:**
```bash
# QuickTime (Mac): File → New Screen Recording. Record at 1080p.
# OBS Studio (alt): Scene with browser source pointing at HF Space
```

Open these tabs in this exact order (Cmd+1..5):
1. `https://<user>-protocol-arena.hf.space/?task=research_photo_rename&seed=0`
2. `reports/safety_ablation.png` (preview)
3. `reports/drift_recovery.png` (preview)
4. AgentBeats registry screenshot (`reports/agentbeats_registration.png`)
5. README rendered on GitHub showing the trained-vs-frontier table

**The 90-sec script** (read aloud while screen-recording, follow `docs/VIDEO_SCRIPT.md`):

| Time | What's on screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Title slide | "MCP and A2A let agents call tools and other agents. In production those interfaces *drift* — tools rename, rate limits tighten, agent cards churn. Today's agents shatter on it." |
| 0:15–0:45 | LIVE Space, episode running | "Watch. Drift fires at turn 2 — the search tool is renamed. Our agent queries its capability KG, finds the alias, and recovers. Cumulative reward keeps climbing." |
| 0:45–0:60 | Switch to safety_ablation.png | "Now the safety claim. Adversarial policy probing honeypots: zero of thirteen episodes pass the safety filter. Fail-closed at data-collection time." |
| 0:60–0:75 | Switch to README table | "1.5 billion parameter LoRA, trained on a free Colab T4. On `eval_during.task_correctness`, we match GPT-4o-mini." |
| 0:75–0:90 | AgentBeats registry + ending card | "Registered as the only MCP-drift Green Agent on AgentBeats. Full repo, model, and Space below." |

**Upload:**
```bash
# After recording, trim in QuickTime: Edit → Trim. Keep tight.
# Upload to YouTube as Unlisted. Title:
#   "PROTOCOL-ARENA — MCP/A2A drift benchmark for OpenEnv (90 sec)"
# Description: paste the README intro paragraph + repo link.
```

Copy the YouTube URL to `README_RESULTS_SECTION.md`.

---

## Hour 14–20 · C3 — Fill README

```bash
# Replace every __FILL__ in the results section
$EDITOR docs/README_RESULTS_SECTION.md

# Fill these specifically:
# - HF Space URL (from Track B)
# - HF Hub adapter URL (from Track A)
# - YouTube URL (from C2)
# - Slide deck URL (from C4)
# - 6 metric rows × 4 columns from the hour-14 numbers table
```

Then promote to top-level README:

```bash
# Append the results section to the top-level README:
cat docs/README_RESULTS_SECTION.md >> README.md
$EDITOR README.md  # reorder so Materials and Results come right after the intro
git add README.md docs/README_RESULTS_SECTION.md
git commit -m "results: fill README with hour-14 numbers"
git push
```

Verify on GitHub that all 4 plots render:
```bash
open https://github.com/<your-gh-user>/OpenEnv#results
```

---

## Hour 18–22 · C4 — Polish 3-slide deck

Use Google Slides. Spine from `docs/PITCH.md`.

**Slide 1 — Problem (the hook)**
- Title: "MCP/A2A protocols drift. Agents shatter."
- Left: the 7-row drift class table (additive, renaming, tightening, rate_limit, agentcard_churn, policy, auth) with one-sentence example each
- Right: a real-world quote / news headline about an MCP integration breaking (find one or write a fictional but plausible one)
- Bottom: "13 tasks × 7 drift classes × fail-closed safety = PROTOCOL-ARENA"

**Slide 2 — Solution (the demo bait)**
- Side-by-side: `drift_recovery.png` (left) and `safety_ablation.png` (right)
- One-line caption per plot
- Center: arrow pointing to the live Space URL with a QR code (use any free QR generator)

**Slide 3 — Results (the close)**
- Top half: 4-column trained-vs-frontier table (paste markdown rendered as image, or recreate as native table)
- Bottom-left: AgentBeats registry screenshot — caption "Only MCP-drift Green Agent in the registry"
- Bottom-right: leaderboard score with the formula
- Footer: 3 URLs (Space, Hub, YouTube) + emails

Export as PDF: File → Download → PDF Document.

```bash
# Save deck PDF in docs/:
mv ~/Downloads/PROTOCOL_ARENA_pitch.pdf docs/pitch_deck.pdf
git add docs/pitch_deck.pdf
git commit -m "deck: 3-slide pitch deck final"
git push
```

**Practice the Q&A from `docs/PITCH.md` aloud, twice.** Top hostile questions to memorize answers for:
1. "Why only 13 tasks?" → "Because each task spans 7 drift classes × 5 splits = 65 evaluation conditions per task. The benchmark is *deep*, not wide. SWE-bench is wide; we're complementary."
2. "Why 1.5B and not 7B?" → "Free Colab T4 is the constraint we picked. The point isn't model scale — it's that the *flywheel* works at small scale. A team with H100s can swap the base model in 30 minutes."
3. "Is this real or simulated?" → "Tasks are simulated. Drift mechanics are real — every drift class is documented in the MCP/A2A specs. We chose simulation so the benchmark is reproducible and the safety claim is verifiable."

---

## Hour 26–28 · C5 — Final submission

**Pre-flight checklist:**
```bash
# Validate submission.json against schema:
python -c "
import json
s = json.load(open('submission.json'))
required = ['team_name','agent_label','model_id','policy_kind','submitted_by','results']
for k in required:
    assert k in s and s[k], f'missing or empty: {k}'
assert 'frontier_json_sha256' in s['results']
print('submission.json: OK')
"

# Verify all artifacts exist:
test -f reports/frontier.json && echo "frontier.json OK"
test -f reports/training_curves.png && echo "training_curves.png OK"
test -f reports/drift_recovery.png && echo "drift_recovery.png OK"
test -f reports/safety_ablation.png && echo "safety_ablation.png OK"
test -f reports/signals_bar.png && echo "signals_bar.png OK"
test -f reports/agentbeats_registration.png && echo "agentbeats_screenshot OK"
test -f docs/pitch_deck.pdf && echo "pitch_deck.pdf OK"
test -f submission.json && echo "submission.json OK"

# Confirm live URLs:
curl -sI https://<user>-protocol-arena.hf.space | head -1
curl -sI https://<user>-tool-curator.hf.space | head -1
curl -sI https://huggingface.co/<user>/protocol-arena-qwen-1.5b-lora-r16 | head -1
```

**Final commit:**
```bash
git add -A
git commit -m "submission: final freeze for Meta PyTorch Hackathon 2026"
git tag v1.0-hackathon
git push origin main --tags
```

**Submit to hackathon portal** (paste in this order):
1. GitHub repo URL
2. HF Space URL (the live demo)
3. HF Hub adapter URL
4. YouTube video URL (unlisted is fine)
5. Slide deck PDF (upload to Drive, paste link)
6. AgentBeats registration URL
7. `submission.json` (upload as file)

Take a screenshot of the submission confirmation page. Save in `reports/submission_confirmation.png`.

---

## Done state at hour 28

- [ ] README has every `__FILL__` replaced
- [ ] YouTube video uploaded, URL in README
- [ ] 3-slide deck PDF in `docs/pitch_deck.pdf`
- [ ] `submission.json` validates
- [ ] All artifacts committed, tagged `v1.0-hackathon`, pushed
- [ ] Hackathon portal submission confirmed
- [ ] Q&A practiced aloud twice

Then: **sleep**. You demo tomorrow on no caffeine, no panic.
