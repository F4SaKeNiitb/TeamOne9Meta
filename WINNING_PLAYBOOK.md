# Winning Playbook — 30 hours, 3 people

This is the single source of truth for the next 30 hours. Everything
in here is required, in this order, with no detours. If you finish a
section early, **don't add features** — start the next section.

The judging weights:
- **Innovation 40%** (already locked — don't touch)
- **Storytelling 30%** (Track C owns)
- **Reward improvement 20%** (Track A owns — most time-sensitive)
- **Pipeline coherence 10%** (covered by the existing repo)

---

## Roles

- **A — TRAINER.** Owns the Colab notebook, the trained checkpoint, the reward-curve PNG, and the HF Hub model upload.
- **B — INTEGRATOR.** Owns the live-frontier eval, the money-plot, the safety-ablation, the Hugging Face Space deployment.
- **C — STORYTELLER.** Owns the demo seed, the spectator narration walkthrough, the YouTube video, the slide deck, and the README results section.

---

## Hour 0 — kickoff (15 min, everyone)

```bash
git pull
pip install -e .
pytest tests/ -q                       # MUST pass 6/6
python -m arena.ui.spectator_web --port 7861
# open http://localhost:7861/?task=research_photo_rename&seed=0
```

If anything is red, fix that first. **Then split.**

Create three branches:
```bash
git checkout -b track-A-trainer
git checkout -b track-B-integrator
git checkout -b track-C-storyteller
```

---

## Track A — Trainer (longest critical path; START IMMEDIATELY)

### A1. Open Colab + GPU
- Upload `notebooks/PROTOCOL_ARENA_Colab.ipynb` to Colab.
- Runtime → Change runtime → **T4 (free) or L4** GPU.
- Edit cell 1: replace `<YOUR-GH-USER>` with your repo URL.

### A2. Run the notebook top-to-bottom
The notebook is **already wired correctly** for online RL: Phase B
calls `model.generate()` → `env.step()` → `env.step()` … through
multi-turn rollouts, scores them with the live env reward, and only
fine-tunes on the high-reward safe top-quartile. **This is what
satisfies the "training loop connects to the environment, not a static
dataset" criterion.** Do not change this.

Expected wall clock: ~60–90 min on T4, ~40 min on L4.

### A3. After the notebook finishes
You should have:
- `reports/sft_loss.csv` — Phase A loss per step
- `reports/rl_curve.csv` — Phase B mean reward per RL iteration
- **`reports/training_curves.png`** ← the headline plot
- `reports/frontier.json` — trained vs rule_based eval
- `outputs/rl_final/` — the final LoRA adapter

Commit all of `reports/*.png`, `reports/*.csv`, and
`reports/frontier.json` to the repo.

### A4. Push the adapter to HF Hub
```python
from huggingface_hub import login; login()         # paste write token
model.push_to_hub('YOUR-ORG/protocol-arena-qwen-1.5b-lora-r16')
tok.push_to_hub('YOUR-ORG/protocol-arena-qwen-1.5b-lora-r16')
```
Save the URL — Track B and C need it.

### A5. If RL diverges or OOMs
**Fall-back protocol — do this fast, don't sink time:**
1. If 1.5B OOMs, change `BASE` to `Qwen/Qwen2.5-0.5B-Instruct`.
2. If RL mean reward goes DOWN, set `N_ITERS = 2` and live with a 2-point curve. A short curve is still a curve.
3. If RL fully fails, ship Phase A SFT only and rename the plot caption to "Phase A — SFT loss". You still have a training story.

**Hard rule: if no `reports/training_curves.png` by hour 14, declare done and switch to helping Track C.** A submission with a real SFT-loss plot beats a submission waiting for GRPO to converge.

---

## Track B — Integrator (run in parallel with A from hour 0)

### B1. Live-frontier eval (needs API keys)
```bash
export OPENAI_API_KEY=sk-…
export ANTHROPIC_API_KEY=sk-ant-…
# Optional: export HF_TOKEN=hf_… for Qwen via HF inference proxy

python -m arena.eval.run_frontier --seeds 0 1 2 --out reports/frontier_live.json
python -m arena.eval.report --in reports/frontier_live.json --out reports/
python scripts/score_submission.py reports/frontier_live.json
```
Save `reports/frontier_live.json` and the scoreboard screenshot.

When Track A finishes, merge the trained-policy block from A's
`reports/frontier.json` into `reports/frontier_live.json` so the same
file has BOTH the trained agent AND the frontier APIs in one ranked
table.

### B2. The money plot
```bash
python scripts/find_killer_seed.py --top 5
# pick the (task, seed) row 1; export it as KILLER_TASK / KILLER_SEED
python scripts/make_money_plot.py \
    --task $KILLER_TASK --seed $KILLER_SEED \
    --policies rule_based keyword \
    --out reports/drift_recovery.png
```
Once Track A pushes the adapter, re-run the money plot WITH the
trained policy:
```bash
python scripts/make_money_plot.py \
    --task $KILLER_TASK --seed $KILLER_SEED \
    --policies rule_based keyword trained:scripts.trained_adapter:policy \
    --out reports/drift_recovery.png
```
You may need to write a tiny `scripts/trained_adapter.py` that loads
the LoRA from HF Hub and exposes a `policy(obs)` callable. It's a
~30-line file that wraps `model.generate()`.

### B3. Safety ablation
```bash
python scripts/run_safety_ablation.py --seeds 0 1 2 \
    --out reports/safety_ablation.png
```
This proves the fail-closed claim: the adversarial policy contributes
zero rows to training data. Commit the PNG.

### B4. Hugging Face Space deploy
```bash
# Already logged into HF from Track A4? Reuse the token.
huggingface-cli repo create protocol-arena --type space --space_sdk docker
git remote add hf https://huggingface.co/spaces/YOUR-USER/protocol-arena
git push hf track-B-integrator:main
```
The existing `Dockerfile` is HF-Spaces-compatible (binds to `$PORT`,
copies all source). Spaces builds it automatically.

If the Space build fails, the most common cause is `pyproject.toml`
having a build-system entry that pulls in cython/wheel. Comment that
out for the Space-only branch.

Save the Space URL: `https://huggingface.co/spaces/YOUR-USER/protocol-arena`.

### B5. Add an HF YAML header to the README
Once you've created the Space, prepend this to `README.md` so HF
Spaces picks up the metadata:

```yaml
---
title: PROTOCOL-ARENA
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: docker
pinned: true
license: apache-2.0
---
```

---

## Track C — Storyteller (start at hour 0; finishes last)

### C1. Pick the killer seed (parallel with B2)
Use `scripts/find_killer_seed.py --top 5`. Pick the row whose drift
event is most narratable — `research_photo_rename` seed 0 is a strong
default because the rename is intuitive even to non-engineers.
Document the choice in `DEMO.md`:

```markdown
# DEMO

Killer seed: research_photo_rename, seed 0.
Drift fires at turn 2: web.search → query_web.
Expected pitch line: "watch the search tool get renamed mid-task and
the agent recover via its long-term memory."
```

### C2. Rehearse the spectator demo (with narration)
```bash
python -m arena.ui.spectator_web --port 7861
# open the killer seed URL — narration strip is at top
```
The new narration overlay translates each turn into plain English. A
non-technical judge can read along while the DAG and bars move.
Practice the demo three times. Time it. Aim for **75 seconds** including drift.

### C3. Record the YouTube video
- Use OBS or Loom or QuickTime → File → New Screen Recording.
- Record the spectator at the killer seed once cleanly.
- Voiceover script is in `docs/VIDEO_SCRIPT.md` — read it verbatim.
- Total length: **90 seconds.** Don't go over 2 min (rule).
- Upload to YouTube as **Unlisted**, save the URL.

### C4. Build the slide deck
- Use the outline in `docs/PITCH.md` — three slides, copy-paste-ready.
- Export as PDF or share as Google Slides public link.

### C5. README results section
After Track A and Track B finish, fill in the README results section
template (in `docs/README_RESULTS_SECTION.md`) with the actual
numbers and embed the four key plots:
- `reports/training_curves.png` (Phase A + Phase B)
- `reports/drift_recovery.png` (the money plot)
- `reports/safety_ablation.png` (fail-closed proof)
- `reports/signals_bar.png` (per-signal breakdown)

Each plot needs a one-line caption directly below it in the README.

### C6. Add the materials links section to README
At the top, near the badge area:

```markdown
## Materials
- **Live demo (HF Space)**: <https://huggingface.co/spaces/YOUR-USER/protocol-arena>
- **Trained model (HF Hub)**: <https://huggingface.co/YOUR-USER/protocol-arena-qwen-1.5b-lora-r16>
- **YouTube walkthrough (90s)**: <https://youtu.be/YOUR-VIDEO-ID>
- **Slides**: <link-to-slides>
- **Training notebook**: [`notebooks/PROTOCOL_ARENA_Colab.ipynb`](notebooks/PROTOCOL_ARENA_Colab.ipynb)
```

This is the section judges look for first.

---

## Hour-by-hour skeleton

| Hour | A (trainer) | B (integrator) | C (storyteller) |
|---|---|---|---|
| 0 | A1 setup Colab | B1 live frontier eval | C1 pick killer seed |
| 1–6 | A2 notebook running | B2 money plot (rule_based+keyword) | C2 rehearse demo |
| 6–10 | A2 finishing | B3 safety ablation; B4 HF Space deploy | C3 record video |
| 10–14 | A3 commit plots; A4 push to Hub | B5 README HF header; help A if blocked | C4 slide deck |
| 14–18 | (free — help C) | B2 re-run money plot WITH trained policy | C5 README results |
| 18–22 | help C with technical deck Q&A | merge frontier_live.json with trained block | C6 materials links |
| 22–26 | rehearse pitch | rehearse pitch | rehearse pitch |
| 26–28 | smoke test on demo laptop | smoke test on demo laptop | smoke test on demo laptop |
| 28–30 | submit + present | submit + present | submit + present |

---

## Submission checklist (final 2 hours)

Tick all of these before submitting. Missing any → "serious disadvantage" per the rules.

- [ ] Repo is **public** on GitHub.
- [ ] `README.md` has the **Materials** section with all 5 links.
- [ ] `README.md` embeds **all four plots** with captions.
- [ ] HF Space builds and serves at the saved URL.
- [ ] HF Hub adapter is uploaded and downloadable.
- [ ] YouTube video is **Unlisted** and under 2 min.
- [ ] Slide deck link works in incognito.
- [ ] `pytest tests/ -q` passes on the demo laptop.
- [ ] `git tag v1.0-hackathon && git push --tags`.
- [ ] **Submission portal:** repo URL + HF Space URL + materials links submitted by deadline. **One submission per team — pick the right URL.**

---

## What NOT to do

- Don't add new tasks, drift classes, or signals. Innovation is locked.
- Don't refactor the env code "while you're here."
- Don't use `git push --force` to main.
- Don't commit large video files to the repo (rules forbid it for HF env submissions).
- Don't submit before A's training plots and B's HF Space are ready, even if you're tired. **Late ≠ disqualified; missing ≠ disadvantaged.**
- Don't paste API keys into commits.
- Don't promise more than the plots show. If trained-vs-frontier shows a 5-point gain, say "5-point gain", not "dramatically beats."
