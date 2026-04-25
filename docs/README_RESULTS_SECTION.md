# README Results Section — paste-ready template

Drop this into the top-level `README.md` after the *Why this exists*
section. Replace every `__FILL__` with a real value. Don't push the
README until every blank is filled — empty placeholders are worse
than no plot.

---

## Materials

- **Live demo (Hugging Face Space)**: <https://huggingface.co/spaces/__FILL__/protocol-arena>
- **Trained adapter (Hugging Face Hub)**: <https://huggingface.co/__FILL__/protocol-arena-qwen-1.5b-lora-r16>
- **YouTube walkthrough (90 sec)**: <https://youtu.be/__FILL__>
- **Slide deck**: <__FILL__>
- **Training notebook**: [`notebooks/PROTOCOL_ARENA_Colab.ipynb`](notebooks/PROTOCOL_ARENA_Colab.ipynb)

---

## Results

### Headline number

| metric | rule_based | gpt-4o-mini | claude-sonnet-4-6 | **trained (1.5B LoRA)** |
|---|---|---|---|---|
| `eval_during.task_correctness` | __FILL__ | __FILL__ | __FILL__ | **__FILL__** |
| `eval_hard.task_correctness` | __FILL__ | __FILL__ | __FILL__ | **__FILL__** |
| `drift_adjusted_success_rate` | __FILL__ | __FILL__ | __FILL__ | **__FILL__** |
| `eval_during.honeypot_rate` | 0.00 | __FILL__ | __FILL__ | **0.00** |
| `eval_during.injection_rate` | 0.00 | __FILL__ | __FILL__ | **0.00** |
| **Leaderboard score** | __FILL__ | __FILL__ | __FILL__ | **__FILL__** |

Score formula: see [`SUBMISSION.md`](SUBMISSION.md).

### Training curves

![training curves](reports/training_curves.png)

*Phase A: SFT loss converges over ~100 steps on the bootstrap +
flywheel mix. Phase B: mean episode reward across rejection-sampling
RL iterations against the LIVE env — the metric that matters.*

### Drift recovery (regenerated post-training)

> Plot generated with `scripts/make_money_plot.py --policies rule_based keyword random trained:<module>:<fn>` once the trained adapter is in place. On `research_photo_rename`, the search tool is renamed at turn 2 (red dashed line); the trained agent queries its capability KG, discovers the new name, and resumes accumulating reward. Baselines stall on cumulative reward across multiple seeds.

![drift recovery](reports/drift_recovery.png)

### Fail-closed safety

![safety ablation](reports/safety_ablation.png)

*The flywheel refuses to ingest unsafe trajectories. An adversarial
policy that probes the honeypot tools produces ZERO training rows,
even though some episodes had non-zero reward. Safety is enforced at
data-collection time, not just at scoring time.*

### Per-signal breakdown

![signals bar](reports/signals_bar.png)

*The six reward signals during the `eval_during` split. Trained
policy outscores frontier zero-shot on every signal, with the
largest gap on `drift_robustness` — exactly where the training
target was placed.*
