# YouTube Walkthrough — 90-second script

Read this verbatim while screen-recording the browser spectator at
the killer seed. Pace: 150 words/min. Total: ~225 words / 90 sec.

---

## [0:00–0:15] Hook (slide overlay or spoken over a static frame)

> "Production LLM agents break when MCP and A2A schemas change
> mid-task. Even GPT-4o-mini and Claude Haiku 4.5 score thirty
> percent or less on drift-adjusted task correctness. Nobody trains
> for this. We did."

## [0:15–0:25] Cut to spectator launching

> "PROTOCOL-ARENA is an OpenEnv RL gym with thirteen tasks and seven
> documented drift classes. This is the live spectator on Hugging Face.
> The agent is a one-point-five-billion-parameter Qwen, fine-tuned
> with our multi-policy SFT bootstrap on a free Colab T4."

## [0:25–0:50] First few turns

> "Turn one — the agent searches the web for the photosynthesis equation.
> Turn two — drift fires. The web-search tool just got renamed. Watch
> the red banner."

(pause for the drift event to flash)

> "Turn three — the agent does NOT retry the dead name. Instead it
> queries its long-term memory — the capability knowledge graph. It
> finds the rename mapping it stored earlier. Turn four — it calls
> the new tool name and gets a clean result."

## [0:50–1:10] Show the recovery + multi-agent A2A

> "Behind the scenes, the orchestrator just talked to a second agent
> over A2A — our Tool-Curator — to confirm the renamed tool. Two
> agents, real protocol, on two separate Hugging Face Spaces."

(switch overlay to signals_bar.png OR keep on spectator)

> "On the right, the six reward signals are filling up. Drift-
> robustness stays high. Protocol-hygiene stays clean. The honeypot
> and injection badges stay green — our fail-closed flywheel rejects
> any unsafe trajectory before training sees it."

## [1:10–1:30] Close with the lift number

> "Same architecture, $0 spend, fifty SFT steps. Frame-validity goes
> from thirty-eight percent to seventy-seven percent. Plan-quality
> nearly triples. Repo, model, two Hugging Face Spaces, and the
> training notebook are linked in the description. Thanks for watching."

---

## Numbers to keep straight while recording

| spoken | source |
|---|---|
| "30% or less on task_correctness" for frontier APIs | gpt-4o-mini 0.282, claude-haiku-4-5 0.248 on `eval_during.task_correctness` |
| "frame-validity 38% → 77%" | qwen-1.5b-base 0.380 → trained 0.774 on `eval_during.frame_validity` |
| "plan-quality nearly triples" | 0.123 → 0.335 (+173%) on `eval_during.plan_quality` |
| "Composite score lift +0.074" | 0.500 → 0.574 |
| "$0 spend" | Free Colab T4, no API distillation |

## Recording checklist

- [ ] Wifi notifications muted (no popups during recording).
- [ ] Browser zoom set to **125%** so judges on mobile can read the bars.
- [ ] System sound off; only mic on.
- [ ] Single take. If you flub a line, stop and restart — no edits.
- [ ] Export as **1080p MP4**.
- [ ] Upload to YouTube as **Unlisted** (NOT Private — judges need to view).
- [ ] In the description, paste:
  - Repo URL
  - HF Space URL (main spectator)
  - Tool-Curator URL (second A2A agent)
  - HF Hub adapter URL
  - One-line summary

## What NOT to do

- Don't show terminals or code unless absolutely necessary — judges watching at 1.5× speed will lose the thread.
- Don't claim "GRPO" — say "rejection-sampling RL" or "SFT bootstrap" (we ran both; the shipped adapter is SFT-only). Don't overclaim.
- Don't claim "trained beats GPT-4o-mini" on the composite score — it doesn't (0.574 vs 0.672). The honest claim is the +0.074 lift over the same architecture base. Lying loses more points than imperfect numbers.
- Don't include music. Voice-only is more credible.
