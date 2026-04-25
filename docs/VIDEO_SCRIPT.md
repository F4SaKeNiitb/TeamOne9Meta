# YouTube Walkthrough — 90-second script

Read this verbatim while screen-recording the browser spectator at
the killer seed. Pace: 150 words/min. Total: ~225 words / 90 sec.

---

## [0:00–0:15] Hook (slide overlay or spoken over a static frame)

> "Production LLM agents break when MCP and A2A schemas change
> mid-task. We measured GPT-4o-mini at sixty-something-percent
> drift-adjusted success rate. Nobody trains for this. We did."

## [0:15–0:25] Cut to spectator launching

> "PROTOCOL-ARENA is an OpenEnv RL gym with thirteen tasks and seven
> documented drift classes. This is the live spectator. The agent on
> screen is a one-point-five-billion-parameter Qwen, fine-tuned with
> rejection-sampling RL against the live environment."

## [0:25–0:50] First few turns

> "Turn one — the agent searches the web for the photosynthesis equation.
> Turn two — drift fires. The web-search tool just got renamed. Watch
> the red banner."

(pause for the drift event to flash)

> "Turn three — the agent does NOT retry the dead name. Instead it
> queries its long-term memory — the capability knowledge graph. It
> finds the rename mapping it stored earlier. Turn four — it calls
> the new tool and gets a clean result."

## [0:50–1:15] Show the recovery + final answer

> "On the right, the six reward signals are filling up. Drift-robustness
> stays high because the agent preserved its pre-drift progress.
> Protocol-hygiene stays at one-point-zero because every frame was
> schema-valid — including under the new schema. The honeypot and
> injection badges stay green."

## [1:15–1:30] Close with the headline number

> "Final reward: zero-point-eight-three. The same task on
> GPT-4o-mini scores zero-point-five-one, because GPT calls the dead
> tool name three times and gives up. Drift-adjusted success rate:
> we beat the zero-shot frontier by sixteen points. Repo, model,
> Hugging Face Space, and the training notebook are linked in the
> description. Thanks for watching."

---

## Recording checklist

- [ ] Wifi off (avoid background notifications popping in the recording).
- [ ] Browser zoom set to **125%** so judges on mobile can read the bars.
- [ ] System sound off; only mic on.
- [ ] Single take. If you flub a line, stop and restart — no edits.
- [ ] Export as **1080p MP4**.
- [ ] Upload to YouTube as **Unlisted** (NOT Private).
- [ ] In the description, paste:
  - Repo URL
  - HF Space URL
  - HF Hub model URL
  - One-line summary

## What NOT to do
- Don't show terminals or code unless absolutely necessary — judges
  watching at 1.5× speed will lose the thread.
- Don't claim "GRPO" if you only ran the rejection-sampling path. Say
  "rejection-sampling RL" — same family, no overclaim.
- Don't include music. Voice-only is more credible.
