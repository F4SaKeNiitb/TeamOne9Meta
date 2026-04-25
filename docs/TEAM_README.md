# Team — read this first (5 min)

You have **3 people, 30 hours, and one goal**: ship a podium-worthy submission to Meta PyTorch Hackathon 2026.

## Pick your track

Open the file matching your role and follow it hour-by-hour. Each track is self-contained — terminal commands inside, not just guidance.

| Track | File | Owns | Critical path? |
|---|---|---|---|
| **A — Trainer** | [`TRACK_A_TRAINER.md`](TRACK_A_TRAINER.md) | Colab notebook, RL, frontier baseline, `submission.json` | ✅ longest |
| **B — Integrator** | [`TRACK_B_INTEGRATOR.md`](TRACK_B_INTEGRATOR.md) | HF Space, multi-agent A2A, AgentBeats registry | ✅ shared |
| **C — Storyteller** | [`TRACK_C_STORYTELLER.md`](TRACK_C_STORYTELLER.md) | Video, README, deck, hackathon submission | drafts in parallel |

## The 2 kill-shots (these are why we win, not just submit)

1. **Track A · Frontier baseline** — gpt-4o-mini and claude-haiku-4-5 in `reports/frontier.json`. Without this, "trained" is meaningless.
2. **Track B · Multi-agent A2A demo** — Tool-Curator agent on a 2nd HF Space, with the A2A call **visible in the spectator UI event log** during the video.

Both must land. Without #1, the trained column is empty. Without #2, you're a single-agent submission in a multi-agent-themed competition.

## Sync points (mandatory, all-hands, slack/voice)

| Hour | Check | If failing |
|---|---|---|
| **14** | Training done? Space live? Frontier cached? | Trigger SFT-only fallback (see Track A bottom section) |
| **22** | Video recorded? Deck draft done? Multi-agent live? | Drop multi-agent if not done; never drop video |
| **28** | All artifacts present? `submission.json` validates? URLs live? | Hard freeze. No more code. |

## Hard rules

- **No new features after hour 14.** Polish only.
- **No code changes after hour 28.** Submit and sleep.
- **`reports/safety_ablation.png` is the headline visual** — adversarial 39 dropped (red) vs rule_based/keyword 39 kept (green). Track B verifies in hour 0–2. The drift-recovery plot is regenerated AFTER Track A delivers a trained policy (otherwise random's variance masks the story).
- **Storytelling is 30%. Reward improvement is 20%.** A clean demo with a smaller model improvement beats a messy demo with a bigger one.

## Communication

- Use a single shared chat. Post checkpoint outputs (logs, screenshots, URLs) so the other tracks can see progress.
- Track A handoff to Track C is the most important — `reports/frontier.json` and adapter URL must arrive by hour 14.
- Track B handoff to Track C — Space URL by hour 6, AgentBeats screenshot by hour 13.

## What's already done (don't redo it)

- ✅ All 13 tasks, 7 drift classes, 6-signal reward — in `arena/`
- ✅ Honeypot + injection adversarial layer
- ✅ Capability KG (SQLite + FTS5)
- ✅ GraphSAGE plan scorer
- ✅ Drift-Adjusted Success Rate + Brier sentinel
- ✅ Spectator UI with narration overlay
- ✅ Killer-seed scanner (drift fires at turn 2, `research_photo_rename` seed 0)
- ✅ Money plot generator
- ✅ Safety ablation (verified: 0/13 unsafe rows kept for adversarial)
- ✅ Colab notebook for online RL on Qwen2.5-1.5B
- ✅ Score submission tool
- ✅ Dockerfile fixed for HF Spaces
- ✅ Pitch, video, deploy, README templates

## What's NOT done (this is your work)

- ❌ Run the Colab notebook end-to-end (Track A)
- ❌ Frontier API runs (Track A, after Track C wires keys)
- ❌ Live HF Space deploy (Track B)
- ❌ Tool-Curator A2A agent (Track B)
- ❌ AgentBeats registration (Track B)
- ❌ 90-sec YouTube video (Track C)
- ❌ Filled README, deck, submission (Track C)

## Start order — right now

```bash
# All three of you in parallel:
# A: open Colab, start Cell 1
# B: cd /Users/manish/Downloads/OpenEnv && python scripts/find_killer_seed.py --top 5
# C: get OPENAI_API_KEY and ANTHROPIC_API_KEY ready
```

Now read your track file. Go.
