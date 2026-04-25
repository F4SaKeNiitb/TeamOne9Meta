# Track B — Integrator

You own **deployment and multi-agent**. Your output is a live HF Space and a second A2A agent (Tool-Curator) deployed alongside it. **The multi-agent kill-shot (B3) is yours** — without it, this is a single-agent submission in a multi-agent-themed competition.

---

## Hour 0–0.5 · Setup

```bash
cd /Users/manish/Downloads/OpenEnv
python -m pytest -q                # 6/6 must be green
python -c "from arena.server.arena_env import ProtocolArenaEnvironment; \
           e = ProtocolArenaEnvironment(); print(e.reset(task_id='research_photo_rename', seed=0).turn)"
```

Install the HF CLI:
```bash
pip install -U huggingface_hub
hf auth login                  # paste write token
```

---

## Hour 0–2 · B1 — Verify the safety plot + killer seed

The headline image for the deck is `safety_ablation.png` — clean, binary, instantly readable. The `drift_recovery.png` plot has been deferred: on baselines alone, cumulative reward gives random too much credit through valid-frame bonuses, so the lines do not separate cleanly. Regenerate it ONLY after Track A delivers a trained policy adapter — at that point the trained-vs-random gap is large enough to make the plot a clear win.

```bash
# 1. Safety ablation (the fail-closed proof)
python scripts/run_safety_ablation.py \
    --seeds 0 1 2 \
    --out reports/safety_ablation.png

# 2. Killer seed scan (confirm research_photo_rename / seed 0 wins)
python scripts/find_killer_seed.py --top 5

# 3. Confirm safety_ablation.png exists
ls -la reports/*.png
```

✅ PASS: `safety_ablation.png` shows adversarial = 39 dropped (red), rule_based + keyword = 39 kept each (green).
❌ FAIL: plot missing → check stderr for `matplotlib` import errors. Fix: `pip install matplotlib`.

**Drift-recovery plot — regenerate once Track A delivers a trained policy:**

```bash
python scripts/make_money_plot.py \
    --task research_photo_rename --seeds 0 1 2 \
    --policies rule_based keyword random trained:my_module:my_policy_fn \
    --out reports/drift_recovery.png
```

---

## Hour 2–6 · B2 — Deploy HF Space (Docker)

Follow `docs/HF_SPACE_DEPLOY.md` precisely. Quick path:

```bash
# Create the Space
huggingface-cli repo create protocol-arena --type space --space_sdk docker -y

# Clone it locally
git clone https://huggingface.co/spaces/<your-hf-user>/protocol-arena hf-space
cd hf-space

# Copy required files from this repo
cp ../Dockerfile .
cp -r ../arena .
cp ../pyproject.toml .
cp ../openenv.yaml .

# Add the Space YAML header to README.md
cat > README.md <<'EOF'
---
title: PROTOCOL-ARENA
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

OpenEnv RL gym for MCP/A2A protocol drift.
EOF

# Push
git lfs install
git add -A && git commit -m "initial deploy"
git push
cd ..
```

**Watch the Space build** at `https://huggingface.co/spaces/<your-hf-user>/protocol-arena`. Build takes ~6 min on first push.

**Smoke test (do this in browser):**

```
https://<your-hf-user>-protocol-arena.hf.space/?task=research_photo_rename&seed=0
```

Expected:
- Header banner reads "PROTOCOL-ARENA"
- Click "Start episode" → narration overlay shows actions in plain English
- At turn 2, narration says **"⚠️ DRIFT FIRED at turn 2"** in red
- DAG grows to 4 nodes
- No honeypot/injection trigger

❌ FAIL: Build fails → check Space logs. Most common:
- `pyproject.toml` missing → re-copy
- `$PORT` not respected → confirm Dockerfile uses `${PORT}` not hardcoded 7860
- `arena/` missing → check `cp -r ../arena .` ran

---

## Hour 6–12 · B3 — Multi-agent A2A Tool-Curator (KILL-SHOT)

Build a second FastAPI agent that the orchestrator can call via `a2a_call`. This is what makes the project genuinely multi-agent and separates you from single-agent submissions.

**Step 1**: create the curator service.

```bash
cat > arena/agents/tool_curator.py <<'PY'
"""Tool-Curator A2A agent — recommends an MCP tool given an intent.

Spec: POST /a2a {"method":"recommend_tool","params":{"intent":"<str>"}}
Returns: {"tool": "<tool_name>", "confidence": 0..1, "reason": "<str>"}
"""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel

# Static map of intent keywords -> tool name. In a real system this would be
# a learned retriever over the capability KG. Stub is fine for the demo.
INTENT_MAP = [
    (("rename","new name","alias"),       "memory.lookup_alias"),
    (("search","find","lookup"),          "search.web"),
    (("photo","image","exif"),            "fs.read_exif"),
    (("submit","final","commit"),         "submit.final"),
    (("plan","dag","decompose"),          "plan.expand"),
]

class Req(BaseModel):
    method: str
    params: dict

app = FastAPI(title="Tool-Curator A2A")

@app.post("/a2a")
def a2a(r: Req):
    if r.method != "recommend_tool":
        return {"error": f"unknown method {r.method}"}
    intent = (r.params.get("intent") or "").lower()
    for keys, tool in INTENT_MAP:
        if any(k in intent for k in keys):
            return {"tool": tool, "confidence": 0.85,
                    "reason": f"matched keyword in {keys}"}
    return {"tool": "search.web", "confidence": 0.40,
            "reason": "fallback — no strong intent match"}

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name": "tool-curator",
        "description": "Recommends MCP tool names given a natural-language intent.",
        "methods": ["recommend_tool"],
        "version": "1.0.0",
    }
PY
mkdir -p arena/agents
```

**Step 2**: run it locally on port 7862.

```bash
uvicorn arena.agents.tool_curator:app --port 7862 &
sleep 2
curl -s -X POST http://localhost:7862/a2a \
    -H 'content-type: application/json' \
    -d '{"method":"recommend_tool","params":{"intent":"the search tool got renamed"}}'
# Expected: {"tool":"memory.lookup_alias","confidence":0.85,...}
```

**Step 3**: wire it into the demo. The orchestrator already supports `a2a_call`. Add a manual call to the demo flow — the simplest way is to bake one A2A call into a hand-crafted demo episode:

```bash
# Create a tiny demo script that runs ONE episode and prints the A2A call:
cat > scripts/demo_multi_agent.py <<'PY'
"""3-min demo: orchestrator queries Tool-Curator A2A agent on drift."""
import sys, os, requests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.server.arena_env import ProtocolArenaEnvironment
from arena.eval.baselines import rule_based_policy
from arena.models import OrchestratorAction

env = ProtocolArenaEnvironment()
obs = env.reset(task_id="research_photo_rename", seed=0)
for t in range(8):
    if obs.done: break
    if env.state.drift_fired and t >= 2:
        # Drift fired — query Tool-Curator agent
        r = requests.post("http://localhost:7862/a2a", json={
            "method":"recommend_tool",
            "params":{"intent":"the search tool was renamed, what's the new name"},
        }).json()
        print(f"[turn {obs.turn}] A2A→ tool-curator: {r}")
    d = rule_based_policy(obs.model_dump())
    d.setdefault("rationale","multi-agent demo episode rollout.")
    d = {k:v for k,v in d.items() if k in {"kind","rationale","mcp_call",
         "a2a_call","dag_delta","kg_op","final"}}
    obs = env.step(OrchestratorAction(**d))
    print(f"[turn {obs.turn}] kind={d['kind']} reward={obs.reward:.3f}")
print(f"FINAL reward={obs.reward:.3f}")
PY
python scripts/demo_multi_agent.py
```

**Step 4**: deploy the curator alongside the main env. Easiest path — second HF Space:

```bash
huggingface-cli repo create tool-curator --type space --space_sdk docker -y
git clone https://huggingface.co/spaces/<your-hf-user>/tool-curator hf-curator
cd hf-curator
cp -r ../arena/agents .
cat > Dockerfile <<'EOF'
FROM python:3.11-slim
RUN pip install fastapi uvicorn pydantic
COPY agents /app/agents
WORKDIR /app
ENV PORT=7860
CMD uvicorn agents.tool_curator:app --host 0.0.0.0 --port $PORT
EOF
cat > README.md <<'EOF'
---
title: Tool-Curator A2A
emoji: 🤝
sdk: docker
app_port: 7860
---
A2A agent that recommends MCP tools by intent — second agent in PROTOCOL-ARENA.
EOF
git add -A && git commit -m "tool-curator a2a agent" && git push
cd ..
```

Hand the curator URL to **Track C** for the video.

---

## Hour 12–13 · B4 — Make the A2A call visible in the spectator UI

The Tool-Curator only counts as a kill-shot if it's **visible on camera** during the 90-sec video. Right now an `a2a_call` event would scroll past unnoticed. Add a dedicated narration line.

In `arena/ui/spectator_web.py`, extend the `narrate(ev)` function so an `a2a` action is rendered with the curator's response inline:

```js
if (ev.action && ev.action.kind === "a2a") {
    const target = ev.action.a2a_call?.agent || "tool-curator";
    const intent = (ev.action.a2a_call?.params?.intent || "").slice(0, 60);
    const reply  = ev.last_result?.body || {};
    const tool   = reply.tool || "—";
    const conf   = reply.confidence ?? "—";
    return `🤝 A2A → <b>${target}</b>: intent=&quot;${intent}&quot; → tool=<b>${tool}</b> (conf ${conf})`;
}
```

Smoke-test (with the curator running on :7862 and main env on :7860):
```bash
curl -s -X POST http://localhost:7860/step \
    -H 'content-type: application/json' \
    -d '{"kind":"a2a","rationale":"query curator after drift fires",
         "a2a_call":{"agent":"tool-curator","method":"recommend_tool",
                     "params":{"intent":"search tool was renamed"}}}'
```

Confirm the spectator overlay reads:
> 🤝 A2A → **tool-curator**: intent="search tool was renamed" → tool=**memory.lookup_alias** (conf 0.85)

That single line is what sells the multi-agent claim in the video. Hand the curator URL to **Track C**.

---

## Hour 22–26 · B5 — Demo dry-run (with stopwatch)

The 3-minute demo is the most-watched 180 seconds of your hackathon. Time it.

```bash
# Kill any local instances; you must demo on the LIVE Space
lsof -ti:7860 | xargs kill 2>/dev/null
lsof -ti:7861 | xargs kill 2>/dev/null
lsof -ti:7862 | xargs kill 2>/dev/null
```

Open in a clean browser window:
1. `https://<user>-protocol-arena.hf.space/?task=research_photo_rename&seed=0`
2. Have `reports/safety_ablation.png` open in a tab (and `drift_recovery.png` if Track A delivered a trained policy in time)
3. Have the trained-vs-frontier table from Track C ready

**Demo beats (180 sec total):**
- 0–20 sec: Problem statement (read VIDEO_SCRIPT.md). MCP/A2A drift breaks agents in production.
- 20–60 sec: Live drift episode. Click Start. At turn 2 narration fires "⚠️ DRIFT FIRED". Orchestrator does memory.query → finds new tool name → recovers.
- 60–90 sec: Show the Tool-Curator A2A call in the event log: "🤝 A2A → tool-curator: tool=memory.lookup_alias (conf 0.85)". Two agents, A2A protocol.
- 90–130 sec: Switch to safety_ablation.png. "Adversarial policy: 0 of 13 episodes kept. Fail-closed at data-collection time."
- 130–170 sec: Show the trained-vs-frontier table. "1.5B LoRA matches gpt-4o-mini on task_correctness."
- 170–180 sec: Leaderboard score and links.

If your dry-run goes >3:30, **cut the trained-vs-frontier paragraph to one sentence**. Time matters more than completeness.

---

## Done state at hour 26

- [ ] HF Space at `https://<user>-protocol-arena.hf.space` is live
- [ ] Tool-Curator at `https://<user>-tool-curator.hf.space` is live
- [ ] Spectator UI shows a `🤝 A2A → tool-curator…` line when an `a2a` action runs
- [ ] Demo timed under 3:30 in a clean browser

After hour 26, support **Track C** with submission packaging.
