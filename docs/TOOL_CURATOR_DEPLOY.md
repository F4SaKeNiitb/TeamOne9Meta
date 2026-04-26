# Tool-Curator — second HF Space deploy

Step-by-step instructions to deploy the **Tool-Curator A2A agent** as a
second Hugging Face Space alongside the main PROTOCOL-ARENA spectator.

This is the **multi-agent kill-shot** for the submission. Without a second
agent, the project mentions A2A but has nothing live to demonstrate. With
this second Space, judges can `curl` two distinct agents on two URLs and
see real A2A protocol traffic between them.

Total deploy time: **~10 min** (most of it is the HF build).

---

## What you're shipping

A FastAPI service that exposes two endpoints:

- `GET /.well-known/agent.json` — A2A discovery / agent card (standard
  endpoint name; recognised by any A2A-aware client)
- `POST /a2a` — body
  ```json
  {"method": "recommend_tool", "params": {"intent": "<natural-language string>"}}
  ```
  → response
  ```json
  {"tool": "<mcp-tool-name>", "confidence": 0.0..1.0, "reason": "<str>"}
  ```

Source code: [`arena/agents/tool_curator.py`](../arena/agents/tool_curator.py).
~70 lines, no PyTorch, no env code — the Docker image is ~50 MB.

---

## Pre-flight: confirm the curator works locally

```bash
cd /Users/manish/Downloads/OpenEnv
python3 -c "
from fastapi.testclient import TestClient
from arena.agents.tool_curator import app
c = TestClient(app)
print('agent_card:', c.get('/.well-known/agent.json').status_code)
print('a2a:', c.post('/a2a',
    json={'method':'recommend_tool','params':{'intent':'rename'}}).json())
"
```

Expected output:

```
agent_card: 200
a2a: {'tool': 'memory.lookup_alias', 'confidence': 0.85, 'reason': "matched keywords ['rename', 'new name', 'alias'] in intent"}
```

If this fails, fix the curator file before deploying.

---

## Step 1 — create the empty Space on HF

Open in your browser: <https://huggingface.co/new-space>

Fill in:

| Field | Value |
|---|---|
| Owner | `Kashishshaikh` |
| Space name | `tool-curator` |
| License | `apache-2.0` |
| Space SDK | **Docker** → **Blank** template |
| Space hardware | CPU basic (free) |
| Visibility | Public |

Click **Create Space**. You'll land on
`https://huggingface.co/spaces/Kashishshaikh/tool-curator`.

> Alternative if you have `huggingface-cli` set up:
> ```bash
> huggingface-cli repo create tool-curator --type space --space_sdk docker -y
> ```

---

## Step 2 — clone the empty Space repo locally

```bash
cd /Users/manish/Downloads
git clone https://huggingface.co/spaces/Kashishshaikh/tool-curator hf-curator
cd hf-curator
```

If git asks for credentials: username = your HF username, password = an HF
**write** token (https://huggingface.co/settings/tokens — create one if you
don't have it).

---

## Step 3 — populate the Space directory

The curator needs only 3 files: a `Dockerfile`, a `README.md` with the
Space metadata header, and the curator Python module.

Run these from inside `hf-curator/`:

```bash
# Bring in the curator code (only the agents/ folder is needed —
# no PyTorch, no env code, ~50 MB image instead of ~500 MB)
mkdir -p agents
cp /Users/manish/Downloads/OpenEnv/arena/agents/__init__.py    agents/__init__.py
cp /Users/manish/Downloads/OpenEnv/arena/agents/tool_curator.py agents/tool_curator.py

# Tiny Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic
COPY agents /app/agents
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn agents.tool_curator:app --host 0.0.0.0 --port ${PORT}"]
EOF

# HF Space metadata header in README
cat > README.md <<'EOF'
---
title: Tool-Curator A2A
emoji: 🤝
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: A2A agent that recommends MCP tools by intent
---

# Tool-Curator (A2A agent)

Second agent in the **PROTOCOL-ARENA** demo. Receives `recommend_tool`
A2A calls and returns an MCP tool name based on natural-language intent.

## Endpoints

- `GET /.well-known/agent.json` — A2A discovery / agent card
- `POST /a2a` — body:
  `{"method": "recommend_tool", "params": {"intent": "<str>"}}`
  → `{"tool": "<name>", "confidence": 0..1, "reason": "<str>"}`

## Try it

```bash
curl -s https://Kashishshaikh-tool-curator.hf.space/.well-known/agent.json
curl -s -X POST https://Kashishshaikh-tool-curator.hf.space/a2a \
    -H 'content-type: application/json' \
    -d '{"method":"recommend_tool","params":{"intent":"search tool was renamed"}}'
```

Paired with the main spectator at
[Kashishshaikh/protocol-arena](https://huggingface.co/spaces/Kashishshaikh/protocol-arena).
EOF
```

Verify the layout:

```bash
ls -la
# Dockerfile
# README.md
# agents/__init__.py
# agents/tool_curator.py
```

---

## Step 4 — push and watch the build

```bash
git add Dockerfile README.md agents/
git commit -m "Tool-Curator A2A agent — recommends MCP tools by intent"
git push
```

If the push prompts for credentials again, paste your HF write token.

Open **https://huggingface.co/spaces/Kashishshaikh/tool-curator** in a
browser. The **Logs** tab shows the build:

| Stage | Time |
|---|---|
| Pulling `python:3.11-slim` | ~30 sec |
| `pip install fastapi uvicorn pydantic` | ~30 sec |
| Container starts → status flips to **Running** | ~2 min total |

Faster than the main Space because there's no PyTorch / matplotlib /
openenv-core in the image.

---

## Step 5 — smoke test

Once the Space is **Running**, run these four `curl`s from your laptop:

```bash
# 1. Agent card (A2A discovery)
curl -s https://Kashishshaikh-tool-curator.hf.space/.well-known/agent.json \
    | python3 -m json.tool

# 2. Rename intent → memory.lookup_alias
curl -s -X POST https://Kashishshaikh-tool-curator.hf.space/a2a \
    -H 'content-type: application/json' \
    -d '{"method":"recommend_tool","params":{"intent":"the search tool was renamed"}}' \
    | python3 -m json.tool

# 3. Photo exif intent → fs.read_exif
curl -s -X POST https://Kashishshaikh-tool-curator.hf.space/a2a \
    -H 'content-type: application/json' \
    -d '{"method":"recommend_tool","params":{"intent":"read photo exif data"}}' \
    | python3 -m json.tool

# 4. Fallback when no keyword matches → search.web (low confidence)
curl -s -X POST https://Kashishshaikh-tool-curator.hf.space/a2a \
    -H 'content-type: application/json' \
    -d '{"method":"recommend_tool","params":{"intent":"asdf qwerty"}}' \
    | python3 -m json.tool
```

All four should return JSON with status 200 in well under a second.

Expected first response (the agent card):

```json
{
  "name": "tool-curator",
  "description": "Recommends MCP tool names given a natural-language intent.",
  "methods": ["recommend_tool"],
  "version": "1.0.0",
  "schema": {
    "recommend_tool": {
      "params": {"intent": "string"},
      "returns": {"tool": "string", "confidence": "number", "reason": "string"}
    }
  }
}
```

---

## Step 6 — verify the README links resolve

Reload the main repo's `README.md` on GitHub. This line should now point
at a live page (no 404):

```
- **Tool-Curator (second A2A agent)**: <https://huggingface.co/spaces/Kashishshaikh/tool-curator>
```

Same for the live API URL `https://Kashishshaikh-tool-curator.hf.space/`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build fails with `COPY agents /app/agents` error | `agents/` folder wasn't copied or has a typo | `ls agents/` from inside `hf-curator` — should show `__init__.py` and `tool_curator.py` |
| Build succeeds but Space stuck on "Building" indefinitely | Port mismatch | Confirm `app_port: 7860` in README header AND `${PORT}` in Dockerfile CMD |
| `/a2a` returns 422 Unprocessable Entity | Pydantic schema mismatch | The `A2ARequest` model requires `method` (string) and `params` (dict). Send both. |
| `curl` returns HTML instead of JSON | You hit `https://huggingface.co/spaces/...` (management page) instead of `https://Kashishshaikh-tool-curator.hf.space/` (live app) | Use the `.hf.space` subdomain for API calls |
| 404 on `/.well-known/agent.json` after rebuild | Old build cached | HF page → **Settings** → **Factory rebuild** |
| `git push` hangs forever | LFS hook waiting for large files; we have none | `git lfs uninstall --local && git push` |

---

## Why this matters

Without this second Space, the submission is structurally a single-agent
project with A2A *mentioned* in the docs.

With it, you can defend three concrete claims to judges:

1. **Two live, distinct agents** — main spectator at
   `Kashishshaikh-protocol-arena.hf.space` + curator at
   `Kashishshaikh-tool-curator.hf.space`. Two URLs, two FastAPI containers.
2. **Real A2A protocol surface** — the `/.well-known/agent.json`
   discovery endpoint and `POST /a2a` JSON-RPC method. Judges can curl
   them. Same protocol an A2A-compliant orchestrator would use.
3. **Demo-recordable multi-agent flow** — orchestrator hits drift on
   `web.search`, queries the curator, gets back `memory.lookup_alias`,
   recovers. The video can show this end-to-end.

The main hackathon page weights "multi-agent" as a category bonus. This
50-line FastAPI service is the cheapest way to satisfy it.

---

## Optional Step 7 — wire the A2A narration into the spectator UI

The Tool-Curator only counts as a kill-shot if it's **visible on camera**
during the 90-sec video. Right now an `a2a` action would scroll past in
the action log unnoticed. To add a dedicated narration line, edit the
`narrate(ev)` function in `arena/ui/spectator_web.py` to add an `a2a`
case:

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

That single line is what sells the multi-agent claim in the video.
