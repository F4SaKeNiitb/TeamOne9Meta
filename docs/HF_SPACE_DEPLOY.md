# Hugging Face Space Deployment Runbook

The submission rules require: **"Push your environment to a Hugging
Face Space so it's discoverable and runnable."** Skipping this drops
you out of the running.

We use a **Docker Space** because the environment runs as a FastAPI
service on port 7860 — exactly what HF Spaces expects.

---

## One-time setup

```bash
pip install --upgrade huggingface_hub
huggingface-cli login          # paste a WRITE token (settings → access tokens)
```

## Create the Space

```bash
huggingface-cli repo create protocol-arena \
    --type space --space_sdk docker
# Output:
#   https://huggingface.co/spaces/<YOUR-USER>/protocol-arena
```

## Add the HF metadata header to README

The first push after Space creation must have a YAML header at the
top of `README.md` so the Space picks up metadata:

```yaml
---
title: PROTOCOL-ARENA
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
short_description: RL gym for LLM agents speaking MCP/A2A under schema drift
---
```

Add it, commit, then push:

```bash
git remote add hf https://huggingface.co/spaces/<YOUR-USER>/protocol-arena
git push hf <your-current-branch>:main
```

## After the build finishes

- The Space landing page shows a green "Running" badge.
- Open the Space → there's a `/docs` route from FastAPI that judges
  can poke. The actual env is reachable at the Space's public URL on
  port 7860.
- Verify with `curl`:
  ```bash
  curl https://<YOUR-USER>-protocol-arena.hf.space/docs
  ```

## If the build fails

The two failure modes we've seen on HF Spaces with Docker:

1. **`pyproject.toml` build-system pulls in cython** — common with
   Pydantic-V2 source builds. Fix: pin to wheels in `pyproject.toml`,
   or remove the `[build-system]` section for the Space-only branch.
2. **Port mismatch** — HF Spaces sets `$PORT` to whatever it wants.
   Our Dockerfile already uses `${PORT}` with a 7860 default; **don't
   hardcode 7860** in `CMD`.

## Linking from README

Add to your top-level README's `## Materials` section:

```markdown
- **Live demo (HF Space)**: <https://huggingface.co/spaces/YOUR-USER/protocol-arena>
```

That single line is what judges look for.

---

## Why Docker, not Gradio/Streamlit

HF Spaces also accepts `sdk: gradio` and `sdk: streamlit`. We don't use
either because:
- Gradio Spaces wrap a Python function — they can't run our FastAPI
  state machine.
- Streamlit doesn't support the SSE streaming our spectator needs.
- Docker Spaces give us full control and let us reuse the existing
  `arena/server/app.py`.

## Verify before submitting

```bash
# from a fresh terminal
curl -sf https://<YOUR-USER>-protocol-arena.hf.space/docs | head -c 200
# Should return: {"openapi":"3.0.2","info":{"title":"PROTOCOL-ARENA"…
```

If that returns HTML or 404, the Space is misconfigured. Don't submit
until it returns FastAPI JSON.
