"""Tool-Curator A2A agent — recommends an MCP tool given a natural-language intent.

This is the SECOND agent in PROTOCOL-ARENA. The orchestrator (running on
the main Space) makes A2A calls to THIS agent during drift events to
discover the renamed/aliased tool. Two agents, A2A protocol — that's
what makes this a multi-agent submission.

Spec:
    POST /a2a            {"method": "recommend_tool",
                          "params": {"intent": "<str>"}}
    →                    {"tool": "<name>", "confidence": 0..1, "reason": "<str>"}

    GET /.well-known/agent.json
    →                    AgentCard JSON for discovery

In a real system this would be a learned retriever over the capability
KG. The keyword stub is fine for the demo — the point is the A2A
protocol surface, not the retrieval quality.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel


# Static intent → tool map. Order matters — earlier rules match first.
INTENT_MAP = [
    (("rename", "new name", "alias"),     "memory.lookup_alias"),
    (("search", "find", "lookup"),         "search.web"),
    (("photo", "image", "exif"),           "fs.read_exif"),
    (("submit", "final", "commit"),        "submit.final"),
    (("plan", "dag", "decompose"),         "plan.expand"),
    (("citation", "cite", "year", "author"), "kb.lookup_fact"),
]


class A2ARequest(BaseModel):
    method: str
    params: dict = {}


app = FastAPI(title="Tool-Curator A2A",
              description="Recommends MCP tools by natural-language intent.",
              version="1.0.0")


_INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8" />
<title>Tool-Curator · A2A agent</title>
<style>
  :root {
    --bg:#0b0f14; --fg:#e6edf3; --muted:#8b949e;
    --green:#3fb950; --amber:#d29922; --red:#f85149;
    --blue:#58a6ff; --panel:#161b22; --border:#30363d;
  }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 -apple-system,"SF Mono",Menlo,monospace;
         background:var(--bg); color:var(--fg); }
  header { padding:14px 20px; border-bottom:1px solid var(--border);
           display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }
  header h1 { margin:0; font-size:16px; letter-spacing:0.5px; }
  header .meta { color:var(--muted); font-size:12px; }
  header .badge { margin-left:auto; padding:2px 10px; border-radius:10px;
                  background:var(--green); color:#000; font-weight:600;
                  font-size:11px; letter-spacing:0.5px; }
  main { max-width:780px; margin:0 auto; padding:20px; }
  section { background:var(--panel); border:1px solid var(--border);
            border-radius:6px; padding:16px; margin-bottom:14px; }
  section h2 { margin:0 0 10px; font-size:13px; color:var(--muted);
               text-transform:uppercase; letter-spacing:0.8px; }
  textarea { width:100%; min-height:60px; padding:10px;
             background:#0d1117; color:var(--fg); border:1px solid var(--border);
             border-radius:4px; font:inherit; resize:vertical; }
  button { margin-top:10px; padding:8px 18px; background:var(--blue);
           color:#000; border:none; border-radius:4px; font:inherit;
           font-weight:600; cursor:pointer; }
  button:hover { background:#79b8ff; }
  pre { margin:0; padding:12px; background:#0d1117; border-radius:4px;
        font-size:12px; overflow-x:auto; color:var(--fg); }
  pre.ok b { color:var(--green); }
  .row { display:flex; gap:10px; flex-wrap:wrap; }
  .row > div { flex:1; min-width:200px; }
  .key { color:var(--muted); }
  a { color:var(--blue); }
  .examples button { margin:4px 4px 0 0; padding:4px 10px; font-size:12px;
                     background:var(--border); color:var(--fg); }
  .examples button:hover { background:#3a4250; }
  footer { color:var(--muted); font-size:12px; padding:14px 20px;
           border-top:1px solid var(--border); margin-top:20px; text-align:center; }
</style>
</head><body>

<header>
  <h1>🤝 Tool-Curator</h1>
  <span class="meta">A2A agent · recommends MCP tools by intent</span>
  <span class="badge">RUNNING</span>
</header>

<main>

<section>
<h2>Try it — type a natural-language intent</h2>
<textarea id="intent" placeholder="e.g. the search tool was renamed mid-session">the search tool was renamed mid-session</textarea>
<div class="examples">
  <span class="key">examples:</span>
  <button onclick="setEx('the search tool was renamed mid-session')">rename</button>
  <button onclick="setEx('read photo exif data')">photo / exif</button>
  <button onclick="setEx('find a citation for the 2017 transformer paper')">citation</button>
  <button onclick="setEx('decompose this task into a plan')">plan</button>
  <button onclick="setEx('asdf qwerty unknown')">unknown (fallback)</button>
</div>
<button onclick="recommend()">▶ recommend tool</button>
</section>

<section>
<h2>Response from <code>POST /a2a</code></h2>
<pre id="response" class="ok">click <b>recommend tool</b> to query the agent.</pre>
</section>

<section>
<h2>Agent card · <code>GET /.well-known/agent.json</code></h2>
<pre id="card">loading…</pre>
</section>

<section>
<h2>Curl examples</h2>
<pre>$ curl -s {{HOST}}/.well-known/agent.json
$ curl -s -X POST {{HOST}}/a2a \\
    -H 'content-type: application/json' \\
    -d '{"method":"recommend_tool","params":{"intent":"search tool was renamed"}}'</pre>
</section>

</main>

<footer>
Pairs with the main spectator at
<a href="https://huggingface.co/spaces/Kashishshaikh/protocol-arena">Kashishshaikh/protocol-arena</a>.
Source: <code>arena/agents/tool_curator.py</code>.
</footer>

<script>
const $ = id => document.getElementById(id);
const setEx = s => { $('intent').value = s; };

// Show actual host in curl examples
document.querySelector('pre').innerHTML =
  document.querySelector('pre').innerHTML.replaceAll('{{HOST}}', location.origin);

// Load agent card on first paint
fetch('/.well-known/agent.json')
  .then(r => r.json())
  .then(d => $('card').textContent = JSON.stringify(d, null, 2))
  .catch(e => $('card').textContent = 'failed to load: ' + e);

async function recommend() {
  const intent = $('intent').value;
  const out = $('response');
  out.textContent = 'querying…';
  try {
    const r = await fetch('/a2a', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({method: 'recommend_tool', params: {intent}}),
    });
    const j = await r.json();
    out.innerHTML = JSON.stringify(j, null, 2)
      .replace(/"tool"/, '<b style="color:var(--green)">"tool"</b>')
      .replace(/"confidence"/, '<b style="color:var(--blue)">"confidence"</b>');
  } catch (e) {
    out.textContent = 'error: ' + e;
  }
}
</script>
</body></html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve a small self-contained UI so the curator looks like a real
    agent in a browser, not a JSON dump. The actual A2A surface is at
    GET /.well-known/agent.json and POST /a2a — those are unchanged.
    The UI calls those same endpoints client-side."""
    return _INDEX_HTML


@app.get("/info")
def info():
    """Original JSON root, kept under /info for any client that was
    relying on it. Most A2A clients use /.well-known/agent.json."""
    return {
        "name": "tool-curator",
        "description": "PROTOCOL-ARENA second agent — recommends MCP tools.",
        "endpoints": {
            "agent_card": "/.well-known/agent.json",
            "a2a":        "POST /a2a",
        },
    }


@app.get("/.well-known/agent.json")
def agent_card():
    """Standard A2A discovery endpoint."""
    return {
        "name":        "tool-curator",
        "description": "Recommends MCP tool names given a natural-language intent.",
        "methods":     ["recommend_tool"],
        "version":     "1.0.0",
        "schema": {
            "recommend_tool": {
                "params": {"intent": "string"},
                "returns": {"tool": "string", "confidence": "number",
                            "reason": "string"},
            },
        },
    }


@app.post("/a2a")
def a2a(req: A2ARequest):
    if req.method != "recommend_tool":
        return {"error": f"unknown method {req.method!r}",
                "supported": ["recommend_tool"]}
    intent = (req.params.get("intent") or "").lower()
    for keys, tool in INTENT_MAP:
        if any(k in intent for k in keys):
            return {
                "tool":       tool,
                "confidence": 0.85,
                "reason":     f"matched keywords {list(keys)} in intent",
            }
    return {
        "tool":       "search.web",
        "confidence": 0.40,
        "reason":     "fallback — no strong intent match; defaulting to search",
    }
