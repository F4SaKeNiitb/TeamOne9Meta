"""Browser spectator — Plan §18.1.

Runs a real PROTOCOL-ARENA episode in a background thread and streams
turn-by-turn state to a single self-contained HTML page over SSE. The
page renders:

  • the live DAG as inline SVG (nodes coloured by status)
  • the six reward signals as horizontal bars
  • safety counters (honeypot / injection) as red badges that flash on trigger
  • a drift-event ticker that highlights the turn drift fires
  • an action log (last 8 turns) with the orchestrator's rationale

Everything ships inline — no CDNs, no static directory. That means the
demo works in an air-gapped judging room. One episode per HTTP
connection: reload the page to start over with a new seed.

Usage:
    python -m arena.ui.spectator_web --port 7861
    open http://localhost:7861/?task=research_photo_rename&seed=0

The terminal spectator (`arena.ui.spectator`) is still the right tool
for replay logs and CI; this one is the demo-day visual.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

from ..server.arena_env import ProtocolArenaEnvironment
from ..models import OrchestratorAction
from ..eval.baselines import rule_based_policy
from ..tasks import ALL_TASKS


PolicyFn = Callable[[Dict[str, Any]], Dict[str, Any]]


_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PROTOCOL-ARENA · live spectator</title>
<style>
  :root {
    --bg: #0b0f14;
    --fg: #e6edf3;
    --muted: #8b949e;
    --green: #3fb950;
    --amber: #d29922;
    --red: #f85149;
    --blue: #58a6ff;
    --panel: #161b22;
    --border: #30363d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 13px/1.45 -apple-system, "SF Mono", Menlo, monospace;
    background: var(--bg); color: var(--fg);
  }
  header {
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  }
  header h1 { margin: 0; font-size: 14px; letter-spacing: 0.5px; }
  header .meta { color: var(--muted); font-size: 12px; }
  header .reward { margin-left: auto; font-size: 18px; font-weight: 600; }
  main {
    display: grid;
    grid-template-columns: 1fr 360px;
    grid-template-rows: 1fr auto;
    gap: 1px;
    height: calc(100vh - 44px);
    background: var(--border);
  }
  section { background: var(--panel); padding: 12px; overflow: auto; }
  section h2 {
    margin: 0 0 8px; font-size: 11px; color: var(--muted);
    letter-spacing: 1px; text-transform: uppercase; font-weight: 500;
  }
  #dag-panel { grid-row: 1 / 3; }
  #dag svg { width: 100%; height: 100%; }
  .node-rect { fill: #21262d; stroke: var(--border); stroke-width: 1.5; }
  .node-rect.ok { fill: #0f5132; stroke: var(--green); }
  .node-rect.fail { fill: #4a1419; stroke: var(--red); }
  .node-rect.pending { fill: #2d2a14; stroke: var(--amber); }
  .node-text { fill: var(--fg); font-size: 11px; }
  .edge { stroke: var(--muted); stroke-width: 1.2; fill: none; marker-end: url(#arrow); }
  .signal-row { display: grid; grid-template-columns: 130px 1fr 50px; gap: 8px;
                align-items: center; margin-bottom: 4px; }
  .signal-row .label { color: var(--muted); font-size: 11px; }
  .bar { height: 14px; background: #21262d; border-radius: 2px; overflow: hidden; }
  .bar > div { height: 100%; background: var(--blue); transition: width 0.3s ease; }
  .bar.penalty > div { background: var(--red); }
  .signal-row .val { text-align: right; font-variant-numeric: tabular-nums; }
  .badges { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; border: 1px solid var(--border); }
  .badge.fired { background: var(--red); color: #fff; border-color: var(--red);
                 animation: pulse 0.6s ease 2; }
  .badge.safe  { color: var(--green); border-color: var(--green); }
  @keyframes pulse { 0% { opacity: 0.4; } 100% { opacity: 1; } }
  #log { font-size: 12px; }
  .log-row { padding: 4px 0; border-bottom: 1px dashed var(--border); }
  .log-row .turn { color: var(--blue); font-weight: 600; }
  .log-row .kind { color: var(--amber); }
  .log-row .rationale { color: var(--muted); font-size: 11px; }
  #drift-ticker { padding: 8px 14px; background: #161b22;
                  border-top: 1px solid var(--border); min-height: 28px;
                  font-size: 12px; color: var(--muted); grid-column: 1 / 3; }
  #drift-ticker.fired { color: var(--red); animation: pulse 0.4s ease 3; }
  #conn { font-size: 11px; color: var(--muted); }
  #conn.live { color: var(--green); }
  #conn.dead { color: var(--red); }
</style>
</head>
<body>
<header>
  <h1>🟢 PROTOCOL-ARENA · live spectator</h1>
  <span class="meta" id="task-meta">connecting…</span>
  <span id="conn">●</span>
  <span class="reward" id="reward">—</span>
</header>
<main>
  <section id="dag-panel">
    <h2>DAG (orchestrator plan)</h2>
    <div id="dag"></div>
  </section>
  <section id="signals-panel">
    <h2>6-signal reward</h2>
    <div id="signals"></div>
    <h2 style="margin-top:14px">Safety</h2>
    <div class="badges" id="safety">
      <span class="badge safe" id="hp-badge">honeypot ✓</span>
      <span class="badge safe" id="inj-badge">injection ✓</span>
    </div>
    <h2 style="margin-top:14px">Action log</h2>
    <div id="log"></div>
  </section>
  <div id="drift-ticker">no drift fired yet.</div>
</main>
<script>
const SIGNAL_KEYS = ["task_correctness","drift_robustness","plan_quality",
                     "protocol_hygiene","efficiency","memory_hit_bonus"];

function $(id) { return document.getElementById(id); }

function renderSignals(signals) {
  const root = $("signals");
  root.innerHTML = "";
  for (const k of SIGNAL_KEYS) {
    const v = signals[k] ?? 0;
    const pct = Math.max(0, Math.min(1, v)) * 100;
    const row = document.createElement("div");
    row.className = "signal-row";
    row.innerHTML = `<span class="label">${k}</span>
      <span class="bar"><div style="width:${pct}%"></div></span>
      <span class="val">${v.toFixed(3)}</span>`;
    root.appendChild(row);
  }
}

function renderDAG(dag) {
  const nodes = (dag && dag.nodes) || [];
  const edges = (dag && dag.edges) || [];
  if (nodes.length === 0) {
    $("dag").innerHTML = '<p style="color:var(--muted)">empty DAG — agent has not planned yet.</p>';
    return;
  }
  const cols = {};
  const depth = {};
  nodes.forEach(n => { depth[n.id] = 0; });
  edges.forEach(([a, b]) => {
    depth[b] = Math.max(depth[b] || 0, (depth[a] || 0) + 1);
  });
  nodes.forEach(n => {
    const d = depth[n.id] || 0;
    (cols[d] = cols[d] || []).push(n);
  });
  const colWidth = 160, rowHeight = 56, padX = 24, padY = 24;
  const maxCol = Math.max(0, ...Object.keys(cols).map(Number));
  const maxRow = Math.max(1, ...Object.values(cols).map(c => c.length));
  const W = padX * 2 + (maxCol + 1) * colWidth;
  const H = padY * 2 + maxRow * rowHeight;
  const pos = {};
  Object.entries(cols).forEach(([d, list]) => {
    list.forEach((n, i) => {
      pos[n.id] = { x: padX + d * colWidth, y: padY + i * rowHeight };
    });
  });
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  // arrow marker
  const defs = document.createElementNS(ns, "defs");
  defs.innerHTML = `<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
    markerUnits="strokeWidth" markerWidth="6" markerHeight="6" orient="auto">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b949e"/></marker>`;
  svg.appendChild(defs);
  edges.forEach(([a, b]) => {
    if (!pos[a] || !pos[b]) return;
    const x1 = pos[a].x + 130, y1 = pos[a].y + 18;
    const x2 = pos[b].x, y2 = pos[b].y + 18;
    const e = document.createElementNS(ns, "path");
    e.setAttribute("class", "edge");
    e.setAttribute("d", `M${x1},${y1} C${x1+30},${y1} ${x2-30},${y2} ${x2},${y2}`);
    svg.appendChild(e);
  });
  nodes.forEach(n => {
    const p = pos[n.id]; if (!p) return;
    const g = document.createElementNS(ns, "g");
    g.setAttribute("transform", `translate(${p.x},${p.y})`);
    const r = document.createElementNS(ns, "rect");
    const status = (n.status || "pending").toLowerCase();
    r.setAttribute("class", `node-rect ${status}`);
    r.setAttribute("width", "130"); r.setAttribute("height", "36");
    r.setAttribute("rx", "4");
    g.appendChild(r);
    const t1 = document.createElementNS(ns, "text");
    t1.setAttribute("class", "node-text");
    t1.setAttribute("x", "8"); t1.setAttribute("y", "15");
    t1.textContent = (n.id || "").slice(0, 16);
    g.appendChild(t1);
    const t2 = document.createElementNS(ns, "text");
    t2.setAttribute("class", "node-text"); t2.setAttribute("fill", "#8b949e");
    t2.setAttribute("x", "8"); t2.setAttribute("y", "29");
    t2.textContent = (n.op || "").slice(0, 18);
    g.appendChild(t2);
    svg.appendChild(g);
  });
  $("dag").innerHTML = "";
  $("dag").appendChild(svg);
}

function appendLog(turn, action, ok) {
  const log = $("log");
  const row = document.createElement("div");
  row.className = "log-row";
  const status = ok === true ? "✓" : ok === false ? "✗" : "·";
  row.innerHTML = `<span class="turn">T${turn}</span>
    <span class="kind">${action.kind || "?"}</span> ${status}
    <div class="rationale">${(action.rationale || "").slice(0, 110)}</div>`;
  log.insertBefore(row, log.firstChild);
  while (log.childElementCount > 8) log.removeChild(log.lastChild);
}

function fireDrift(turn, drift) {
  const t = $("drift-ticker");
  t.textContent = `🔴 T${turn}: drift fired — ${drift}`;
  t.classList.add("fired");
  setTimeout(() => t.classList.remove("fired"), 1800);
}

function setSafety(hp, inj) {
  const hb = $("hp-badge"), ib = $("inj-badge");
  hb.textContent = `honeypot ${hp ? "✗ TRIGGERED" : "✓"}`;
  hb.className = "badge " + (hp ? "fired" : "safe");
  ib.textContent = `injection ${inj ? "✗ FOLLOWED" : "✓"}`;
  ib.className = "badge " + (inj ? "fired" : "safe");
}

function setConn(state) {
  const c = $("conn");
  c.className = state;
  c.textContent = state === "live" ? "● live" : state === "dead" ? "● disconnected" : "● …";
}

const params = new URLSearchParams(location.search);
const task = params.get("task") || "research_photo_rename";
const seed = params.get("seed") || "0";
$("task-meta").textContent = `task=${task} · seed=${seed}`;

const es = new EventSource(`/events?task=${encodeURIComponent(task)}&seed=${seed}`);
es.onopen = () => setConn("live");
es.onerror = () => setConn("dead");
es.onmessage = (msg) => {
  let ev;
  try { ev = JSON.parse(msg.data); } catch { return; }
  if (ev.kind === "init") {
    $("task-meta").textContent =
      `task=${ev.task_id} · seed=${ev.seed} · max_turns=${ev.max_turns}`;
    renderSignals({});
    renderDAG({});
  }
  if (ev.kind === "turn" || ev.kind === "final") {
    $("reward").textContent = (ev.reward ?? 0).toFixed(3);
    $("task-meta").textContent =
      `task=${ev.task_id} · seed=${ev.seed} · turn ${ev.turn}/${ev.max_turns}`;
    renderSignals(ev.signals || {});
    renderDAG(ev.dag || {});
    setSafety(ev.honeypot_triggered, ev.injection_followed);
    if (ev.action) appendLog(ev.turn, ev.action, ev.last_ok);
    if (ev.drift_fired_this_turn) fireDrift(ev.turn, ev.drift_fired_this_turn);
  }
  if (ev.kind === "final") {
    $("conn").className = "live"; $("conn").textContent = "● done";
    es.close();
  }
};
</script>
</body>
</html>
"""


def _action_dict_clean(d: Dict[str, Any]) -> Dict[str, Any]:
    keep = {"kind", "rationale", "mcp_call", "a2a_call", "dag_delta",
            "kg_op", "final", "rewind_n", "confidence"}
    return {k: v for k, v in d.items() if k in keep}


def _run_episode(task_id: str, seed: int, max_turns: int,
                 policy: PolicyFn, q: "queue.Queue[Any]") -> None:
    """Worker thread — drives the env and pushes SSE events into `q`."""
    try:
        env = ProtocolArenaEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        prior_drift_count = 0
        q.put({
            "kind": "init", "task_id": task_id, "seed": seed,
            "max_turns": obs.max_turns,
        })
        steps = 0
        while not obs.done and steps < max_turns:
            obs_dict = obs.model_dump()
            decision = policy(obs_dict)
            decision.setdefault("rationale",
                                "spectator-run baseline action.")
            if len(decision["rationale"]) < 20:
                decision["rationale"] = (decision["rationale"] + " " * 25)[:40]
            decision = _action_dict_clean(decision)
            obs = env.step(OrchestratorAction(**decision))
            steps += 1

            # Detect a drift event firing on this turn.
            drift_now = ""
            try:
                fired = list(env.state.drift_fired or [])
                if len(fired) > prior_drift_count:
                    new = fired[prior_drift_count:]
                    drift_now = "; ".join(str(x) for x in new)
                    prior_drift_count = len(fired)
            except Exception:
                pass

            last = obs.last_result
            q.put({
                "kind": "turn",
                "task_id": task_id, "seed": seed,
                "turn": obs.turn, "max_turns": obs.max_turns,
                "action": {"kind": decision.get("kind"),
                           "rationale": decision.get("rationale", "")},
                "dag": obs.dag_state.model_dump() if obs.dag_state else {},
                "signals": obs.reward_signals or {},
                "reward": obs.reward,
                "feedback": (obs.feedback or "")[:200],
                "honeypot_triggered": bool(env.state.honeypot_triggered),
                "injection_followed": bool(env.state.injection_followed),
                "last_ok": bool(last.ok) if last else None,
                "drift_fired_this_turn": drift_now,
            })

        q.put({
            "kind": "final",
            "task_id": task_id, "seed": seed,
            "turn": obs.turn, "max_turns": obs.max_turns,
            "dag": obs.dag_state.model_dump() if obs.dag_state else {},
            "signals": obs.reward_signals or {},
            "reward": obs.reward,
            "honeypot_triggered": bool(env.state.honeypot_triggered),
            "injection_followed": bool(env.state.injection_followed),
            "drift_fired_this_turn": "",
        })
    except Exception as e:
        q.put({"kind": "final", "error": str(e), "reward": 0.0,
               "task_id": task_id, "seed": seed,
               "turn": 0, "max_turns": 0,
               "signals": {}, "dag": {},
               "honeypot_triggered": False, "injection_followed": False})
    finally:
        q.put(None)


def build_app(policy: PolicyFn = rule_based_policy, max_turns: int = 12):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

    app = FastAPI(title="protocol-arena-spectator")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX_HTML

    @app.get("/tasks")
    def list_tasks():
        return JSONResponse({"tasks": list(ALL_TASKS.keys())})

    @app.get("/events")
    def events(task: str = "research_photo_rename", seed: int = 0):
        if task not in ALL_TASKS:
            return JSONResponse({"error": f"unknown task {task!r}"},
                                status_code=400)
        q: "queue.Queue[Any]" = queue.Queue(maxsize=64)
        threading.Thread(
            target=_run_episode,
            args=(task, int(seed), max_turns, policy, q),
            daemon=True,
        ).start()

        def gen():
            yield "retry: 1500\n\n"
            while True:
                try:
                    ev = q.get(timeout=8.0)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                if ev is None:
                    return
                yield f"data: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PROTOCOL-ARENA browser spectator")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed; run `pip install uvicorn fastapi`",
              file=sys.stderr)
        return 1

    app = build_app(max_turns=args.max_turns)
    print(f"[spectator] open http://{args.host}:{args.port}/?task=research_photo_rename&seed=0")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
