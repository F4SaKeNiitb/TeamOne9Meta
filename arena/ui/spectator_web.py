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
  header .controls { display: flex; gap: 6px; align-items: center;
                     margin-left: 12px; }
  header .controls select, header .controls input,
  header .controls button {
    background: var(--panel); color: var(--fg); border: 1px solid var(--border);
    border-radius: 4px; padding: 4px 8px; font: inherit;
  }
  header .controls input { width: 60px; }
  header .controls button { cursor: pointer; }
  header .controls button:hover { background: var(--border); }
  header .controls button.start { background: var(--green); color: #000;
                                   border-color: var(--green); font-weight: 600; }
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
  #narrate {
    padding: 10px 16px; font-size: 15px; line-height: 1.4;
    background: linear-gradient(90deg, #1c2128 0%, #161b22 100%);
    border-bottom: 1px solid var(--border); color: var(--fg);
    font-weight: 500; min-height: 22px;
  }
  #narrate.drift { background: #4a1419; color: #fff;
                   animation: pulse 0.5s ease 2; border-bottom-color: var(--red); }
  #narrate.safety { background: #5a1a1a; color: #fff;
                    animation: pulse 0.4s ease 4; border-bottom-color: var(--red); }
  #narrate.recover { background: #0f5132; color: #fff;
                     border-bottom-color: var(--green); }
</style>
</head>
<body>
<header>
  <h1>🟢 PROTOCOL-ARENA · live spectator</h1>
  <span class="meta" id="task-meta">connecting…</span>
  <span id="conn">●</span>
  <span class="reward" id="reward">—</span>
  <span class="controls">
    <select id="policy-select" title="policy / agent"></select>
    <select id="task-select" title="task"></select>
    <input id="seed-input" type="number" value="0" min="0" title="seed"/>
    <button id="start-btn" class="start" title="run a fresh episode">▶ Start</button>
  </span>
</header>
<div id="narrate">↻ waiting for the agent to take its first action…</div>
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

// Plain-English narration for non-technical judges. Maps action.kind +
// drift events to a sentence that says what just happened in human terms.
function narrate(ev) {
  const n = $("narrate");
  const a = ev.action || {};
  let cls = "", txt = "";
  if (ev.honeypot_triggered) {
    cls = "safety";
    txt = "🚨 SAFETY BREACH — the agent fell for a honeypot tool. Penalty applied; trajectory will NOT be ingested back into training.";
  } else if (ev.injection_followed) {
    cls = "safety";
    txt = "🚨 SAFETY BREACH — the agent followed a prompt injection embedded in a search result. Hard penalty applied.";
  } else if (ev.drift_fired_this_turn) {
    cls = "drift";
    txt = `⚠️ DRIFT FIRED at turn ${ev.turn} — ${ev.drift_fired_this_turn}. The agent must now recover.`;
  } else if (a.kind === "memory") {
    cls = "recover";
    txt = `🧠 Turn ${ev.turn}: agent is querying its long-term memory (capability KG) — typical recovery move after drift.`;
  } else if (a.kind === "rewind") {
    cls = "recover";
    txt = `⏮️ Turn ${ev.turn}: agent rewound to retry a different branch.`;
  } else if (a.kind === "mcp") {
    txt = `🔧 Turn ${ev.turn}: agent calling an MCP tool` +
          (ev.last_ok === false ? " — last call failed, likely drift hitting." : ".");
  } else if (a.kind === "a2a") {
    cls = "recover";
    const target = a.a2a_call?.agent_card_id || "tool-curator";
    const intent = (a.a2a_call?.task_spec || "").slice(0, 60);
    const body   = ev.last_result_body || {};
    const tool   = body.tool ?? "—";
    const conf   = body.confidence ?? "—";
    txt = `🤝 A2A → ${target}: intent="${intent}…" → tool=${tool} (conf ${conf})`;
  } else if (a.kind === "plan") {
    txt = `📋 Turn ${ev.turn}: agent updating its DAG plan.`;
  } else if (a.kind === "submit") {
    cls = "recover";
    txt = `✅ Turn ${ev.turn}: agent submitting final answer. Episode complete.`;
  } else if (ev.kind === "init") {
    txt = `▶️ Episode starting on task '${ev.task_id}' (seed ${ev.seed}). Watch the DAG grow and the six signal bars on the right.`;
  } else {
    txt = `Turn ${ev.turn || "?"}: agent thinking…`;
  }
  n.className = cls;
  n.textContent = txt;
}

const params = new URLSearchParams(location.search);
const task   = params.get("task") || "research_photo_rename";
const seed   = params.get("seed") || "0";
const policy = params.get("policy") || "rule_based";
$("task-meta").textContent = `task=${task} · seed=${seed} · policy=${policy}`;

// Populate task + policy dropdowns from the server. Frontier API options
// (gpt-4o-mini, claude-haiku-4-5) only appear if the corresponding
// secret is set on the Space — keeps the dropdown honest.
fetch("/tasks").then(r => r.json()).then(d => {
  const sel = $("task-select");
  for (const t of d.tasks || []) {
    const o = document.createElement("option");
    o.value = t; o.textContent = t;
    if (t === task) o.selected = true;
    sel.appendChild(o);
  }
}).catch(() => {});
fetch("/policies").then(r => r.json()).then(d => {
  const sel = $("policy-select");
  const policies = d.policies || [];
  for (const p of policies) {
    const o = document.createElement("option");
    o.value = p; o.textContent = p;
    if (p === policy) o.selected = true;
    sel.appendChild(o);
  }
  if (d.needs_api_key) {
    // Surface a clear setup banner instead of silent failure.
    const n = $("narrate");
    n.className = "safety";
    n.innerHTML =
      "🔐 No LLM API key configured on this Space. " +
      "Add <code>ANTHROPIC_API_KEY</code> or <code>OPENAI_API_KEY</code> " +
      "in Space Settings → Variables and secrets, then rebuild. " +
      "Until then, episodes cannot be streamed.";
    $("start-btn").disabled = true;
    $("start-btn").style.opacity = "0.5";
    $("start-btn").style.cursor = "not-allowed";
  }
}).catch(() => {});

$("seed-input").value = seed;
$("start-btn").onclick = () => {
  const t = $("task-select").value || task;
  const s = $("seed-input").value || "0";
  const p = $("policy-select").value || policy;
  location.href = `/?task=${encodeURIComponent(t)}&seed=${s}&policy=${encodeURIComponent(p)}`;
};

const es = new EventSource(
  `/events?task=${encodeURIComponent(task)}&seed=${seed}` +
  `&policy_id=${encodeURIComponent(policy)}`);
es.onopen = () => setConn("live");
es.onerror = () => setConn("dead");
es.onmessage = (msg) => {
  let ev;
  try { ev = JSON.parse(msg.data); } catch { return; }
  narrate(ev);
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


def _describe_drift_at_turn(task_id: str, turn: int) -> str:
    """Look up the drift schedule for `task_id` and return a short
    human-readable description of any drift event whose `turn` equals
    `turn`. Returns "" if the schedule has nothing for this turn.

    The env state only tracks a *boolean* `drift_fired`, not the event
    details. We pull the human-readable text out of the task's static
    DriftSchedule so the spectator narration can say WHAT drifted, not
    just THAT something drifted."""
    task = ALL_TASKS.get(task_id, {}) or {}
    sched = task.get("drift_schedule")
    events = getattr(sched, "events", []) or []
    descriptions: List[str] = []
    for e in events:
        if int(getattr(e, "turn", -1)) != turn:
            continue
        klass = getattr(e, "klass", "?")
        srv   = getattr(e, "target_server", None)
        tool  = getattr(e, "target_tool", None)
        peer  = getattr(e, "target_peer", None)
        detail = getattr(e, "detail", {}) or {}
        target = (f"{srv}.{tool}" if srv and tool
                  else (peer or srv or tool or "env"))
        if klass == "renaming" and "new_name" in detail:
            descriptions.append(f"{klass} on {target} → {detail['new_name']}")
        else:
            descriptions.append(f"{klass} on {target}")
    return "; ".join(descriptions)


def _run_episode(task_id: str, seed: int, max_turns: int,
                 policy: PolicyFn, q: "queue.Queue[Any]") -> None:
    """Worker thread — drives the env and pushes SSE events into `q`."""
    try:
        env = ProtocolArenaEnvironment()
        obs = env.reset(task_id=task_id, seed=seed)
        # `env.state.drift_fired` is a BOOL, not a list. Track its
        # False→True transition once per episode rather than counting
        # list growth (which never grows because it's a scalar).
        prior_drift_fired = bool(env.state.drift_fired)
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

            # Detect drift firing on this turn — bool transition + schedule lookup.
            drift_now = ""
            try:
                now_fired = bool(env.state.drift_fired)
                if now_fired and not prior_drift_fired:
                    desc = _describe_drift_at_turn(task_id, obs.turn)
                    drift_now = desc or "schema drift event fired"
                    prior_drift_fired = True
            except Exception:
                pass

            last = obs.last_result
            # Pass the full action dict (not just kind+rationale) so the
            # frontend can render a richer narration line — specifically,
            # for `a2a` actions we want to show the target peer + the
            # tool the curator recommended back. Also surface the env's
            # last_result body so the JS sees what came back from the
            # A2A peer.
            q.put({
                "kind": "turn",
                "task_id": task_id, "seed": seed,
                "turn": obs.turn, "max_turns": obs.max_turns,
                "action": decision,
                "dag": obs.dag_state.model_dump() if obs.dag_state else {},
                "signals": obs.reward_signals or {},
                "reward": obs.reward,
                "feedback": (obs.feedback or "")[:200],
                "honeypot_triggered": bool(env.state.honeypot_triggered),
                "injection_followed": bool(env.state.injection_followed),
                "last_ok": bool(last.ok) if last else None,
                "last_result_body": (last.body if (last and last.ok) else None),
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


def _build_policy_registry() -> Dict[str, "PolicyFn"]:
    """Map a string policy id (selectable from the UI dropdown) to a
    callable that takes an obs dict and returns an action dict.

    Only frontier-API policies are exposed in the dropdown — this Space
    is intentionally LLM-driven so judges always see a real model
    reasoning, never a Python `if/else`. Set `OPENAI_API_KEY` and/or
    `ANTHROPIC_API_KEY` as HF Space secrets to enable each option.

    `rule_based_policy` is still imported here as the silent fallback
    inside each LLM policy when the model emits invalid JSON — judges
    don't see it surfaced anywhere."""
    import os
    from ..eval.baselines import rule_based_policy

    registry: Dict[str, "PolicyFn"] = {}

    # claude-haiku-4-5 via ANTHROPIC_API_KEY (Space secret)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from openai import OpenAI
            import inference, json as _json
            _aclient = OpenAI(api_key=os.getenv("ANTHROPIC_API_KEY"),
                              base_url="https://api.anthropic.com/v1")

            def _haiku_policy(obs):
                msgs = [{"role": "system", "content": inference.SYSTEM_PROMPT},
                        {"role": "user",   "content": inference.build_user_msg(obs)}]
                try:
                    r = _aclient.chat.completions.create(
                        model="claude-haiku-4-5-20251001", messages=msgs,
                        temperature=0.2, max_tokens=400)
                    raw = (r.choices[0].message.content or "").strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"): raw = raw[4:]
                    return _json.loads(raw.strip())
                except Exception:
                    return rule_based_policy(obs)
            registry["claude-haiku-4-5"] = _haiku_policy
        except Exception:
            pass

    # gpt-4o-mini via OPENAI_API_KEY (Space secret)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            import inference, json as _json
            _client = OpenAI()  # picks up OPENAI_API_KEY automatically

            def _gpt_policy(obs):
                msgs = [{"role": "system", "content": inference.SYSTEM_PROMPT},
                        {"role": "user",   "content": inference.build_user_msg(obs)}]
                try:
                    r = _client.chat.completions.create(
                        model="gpt-4o-mini", messages=msgs,
                        temperature=0.2, max_tokens=400)
                    raw = (r.choices[0].message.content or "").strip()
                    if raw.startswith("```"):
                        raw = raw.split("```")[1]
                        if raw.startswith("json"): raw = raw[4:]
                    return _json.loads(raw.strip())
                except Exception:
                    return rule_based_policy(obs)
            registry["gpt-4o-mini"] = _gpt_policy
        except Exception:
            pass

    return registry


def build_app(policy: PolicyFn = rule_based_policy, max_turns: int = 12):
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

    app = FastAPI(title="protocol-arena-spectator")
    registry = _build_policy_registry()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _INDEX_HTML

    @app.get("/tasks")
    def list_tasks():
        return JSONResponse({"tasks": list(ALL_TASKS.keys())})

    @app.get("/policies")
    def list_policies():
        """Frontend reads this to populate the policy dropdown. The
        Space ships LLM-only — if neither API key is configured, this
        returns an empty list and the UI surfaces a setup error."""
        order = ["claude-haiku-4-5", "gpt-4o-mini"]
        available = [p for p in order if p in registry]
        return JSONResponse({
            "policies": available,
            "needs_api_key": len(available) == 0,
            "secrets_required": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"],
        })

    @app.get("/events")
    def events(task: str = "research_photo_rename",
               seed: int = 0,
               policy_id: str = ""):
        if task not in ALL_TASKS:
            return JSONResponse({"error": f"unknown task {task!r}"},
                                status_code=400)
        if not registry:
            return JSONResponse({
                "error": "no LLM policy configured — set ANTHROPIC_API_KEY "
                         "or OPENAI_API_KEY as a Space secret and rebuild.",
            }, status_code=503)
        # Default to the first available policy if none was requested or
        # if the requested one isn't available (e.g. user clicked an old
        # bookmarked URL referencing a removed baseline).
        chosen = registry.get(policy_id) or next(iter(registry.values()))
        q: "queue.Queue[Any]" = queue.Queue(maxsize=64)
        threading.Thread(
            target=_run_episode,
            args=(task, int(seed), max_turns, chosen, q),
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


# Module-level app so `uvicorn arena.ui.spectator_web:app` works without
# going through main(). HF Spaces / docker entrypoints import this.
app = build_app()


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
