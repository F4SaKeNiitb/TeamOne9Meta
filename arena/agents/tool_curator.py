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


@app.get("/")
def root():
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
