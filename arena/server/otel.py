"""Tracing wrapper — Plan §18.2.

Tries real OpenTelemetry (OTLP → Jaeger / Tempo / grafana) first; falls
back to an in-memory JSON tracer when the SDK isn't installed. Every
span remains JSON-serialisable so the replay script reconstructs the
episode deterministically regardless of backend.

Environment variables:
  OTEL_EXPORTER_OTLP_ENDPOINT   : e.g. http://localhost:4318
  OTEL_SERVICE_NAME             : default "protocol-arena"

If the endpoint is unreachable, we warn once and keep the in-memory
backend so demos don't hard-fail in air-gapped judging rooms.
"""

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


_OTLP_READY = False
_OTLP_WARNED = False
_real_tracer = None


def _maybe_init_otlp():
    """One-time attempt to wire up real OTLP. Silent when unavailable."""
    global _OTLP_READY, _OTLP_WARNED, _real_tracer
    if _OTLP_READY or _OTLP_WARNED:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint:
        _OTLP_WARNED = True
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "protocol-arena"),
        })
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        ))
        trace.set_tracer_provider(provider)
        _real_tracer = trace.get_tracer("arena")
        _OTLP_READY = True
    except Exception as e:
        print(f"[otel] OTLP disabled: {e}", file=sys.stderr)
        _OTLP_WARNED = True


@dataclass
class Span:
    name: str
    start: float
    end: Optional[float] = None
    attrs: Dict[str, Any] = field(default_factory=dict)
    children: List["Span"] = field(default_factory=list)


class ArenaTracer:
    """Episode-scoped tracer.

    Each episode gets a unique trace_id. Every span is JSON-serialisable
    so the replay script can reconstruct the episode deterministically.
    If OTLP is configured, spans also fan out to the real SDK so they
    show up in Jaeger / Grafana Tempo for the live demo.
    """

    def __init__(self):
        _maybe_init_otlp()
        self.trace_id: str = uuid.uuid4().hex[:16]
        self.spans: List[Span] = []
        self._stack: List[Span] = []
        self._otlp_stack: List[Any] = []

    def reset(self):
        self.trace_id = uuid.uuid4().hex[:16]
        self.spans.clear()
        self._stack.clear()
        self._otlp_stack.clear()

    def start(self, name: str, **attrs) -> Span:
        span = Span(name=name, start=time.time(), attrs=dict(attrs))
        if self._stack:
            self._stack[-1].children.append(span)
        else:
            self.spans.append(span)
        self._stack.append(span)
        if _OTLP_READY and _real_tracer is not None:
            ctx = _real_tracer.start_as_current_span(name, attributes={
                str(k): _otel_attr(v) for k, v in attrs.items()
            })
            self._otlp_stack.append(ctx.__enter__())
        return span

    def end(self, span: Span, **attrs):
        span.end = time.time()
        span.attrs.update(attrs)
        if self._stack and self._stack[-1] is span:
            self._stack.pop()
        if _OTLP_READY and self._otlp_stack:
            otlp_span = self._otlp_stack.pop()
            try:
                for k, v in attrs.items():
                    otlp_span.set_attribute(str(k), _otel_attr(v))
                otlp_span.end()
            except Exception:
                pass

    def event(self, name: str, **attrs):
        self.start(name, **attrs).end = time.time()
        if self._stack and self._stack[-1].name == name:
            self._stack.pop()
        if _OTLP_READY and self._otlp_stack:
            try:
                self._otlp_stack.pop().end()
            except Exception:
                pass

    def to_json(self) -> str:
        def _ser(sp: Span):
            return {
                "name": sp.name,
                "start": sp.start,
                "end": sp.end,
                "attrs": sp.attrs,
                "children": [_ser(c) for c in sp.children],
            }
        return json.dumps({"trace_id": self.trace_id,
                           "spans": [_ser(s) for s in self.spans]})


def _otel_attr(v: Any) -> Any:
    """OTel attribute values must be primitives."""
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return json.dumps(v, default=str)[:512]
    except Exception:
        return str(v)[:512]
