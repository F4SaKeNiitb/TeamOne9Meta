"""Dedup primitives — SHA-256 + Jaccard (+ semantic if torch is available)."""

import hashlib
import json
from typing import Any, Set


def artifact_hash(spec: Any) -> str:
    content = json.dumps(spec, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def tokenize(text: str) -> Set[str]:
    return {w for w in text.lower().split() if len(w) > 2}


def jaccard(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
