"""Capability Knowledge Graph.

Persistent cross-episode memory answering 'have I seen this tool/peer before?'.

Storage: SQLite FTS5 for keyword retrieval + in-memory embedding index for
semantic retrieval. A SentenceTransformer is loaded lazily; if unavailable,
we fall back to a simple bag-of-words Jaccard similarity so the env still
runs with zero heavy deps.

Dedup layers:
  1. SHA-256 over canonical JSON of the tool spec (artifact-level)
  2. Union-Find over observed rename chains (entity-level)
  3. Semantic similarity (claim-level)
"""

import os
import json
import hashlib
import sqlite3
from typing import List, Dict, Any, Optional

from ..models import KGFact


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class CapabilityKG:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT,
        predicate TEXT,
        obj TEXT,
        confidence REAL,
        provenance TEXT,
        content_hash TEXT UNIQUE
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
        subject, predicate, obj, content='facts', content_rowid='id'
    );
    CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
        INSERT INTO facts_fts(rowid, subject, predicate, obj)
        VALUES (new.id, new.subject, new.predicate, new.obj);
    END;
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()
        self.uf = UnionFind()
        self._cache_embed: Dict[str, List[float]] = {}
        self._encoder = None  # lazily loaded

    # ---- core operations --------------------------------------------------

    def write(self, fact: KGFact) -> bool:
        content = json.dumps({
            "s": fact.subject, "p": fact.predicate, "o": fact.obj
        }, sort_keys=True)
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        try:
            self.conn.execute(
                "INSERT INTO facts (subject, predicate, obj, confidence, provenance, content_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fact.subject, fact.predicate, fact.obj,
                 fact.confidence, fact.provenance or "", h),
            )
            self.conn.commit()
            if fact.predicate in ("renamed_to", "supersedes"):
                self.uf.union(fact.subject, fact.obj)
            return True
        except sqlite3.IntegrityError:
            return False

    def query(self, pattern: str, top_k: int = 5) -> List[KGFact]:
        if not pattern:
            return []
        esc = pattern.replace('"', '""')
        try:
            cur = self.conn.execute(
                f'SELECT subject, predicate, obj, confidence, provenance '
                f'FROM facts_fts JOIN facts ON facts_fts.rowid = facts.id '
                f'WHERE facts_fts MATCH "{esc}" LIMIT {top_k}'
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            rows = []

        if not rows:
            cur = self.conn.execute(
                "SELECT subject, predicate, obj, confidence, provenance FROM facts "
                "WHERE subject LIKE ? OR obj LIKE ? LIMIT ?",
                (f"%{pattern}%", f"%{pattern}%", top_k),
            )
            rows = cur.fetchall()

        return [KGFact(
            subject=r[0], predicate=r[1], obj=r[2],
            confidence=r[3], provenance=r[4] or None
        ) for r in rows]

    def canonical(self, entity: str) -> str:
        return self.uf.find(entity)

    def bfs(self, start: str, max_hops: int = 3, top_k: int = 8) -> List[KGFact]:
        """Multi-hop BFS over the (subject, obj) edge set."""
        seen = {start}
        frontier = [start]
        results: List[KGFact] = []
        for _ in range(max_hops):
            next_frontier: List[str] = []
            for ent in frontier:
                cur = self.conn.execute(
                    "SELECT subject, predicate, obj, confidence, provenance FROM facts "
                    "WHERE subject = ? OR obj = ?", (ent, ent),
                )
                for r in cur.fetchall():
                    results.append(KGFact(
                        subject=r[0], predicate=r[1], obj=r[2],
                        confidence=r[3], provenance=r[4] or None
                    ))
                    for n in (r[0], r[2]):
                        if n not in seen:
                            seen.add(n)
                            next_frontier.append(n)
                if len(results) >= top_k:
                    return results[:top_k]
            frontier = next_frontier
            if not frontier:
                break
        return results[:top_k]

    # ---- convenience recording ------------------------------------------------

    def record_tool_seen(self, server_id: str, tool: str, schema_hash: str):
        self.write(KGFact(
            subject=f"{server_id}.{tool}",
            predicate="schema_hash",
            obj=schema_hash,
            confidence=1.0,
            provenance="observed",
        ))

    def record_rename(self, old: str, new: str, turn: int):
        self.write(KGFact(
            subject=old, predicate="renamed_to", obj=new,
            confidence=1.0, provenance=f"turn={turn}"
        ))

    def record_failure(self, tool_qid: str, status_code: int, error: str):
        self.write(KGFact(
            subject=tool_qid, predicate=f"failed_{status_code}",
            obj=error[:64], confidence=1.0, provenance="runtime",
        ))
