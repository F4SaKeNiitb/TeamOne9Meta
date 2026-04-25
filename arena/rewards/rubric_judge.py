"""Rubric-based rationale quality check.

Rule-based today (keyword coverage + length + avoids filler). Swap in an
LLM-judge at any time; the signature stays the same.
"""

from typing import List


FILLER = {"maybe", "perhaps", "probably", "i think", "not sure", "guess"}


def rationale_quality(rationale: str, oracle_matchers: List[str]) -> float:
    if not rationale:
        return 0.0
    text = rationale.lower()
    words = text.split()
    length_ok = len(words) >= 30
    coverage = 0.0
    if oracle_matchers:
        hits = sum(1 for m in oracle_matchers if m.lower() in text)
        coverage = hits / len(oracle_matchers)
    filler_count = sum(1 for f in FILLER if f in text)
    filler_pen = min(0.3, 0.05 * filler_count)
    base = 0.6 * coverage + (0.3 if length_ok else 0.0)
    return round(max(0.0, min(1.0, base + 0.1 - filler_pen)), 3)
