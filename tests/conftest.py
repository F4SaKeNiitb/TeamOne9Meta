"""Pytest configuration — ensures the repo root is on sys.path so that
`arena` and `protocol_arena` can be imported without requiring an editable
install at test time.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
