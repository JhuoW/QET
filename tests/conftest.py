"""Shared test setup. Tests (never production code) may import the oracle `reference/qet_ref.py` as `qet_ref`."""

import sys
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "reference"
if str(REFERENCE) not in sys.path:
    sys.path.insert(0, str(REFERENCE))
