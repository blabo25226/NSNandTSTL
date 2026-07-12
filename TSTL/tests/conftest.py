"""Pytest path setup for TSTL/src imports."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
