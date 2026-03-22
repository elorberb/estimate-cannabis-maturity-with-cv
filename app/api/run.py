"""Cross-platform API runner — works on Mac, Linux, and Windows.

Usage:
    uv run python run.py

Equivalent to: PYTHONPATH=src uvicorn src.main:app --reload
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")

subprocess.run(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "src.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ],
    check=False,
)
