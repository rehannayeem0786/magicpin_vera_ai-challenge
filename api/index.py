"""Vercel serverless entrypoint for Vera Pro.

Vercel's Python runtime imports this module and serves the ASGI ``app``
object it finds here. Importing ``bot`` is safe: its uvicorn entry point
is guarded by ``if __name__ == "__main__"`` (bot.py, end of file), so no
second server is started.

The sys.path shim keeps the repo root importable regardless of how the
runtime lays out the task directory, so ``bot.py`` and its sibling
modules (composer, prompts, validators, ...) resolve cleanly.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bot import app  # noqa: E402  (ASGI app discovered by Vercel)
