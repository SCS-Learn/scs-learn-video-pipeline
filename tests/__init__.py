"""Tests for the lecture scanner (src/scan).

Plain stdlib `unittest`, no pytest, no network, no PSC, no reliance on data/.
Run from the repository root:

    .venv/bin/python -m unittest discover -s tests -v

The repository root is put on sys.path here so that `from src.scan import ...`
resolves however unittest happens to have chosen its top-level directory.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
