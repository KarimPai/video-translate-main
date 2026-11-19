"""Pytest configuration for the video translator test suite."""
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'scripts' and 'src' are importable.
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
