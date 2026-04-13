"""
Pytest configuration for engine/ unit tests.

Adds the engine/ directory to sys.path so all engine modules are importable
regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

# engine/ root → two levels up from this file (tests/conftest.py)
ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))
