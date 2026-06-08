"""Make `phase1` and `activation-extract` importable from tests."""

import sys
from pathlib import Path

_HERE = Path(__file__).parent
_ACTIVATION_EXTRACT = Path.home() / "activation-extract"

for p in (_HERE, _ACTIVATION_EXTRACT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
