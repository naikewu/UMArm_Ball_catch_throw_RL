"""Fresh PPO ball-catch training package for the fitted UMArm twin."""
from __future__ import annotations

import sys
from pathlib import Path

# The fitted plant remains owned by the parent workspace's ``control`` package.
_workspace_root = str(Path(__file__).resolve().parents[2])
if _workspace_root not in sys.path:
    sys.path.insert(0, _workspace_root)
