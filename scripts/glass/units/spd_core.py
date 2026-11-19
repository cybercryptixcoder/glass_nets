"""Re-export of the SPD engine, which lives in lab/spd.py since CHANGES A1 (kept so the U19 experiment scripts run)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "lab"))
from spd import *            # noqa: F401,F403,E402
from spd import _run         # noqa: F401,E402
