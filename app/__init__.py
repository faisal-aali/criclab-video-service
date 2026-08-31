# Cric-Lab backend package
import os
from pathlib import Path

# Serverless (Vercel) has no display; matplotlib/OpenCV must not load GUI backends.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_mpl = Path("/tmp/matplotlib")
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))


