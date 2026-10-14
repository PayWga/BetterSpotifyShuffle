"""Keep Numba's compilation cache inside the test workspace."""
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / '.test-runs' / 'numba-cache'))
