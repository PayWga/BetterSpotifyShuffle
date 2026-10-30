"""Keep Numba's compilation cache inside the test workspace."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1] / '.test-runs' / 'numba-cache'))


@pytest.fixture(autouse=True)
def isolate_private_spotify_proxy(monkeypatch, tmp_path_factory):
    """Tests never read or route through the user's real private proxy config."""
    from scripts import spotify_proxy
    monkeypatch.setattr(spotify_proxy, 'PROXY_CONFIG_PATH',
                        tmp_path_factory.getbasetemp() / 'absent-private-proxy.json')


@pytest.fixture
def metadata_clock(monkeypatch):
    """Keep offline metadata integration tests fast and isolated from real time."""
    from scripts import spotify_metadata as sm
    now = [0.0]
    def sleep(seconds):
        now[0] += seconds
    monkeypatch.setattr(sm, 'time', SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    monkeypatch.setattr(sm, '_last_request_started', None)
