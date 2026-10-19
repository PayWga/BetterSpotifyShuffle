"""Persistent budget tests use only temporary state and simulated HTTP."""
import json

import pytest

from scripts import spotify_metadata as sm
from scripts import data_merging as dm


def test_budget_counts_attempts_across_instances_and_expires(tmp_path):
    now = [100000.0]
    path = tmp_path / 'budget.json'
    guard = sm.SpotifyRequestBudget(path, limit=2, clock=lambda: now[0])
    guard.reserve()
    sm.SpotifyRequestBudget(path, limit=2, clock=lambda: now[0]).reserve()
    with pytest.raises(sm.SpotifyBudgetBlocked):
        guard.reserve()
    now[0] += 86401
    guard.reserve()
    assert len(json.loads(path.read_text())['attempts']) == 1


def test_default_does_not_impose_an_invented_daily_cap(tmp_path):
    guard = sm.SpotifyRequestBudget(tmp_path / 'budget.json')
    with guard.locked():
        guard.write({'version': 1, 'attempts': [guard.clock()] * 600,
                     'blocked_until': 0, 'reason': ''})
    guard.reserve()
    assert guard.summary()['limit_per_24h'] is None


def test_sustained_pacing_and_batch_rest_survive_restarts(tmp_path):
    now, starts = [100000.0], []
    def sleep(seconds):
        now[0] += seconds
    path = tmp_path / 'budget.json'
    for _ in range(4):
        guard = sm.SpotifyRequestBudget(path, interval=10, rest_every=2, rest_seconds=300,
                                        clock=lambda: now[0], sleeper=sleep)
        guard.reserve()
        starts.append(now[0])
    assert starts == [100000, 100010, 100310, 100320]


def test_cooldown_persists_and_cannot_be_shortened(tmp_path):
    now = [100000.0]
    path = tmp_path / 'budget.json'
    guard = sm.SpotifyRequestBudget(path, clock=lambda: now[0])
    guard.pause(80000, 'QUOTA_EXCEEDED')
    guard.pause(1, 'rate limit')
    restarted = sm.SpotifyRequestBudget(path, clock=lambda: now[0])
    with pytest.raises(sm.SpotifyBudgetBlocked, match='QUOTA_EXCEEDED'):
        restarted.reserve()
    now[0] += 80001
    restarted.reserve()


def test_budget_corruption_fails_closed(tmp_path):
    path = tmp_path / 'budget.json'
    path.write_text('{broken')
    with pytest.raises(sm.SpotifyBudgetBlocked, match='unreadable'):
        sm.SpotifyRequestBudget(path).reserve()


def test_transport_counts_failed_http_attempts_before_retries(tmp_path, monkeypatch):
    from urllib3.connectionpool import HTTPConnectionPool
    from urllib3.response import HTTPResponse
    from io import BytesIO
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        return HTTPResponse(body=BytesIO(b'{}'), status=503, preload_content=False)
    monkeypatch.setattr(HTTPConnectionPool, '_make_request', response)
    guard = sm.SpotifyRequestBudget(tmp_path / 'budget.json', limit=2)
    session = sm.metadata_session(guard)
    session.trust_env = False
    try:
        session.get('http://offline.invalid/test')
        session.get('http://offline.invalid/test')
        with pytest.raises(sm.SpotifyBudgetBlocked):
            session.get('http://offline.invalid/test')
        assert len(calls) == 2
    finally:
        session.close()


@pytest.mark.parametrize('header', ['80000', None])
def test_transport_persists_429_and_blocks_restart(tmp_path, monkeypatch, header):
    from urllib3.connectionpool import HTTPConnectionPool
    from urllib3.response import HTTPResponse
    from io import BytesIO
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        return HTTPResponse(body=BytesIO(b'{"error":{"reason":"QUOTA_EXCEEDED"}}'),
                            status=429, headers={} if header is None else {'Retry-After': header},
                            preload_content=False)
    monkeypatch.setattr(HTTPConnectionPool, '_make_request', response)
    path = tmp_path / 'budget.json'
    session = sm.metadata_session(sm.SpotifyRequestBudget(path))
    session.trust_env = False
    try:
        with pytest.raises(sm.SpotifyBudgetBlocked, match='QUOTA_EXCEEDED'):
            session.get('http://offline.invalid/test')
        with pytest.raises(sm.SpotifyBudgetBlocked):
            sm.SpotifyRequestBudget(path).reserve()
        assert len(calls) == 1
    finally:
        session.close()


def test_factory_checks_cooldown_before_authentication(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, 'PRIVATE_DIR', tmp_path)
    sm.SpotifyRequestBudget(tmp_path / 'spotify_request_budget.json').pause(80000, 'reported cooldown')
    def forbidden(**kwargs):
        pytest.fail('Cooldown must block before authentication or reading credentials')
    monkeypatch.setattr(dm, 'SpotifyOAuth', forbidden)
    with pytest.raises(sm.SpotifyBudgetBlocked):
        dm.get_spotify_client()


def test_shared_budget_reservation_is_atomic_between_threads(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    path = tmp_path / 'budget.json'
    def attempt(_):
        try:
            sm.SpotifyRequestBudget(path, limit=3).reserve()
            return True
        except sm.SpotifyBudgetBlocked:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(10))) == 3


def test_budget_stop_preserves_master_checkpoint(tmp_path):
    path, checkpoint = tmp_path / 'master.csv', tmp_path / 'state.json'
    a, b = 'A' * 22, 'B' * 22
    path.write_text(f'id,name\n{a},\n{b},\n')
    before = path.read_bytes()
    class Spotify:
        def track(self, tid):
            if tid == b:
                raise sm.SpotifyBudgetBlocked('Local budget exhausted')
            return {'id': tid, 'name': 'Fetched'}
    with pytest.raises(sm.SpotifyBudgetBlocked):
        dm.backfill_metadata(path, checkpoint, sp=Spotify(), fetch_only=True, fields=['name'])
    assert json.loads(checkpoint.read_text())['records'][a]['name'] == 'Fetched'
    assert path.read_bytes() == before


def test_429_body_read_failure_still_persists_cooldown(tmp_path, monkeypatch):
    import requests
    from io import BytesIO
    calls = []
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = '80000'
    response._content = b''
    response.raw = BytesIO(b'')
    def broken_json(**kwargs):
        raise requests.exceptions.ChunkedEncodingError('truncated quota response')
    response.json = broken_json
    def send(*args, **kwargs):
        calls.append(1)
        return response
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', send)
    path = tmp_path / 'budget.json'
    session = sm.metadata_session(sm.SpotifyRequestBudget(path))
    try:
        with pytest.raises(sm.SpotifyBudgetBlocked):
            session.get('https://offline.invalid/test')
        with pytest.raises(sm.SpotifyBudgetBlocked):
            sm.SpotifyRequestBudget(path).reserve()
        assert len(calls) == 1
    finally:
        session.close()


def test_429_diagnostics_preserve_nonquota_reason(tmp_path, monkeypatch, capsys):
    import requests
    response = requests.Response()
    response.status_code = 429
    response.headers['Retry-After'] = '80000'
    response._content = b'{"error":{"reason":"RATE_LIMIT_EXCEEDED"}}'
    response._content_consumed = True
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', lambda *args, **kwargs: response)
    session = sm.metadata_session(sm.SpotifyRequestBudget(tmp_path / 'budget.json'))
    try:
        with pytest.raises(sm.SpotifyBudgetBlocked):
            session.get('https://offline.invalid/test')
        assert 'reason=RATE_LIMIT_EXCEEDED' in capsys.readouterr().out
    finally:
        session.close()
