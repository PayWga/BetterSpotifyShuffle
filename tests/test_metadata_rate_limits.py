"""Rate limits stop the whole metadata job without spending track attempts."""
import json

import pandas as pd
import pytest
from spotipy.exceptions import SpotifyException

from scripts import data_merging as dm
from scripts import spotify_metadata as sm


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    waits = []
    monkeypatch.setattr(sm.time, 'sleep', waits.append)
    return waits


@pytest.mark.parametrize('header', ['86160', None, 'invalid', '-1', 'nan', 'inf'])
def test_long_or_unknown_limit_stops_without_retry(header, no_wait):
    calls = []
    def limited():
        calls.append(1)
        raise SpotifyException(429, -1, 'synthetic', headers={} if header is None else {'retry-after': header})
    with pytest.raises(sm.SpotifyCooldown) as caught:
        sm.request(limited)
    assert len(calls) == 1
    assert not no_wait
    if header == '86160':
        assert caught.value.retry_after == 86160
        assert caught.value.retry_at.tzinfo is not None
        assert 'UTC' in str(caught.value)
    else:
        assert caught.value.retry_at is None


def test_short_limit_waits_then_returns_with_conservative_pacing(no_wait):
    calls = []
    def fetch():
        calls.append(1)
        if len(calls) == 1:
            raise SpotifyException(429, -1, 'synthetic', headers={'Retry-After': '2'})
        return 'ok'
    assert sm.request(fetch) == 'ok'
    assert no_wait == [2, 1]


def test_repeated_short_limits_stop_instead_of_moving_to_next_track(no_wait):
    calls = []
    def limited():
        calls.append(1)
        raise SpotifyException(429, -1, 'synthetic', headers={'Retry-After': '2'})
    with pytest.raises(sm.SpotifyCooldown):
        sm.request(limited)
    assert len(calls) == 4
    assert no_wait == [2, 2, 2]


@pytest.mark.parametrize('kind', ['backlog', 'master'])
def test_limit_checkpoints_completed_work_and_never_requests_later_tracks(tmp_path, kind):
    ids = ['A' * 22, 'B' * 22, 'C' * 22]
    path = tmp_path / 'catalog.csv'
    path.write_text(('track_id,attempts,last_error\n' if kind == 'backlog' else 'id,attempts,last_error\n')
                    + ''.join(f'{tid},4,no_matching_recording\n' for tid in ids), encoding='utf-8')
    class Spotify:
        calls = []
        def track(self, tid):
            self.calls.append(tid)
            if tid == ids[1]:
                raise SpotifyException(429, -1, 'synthetic', headers={'Retry-After': '86160'})
            return {'id': tid, 'name': 'Song', 'duration_ms': 180000, 'artists': [{'name': 'Artist'}]}
    sp = Spotify()
    checkpoint = tmp_path / 'checkpoint.json'
    with pytest.raises(sm.SpotifyCooldown):
        if kind == 'backlog':
            dm.backfill_backlog_metadata(path, sp=sp)
        else:
            dm.backfill_metadata(path, checkpoint, sp=sp)
    assert sp.calls == ids[:2]
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(rows['attempts']) == ['4'] * 3
    assert list(rows['last_error']) == ['no_matching_recording'] * 3
    if kind == 'backlog':
        assert rows.iloc[0]['duration_ms'] == '180000'
        assert rows.iloc[1]['duration_ms'] == ''
    else:
        assert json.loads(checkpoint.read_text())['records'][ids[0]]['duration_ms'] == 180000


def test_metadata_transport_preserves_429_response_headers():
    session = sm.metadata_session()
    try:
        for scheme in ('https://', 'http://'):
            retry = session.get_adapter(scheme).max_retries
            assert retry.total == 0
            assert retry.raise_on_status is False
    finally:
        session.close()


def test_spotipy_receives_original_429_headers_without_network(monkeypatch):
    from io import BytesIO
    import spotipy
    from urllib3.connectionpool import HTTPConnectionPool
    from urllib3.response import HTTPResponse

    def response(*args, **kwargs):
        return HTTPResponse(body=BytesIO(b'{"error":{"status":429,"message":"synthetic"}}'),
                            status=429, headers={'Retry-After': '86160'}, preload_content=False)
    monkeypatch.setattr(HTTPConnectionPool, '_make_request', response)
    session = sm.metadata_session()
    session.trust_env = False
    client = spotipy.Spotify(auth='synthetic', requests_session=session)
    try:
        with pytest.raises(sm.SpotifyCooldown) as caught:
            sm.request(lambda: client._internal_call('GET', 'http://offline.invalid/test', None, {}))
        assert caught.value.retry_after == 86160
    finally:
        session.close()


def test_cli_rate_limit_exit_is_readable_and_nonzero(capsys):
    def stopped():
        raise sm.SpotifyCooldown(86160)
    with pytest.raises(SystemExit) as caught:
        dm.run_metadata_action(stopped)
    assert caught.value.code == 2
    message = capsys.readouterr().out
    assert 'UTC' in message
    assert 'checkpoints are retained' in message


def test_client_factory_uses_header_preserving_session(tmp_path, monkeypatch):
    (tmp_path / 'spotify_credentials.json').write_text(json.dumps({
        'client_id': 'synthetic', 'client_secret': 'synthetic', 'redirect_uri': 'http://localhost'}))
    monkeypatch.setattr(dm, 'PRIVATE_DIR', tmp_path)
    monkeypatch.setattr(dm, 'SpotifyOAuth', lambda **kwargs: object())
    client = dm.get_spotify_client()
    try:
        assert client._session.get_adapter('https://').max_retries.raise_on_status is False
    finally:
        client._session.close()


def test_sync_preserves_successes_before_cooldown(tmp_path, monkeypatch):
    ids = ['A' * 22, 'B' * 22, 'C' * 22]
    master, history = tmp_path / 'master.csv', tmp_path / 'history.csv'
    master.write_text('id,name,tempo\nexisting,Keep,120\n')
    history.write_text('track_id\n' + '\n'.join(ids) + '\n')
    class Spotify:
        calls = []
        def track(self, tid):
            self.calls.append(tid)
            if tid == ids[1]:
                raise SpotifyException(429, -1, 'synthetic', headers={'Retry-After': '86160'})
            return {'id': tid, 'name': 'Song', 'duration_ms': 180000}
    sp = Spotify()
    monkeypatch.setattr(dm, 'MASTER_DB', master)
    monkeypatch.setattr(dm, 'HISTORY_DB', history)
    monkeypatch.setattr(dm, 'get_spotify_client', lambda: sp)
    with pytest.raises(sm.SpotifyCooldown):
        dm.sync_metadata()
    rows = pd.read_csv(master, dtype=str, keep_default_na=False).set_index('id')
    assert rows.loc[ids[0], 'name'] == 'Song'
    assert ids[1] not in rows.index
    assert rows.loc['existing', 'tempo'] == '120'
    assert sp.calls == ids[:2]
