import importlib
import json
from types import SimpleNamespace

import pytest


def api():
    assert importlib.util.find_spec('scripts.spotify_gateway'), 'shared gateway missing'
    return importlib.import_module('scripts.spotify_gateway')


def test_factory_uses_private_credentials_and_cache_without_auth_requests(tmp_path, monkeypatch):
    m = api()
    credentials = tmp_path / 'credentials.json'
    content = '{"client_id":"synthetic-id","client_secret":"synthetic-secret","redirect_uri":"http://127.0.0.1:8888/callback"}'
    credentials.write_text(content, encoding='utf-8')
    captured = {}
    def oauth(**kwargs):
        captured['oauth'] = kwargs
        return object()
    def spotify(**kwargs):
        captured['spotify'] = kwargs
        return SimpleNamespace(current_playback=lambda: {'item': None})
    monkeypatch.setattr(m, 'SpotifyOAuth', oauth)
    monkeypatch.setattr(m.spotipy, 'Spotify', spotify)
    client = m.create_client(credentials_path=credentials, cache_path=tmp_path / 'cache',
                             budget_path=tmp_path / 'budget.json')
    assert client.current_playback() == {'item': None}
    assert captured['oauth']['client_id'] == 'synthetic-id'
    assert captured['oauth']['client_secret'] == 'synthetic-secret'
    assert captured['oauth']['cache_handler'].cache_path == str(tmp_path / 'cache')
    assert captured['spotify']['retries'] == 0
    assert captured['spotify']['status_retries'] == 0
    assert captured['oauth']['requests_session'] is captured['spotify']['requests_session']
    assert credentials.read_text(encoding='utf-8') == content
    captured['spotify']['requests_session'].close()


def test_defaults_use_existing_sensitive_info_locations():
    m = api()
    assert m.CREDS_PATH.name == 'spotify_credentials.json'
    assert m.CREDS_PATH.parent.name == 'sensitive_info'
    assert m.CACHE_PATH.name == '.spotify_cache'
    assert m.BUDGET_PATH.name == 'spotify_request_budget.json'


def test_cooldown_blocks_before_reading_credentials(tmp_path, monkeypatch):
    m = api()
    from scripts.spotify_budget import SpotifyRequestBudget, SpotifyBudgetBlocked
    budget = tmp_path / 'budget.json'
    SpotifyRequestBudget(budget).pause(600, 'QUOTA_EXCEEDED')
    with pytest.raises(SpotifyBudgetBlocked):
        m.create_client(credentials_path=tmp_path / 'does-not-exist', cache_path=tmp_path / 'cache',
                        budget_path=budget)


def test_daemon_uses_one_observation_per_iteration_and_adaptive_delays():
    api()
    from scripts import playback_daemon as d
    class Client:
        def __init__(self):
            self.calls = 0
        def current_playback(self):
            progress = [0, 15000, 30000, 45000, 120000][self.calls]
            self.calls += 1
            return {'item': {'id': 'a', 'duration_ms': 180000, 'name': 'A', 'artists': []},
                    'progress_ms': progress, 'is_playing': True,
                    'device': {'id': 'pc'}, 'context': {'uri': 'spotify:playlist:x'}}
    client = Client()
    now, delays, evidence, history = [0.], [], [], []
    def sleep(delay):
        delays.append(delay)
        now[0] += delay
    d.run_polling(client, clock=lambda: now[0], sleeper=sleep, max_polls=5,
                  event_sink=evidence.append, history_sink=lambda *args: history.append(args))
    assert client.calls == 5
    assert delays == [15., 15., 15., 75.]
    assert len([e for e in evidence if e.kind == 'feedback' and e.outcome == 'accepted']) == 1
    assert len(history) == 1
    assert history[0][1] == 120000


def test_inferred_evidence_is_json_serializable_and_kept_out_of_legacy_history(tmp_path):
    api()
    from scripts import playback_daemon as d
    from shufflerec.playback import PlaybackTracker, Snapshot, SubmittedTrack
    tracker = PlaybackTracker()
    tracker.set_submitted([SubmittedTrack(i, t, 180000) for i, t in enumerate('abc', 1)], version='v')
    tracker.observe(Snapshot('a', 180000, 0, True, 0, 'pc', 'x'))
    events = tracker.observe(Snapshot('c', 180000, 0, True, 15, 'pc', 'x'), continuity=True)
    path = tmp_path / 'evidence.jsonl'
    for event in events:
        d.write_evidence(event, path=path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    b = next(row for row in rows if row['track_id'] == 'b')
    assert b['inferred'] and b['upper_bound_ms'] == 15000 and b['listened_ms'] == 0
    assert b['timestamp'].endswith('+00:00')


def test_pause_gaps_cannot_make_an_observed_skip_look_confidently_early():
    from shufflerec.playback import PlaybackTracker, Snapshot
    tracker = PlaybackTracker()
    tracker.observe(Snapshot('a', 180000, 0, True, 0, 'pc', 'x'))
    tracker.observe(Snapshot('a', 180000, 29000, False, 30, 'pc', 'x'))
    tracker.observe(Snapshot('a', 180000, 58000, True, 60, 'pc', 'x'))
    events = tracker.observe(Snapshot('b', 180000, 0, True, 61, 'pc', 'x'))
    a = next(e for e in events if e.track_id == 'a')
    assert a.upper_bound_ms >= 60000
    assert a.weak and a.weight == .5


def test_polling_factory_reuses_shared_gateway_and_private_path_settings(tmp_path, monkeypatch):
    m = api()
    from scripts import polling
    captured = {}
    client = object()
    def factory(**kwargs):
        captured.update(kwargs)
        return client
    monkeypatch.setattr(m, 'create_client', factory)
    monkeypatch.setattr(polling, 'CREDS_PATH', tmp_path / 'credentials.json')
    monkeypatch.setattr(polling, 'CACHE_PATH', tmp_path / 'cache')
    assert polling.get_spotify_client() is client
    assert captured['credentials_path'] == tmp_path / 'credentials.json'
    assert captured['cache_path'] == tmp_path / 'cache'


@pytest.mark.parametrize('completion,reason,event', [
    (.14, 'track_change', 'GHOST_SKIP'), (.15, 'track_change', 'NEUTRAL'),
    (.25, 'track_change', 'NEUTRAL'), (.70, 'track_change', 'PLAY'),
    (.01, 'daemon_stopped', 'NEUTRAL'),
])
def test_historical_event_classification_stays_compatible(completion, reason, event):
    from scripts.polling import classify_live_event
    assert classify_live_event(completion, reason)[0] == event


def test_adaptive_end_guard_preserves_historical_play_for_continuous_full_song():
    from scripts.playback_daemon import run_polling
    from scripts.polling import classify_live_event
    now, history, times = [0.], [], []
    class Client:
        def current_playback(self):
            times.append(now[0])
            a = now[0] < 180
            return {'item': {'id': 'a' if a else 'b', 'duration_ms': 180000},
                    'progress_ms': now[0] * 1000 if a else 0, 'is_playing': True,
                    'device': {'id': 'pc'}, 'context': {'uri': 'x'}}
    def sleep(delay):
        now[0] += delay
    run_polling(Client(), clock=lambda: now[0], sleeper=sleep, max_polls=7,
                history_sink=lambda *args: history.append(args))
    a = next(row for row in history if row[0]['id'] == 'a')
    assert 178. in times
    assert a[1] >= 126000
    assert classify_live_event(a[1] / 180000, a[-1])[0] == 'PLAY'


@pytest.mark.parametrize('mode', ['outage', 'transfer', 'uncertain_threshold'])
def test_uncertain_timing_is_never_written_as_historical_early_skip(mode):
    from scripts.playback_daemon import run_polling
    from scripts.polling import classify_live_event
    from requests.exceptions import ConnectionError
    now, history = [0.], []
    class Client:
        calls = 0
        def current_playback(self):
            self.calls += 1
            if mode == 'outage' and self.calls == 2:
                raise ConnectionError('synthetic outage')
            first = self.calls == 1
            return {'item': {'id': 'a' if first else 'b', 'duration_ms': 60000},
                    'progress_ms': 0, 'is_playing': True,
                    'device': {'id': 'pc' if first or mode != 'transfer' else 'phone'},
                    'context': {'uri': 'x'}}
    def sleep(delay):
        now[0] += delay
    run_polling(Client(), clock=lambda: now[0], sleeper=sleep,
                max_polls=3 if mode == 'outage' else 2,
                history_sink=lambda *args: history.append(args))
    a = next(row for row in history if row[0]['id'] == 'a')
    assert classify_live_event(a[1] / 60000, a[-1])[0] == 'NEUTRAL'


def test_unknown_track_duration_is_logged_as_missing_and_neutral(tmp_path, monkeypatch):
    import csv
    from scripts import polling
    path = tmp_path / 'history.csv'
    monkeypatch.setattr(polling, 'CSV_FILE', path)
    polling.init_csv()
    polling.log_track({'id': 'unknown', 'name': 'Synthetic', 'duration_ms': None, 'artists': []},
                      1000, 'x', 'playlist', 'uncertain_transition')
    with path.open(newline='', encoding='utf-8') as stream:
        row = next(csv.DictReader(stream))
    assert row['duration_ms'] == '' and row['completion_rate'] == ''
    assert row['event_type'] == 'NEUTRAL'
    assert row['ms_played'] == '1000'
