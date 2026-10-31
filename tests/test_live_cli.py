import importlib
import json

import pandas as pd
import pytest

from shufflerec.features import FeaturePreprocessor
from shufflerec.model import ModelConfig, SpotifySASRec
from shufflerec.training import save_checkpoint


def api():
    assert importlib.util.find_spec('shufflerec.live'), 'live entry point missing'
    return importlib.import_module('shufflerec.live')


def prepare(tmp_path):
    frame = pd.DataFrame([dict(id=str(i), artists='Artist', duration_ms=180000, tempo=80 + i)
                          for i in range(20)])
    prep = FeaturePreprocessor().fit(frame, frame.id)
    model = SpotifySASRec(prep, max_length=3,
                         config=ModelConfig(d_model=8, nhead=2, feedforward=16, dropout=0))
    save_checkpoint(tmp_path / 'model.pt', model, prep)
    frame.to_csv(tmp_path / 'catalog.csv', index=False)
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps({'checkpoint': 'model.pt', 'catalog': 'catalog.csv',
                               'request_interval_seconds': 0}), encoding='utf-8')
    return path


def spotify():
    class Client:
        def __init__(self): self.queued = []; self.writes = []
        def current_playback(self):
            return {'item': {'id': '0', 'name': 'Synthetic', 'type': 'track', 'duration_ms': 180000,
                             'artists': [{'id': 'artist', 'name': 'Artist'}]},
                    'progress_ms': 0, 'is_playing': True, 'device': {'id': 'pc'},
                    'context': {'type': 'playlist', 'uri': 'spotify:playlist:source'}}
        def playlist(self, identity, fields=None): return {'id': identity, 'snapshot_id': 'v1'}
        def playlist_items(self, identity, **kwargs):
            return {'items': [{'item': {'id': str(i), 'type': 'track', 'duration_ms': 180000,
                                      'artists': [{'name': 'Artist', 'id': 'artist'}]}} for i in range(20)], 'next': None}
        def queue(self): return {'currently_playing': {'id': '0'}, 'queue': [{'id': t, 'type': 'track'} for t in self.queued]}
        def add_to_queue(self, uri, device_id=None):
            self.writes.append((uri, device_id)); self.queued.append(uri.rsplit(':', 1)[-1])
    return Client()


def test_real_checkpoint_to_spotify_queue_through_live_entry_point(tmp_path, capsys):
    m = api(); settings = prepare(tmp_path); client = spotify()
    status = m.run_pipeline(settings_path=settings, private_dir=tmp_path, client=client, once=True)
    assert status['upcoming'] == 5
    assert len(client.writes) == 5 and len(set(client.queued)) == 5
    assert (tmp_path / 'queue_commands.jsonl').exists()
    assert (tmp_path / 'playback_evidence.jsonl').exists()
    assert not (tmp_path / 'spotify_credentials.json').exists()


def test_dry_run_plans_with_real_checkpoint_without_queue_or_history_writes(tmp_path):
    m = api(); settings = prepare(tmp_path); client = spotify()
    status = m.run_pipeline(settings_path=settings, private_dir=tmp_path, client=client, once=True, dry_run=True)
    assert len(status['queue']) == 5
    assert client.writes == []
    assert not (tmp_path / 'queue_commands.jsonl').exists()
    assert not (tmp_path / 'spotify_listening_history.csv').exists()


def test_lease_prevents_two_playback_owners_and_releases_without_deleting_file(tmp_path):
    api()
    from scripts.playback_lease import PlaybackLease
    path = tmp_path / 'owner.lock'
    with PlaybackLease(path):
        with pytest.raises(RuntimeError, match='already running'):
            with PlaybackLease(path):
                pytest.fail('two owners acquired the lease')
    with PlaybackLease(path):
        assert path.exists()


def test_existing_polling_daemon_respects_live_lease_before_authentication(tmp_path, monkeypatch):
    api()
    from scripts import polling
    from scripts.playback_lease import PlaybackLease
    monkeypatch.setattr(polling, 'PRIVATE_DIR', tmp_path)
    monkeypatch.setattr(polling, 'init_csv', lambda: pytest.fail('locked daemon must not initialize history'))
    with PlaybackLease(tmp_path / 'playback_owner.lock'):
        with pytest.raises(RuntimeError, match='already running'):
            polling.start_daemon()


def test_cleanup_error_still_closes_owned_http_session(tmp_path, monkeypatch):
    m = api(); settings = prepare(tmp_path)
    class Session:
        closed = False
        def close(self): self.closed = True
    client = spotify(); client._session = Session()
    monkeypatch.setattr(m, 'create_client', lambda **kwargs: client)
    real_close = m.LiveController.close
    def close(app):
        real_close(app)
        raise OSError('synthetic cleanup error')
    monkeypatch.setattr(m.LiveController, 'close', close)
    with pytest.raises(OSError):
        m.run_pipeline(settings_path=settings, private_dir=tmp_path, once=True, dry_run=True)
    assert client._session.closed


@pytest.mark.parametrize('kind,expected', [('cooldown', 'spotify_cooldown'), ('oauth', 'authorization_required')])
def test_startup_auth_and_cooldown_errors_are_actionable_without_private_details(kind, expected, monkeypatch, capsys):
    m = api()
    from scripts.spotify_budget import SpotifyBudgetBlocked
    from spotipy.exceptions import SpotifyOauthError
    error = SpotifyBudgetBlocked('synthetic private detail') if kind == 'cooldown' else SpotifyOauthError('invalid_grant', 'synthetic private detail')
    def fail(**kwargs): raise error
    monkeypatch.setattr(m, 'run_pipeline', fail)
    assert m.main([]) == 2
    output = capsys.readouterr().out
    assert json.loads(output)['state'] == expected
    assert 'synthetic private detail' not in output
