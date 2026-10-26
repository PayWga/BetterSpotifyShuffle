"""Authentication outages must not exhaust a synthetic backlog's attempts."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import feature_extraction as fe


@pytest.mark.parametrize('during_search', [False, True])
def test_authentication_challenge_stops_candidate_and_search_retries(tmp_path, during_search):
    from yt_dlp.utils import DownloadError
    calls = []

    class Downloader:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            calls.append(download)
            if download or during_search:
                raise DownloadError("Sign in to confirm you're not a bot")
            return {'entries': [
                {'title': 'Artist - Example Song', 'url': 'https://example.org/one'},
                {'title': 'Artist - Example Song', 'url': 'https://example.org/two'}]}

    with patch('yt_dlp.YoutubeDL', Downloader):
        with pytest.raises(RuntimeError, match='authentication_required'):
            fe.acquire_audio({'track_id': 'track', 'track_name': 'Example Song',
                              'artist_name': 'Artist'}, tmp_path)
    assert calls == ([False] if during_search else [False, True])


@pytest.mark.parametrize('reason, expected_requests', [
    ('download_failed: authentication_required', 3),
    ('download_failed: session_reload_required', 3),
    ('download_failed: js_challenge_failed', 3),
    ('search_failed: http_429', 1)])
def test_block_stops_new_requests_preserves_attempts_and_saves_ready_audio(tmp_path, reason, expected_requests):
    master, backlog, audio = tmp_path / 'master.csv', tmp_path / 'backlog.csv', tmp_path / 'audio'
    audio.mkdir()
    (audio / 'cached.wav').write_bytes(b'fixture')
    pd.DataFrame([{'id': 'cached', 'tempo': float('nan')}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': tid, 'track_name': 'Song', 'artist_name': 'Artist',
                   'attempts': 1, 'last_error': ''}
                  for tid in ['cached'] + [f'track{i}' for i in range(8)]]).to_csv(backlog, index=False)
    requests = []

    def acquire(item, directory):
        requests.append(item['track_id'])
        raise RuntimeError(reason)

    with patch.object(fe, 'acquire_audio', side_effect=acquire), patch.object(
        fe, 'extract_audio_features', return_value={'tempo': 123.}):
        stats = fe.process(backlog, master, audio, acquisition_workers=1,
                           analysis_workers=1, batch_size=1)
    assert len(requests) == expected_requests
    remaining = pd.read_csv(backlog).set_index('track_id')
    assert len(remaining) == 8
    assert remaining['attempts'].eq(1).all()
    assert remaining.loc[requests[0], 'last_error'] == reason
    assert pd.read_csv(master).set_index('id').loc['cached', 'tempo'] == 123.
    assert stats['saved'] == 1
    assert stats['failed'] == 0
    assert stats['deferred'] == 8


def test_isolated_challenge_does_not_stop_other_tracks(tmp_path):
    master, backlog, audio = tmp_path / 'master.csv', tmp_path / 'backlog.csv', tmp_path / 'audio'
    audio.mkdir()
    pd.DataFrame([{'id': 'existing'}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': tid, 'track_name': 'Song', 'artist_name': 'Artist'}
                  for tid in ['blocked', 'good', 'missing']]).to_csv(backlog, index=False)

    def acquire(item, directory):
        if item['track_id'] == 'blocked':
            raise RuntimeError('download_failed: authentication_required')
        if item['track_id'] == 'missing':
            raise RuntimeError('no_matching_recording')
        return Path(directory) / 'good.wav', 'synthetic'

    with patch.object(fe, 'acquire_audio', side_effect=acquire), patch.object(
        fe, 'extract_audio_features', return_value={'tempo': 120.}):
        stats = fe.process(backlog, master, audio, workers=1)
    remaining = pd.read_csv(backlog).set_index('track_id')
    assert remaining.loc['blocked', 'attempts'] == 0
    assert remaining.loc['missing', 'attempts'] == 1
    assert stats['saved'] == 1
    assert stats['failed'] == 1
    assert stats['deferred'] == 1
