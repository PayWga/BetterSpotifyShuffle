"""Recovery behavior uses synthetic search results and temporary catalogs."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import feature_extraction as fe


@pytest.mark.parametrize('annotation', ['Audio Only', 'Full Audio', 'Official Audio Only'])
def test_harmless_audio_annotations_are_accepted(annotation):
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist'},
                               {'title': f'Artist - Song ({annotation})'})
    assert score is not None


@pytest.mark.parametrize('annotation', ['Audio Only Live', 'Full Audio Cover', 'Another Song'])
def test_annotations_do_not_hide_recording_conflicts(annotation):
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist'},
                               {'title': f'Artist - Song ({annotation})'})
    assert score is None


@pytest.mark.parametrize('resolved_artists,expected', [(['Main', 'Guest'], True), (['Main'], True)])
def test_missing_flat_artist_credit_is_resolved_before_downloading(tmp_path, resolved_artists, expected):
    downloads, resolutions = [], []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if url.startswith('ytsearch'):
                return {'entries': [{'title': 'Main - Song', 'url': 'https://example.org/song'}]}
            if not download:
                resolutions.append(url)
                return {'title': 'Main - Song', 'artists': resolved_artists, 'duration': 180}
            downloads.append(url)
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return {'title': 'Main - Song', 'artists': resolved_artists, 'duration': 180}

    with patch('yt_dlp.YoutubeDL', Downloader):
        if expected:
            path, _ = fe.acquire_audio({'track_id': 'track', 'track_name': 'Song',
                                        'artist_name': 'Main, Guest', 'duration_ms': 180000}, tmp_path)
            assert path.read_bytes() == b'fixture'
        else:
            with pytest.raises(RuntimeError, match='no_matching_recording'):
                fe.acquire_audio({'track_id': 'track', 'track_name': 'Song',
                                  'artist_name': 'Main, Guest', 'duration_ms': 180000}, tmp_path)
    assert resolutions == ['https://example.org/song']
    assert downloads == (['https://example.org/song'] if expected else [])


def test_primary_artist_fallback_recovers_without_losing_guest_checks(tmp_path):
    queries = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if not download:
                queries.append(url)
                if len(queries) < 3:
                    return {'entries': []}
                return {'entries': [{'title': 'Main feat. Guest - Song', 'url': 'https://example.org/good'}]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return {'title': 'Main feat. Guest - Song'}

    with patch('yt_dlp.YoutubeDL', Downloader):
        path, _ = fe.acquire_audio({'track_id': 'track', 'track_name': 'Song',
                                    'artist_name': 'Main, Guest'}, tmp_path)
    assert path.read_bytes() == b'fixture'
    assert queries[2] == 'ytsearch15:Main song'


def test_failed_search_logs_bounded_candidate_examples(tmp_path, caplog):
    entries = [{'title': f'Other Artist - Different Song {i}', 'uploader': 'Other Artist',
                'url': f'https://example.org/{i}'} for i in range(20)]
    with patch('yt_dlp.YoutubeDL') as downloader:
        downloader.return_value.__enter__.return_value.extract_info.return_value = {'entries': entries}
        with pytest.raises(RuntimeError, match='no_matching_recording'):
            fe.acquire_audio({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Artist'}, tmp_path)
    examples = [r.getMessage() for r in caplog.records if 'Rejected candidate for' in r.getMessage()]
    assert len(examples) == 5
    assert 'Different Song 0' in examples[0]
    assert 'https://example.org/0' in examples[0]
    assert 'reason=title' in examples[0]


def test_rejection_diagnostics_strip_url_credentials_and_tokens(tmp_path, caplog):
    with patch('yt_dlp.YoutubeDL') as downloader:
        downloader.return_value.__enter__.return_value.extract_info.return_value = {
            'entries': [{'title': 'Different Song', 'url': 'https://user:SECRET@example.org/video?token=TOKEN#FRAGMENT'}]}
        with pytest.raises(RuntimeError, match='no_matching_recording'):
            fe.acquire_audio({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Artist'}, tmp_path)
    assert 'url=https://example.org/video' in caplog.text
    assert 'SECRET' not in caplog.text
    assert 'TOKEN' not in caplog.text
    assert 'FRAGMENT' not in caplog.text


@pytest.mark.parametrize('message,expected', [
    ('Video not available in your country', 'geo_restricted'),
    ('The uploader has not made this video available in your country', 'geo_restricted'),
    ('This video is blocked in your region', 'geo_restricted'),
    ('The page needs to be reloaded.', 'session_reload_required'),
    ('Signature solving failed', 'js_challenge_failed'),
    ('Video not available', 'unavailable'),
])
def test_download_errors_preserve_actionable_reason(message, expected):
    assert fe.acquisition_error_code(RuntimeError(message)) == expected


def test_retry_failed_moves_persistent_failures_behind_less_attempted_tracks(tmp_path):
    master, backlog = tmp_path / 'master.csv', tmp_path / 'backlog.csv'
    pd.DataFrame([{'id': 'existing'}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': tid, 'track_name': 'Song', 'artist_name': 'Artist', 'attempts': 3}
                  for tid in ['one', 'two', 'three', 'four']]).to_csv(backlog, index=False)
    requested = []

    def acquire(item, audio_dir):
        requested.append(item['track_id'])
        raise RuntimeError('no_matching_recording')

    with patch.object(fe, 'acquire_audio', side_effect=acquire):
        fe.process(backlog, master, tmp_path / 'audio', retry_failed=True, limit=2, workers=1)
        fe.process(backlog, master, tmp_path / 'audio', retry_failed=True, limit=2, workers=1)
    assert requested == ['one', 'two', 'three', 'four']
    assert pd.read_csv(backlog)['attempts'].tolist() == [4, 4, 4, 4]


def test_guest_credit_without_primary_artist_never_resolves_or_downloads(tmp_path):
    calls = []

    class Downloader:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            calls.append((url, download))
            return {'entries': [{'title': 'Guest - Song', 'url': 'https://example.org/wrong'}]}

    with patch('yt_dlp.YoutubeDL', Downloader), pytest.raises(RuntimeError, match='no_matching_recording'):
        fe.acquire_audio({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Main, Guest'}, tmp_path)
    assert calls
    assert all(url.startswith('ytsearch') and not download for url, download in calls)


def test_download_losing_previously_resolved_guest_is_not_published(tmp_path):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if url.startswith('ytsearch'):
                return {'entries': [{'title': 'Main - Song', 'url': 'https://example.org/song'}]}
            if not download:
                return {'title': 'Main - Song', 'artists': ['Main', 'Guest']}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'wrong')
            return {'title': 'Main - Song', 'artists': ['Main']}

    with patch('yt_dlp.YoutubeDL', Downloader), pytest.raises(RuntimeError, match='resolved_recording_mismatch: artist'):
        fe.acquire_audio({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Main, Guest'}, tmp_path)
    assert not (tmp_path / 'track.wav').exists()
    assert not (tmp_path / 'track.source.json').exists()


@pytest.mark.parametrize('message,code', [
    ('The page needs to be reloaded.', 'session_reload_required'),
    ('Signature solving failed', 'js_challenge_failed')])
def test_service_error_during_metadata_resolution_stops_fallback_requests(tmp_path, message, code):
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
            calls.append((url, download))
            if url.startswith('ytsearch'):
                return {'entries': [{'title': 'Main - Song', 'url': 'https://example.org/song'}]}
            raise DownloadError(message)

    with patch('yt_dlp.YoutubeDL', Downloader), pytest.raises(RuntimeError, match=code):
        fe.acquire_audio({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Main, Guest'}, tmp_path)
    assert len(calls) == 2
    assert calls[1] == ('https://example.org/song', False)


def test_structured_creator_credits_can_identify_collaborators():
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Main, Guest'},
                               {'title': 'Main - Song', 'creators': ['Main', 'Guest']})
    assert score is not None


@pytest.mark.parametrize('annotation', ['Audio Only', 'Full Audio'])
def test_audio_annotations_do_not_allow_a_conflicting_remaster_year(annotation):
    score, _ = fe.match_result({'track_name': 'Song - 2011 Remaster', 'artist_name': 'Artist'},
                               {'title': f'Artist - Song (Remastered 2012) ({annotation})'})
    assert score is None
