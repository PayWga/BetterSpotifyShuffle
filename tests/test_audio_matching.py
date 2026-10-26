"""Search and matching regressions; all online responses are synthetic."""
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import feature_extraction as fe


@pytest.mark.parametrize('requested,artist,title,channel,expected', [
    ('Example Song', 'Main Artist, Guest Artist', 'Main Artist feat. Guest Artist - Example Song', '', True),
    ('Example Song', 'Main Artist, Guest Artist', 'Main Artist - Example Song', '', True),
    ('Example Song (feat. Guest Artist)', 'Main Artist', 'Main Artist ft. Guest Artist - Example Song', '', True),
    ('Example Song (feat. Guest Artist)', 'Main Artist', 'Main Artist - Example Song', '', True),
    ('Example Song - 2011 Remaster', 'Main Artist', 'Main Artist - Example Song (Remastered 2011)', '', True),
    ('Example Song - 2011 Remaster', 'Main Artist', 'Main Artist - Example Song (Remastered 2011) (Official Audio)', '', True),
    ('Example Song - 2011 Remaster', 'Main Artist', 'Main Artist - Example Song', '', True),
    ("Don't Stop", 'Main Artist', 'Main Artist - Dont Stop', '', True),
    ('Example Song', 'Main Artist', 'Example Song (Official Video)', 'Main ArtistVEVO', True),
    ('Example Song', 'Main Artist', 'Main Artist - Example Song (Live)', '', False),
    ('Example Song', 'Main Artist', 'Main Artist - Example Song (Cover)', '', False),
    ('Example Song', 'Main Artist', 'Other Artist - Example Song', '', False),
    ('Example Song', 'Main Artist', 'Main Artist - Another Example Song', '', False),
    ('Example Song - 2011 Remaster', 'Main Artist', 'Main Artist - Example Song (Remastered 2012)', '', False),
])
def test_safe_metadata_formatting_variations(requested, artist, title, channel, expected):
    item = {'track_name': requested, 'artist_name': artist, 'duration_ms': 180000}
    entry = {'title': title, 'channel': channel, 'duration': 180}
    assert (fe.match_score(item, entry) is not None) is expected


def test_fallback_search_recovers_when_first_query_only_finds_wrong_recordings(tmp_path):
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
                if len(queries) == 1:
                    return {'entries': [{'title': 'Artist - Example Song (Live)', 'url': 'https://example.org/wrong'}]}
                return {'entries': [{'title': 'Artist - Example Song', 'url': 'https://example.org/right'}]}
            assert url == 'https://example.org/right'
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return {'webpage_url': url, 'title': 'Artist - Example Song'}

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, source = fe.acquire_audio({'track_id': 'track', 'track_name': 'Example Song', 'artist_name': 'Artist'}, tmp_path)
    assert audio.read_bytes() == b'fixture'
    assert source == 'https://example.org/right'
    assert len(queries) == 2


def test_no_matching_recording_has_specific_error_instead_of_audio_not_found(tmp_path):
    with patch('yt_dlp.YoutubeDL') as downloader:
        downloader.return_value.__enter__.return_value.extract_info.return_value = {
            'entries': [{'title': 'Other Artist - Example Song', 'url': 'https://example.org/wrong'}]}
        with pytest.raises(RuntimeError, match='no_matching_recording'):
            fe.acquire_audio({'track_id': 'track', 'track_name': 'Example Song', 'artist_name': 'Main Artist'}, tmp_path)


def test_empty_search_results_have_specific_error(tmp_path):
    with patch('yt_dlp.YoutubeDL') as downloader:
        downloader.return_value.__enter__.return_value.extract_info.return_value = {'entries': []}
        with pytest.raises(RuntimeError, match='no_search_results'):
            fe.acquire_audio({'track_id': 'track', 'track_name': 'Example Song', 'artist_name': 'Artist'}, tmp_path)


@pytest.mark.parametrize('message,expected', [
    ('HTTP Error 403: Forbidden', 'http_403'),
    ('HTTP Error 429: Too Many Requests', 'http_429'),
    ('Sign in to confirm you are not a bot', 'authentication_required'),
    ('Video not available in your country', 'geo_restricted'),
    ('Connection timed out', 'network_timeout'),
])
def test_download_errors_have_actionable_categories(message, expected):
    assert fe.acquisition_error_code(RuntimeError(message)) == expected


def test_resolved_duration_mismatch_is_not_published_to_cache(tmp_path):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if not download:
                return {'entries': [
                    {'title': 'Artist - Example Song', 'url': 'https://example.org/bad'},
                    {'title': 'Artist - Example Song', 'url': 'https://example.org/good'},
                ]}
            bad = url.endswith('bad')
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'wrong' if bad else b'right')
            return {'webpage_url': url, 'title': 'Artist - Example Song', 'duration': 600 if bad else 180}

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, source = fe.acquire_audio({'track_id': 'track', 'track_name': 'Example Song',
                                         'artist_name': 'Artist', 'duration_ms': 180000}, tmp_path)
    assert audio.read_bytes() == b'right'
    assert source == 'https://example.org/good'


def test_resolved_validation_uses_selected_candidate_not_last_search_entry(tmp_path):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if not download:
                return {'entries': [
                    {'title': 'Artist - Example Song', 'uploader': 'Artist', 'duration': 180,
                     'url': 'https://example.org/good'},
                    {'title': 'Other Artist - Other Song', 'uploader': 'Other Artist',
                     'duration': 600, 'is_live': True, 'url': 'https://example.org/wrong'},
                ]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'right')
            return {'webpage_url': url, 'title': 'Example Song'}

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, source = fe.acquire_audio({'track_id': 'track', 'track_name': 'Example Song',
                                         'artist_name': 'Artist', 'duration_ms': 180000}, tmp_path)
    assert audio.read_bytes() == b'right'
    assert source == 'https://example.org/good'


@pytest.mark.parametrize('status', [403, 429])
def test_http_errors_retry_new_candidates_but_do_not_hammer_rate_limits(tmp_path, status):
    from yt_dlp.utils import DownloadError
    searches, downloads = [], []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if not download:
                searches.append(url)
                entries = [{'title': 'Artist - Example Song', 'url': 'https://example.org/bad'}]
                if len(searches) > 1:
                    entries.append({'title': 'Artist - Example Song', 'url': 'https://example.org/good'})
                return {'entries': entries}
            downloads.append(url)
            if url.endswith('bad'):
                raise DownloadError(f'HTTP Error {status}: synthetic response')
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'right')
            return {'webpage_url': url, 'title': 'Artist - Example Song'}

    item = {'track_id': 'track', 'track_name': 'Example Song', 'artist_name': 'Artist'}
    with patch('yt_dlp.YoutubeDL', Downloader):
        if status == 429:
            with pytest.raises(RuntimeError, match='http_429'):
                fe.acquire_audio(item, tmp_path)
            assert len(searches) == 1
            assert downloads == ['https://example.org/bad']
        else:
            audio, _ = fe.acquire_audio(item, tmp_path)
            assert audio.read_bytes() == b'right'
            assert downloads == ['https://example.org/bad', 'https://example.org/good']
