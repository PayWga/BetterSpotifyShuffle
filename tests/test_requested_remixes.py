"""A requested remix is eligible; an original must never borrow its audio."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import feature_extraction as fe


@pytest.mark.parametrize('requested,upload', [
    ('Song (Remix)', 'Artist - Song (Remix) (Lyrics)'),
    ('Song - DJ Max Remix', 'Artist - Song (DJ Max Remix) [Unofficial Lyrics]'),
    ('Song - DJ Max Remix', 'Artist - Song (DJ Max RMX)'),
    ('Song (Club Mix)', 'Artist - Song (Club Mix)'),
])
def test_requested_remix_can_match_its_upload(requested, upload):
    score, _ = fe.match_result({'track_name': requested, 'artist_name': 'Artist', 'duration_ms': 180000},
                               {'title': upload, 'duration': 180})
    assert score is not None


@pytest.mark.parametrize('requested,upload', [
    ('Song', 'Artist - Song (Remix)'),
    ('Song (Remix)', 'Artist - Song (Lyrics)'),
    ('Song - DJ Max Remix', 'Artist - Song (DJ Other Remix)'),
    ('Song (Club Mix)', 'Artist - Song (Dub Mix)'),
])
def test_original_and_different_remix_are_not_interchangeable(requested, upload):
    score, _ = fe.match_result({'track_name': requested, 'artist_name': 'Artist', 'duration_ms': 180000},
                               {'title': upload, 'duration': 180})
    assert score is None


def test_requested_remix_is_downloaded_and_published(tmp_path):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            entry = {'title': 'Artist - Song (DJ Max Remix)', 'duration': 180,
                     'url': 'https://example.org/remix'}
            if not download:
                return {'entries': [entry]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'requested remix')
            return entry

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, _ = fe.acquire_audio({'track_id': 'track', 'track_name': 'Song - DJ Max Remix',
                                     'artist_name': 'Artist', 'duration_ms': 180000}, tmp_path)
    assert audio.read_bytes() == b'requested remix'
    assert json.loads(audio.with_suffix('.source.json').read_text(encoding='utf-8'))['match_evidence'] == 'matched'


@pytest.mark.parametrize('cached_title,expected', [
    ('Artist - Song (DJ Max Remix)', True),
    ('Artist - Song (DJ Other Remix)', False),
    ('Artist - Song', False),
])
def test_requested_remix_cache_respects_named_version(tmp_path, cached_title, expected):
    (tmp_path / 'track.wav').write_bytes(b'fixture')
    (tmp_path / 'track.source.json').write_text(json.dumps({'title': cached_title}), encoding='utf-8')
    prepared, _, _ = fe.prepare_track({'track_id': 'track', 'track_name': 'Song - DJ Max Remix',
                                      'artist_name': 'Artist'}, tmp_path, local_only=True, dry_run=False)
    assert (prepared is not None) is expected


@pytest.mark.parametrize('field,value', [
    ('track', 'Song (DJ Other Remix)'),
    ('alt_title', 'Song (DJ Other Remix)'),
    ('version', 'DJ Other Remix'),
    ('remixed_by', 'DJ Other'),
    ('remix_artist', 'DJ Other'),
])
def test_matching_remix_title_cannot_override_conflicting_resolved_metadata(field, value):
    score, _ = fe.match_result({'track_name': 'Song - DJ Max Remix', 'artist_name': 'Artist'},
                               {'title': 'Artist - Song (DJ Max Remix)', field: value})
    assert score is None


@pytest.mark.parametrize('remixer,expected', [('DJ Max', True), ('DJ Other', False)])
def test_featured_credit_does_not_hide_named_remix_identity(remixer, expected):
    score, _ = fe.match_result({'track_name': 'Song (feat. Guest) - DJ Max Remix',
                               'artist_name': 'Artist', 'duration_ms': 180000},
                              {'title': f'Artist - Song (feat. Guest) - {remixer} Remix', 'duration': 180})
    assert (score is not None) is expected


@pytest.mark.parametrize('artist,title,channel', [
    ('Artist', 'Song (Remix)', 'Artist - Topic'),
    ('Main, Guest', 'Main - Song (Remix)', 'Random Fan'),
])
def test_published_remix_cache_preserves_evidence_needed_for_reuse(tmp_path, artist, title, channel):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            entry = {'title': title, 'channel': channel, 'duration': 180,
                     'url': 'https://example.org/remix'}
            if url.startswith('ytsearch'):
                return {'entries': [entry]}
            if download:
                Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'requested remix')
            return entry

    item = {'track_id': 'track', 'track_name': 'Song (Remix)', 'artist_name': artist, 'duration_ms': 180000}
    with patch('yt_dlp.YoutubeDL', Downloader):
        fe.acquire_audio(item, tmp_path)
    prepared, error, _ = fe.prepare_track(item, tmp_path, local_only=True, dry_run=False)
    assert error is None
    assert prepared['audio'].read_bytes() == b'requested remix'


@pytest.mark.parametrize('field', ['track', 'alt_title'])
def test_featured_credit_cannot_hide_conflicting_resolved_remix(field):
    score, _ = fe.match_result({'track_name': 'Song (feat. Guest) - DJ Max Remix',
                               'artist_name': 'Artist', 'duration_ms': 180000},
                              {'title': 'Artist - Song (feat. Guest) - DJ Max Remix',
                               field: 'Song - DJ Other Remix', 'duration': 180})
    assert score is None
