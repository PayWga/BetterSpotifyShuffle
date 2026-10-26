"""Permissive upload presentation, rejecting remixes for original tracks."""
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts import feature_extraction as fe


@pytest.mark.parametrize('upload_title', [
    'Artist - Song (Unofficial Lyrics Video)',
    'Artist - Song [Lyrics by Fan Channel]',
    'Artist - Song (Lyrics / Letra / Paroles)',
    'Artist - Song (Fan Made Lyric Video)',
    'Artist - Song (Reupload)',
    'Artist - Song (Original Mix)',
])
def test_lyric_and_unofficial_reupload_presentations_match(upload_title):
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist', 'duration_ms': 180000},
                               {'title': upload_title, 'uploader': 'Random Music Fan', 'duration': 181})
    assert score is not None


@pytest.mark.parametrize('upload_title', [
    'Artist - Song (Lyrics) [DJ Remix]',
    'Artist - Song (Lyrics by DJ Remix)',
    'Artist - Song (Unofficial RMX)',
    'Artist - Song (Re-Mix)',
    'Artist - Song (Bootleg)',
    'Artist - Song (Mashup)',
    'Artist - Song (Club Mix)',
    'Artist - Song (Lyrics) (Slowed + Reverb)',
    'Artist - Song (Lyrics) (Sped Up)',
    'Artist - Song (Lyrics) (Nightcore)',
    'Artist - Song (Lyrics) (Cover)',
])
def test_lyric_labels_never_hide_changed_recordings(upload_title):
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist'}, {'title': upload_title})
    assert score is None


@pytest.mark.parametrize('field', ['title', 'track', 'alt_title', 'version'])
def test_remix_metadata_is_rejected_for_non_remix_request(field):
    entry = {'title': 'Artist - Song', field: 'Artist - Song (Remix)', 'duration': 180}
    score, reason = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist'}, entry)
    assert score is None
    assert reason == 'remix'


@pytest.mark.parametrize('metadata', [
    {'track': 'Song (Remix)'}, {'alt_title': 'Song (RMX)'}, {'version': 'DJ Remix'},
])
def test_hidden_remix_metadata_overrides_clean_upload_title(metadata):
    score, reason = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist'},
                                    {'title': 'Artist - Song (Lyrics)', **metadata})
    assert score is None
    assert reason == 'remix'


def test_missing_remaster_label_can_match_with_duration_and_audit_note():
    score, reason = fe.match_result({'track_name': 'Song - 2011 Remaster', 'artist_name': 'Artist',
                                     'duration_ms': 180000},
                                    {'title': 'Artist - Song (Lyrics)', 'duration': 181})
    assert score is not None
    assert reason == 'matched_unverified_remaster'


@pytest.mark.parametrize('duration', [None, 196])
def test_unlabelled_remaster_requires_close_known_duration(duration):
    score, _ = fe.match_result({'track_name': 'Song - 2011 Remaster', 'artist_name': 'Artist',
                               'duration_ms': 180000}, {'title': 'Artist - Song (Lyrics)', 'duration': duration})
    assert score is None


def test_missing_guest_credit_requires_close_duration_and_audit_note():
    score, reason = fe.match_result({'track_name': 'Song', 'artist_name': 'Main, Guest', 'duration_ms': 180000},
                                    {'title': 'Main - Song (Lyrics)', 'duration': 181})
    assert score is not None
    assert reason == 'matched_primary_artist_duration'


@pytest.mark.parametrize('duration', [None, 196])
def test_missing_guest_credit_still_rejected_without_close_duration(duration):
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Main, Guest', 'duration_ms': 180000},
                               {'title': 'Main - Song (Lyrics)', 'duration': duration})
    assert score is None


def test_lyric_wrapper_cannot_hide_different_song_title():
    score, _ = fe.match_result({'track_name': 'Song', 'artist_name': 'Artist'},
                               {'title': 'Artist - Another Song (Unofficial Lyrics)'})
    assert score is None


def test_download_remix_metadata_prevents_cache_publication(tmp_path):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if not download:
                return {'entries': [{'title': 'Artist - Song (Lyrics)', 'url': 'https://example.org/song'}]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'wrong')
            return {'title': 'Artist - Song (Lyrics)', 'track': 'Song (Remix)'}

    with patch('yt_dlp.YoutubeDL', Downloader), pytest.raises(RuntimeError, match='resolved_recording_mismatch: remix'):
        fe.acquire_audio({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Artist'}, tmp_path)
    assert not (tmp_path / 'track.wav').exists()


def test_sidecar_records_relaxed_identity_evidence(tmp_path):
    import json

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            entry = {'title': 'Artist - Song (Lyrics)', 'duration': 181, 'url': 'https://example.org/song'}
            if not download:
                return {'entries': [entry]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return entry

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, _ = fe.acquire_audio({'track_id': 'track', 'track_name': 'Song - 2011 Remaster',
                                     'artist_name': 'Artist', 'duration_ms': 180000}, tmp_path)
    metadata = json.loads(audio.with_suffix('.source.json').read_text(encoding='utf-8'))
    assert metadata['match_evidence'] == 'matched_unverified_remaster'


def test_requested_remix_can_use_track_id_cached_audio_without_sidecar(tmp_path):
    (tmp_path / 'track.wav').write_bytes(b'fixture')
    prepared, error, _ = fe.prepare_track({'track_id': 'track', 'track_name': 'Song (Remix)'},
                                         tmp_path, local_only=True, dry_run=False)
    assert prepared is not None
    assert error is None


def test_cached_audio_labelled_remix_is_ignored(tmp_path):
    import json
    (tmp_path / 'track.wav').write_bytes(b'fixture')
    (tmp_path / 'track.source.json').write_text(json.dumps({'title': 'Artist - Song (Remix)',
                                                         'url': 'https://example.org/remix'}), encoding='utf-8')
    prepared, error, _ = fe.prepare_track({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Artist'},
                                         tmp_path, local_only=True, dry_run=False)
    assert prepared is None
    assert error == 'audio_not_found'
    assert (tmp_path / 'track.wav').read_bytes() == b'fixture'


def test_obsolete_remix_exclusions_are_eligible_again_even_above_attempt_limit(tmp_path):
    import pandas as pd
    master, backlog = tmp_path / 'master.csv', tmp_path / 'backlog.csv'
    pd.DataFrame([{'id': 'existing'}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': 'remix', 'attempts': 4, 'last_error': 'excluded_remix'},
                  {'track_id': 'ordinary', 'attempts': 0, 'last_error': ''}]).to_csv(backlog, index=False)
    stats = fe.process(backlog, master, tmp_path / 'audio', local_only=True)
    assert stats['eligible'] == 2
    rows = pd.read_csv(backlog).set_index('track_id')
    assert rows.loc['remix', 'attempts'] == 5


def test_final_lyric_search_drops_missing_remaster_words(tmp_path):
    queries = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            entry = {'title': 'Artist - Song (Unofficial Lyrics)', 'duration': 181,
                     'url': 'https://example.org/song'}
            if not download:
                queries.append(url)
                return {'entries': [entry] if url == 'ytsearch15:Artist song lyrics' else []}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return entry

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, _ = fe.acquire_audio({'track_id': 'track', 'track_name': 'Song - 2011 Remaster',
                                     'artist_name': 'Artist', 'duration_ms': 180000}, tmp_path)
    assert audio.read_bytes() == b'fixture'
    assert queries[-1] == 'ytsearch15:Artist song lyrics'


@pytest.mark.parametrize('block_audio_replace', [False, True])
def test_online_replacement_preserves_cached_remix_and_safe_provenance(tmp_path, block_audio_replace):
    import json
    audio = tmp_path / 'track.wav'
    source = tmp_path / 'track.source.json'
    audio.write_bytes(b'old remix')
    source.write_text(json.dumps({'title': 'Artist - Song (Remix)', 'url': 'https://example.org/remix'}),
                      encoding='utf-8')
    original_source = source.read_bytes()
    original_replace = fe.os.replace

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            if not download:
                return {'entries': [{'title': 'Artist - Song (Lyrics)', 'url': 'https://example.org/original'}]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'new original')
            return {'title': 'Artist - Song (Lyrics)', 'webpage_url': url}

    def replace(src, dst):
        if block_audio_replace and Path(dst) == audio:
            raise PermissionError('synthetic audio lock')
        return original_replace(src, dst)

    with patch('yt_dlp.YoutubeDL', Downloader), patch.object(fe.os, 'replace', side_effect=replace):
        prepared, error, _ = fe.prepare_track({'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Artist'},
                                             tmp_path, local_only=False, dry_run=False)
    if block_audio_replace:
        assert prepared is None
        assert audio.read_bytes() == b'old remix'
        assert source.read_bytes() == original_source
    else:
        assert error is None
        assert audio.read_bytes() == b'new original'
        preserved = next(tmp_path.glob('.track_previous_*'))
        assert (preserved / 'track.wav').read_bytes() == b'old remix'
        assert (preserved / 'track.source.json').read_bytes() == original_source
