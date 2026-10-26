import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts import feature_extraction as fe


@pytest.fixture
def catalog(tmp_path):
    master = tmp_path / 'master.csv'
    backlog = tmp_path / 'backlog.csv'
    pd.DataFrame([{'id': 'song1', 'name': 'Example Song', 'artists': "['Example Artist']",
                   'duration_ms': 180000, 'tempo': 123., 'album': 'Keep me'}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': 'song1', 'attempts': 0}]).to_csv(backlog, index=False)
    return master, backlog, tmp_path / 'audio'


def test_process_acquires_audio_using_master_metadata_and_preserves_values(catalog):
    master, backlog, audio_dir = catalog

    def acquire(item, directory):
        assert item['track_name'] == 'Example Song'
        assert item['artist_name'] == "['Example Artist']"
        assert float(item['duration_ms']) == 180000
        directory.mkdir()
        audio = directory / 'song1.wav'
        audio.write_bytes(b'fixture')
        return audio, 'https://www.youtube.com/watch?v=example'

    with patch.object(fe, 'acquire_audio', side_effect=acquire, create=True), patch.object(
        fe, 'extract_audio_features', return_value={'tempo': 100., 'local_rms_db': -12.}
    ):
        stats = fe.process(backlog, master, audio_dir)
    assert stats == {'eligible': 1, 'saved': 1, 'failed': 0}
    saved = pd.read_csv(master).iloc[0]
    assert saved['tempo'] == 123.
    assert saved['album'] == 'Keep me'
    assert saved['local_rms_db'] == -12.
    assert saved['local_rms_db_source'] == fe.EXTRACTOR_VERSION
    assert saved['local_rms_db_audio_source'] == 'https://www.youtube.com/watch?v=example'
    assert pd.read_csv(backlog).empty


def test_dry_run_never_downloads_or_changes_files(catalog):
    master, backlog, audio_dir = catalog
    before = (master.read_bytes(), backlog.read_bytes())
    with patch.object(fe, 'acquire_audio', side_effect=AssertionError('network'), create=True):
        stats = fe.process(backlog, master, audio_dir, dry_run=True)
    assert stats['saved'] == 0
    assert (master.read_bytes(), backlog.read_bytes()) == before
    assert not audio_dir.exists()


def test_failed_download_retains_backlog_and_records_reason(catalog):
    master, backlog, audio_dir = catalog
    before = master.read_bytes()
    with patch.object(fe, 'acquire_audio', side_effect=RuntimeError('download_failed'), create=True):
        stats = fe.process(backlog, master, audio_dir)
    assert stats['failed'] == 1
    assert master.read_bytes() == before
    row = pd.read_csv(backlog).iloc[0]
    assert row['attempts'] == 1
    assert 'download_failed' in row['last_error']


@pytest.mark.parametrize('title,artist,duration,expected', [
    ('Example Artist - Example Song (Official Audio)', 'Example Artist - Topic', 181, True),
    ('Example Song', 'Example Artist - Topic', 180, True),
    ('Example Artist - Example Song (Live)', 'Example Artist', 180, False),
    ('Example Artist - Example Song (Cover)', 'Example Artist', 180, False),
    ('Example Artist - Example Song (Sped Up)', 'Example Artist', 180, False),
    ('Example Artist - Example Song', 'Example Artist', 600, False),
    ('Other Artist - Example Song', 'Other Artist', 180, False),
    ('Example Artist - Another Song', 'Example Artist', 180, False),
])
def test_recording_match_rejects_wrong_artist_version_or_duration(title, artist, duration, expected):
    item = {'track_name': 'Example Song', 'artist_name': "['Example Artist']", 'duration_ms': 180000}
    entry = {'title': title, 'uploader': artist, 'duration': duration}
    assert (fe.match_score(item, entry) is not None) is expected


def test_search_downloads_matching_candidate_and_caches_source(tmp_path):
    item = {'track_id': 'song1', 'track_name': 'Example Song', 'artist_name': 'Example Artist'}

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
                    {'title': 'Example Artist - Example Song (Live)', 'url': 'https://example.org/wrong'},
                    {'title': 'Example Artist - Example Song', 'url': 'https://example.org/right'},
                ]}
            assert url == 'https://example.org/right'
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return {'webpage_url': url, 'title': 'Example Artist - Example Song'}

    with patch('yt_dlp.YoutubeDL', Downloader):
        path, source = fe.acquire_audio(item, tmp_path)
    assert path == tmp_path / 'song1.wav'
    assert source == 'https://example.org/right'
    assert json.loads((tmp_path / 'song1.source.json').read_text())['url'] == source


def test_last_error_column_accepts_text_after_empty_csv_roundtrip(catalog):
    master, backlog, audio_dir = catalog
    pd.DataFrame([{'track_id': 'song1', 'attempts': 0, 'last_error': np.nan}]).to_csv(backlog, index=False)
    with patch.object(fe, 'acquire_audio', return_value=(None, None), create=True):
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            fe.process(backlog, master, audio_dir)
    assert pd.read_csv(backlog).iloc[0]['last_error'] == 'audio_not_found'


def test_real_signal_extracts_finite_local_measurements(tmp_path):
    sr = 22050
    t = np.arange(sr * 12) / sr
    signal = .02 * np.sin(2 * np.pi * 440 * t)
    signal += fe.librosa.clicks(times=np.arange(.25, 12, .5), sr=sr, length=len(t))
    path = tmp_path / 'signal.wav'
    sf.write(path, signal, sr)
    features = fe.extract_audio_features(path)
    assert features is not None
    for col in fe.LOCAL_COLUMNS:
        assert np.isfinite(features[col]), col
    assert features['tempo'] == pytest.approx(120, abs=5)
    assert 'energy' not in features


@pytest.mark.parametrize('signal', [np.zeros(22050), np.ones(100)])
def test_silence_and_short_audio_are_rejected(tmp_path, signal):
    path = tmp_path / 'invalid.wav'
    sf.write(path, signal, 22050)
    assert fe.extract_audio_features(path) is None


def test_master_write_failure_does_not_drop_backlog(catalog):
    master, backlog, audio_dir = catalog
    audio_dir.mkdir()
    (audio_dir / 'song1.wav').write_bytes(b'fixture')
    before = backlog.read_bytes()
    with patch.object(fe, 'extract_audio_features', return_value={'tempo': 100.}), patch.object(
        fe, 'save_batch', side_effect=OSError('disk full')
    ), pytest.raises(OSError, match='disk full'):
        fe.process(backlog, master, audio_dir)
    assert backlog.read_bytes() == before


def test_blank_existing_provenance_accepts_extractor_name(catalog):
    master, _, _ = catalog
    df = pd.read_csv(master)
    df['local_rms_db'] = np.nan
    df['local_rms_db_source'] = np.nan
    df['local_rms_db_audio_source'] = np.nan
    df.to_csv(master, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        fe.save_batch([{'track_id': 'song1', 'local_rms_db': -12., 'audio_source': 'fixture.wav'}], master)
    row = pd.read_csv(master).iloc[0]
    assert row['local_rms_db_source'] == fe.EXTRACTOR_VERSION
    assert row['local_rms_db_audio_source'] == 'fixture.wav'


def test_new_track_preserves_backlog_metadata(catalog):
    master, _, _ = catalog
    fe.save_batch([{'track_id': 'new_song', 'track_name': 'New Song', 'artist_name': 'New Artist',
                   'album': 'New Album', 'duration_ms': 200000, 'release_date': '2020-01-01',
                   'attempts': 2, 'last_error': '', 'local_rms_db': -12.}], master)
    row = pd.read_csv(master).set_index('id').loc['new_song']
    assert row['name'] == 'New Song'
    assert row['album'] == 'New Album'
    assert row['duration_ms'] == 200000
    assert row['release_date'] == '2020-01-01'
    assert 'attempts' not in row


def test_local_only_and_limit_preserve_unprocessed_items(catalog):
    master, backlog, audio_dir = catalog
    pd.DataFrame([{'track_id': 'song1'}, {'track_id': 'song2'}]).to_csv(backlog, index=False)
    with patch.object(fe, 'acquire_audio', side_effect=AssertionError('network')):
        stats = fe.process(backlog, master, audio_dir, local_only=True, limit=1)
    assert stats == {'eligible': 1, 'saved': 0, 'failed': 1}
    rows = pd.read_csv(backlog).set_index('track_id')
    assert rows.loc['song1', 'attempts'] == 1
    assert rows.loc['song2', 'attempts'] == 0


def test_exhausted_items_require_explicit_retry(catalog):
    master, backlog, audio_dir = catalog
    pd.DataFrame([{'track_id': 'song1', 'attempts': 3}]).to_csv(backlog, index=False)
    before = backlog.read_bytes()
    assert fe.process(backlog, master, audio_dir, local_only=True)['eligible'] == 0
    assert backlog.read_bytes() == before
    assert fe.process(backlog, master, audio_dir, local_only=True, retry_failed=True)['eligible'] == 1
    assert pd.read_csv(backlog).iloc[0]['attempts'] == 4


def test_download_enables_installed_node_runtime(tmp_path):
    item = {'track_id': 'song1', 'track_name': 'Example Song', 'artist_name': 'Example Artist'}
    with patch('yt_dlp.YoutubeDL') as downloader, patch.object(fe.shutil, 'which', return_value='executable'):
        downloader.return_value.__enter__.return_value.extract_info.return_value = {'entries': []}
        with pytest.raises(RuntimeError, match='no_search_results'):
            fe.acquire_audio(item, tmp_path)
        assert 'node' in downloader.call_args.args[0]['js_runtimes']


def test_invalid_ids_cannot_escape_audio_directory(tmp_path):
    with pytest.raises(ValueError, match='invalid_track_id'):
        fe.find_local_audio('../outside', tmp_path)


def test_metadata_text_is_not_coerced_when_rewriting_master(catalog):
    master, _, _ = catalog
    master.write_text('id,name,label,country,tempo\nsong1,Example,00123,NA,\n', encoding='utf-8')
    fe.save_batch([{'track_id': 'song1', 'tempo': 120.}], master)
    row = pd.read_csv(master, dtype=str, keep_default_na=False).iloc[0]
    assert row['label'] == '00123'
    assert row['country'] == 'NA'
    assert float(row['tempo']) == 120.


def test_backlog_metadata_survives_failure_update(catalog):
    master, backlog, audio_dir = catalog
    backlog.write_text('track_id,track_name,catalog_code\nsong1,NA,0007\n', encoding='utf-8')
    fe.process(backlog, master, audio_dir, local_only=True)
    row = pd.read_csv(backlog, dtype=str, keep_default_na=False).iloc[0]
    assert row['track_name'] == 'NA'
    assert row['catalog_code'] == '0007'


def test_numeric_nan_marker_can_be_filled_but_existing_sentinel_is_preserved(catalog):
    master, _, _ = catalog
    master.write_text('id,name,tempo,key\nsong1,Example,NaN,-1\n', encoding='utf-8')
    fe.save_batch([{'track_id': 'song1', 'tempo': 120., 'key': 4.}], master)
    row = pd.read_csv(master).iloc[0]
    assert row['tempo'] == 120.
    assert row['key'] == -1.


def test_failed_candidate_cannot_poison_next_candidate_cache(tmp_path):
    item = {'track_id': 'song1', 'track_name': 'Example Song', 'artist_name': 'Example Artist'}

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
                    {'title': 'Example Artist - Example Song', 'url': 'https://example.org/first'},
                    {'title': 'Example Artist - Example Song', 'url': 'https://example.org/second'},
                ]}
            output = Path(self.options['outtmpl'].replace('%(ext)s', 'wav'))
            if not output.exists():
                output.write_bytes(b'first' if url.endswith('first') else b'second')
            if url.endswith('first'):
                raise RuntimeError('conversion interrupted')
            return {'webpage_url': url, 'title': 'Example Artist - Example Song'}

    with patch('yt_dlp.YoutubeDL', Downloader):
        path, source = fe.acquire_audio(item, tmp_path)
    assert path.read_bytes() == b'second'
    assert source == 'https://example.org/second'
