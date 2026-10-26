"""Placeholder migrations use synthetic catalogs, never the real master table."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import data_merging as dm


@pytest.mark.parametrize('column', [
    'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'key', 'mode', 'time_signature',
    'duration_ms', 'track_number', 'disc_number', 'year', 'release_year',
    'popularity', 'stream_count', 'explicit', 'local_onset_strength',
    'local_spectral_centroid', 'local_spectral_contrast', 'local_beat_regularity',
])
def test_only_exact_minus_one_is_cleared_in_nonnegative_features(column):
    original = pd.DataFrame({column: [-1, '-1.0', ' -1.000 ', 0, 1, -0.999, -1.5, np.nan, 'unknown']})
    before = original.copy(deep=True)
    cleaned, counts = dm.clear_feature_placeholders(original)
    assert cleaned[column].iloc[:3].isna().all()
    pd.testing.assert_series_equal(cleaned[column].iloc[3:], original[column].iloc[3:])
    pd.testing.assert_frame_equal(original, before)
    assert counts == {column: 3}


def test_signed_measurements_and_unknown_columns_are_preserved():
    original = pd.DataFrame({
        'id': ['track2', 'track1'], 'tempo': [-1, 0], 'key': [0, 11],
        'mode': [0, 1], 'loudness': [-1., -12.], 'local_rms_db': [-1., -18.],
        'label': ['-1', '0007'], 'custom_feature': [-1, 5],
        'tempo_source': ['legacy', 'spotify'],
    })
    cleaned, counts = dm.clear_feature_placeholders(original)
    assert counts == {'tempo': 1}
    assert pd.isna(cleaned.loc[0, 'tempo'])
    pd.testing.assert_frame_equal(cleaned.drop(columns='tempo'), original.drop(columns='tempo'))
    assert cleaned.loc[1, 'tempo'] == 0


def test_existing_general_cleaner_does_not_erase_valid_loudness():
    cleaned = dm.clean_track_features(pd.DataFrame({'tempo': [-1.], 'loudness': [-1.]}))
    assert pd.isna(cleaned.loc[0, 'tempo'])
    assert cleaned.loc[0, 'loudness'] == -1.


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,name,label,tempo,key,loudness,local_rms_db,tempo_source\n'
                    '002,NA,0007,-1,-1,-1,-1,legacy\n'
                    '001,-1,-1,0,0,-12,-20,spotify\n', encoding='utf-8')
    return path


def test_dry_run_reports_changes_without_writing(catalog, capsys):
    before = catalog.read_bytes()
    assert dm.clear_placeholders(catalog, dry_run=True) == {'tempo': 1, 'key': 1}
    assert catalog.read_bytes() == before
    assert 'DRY RUN' in capsys.readouterr().out


def test_cleanup_preserves_metadata_and_is_idempotent(catalog):
    assert dm.clear_placeholders(catalog) == {'tempo': 1, 'key': 1}
    row = pd.read_csv(catalog, dtype=str, keep_default_na=False).iloc[0]
    assert row['id'] == '002'
    assert row['name'] == 'NA'
    assert row['label'] == '0007'
    assert row['tempo'] == ''
    assert row['key'] == ''
    assert row['loudness'] == '-1'
    assert row['local_rms_db'] == '-1'
    assert row['tempo_source'] == 'legacy'
    before = catalog.read_bytes()
    assert dm.clear_placeholders(catalog) == {}
    assert catalog.read_bytes() == before


def test_cleanup_does_not_remove_or_normalize_ids(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,tempo\nrepeat,-1\nrepeat,120\n,-1\n', encoding='utf-8')
    dm.clear_placeholders(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(df['id']) == ['repeat', 'repeat', '']
    assert list(df['tempo']) == ['', '120', '']


def test_failed_write_preserves_original_catalog(catalog, monkeypatch):
    before = catalog.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail_write)
    with pytest.raises(OSError, match='disk full'):
        dm.clear_placeholders(catalog)
    assert catalog.read_bytes() == before


def test_cli_targets_only_the_selected_test_catalog(catalog):
    script = Path(dm.__file__).resolve()
    before = catalog.read_bytes()
    args = [sys.executable, str(script), '--clear-placeholders', '--master', str(catalog)]
    dry = subprocess.run([*args, '--dry-run'], capture_output=True, text=True)
    assert dry.returncode == 0, dry.stderr
    assert catalog.read_bytes() == before
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert pd.read_csv(catalog)['tempo'].isna().iloc[0]
