"""Unified offline backfill uses synthetic sources, never private datasets."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from scripts import data_merging as dm

A, B, C = 'A' * 22, 'B' * 22, 'C' * 22


def setup(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'sensitive_info' / 'history').mkdir(parents=True)
    path = tmp_path / 'data' / 'master_track_table.csv'
    return path


def rows(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index('id')


def test_aliases_dates_are_offline_missing_only_and_idempotent(tmp_path, monkeypatch):
    path = setup(tmp_path)
    path.write_text(f'id,name,artists,album,year,release_date,tempo,tempo_source,label,country\n'
                    f'{B},Title,Artist,Album,,2001-04,-1,local,0007,NA\n'
                    f'{A},Other,Other Artist,Other Album,1990,1990-01-02,120,spotify,,\n')
    def forbidden(*args, **kwargs):
        pytest.fail('Local backfill must not authenticate or call Spotify')
    monkeypatch.setattr(dm, 'get_spotify_client', forbidden)
    result = dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[B, 'year'] == '2001'
    assert saved.loc[B, 'release_date_precision'] == 'month'
    assert saved.loc[A, 'release_date_precision'] == 'day'
    assert saved.loc[B, 'artist_name'] == 'Artist'
    assert saved.loc[B, 'artist_name_metadata_source'] == 'master:artists'
    assert list(saved.index) == [B, A]
    assert saved.loc[B, 'tempo'] == '-1'
    assert saved.loc[B, 'tempo_source'] == 'local'
    assert saved.loc[B, 'label'] == '0007'
    assert saved.loc[B, 'country'] == 'NA'
    assert result['updated_cells'] > 0
    before = path.read_bytes()
    assert dm.backfill_local_metadata(path, source_root=tmp_path)['updated_cells'] == 0
    assert path.read_bytes() == before


def test_history_and_raw_export_conflicts_and_duration_provenance(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,name,track_name,artists,artist_name,album,duration_ms,explicit\n'
                    f'{A},,,,,,,\n{B},,,,,,,\n{C},,,,,,,\n')
    private = tmp_path / 'sensitive_info'
    (private / 'spotify_listening_history.csv').write_text(
        'track_id,track_name,artist,event_source,duration_ms,ms_played\n'
        f'{A},One,Artist,polling,120000,1000\n'
        f'{B},Two,Artist,extended_history,130000,130000\n'
        f'{C},Conflicting,Artist,polling,140000,1000\n')
    (private / 'spotify_listening_history - Copy.csv').write_text(
        f'track_id,track_name,duration_ms\n{C},Different,150000\n')
    raw = private / 'history' / 'Streaming_History_Audio_test.json'
    raw.write_text(json.dumps([
        {'spotify_track_uri': f'spotify:track:{A}', 'master_metadata_album_album_name': 'Album',
         'master_metadata_track_name': 'One', 'ts': '2024-01-01', 'ms_played': 12345},
        {'spotify_track_uri': f'spotify:track:{B}', 'master_metadata_album_album_name': 'Album Two'},
    ]))
    result = dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'track_name'] == 'One'
    assert saved.loc[A, 'album_name'] == 'Album'
    assert saved.loc[A, 'duration_ms'] == '120000'
    assert saved.loc[B, 'duration_ms'] == ''
    assert saved.loc[B, 'history_duration_candidate_ms'] == '130000'
    assert saved.loc[C, 'track_name'] == ''
    assert 'year' not in saved or saved['year'].eq('').all()
    assert saved['explicit'].eq('').all()
    assert result['conflicts']['track_name'] >= 1


def test_saved_spotify_results_and_backlog_are_reused_without_operational_fields(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,name,duration_ms,tempo\n{A},Keep,,111\n{B},,,\n')
    checkpoint = tmp_path / 'data' / 'basic.spotify-metadata.json'
    checkpoint.write_text(json.dumps({'version': 1, 'fields': ['name','explicit','duration_ms','year'],
        'records': {A: {'name': 'Overwrite', 'explicit': False, 'duration_ms': 123000, 'year': 1999,
                        'tempo': 999, 'attempts': 8}, 'unknown': {'name': 'Do not insert'}}}))
    backlog = tmp_path / 'data' / 'dsp_backlog.csv'
    backlog.write_text('track_id,spotify_track_name,spotify_artist_names,duration_ms,duration_ms_source,isrc,spotify_metadata_source,spotify_resolved_id,spotify_metadata_error,attempts\n'
                       f'{B},Fetched Song,"[""Artist""]",150000,spotify_track_api,TEST123,spotify_track_api,{B},,3\n')
    original = checkpoint.read_bytes(), backlog.read_bytes()
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'name'] == 'Keep'
    assert saved.loc[A, 'duration_ms'] == '123000'
    assert saved.loc[A, 'explicit'] == 'False'
    assert saved.loc[A, 'tempo'] == '111'
    assert saved.loc[B, 'track_name'] == 'Fetched Song'
    assert saved.loc[B, 'duration_ms'] == '150000'
    assert 'unknown' not in saved.index
    assert 'attempts' not in saved
    assert original == (checkpoint.read_bytes(), backlog.read_bytes())


def test_dry_run_has_no_writes_and_detects_master_change(tmp_path, monkeypatch):
    path = setup(tmp_path)
    path.write_text(f'id,name\n{A},Title\n')
    before = path.read_bytes()
    result = dm.backfill_local_metadata(path, source_root=tmp_path, dry_run=True)
    assert result['updated_cells'] > 0
    assert path.read_bytes() == before
    from scripts import local_backfill as lb
    original = lb.fill_aliases
    def concurrent_write(df, counts):
        original(df, counts)
        path.write_text(f'id,name\n{A},Changed elsewhere\n')
    monkeypatch.setattr(lb, 'fill_aliases', concurrent_write)
    with pytest.raises(RuntimeError, match='changed'):
        dm.backfill_local_metadata(path, source_root=tmp_path)
    assert rows(path).loc[A, 'name'] == 'Changed elsewhere'


def test_cli_is_one_offline_command_for_selected_master(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,name\n{A},Title\n')
    args = [sys.executable, str(Path(dm.__file__).resolve()), '--backfill-local',
            '--master', str(path), '--local-source-root', str(tmp_path)]
    before = path.read_bytes()
    preview = subprocess.run([*args, '--dry-run'], capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert path.read_bytes() == before
    applied = subprocess.run(args, capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    assert rows(path).loc[A, 'track_name'] == 'Title'


@pytest.mark.usefixtures('metadata_clock')
@pytest.mark.parametrize('old_duration', ['', '180000'])
def test_api_duration_provenance_is_field_specific(tmp_path, old_duration):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms\n{A},\n')
    backlog = tmp_path / 'data' / 'dsp_backlog.csv'
    backlog.write_text(f'track_id,duration_ms\n{A},{old_duration}\n')
    class Spotify:
        def track(self, tid):
            return {'id': tid, 'name': 'Song', 'artists': [{'name': 'Artist'}], 'duration_ms': 123000}
    dm.backfill_backlog_metadata(backlog, sp=Spotify())
    saved_backlog = pd.read_csv(backlog, dtype=str, keep_default_na=False).iloc[0]
    if not old_duration:
        assert saved_backlog['duration_ms_source'] == 'spotify_track_api'
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    if old_duration:
        assert saved.loc[A, 'duration_ms'] == ''
        assert saved.loc[A, 'history_duration_candidate_ms'] == '180000'
    else:
        assert saved.loc[A, 'duration_ms'] == '123000'


def test_invalid_or_noncalendar_dates_do_not_create_year_or_precision(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,release_date\n{A},2020-W01-1\n{B},2020-02-30\n{C},2020-13\n')
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert 'year' not in saved or saved['year'].eq('').all()
    assert 'release_date_precision' not in saved or saved['release_date_precision'].eq('').all()
