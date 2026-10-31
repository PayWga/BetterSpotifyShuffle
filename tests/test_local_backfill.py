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


def completed_plays(tid, durations):
    return [{'spotify_track_uri': f'spotify:track:{tid}',
             'ts': f'2024-01-01T00:{i:02d}:00Z',
             'reason_end': 'trackdone', 'ms_played': value}
            for i, value in enumerate(durations)]


def write_audio_history(tmp_path, events, name='test'):
    path = tmp_path / 'sensitive_info' / 'history' / f'Streaming_History_Audio_{name}.json'
    path.write_text(json.dumps(events))
    return path


def test_completed_play_estimate_reduces_duration_api_queue_and_is_idempotent(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms,explicit\n{A},,\n')
    raw = write_audio_history(tmp_path, completed_plays(A, [180000, 180000, 180100, 180000, 90000]))
    original = raw.read_bytes()
    before = dm.backfill_metadata(path, fields=['duration_ms'], fetch_only=True, dry_run=True)
    assert before['api_needed'] == 1
    preview = dm.backfill_local_metadata(path, source_root=tmp_path, dry_run=True)
    assert preview['columns'].get('duration_ms', 0) == 1
    assert rows(path).loc[A, 'duration_ms'] == ''
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'duration_ms'] == '180000'
    assert saved.loc[A, 'duration_ms_metadata_source'] == 'history_completed_play_median'
    assert saved.loc[A, 'duration_ms_is_estimated'] == 'True'
    assert saved.loc[A, 'duration_ms_estimate_observations'] == '5'
    assert saved.loc[A, 'duration_ms_estimate_support'] == '4'
    assert raw.read_bytes() == original
    assert dm.backfill_metadata(path, fields=['duration_ms'], fetch_only=True,
                                dry_run=True)['api_needed'] == 0
    assert dm.backfill_metadata(path, fields=['duration_ms', 'explicit'], fetch_only=True,
                                dry_run=True)['api_needed'] == 1
    snapshot = path.read_bytes()
    assert dm.backfill_local_metadata(path, source_root=tmp_path)['updated_cells'] == 0
    assert path.read_bytes() == snapshot


@pytest.mark.parametrize('durations', [
    [180000],
    [180000, 190000],
    [180000, 180000, 90000],
    [180000, 180000, 180000, 90000],
    [0, -1, None, 'NaN', True, 180000.5],
])
def test_insufficient_inconsistent_or_invalid_completed_plays_stay_missing(tmp_path, durations):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms\n{A},\n')
    write_audio_history(tmp_path, completed_plays(A, durations))
    dm.backfill_local_metadata(path, source_root=tmp_path)
    assert rows(path).loc[A, 'duration_ms'] == ''


@pytest.mark.parametrize('durations, expected', [
    ([180000, 180000], '180000'),
    ([180000, 180100], '180050'),
])
def test_two_agreeing_completed_plays_supply_labelled_median(tmp_path, durations, expected):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms\n{A},\n')
    write_audio_history(tmp_path, completed_plays(A, durations))
    result = dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'duration_ms'] == expected
    assert saved.loc[A, 'duration_ms_is_estimated'] == 'True'
    assert saved.loc[A, 'duration_ms_estimate_observations'] == '2'
    assert saved.loc[A, 'duration_ms_estimate_support'] == '2'
    assert result['estimated_durations'] == 1


def test_duplicate_exports_and_conflicting_event_copies_do_not_inflate_evidence(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms\n{A},\n{B},\n{C},\n')
    a = completed_plays(A, [180000])
    b = completed_plays(B, [180000, 180000])
    c = completed_plays(C, [180000, 180000])
    write_audio_history(tmp_path, a + b + c, 'one')
    conflicting = dict(b[0], ms_played=90000)
    invalid_timestamps = [dict(c[0], ts='bad'), dict(c[0], ts=''), dict(c[0], ts=None)]
    # Equivalent timezone notation is the same event, not another observation.
    equivalent = dict(a[0], ts='2024-01-01T00:00:00+00:00')
    write_audio_history(tmp_path, a + [equivalent, conflicting] + invalid_timestamps, 'two')
    # A timestamp is needed to deduplicate the listening evidence.
    write_audio_history(tmp_path, [dict(e, ts=None) for e in completed_plays(A, [180000] * 5)], 'three')
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'duration_ms'] == ''
    assert saved.loc[B, 'duration_ms'] == ''
    assert saved.loc[C, 'duration_ms'] == '180000'
    assert saved.loc[C, 'duration_ms_estimate_observations'] == '2'


def test_estimates_preserve_existing_values_and_defer_to_confirmed_cache(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms,duration_ms_metadata_source\n{A},190000,existing\n{B},,\n{C},,\n')
    write_audio_history(tmp_path, sum((completed_plays(tid, [180000] * 3)
                                     for tid in (A, B, C, 'unknown')), []))
    (tmp_path / 'data' / 'basic.spotify-metadata.json').write_text(json.dumps({
        'version': 1, 'fields': ['duration_ms'], 'records': {B: {'duration_ms': 200000}}}))
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'duration_ms'] == '190000'
    assert saved.loc[A, 'duration_ms_metadata_source'] == 'existing'
    assert saved.loc[B, 'duration_ms'] == '200000'
    assert saved.loc[B, 'duration_ms_metadata_source'] == 'spotify_cache:basic.spotify-metadata.json'
    assert saved.loc[B, 'duration_ms_is_estimated'] == ''
    assert saved.loc[C, 'duration_ms'] == '180000'
    assert len(saved) == 3


def test_skips_and_circular_history_percentages_cannot_supply_estimates(tmp_path):
    path = setup(tmp_path)
    path.write_text(f'id,duration_ms\n{A},\n')
    write_audio_history(tmp_path, [dict(e, reason_end='fwdbtn')
                                 for e in completed_plays(A, [180000] * 5)])
    (tmp_path / 'sensitive_info' / 'listening_history.csv').write_text(
        f'track_id,duration_ms,ms_played,percent_completed\n{A},210000,105000,0.5\n')
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'duration_ms'] == ''
    assert saved.loc[A, 'history_duration_candidate_ms'] == '210000'


def test_backlog_local_backfill_uses_same_labelled_duration_estimates(tmp_path):
    setup(tmp_path)
    path = tmp_path / 'data' / 'dsp_backlog.csv'
    path.write_text(f'track_id,duration_ms\n{A},\n')
    write_audio_history(tmp_path, completed_plays(A, [180000] * 2))
    dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    saved = pd.read_csv(path, dtype=str, keep_default_na=False).set_index('track_id')
    assert saved.loc[A, 'duration_ms'] == '180000'
    assert saved.loc[A, 'duration_ms_is_estimated'] == 'True'


def test_new_estimate_refreshes_orphaned_evidence_for_missing_duration(tmp_path):
    path = setup(tmp_path)
    path.write_text('id,duration_ms,duration_ms_is_estimated,duration_ms_estimate_observations,'
                    'duration_ms_estimate_support\n'
                    f'{A},,False,20,19\n{B},190000,False,20,19\n')
    write_audio_history(tmp_path, completed_plays(A, [180000] * 3)
                        + completed_plays(B, [180000] * 3))
    dm.backfill_local_metadata(path, source_root=tmp_path)
    saved = rows(path)
    assert saved.loc[A, 'duration_ms'] == '180000'
    assert saved.loc[A, 'duration_ms_is_estimated'] == 'True'
    assert saved.loc[A, 'duration_ms_estimate_observations'] == '3'
    assert saved.loc[A, 'duration_ms_estimate_support'] == '3'
    assert saved.loc[B, 'duration_ms'] == '190000'
    assert saved.loc[B, 'duration_ms_is_estimated'] == 'False'
    assert saved.loc[B, 'duration_ms_estimate_observations'] == '20'
