"""Backlog-only matching metadata is recovered without touching a live fetch job."""
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
    path = tmp_path / 'data' / 'dsp_backlog.csv'
    path.write_text(f'track_id,track_name,artist_name,duration_ms,attempts,last_error,label\n'
                    f'{A},Song (feat. Guest),Main,,4,no_matching_recording,0007\n'
                    f'{B},,,,2,download_failed,NA\n'
                    f'{C},Keep,Keep Artist,150000,3,no_search_results,\n')
    return path


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index('track_id')


def test_history_fills_backlog_only_and_improves_audio_match(tmp_path, monkeypatch):
    from scripts import feature_extraction as fe
    path = setup(tmp_path)
    history = tmp_path / 'sensitive_info' / 'listening_history.csv'
    history.write_text('track_id,track_name,artist_name,duration_ms,event_source,ms_played\n'
                       f'{A},Song (feat. Guest),Main,180000,live_polling,1000\n'
                       f'{B},Other,Other Artist,120000,extended_history,120000\n'
                       f'{C},Overwrite,Wrong,160000,live_polling,1000\n')
    checkpoint = tmp_path / 'data' / 'active.spotify-metadata.json'
    checkpoint.write_text('incomplete active checkpoint; must not even be opened')
    def forbidden(*args, **kwargs):
        pytest.fail('Offline backlog action must not authenticate')
    monkeypatch.setattr(dm, 'get_spotify_client', forbidden)
    sources_before = history.read_bytes(), checkpoint.read_bytes()
    entry = {'title': 'Main - Song (Official Audio)', 'uploader': 'Main', 'duration': 180}
    item = read(path).loc[A].to_dict()
    assert fe.match_result(item, entry)[0] is None
    stats = dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    rows = read(path)
    assert rows.loc[A, 'duration_ms'] == '180000'
    assert fe.match_result(rows.loc[A].to_dict(), entry)[0] is not None
    assert rows.loc[B, 'track_name'] == 'Other'
    assert rows.loc[B, 'duration_ms'] == ''
    assert rows.loc[B, 'history_duration_candidate_ms'] == '120000'
    assert rows.loc[C, 'track_name'] == 'Keep'
    assert rows.loc[C, 'duration_ms'] == '150000'
    assert list(rows['attempts']) == ['4', '2', '3']
    assert list(rows['last_error']) == ['no_matching_recording', 'download_failed', 'no_search_results']
    assert list(rows['label']) == ['0007', 'NA', '']
    assert rows.loc[A, 'duration_ms_metadata_source'].startswith('history_csv:')
    assert (history.read_bytes(), checkpoint.read_bytes()) == sources_before
    assert list(rows.index) == [A, B, C]
    assert stats['updated_cells'] > 0
    before = path.read_bytes()
    assert dm.backfill_local_backlog_metadata(path, source_root=tmp_path)['updated_cells'] == 0
    assert path.read_bytes() == before


def test_master_and_raw_history_supply_metadata_and_full_artist_list(tmp_path):
    path = setup(tmp_path)
    master = tmp_path / 'data' / 'master_track_table.csv'
    master.write_text('id,name,artists,duration_ms,duration_ms_source,spotify_artist_names\n'
                      f'{A},Song (feat. Guest),Main,180000,spotify_track_api,"[""Main"",""Guest""]"\n'
                      f'{B},Recovered,Artist,123000,,\n')
    raw = tmp_path / 'sensitive_info' / 'history' / 'Streaming_History_Audio.json'
    raw.write_text(json.dumps([{'spotify_track_uri': f'spotify:track:{B}',
                               'master_metadata_track_name': 'Recovered',
                               'master_metadata_album_artist_name': 'Artist',
                               'master_metadata_album_album_name': 'Album', 'ms_played': 999999}]))
    before = master.read_bytes(), raw.read_bytes()
    dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    rows = read(path)
    assert json.loads(rows.loc[A, 'spotify_artist_names']) == ['Main', 'Guest']
    assert rows.loc[A, 'duration_ms'] == '180000'
    assert rows.loc[B, 'album_name'] == 'Album'
    assert rows.loc[B, 'duration_ms'] == ''
    assert rows.loc[B, 'history_duration_candidate_ms'] == '123000'
    assert before == (master.read_bytes(), raw.read_bytes())


def test_checkpoint_reuse_requires_explicit_opt_in(tmp_path):
    path = setup(tmp_path)
    checkpoint = tmp_path / 'data' / 'saved.spotify-metadata.json'
    checkpoint.write_text(json.dumps({'version': 1, 'fields': ['duration_ms'],
                                      'records': {A: {'duration_ms': 180000}}}))
    dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    assert 'duration_ms' in read(path) and read(path).loc[A, 'duration_ms'] == ''
    before = checkpoint.read_bytes()
    dm.backfill_local_backlog_metadata(path, source_root=tmp_path, include_checkpoints=True)
    assert read(path).loc[A, 'duration_ms'] == '180000'
    assert checkpoint.read_bytes() == before


def test_history_retains_actual_structured_artist_credits(tmp_path):
    path = setup(tmp_path)
    history = tmp_path / 'sensitive_info' / 'listening_history.csv'
    pd.DataFrame([{'track_id': A, 'spotify_track_name': 'Song (feat. Guest)',
                   'spotify_artist_names': '["Main, Incorporated", "Guest"]'},
                  {'track_id': B, 'spotify_artist_names': 'invalid JSON'}]).to_csv(history, index=False)
    dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    rows = read(path)
    assert json.loads(rows.loc[A, 'spotify_artist_names']) == ['Main, Incorporated', 'Guest']
    assert rows.loc[A, 'artist_name'] == 'Main'
    assert rows.loc[B, 'spotify_artist_names'] == ''


def test_conflicting_confirmed_durations_are_not_guessed(tmp_path):
    path = setup(tmp_path)
    (tmp_path / 'sensitive_info' / 'history.csv').write_text(
        f'track_id,duration_ms,event_source\n{A},180000,live_polling\n{A},185000,live_polling\n')
    result = dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    assert read(path).loc[A, 'duration_ms'] == ''
    assert result['conflicts']['duration_ms'] == 1


def test_backlog_dry_run_and_external_edit_guard(tmp_path, monkeypatch):
    from scripts import local_backfill as lb
    path = setup(tmp_path)
    before = path.read_bytes()
    dm.backfill_local_backlog_metadata(path, source_root=tmp_path, dry_run=True)
    assert path.read_bytes() == before
    original = lb.fill_aliases
    def external(df, counts, *args, **kwargs):
        original(df, counts, *args, **kwargs)
        path.write_text(f'track_id,track_name\n{A},External update\n')
    monkeypatch.setattr(lb, 'fill_aliases', external)
    with pytest.raises(RuntimeError, match='changed'):
        dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    assert read(path).loc[A, 'track_name'] == 'External update'


def test_backlog_cli_uses_selected_file_offline(tmp_path):
    path = setup(tmp_path)
    args = [sys.executable, str(Path(dm.__file__)), '--backfill-backlog-local',
            '--backlog', str(path), '--local-source-root', str(tmp_path)]
    before = path.read_bytes()
    preview = subprocess.run([*args, '--dry-run'], cwd=tmp_path, capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert path.read_bytes() == before
    applied = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert applied.returncode == 0, applied.stderr
    assert 'track_id' in read(path).reset_index()


@pytest.mark.parametrize('ids', [f'{A}\n{A}', '""'])
def test_invalid_backlog_ids_abort_without_writing(tmp_path, ids):
    path = setup(tmp_path)
    path.write_text('track_id\n' + ids + '\n')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        dm.backfill_local_backlog_metadata(path, source_root=tmp_path)
    assert path.read_bytes() == before
