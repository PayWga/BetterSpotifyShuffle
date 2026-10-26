"""Backlog enrichment must not require a master row or rewrite user data."""
import json
import pandas as pd
import pytest
from spotipy.exceptions import SpotifyException

from scripts import data_merging as dm

pytestmark = pytest.mark.usefixtures('metadata_clock')

A, B = 'A' * 22, 'B' * 22


class Spotify:
    def __init__(self):
        self.calls = []

    def track(self, tid):
        self.calls.append(tid)
        return {'id': tid, 'name': 'Song (with Guest)', 'duration_ms': 180000,
                'artists': [{'name': 'Main'}, {'name': 'Guest'}],
                'external_ids': {'isrc': 'TEST123'}}


def backlog(tmp_path):
    path = tmp_path / 'backlog.csv'
    path.write_text(f'track_id,track_name,artist_name,attempts,last_error,label,duration_ms\n'
                    f'{A},Keep title,Main,4,no_matching_recording,0007,\n'
                    f'{B},Other song,Other,3,no_search_results,NA,190000\n', encoding='utf-8')
    return path


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index('track_id')


def test_backlog_enrichment_preserves_identity_counters_and_nonmissing_values(tmp_path):
    path, sp = backlog(tmp_path), Spotify()
    stats = dm.backfill_backlog_metadata(path, sp=sp)
    rows = read(path)
    assert rows.loc[A, 'duration_ms'] == '180000'
    assert rows.loc[B, 'duration_ms'] == '190000'
    assert rows.loc[A, 'track_name'] == 'Keep title'
    assert rows.loc[A, 'artist_name'] == 'Main'
    assert rows.loc[A, 'attempts'] == '4'
    assert rows.loc[A, 'last_error'] == 'no_matching_recording'
    assert rows.loc[A, 'label'] == '0007'
    assert rows.loc[B, 'label'] == 'NA'
    assert json.loads(rows.loc[A, 'spotify_artist_names']) == ['Main', 'Guest']
    assert rows.loc[A, 'spotify_track_name'] == 'Song (with Guest)'
    assert rows.loc[A, 'isrc'] == 'TEST123'
    assert rows.loc[A, 'spotify_metadata_checked_at'].endswith('+00:00')
    assert stats['fetched'] == 2


def test_rerun_resumes_and_limit_counts_new_requests(tmp_path):
    path, sp = backlog(tmp_path), Spotify()
    dm.backfill_backlog_metadata(path, sp=sp, limit=1)
    dm.backfill_backlog_metadata(path, sp=sp)
    assert sp.calls == [A, B]
    before = path.read_bytes()
    dm.backfill_backlog_metadata(path, sp=sp)
    assert sp.calls == [A, B]
    assert path.read_bytes() == before


def test_dry_run_never_authenticates_or_writes(tmp_path, monkeypatch):
    path = backlog(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(dm, 'get_spotify_client', lambda: pytest.fail('authenticated'))
    assert dm.backfill_backlog_metadata(path, dry_run=True)['eligible'] == 2
    assert path.read_bytes() == before


def test_external_backlog_update_aborts_instead_of_overwriting(tmp_path):
    path = backlog(tmp_path)

    class Concurrent(Spotify):
        def track(self, tid):
            path.write_text('track_id,track_name\nexternal,Keep this update\n', encoding='utf-8')
            return super().track(tid)

    with pytest.raises(RuntimeError, match='changed'):
        dm.backfill_backlog_metadata(path, sp=Concurrent(), limit=1)
    assert 'Keep this update' in path.read_text()


def test_auth_failure_preserves_completed_metadata_and_attempts(tmp_path):
    path = backlog(tmp_path)

    class Unauthorized(Spotify):
        def track(self, tid):
            if tid == B:
                raise SpotifyException(401, -1, 'synthetic auth failure')
            return super().track(tid)

    with pytest.raises(SpotifyException):
        dm.backfill_backlog_metadata(path, sp=Unauthorized())
    assert read(path).loc[A, 'duration_ms'] == '180000'
    assert read(path).loc[A, 'attempts'] == '4'


def test_unrelated_spotify_response_does_not_fill_recording_metadata(tmp_path):
    path = backlog(tmp_path)

    class Wrong(Spotify):
        def track(self, tid):
            return super().track('C' * 22)

    dm.backfill_backlog_metadata(path, sp=Wrong(), limit=1)
    assert read(path).loc[A, 'duration_ms'] == ''


def test_backlog_cli_dry_run_uses_supplied_file_without_credentials(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    path = backlog(tmp_path)
    before = path.read_bytes()
    result = subprocess.run([sys.executable, str(Path(dm.__file__)), '--backfill-backlog',
                             '--backlog', str(path), '--dry-run', '--limit', '1'],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'DRY RUN' in result.stdout
    assert path.read_bytes() == before


@pytest.mark.parametrize('linked_id,accepted', [(A, True), (B, False)])
def test_relinked_response_requires_requested_original_id(tmp_path, linked_id, accepted):
    path = backlog(tmp_path)

    class Relinked(Spotify):
        def track(self, tid):
            track = super().track('C' * 22)
            track['linked_from'] = {'id': linked_id}
            return track

    dm.backfill_backlog_metadata(path, sp=Relinked(), limit=1)
    row = read(path).loc[A]
    assert (row['duration_ms'] == '180000') is accepted
    if accepted:
        assert row['spotify_resolved_id'] == 'C' * 22
    else:
        assert row['spotify_metadata_error'] == 'recording_identity_mismatch'


def test_periodic_checkpoint_is_visible_before_next_request(tmp_path, monkeypatch):
    path = tmp_path / 'backlog.csv'
    ids = [f'{number:022d}' for number in range(26)]
    path.write_text('track_id\n' + '\n'.join(ids) + '\n', encoding='utf-8')
    monkeypatch.setattr('scripts.spotify_metadata.time.sleep', lambda _: None)

    class InspectCheckpoint(Spotify):
        def track(self, tid):
            if tid == ids[25]:
                persisted = read(path)
                assert persisted.loc[ids[24], 'duration_ms'] == '180000'
                assert persisted.loc[tid, 'duration_ms'] == ''
            return super().track(tid)

    stats = dm.backfill_backlog_metadata(path, sp=InspectCheckpoint())
    assert stats['fetched'] == 26
    assert read(path).loc[ids[25], 'duration_ms'] == '180000'
