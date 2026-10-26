"""Spotify enrichment uses synthetic catalogs and an offline API boundary."""
import json

import pandas as pd
import pytest
from spotipy.exceptions import SpotifyException

from scripts import data_merging as dm

pytestmark = pytest.mark.usefixtures('metadata_clock')


A = 'A' * 22
B = 'B' * 22


class Spotify:
    def __init__(self):
        self.calls = []

    def track(self, tid):
        self.calls.append(tid)
        return {'id': tid, 'name': 'Fetched title', 'artists': [{'name': 'Artist', 'id': 'artist'}],
                'album': {'id': 'album', 'name': 'Album', 'release_date': '2020-03',
                          'release_date_precision': 'month'},
                'duration_ms': 123000, 'explicit': False, 'track_number': 1, 'disc_number': 1,
                'popularity': 0, 'external_ids': {'isrc': 'TEST123'}}


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text(f'id,name,artists,artist_name,year,tempo,tempo_source,label,country\n'
                    f'{A},Keep me,,,,-1,local,0007,NA\n'
                    f'{B},,,,,120,spotify,,\n', encoding='utf-8')
    return path


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).set_index('id')


def test_backfill_fills_only_gaps_and_preserves_csv_text(catalog, tmp_path):
    sp = Spotify()
    stats = dm.backfill_metadata(catalog, tmp_path / 'checkpoint.json', sp=sp)
    rows = read(catalog)
    assert rows.loc[A, 'name'] == 'Keep me'
    assert rows.loc[A, 'artist_name'] == 'Artist'
    assert rows.loc[A, 'year'] == '2020'
    assert rows.loc[A, 'tempo'] == '-1'
    assert rows.loc[A, 'tempo_source'] == 'local'
    assert rows.loc[A, 'label'] == '0007'
    assert rows.loc[A, 'country'] == 'NA'
    assert rows.loc[B, 'explicit'] == 'False'
    assert rows.loc[B, 'popularity'] == '0'
    assert stats['updated_cells'] > 0


def test_fetch_only_resumes_and_apply_reads_latest_catalog(catalog, tmp_path):
    checkpoint = tmp_path / 'checkpoint.json'
    before = catalog.read_bytes()
    sp = Spotify()
    dm.backfill_metadata(catalog, checkpoint, sp=sp, fetch_only=True, limit=1)
    assert catalog.read_bytes() == before
    assert len(sp.calls) == 1
    dm.backfill_metadata(catalog, checkpoint, sp=sp, fetch_only=True)
    assert sp.calls == [A, B]
    catalog.write_text(catalog.read_text().replace('-1,local', '125,new_extractor'), encoding='utf-8')
    dm.apply_metadata_checkpoint(catalog, checkpoint)
    assert read(catalog).loc[A, 'tempo'] == '125'
    before = catalog.read_bytes()
    assert dm.apply_metadata_checkpoint(catalog, checkpoint)['updated_cells'] == 0
    assert catalog.read_bytes() == before


def test_recent_listening_priority_uses_latest_utc_event_and_resumes(tmp_path):
    path = tmp_path / 'master.csv'
    c, d, e = 'C' * 22, 'D' * 22, 'E' * 22
    path.write_text('id,duration_ms\n' + ''.join(f'{tid},\n' for tid in [A, B, c, d, e]))
    history = tmp_path / 'history.csv'
    history.write_text('track_id,timestamp\n'
                       f'{A},2020-01-01T00:00:00Z\n'
                       f'{B},2026-01-02T00:30:00+03:00\n'
                       f'{A},2026-01-01T23:00:00Z\n'
                       f'{c},2026-01-01 21:30:00\n'
                       f'{d},invalid\nunknown,2030-01-01T00:00:00Z\n')
    original = path.read_bytes(), history.read_bytes()
    checkpoint = tmp_path / 'state.json'
    sp = Spotify()
    # Catalog order is unchanged without the flag.
    dm.backfill_metadata(path, checkpoint, fields=['duration_ms'], sp=sp,
                         fetch_only=True, limit=1)
    assert sp.calls == [A]
    dm.backfill_metadata(path, checkpoint, fields=['duration_ms'], sp=sp,
                         fetch_only=True, recent_first=True, history_path=history, limit=1)
    assert sp.calls == [A, B]
    dm.backfill_metadata(path, checkpoint, fields=['duration_ms'], sp=sp,
                         fetch_only=True, recent_first=True, history_path=history)
    # B and C have equal UTC times. D and E have no usable history.
    assert sp.calls == [A, B, c, d, e]
    assert original == (path.read_bytes(), history.read_bytes())


def test_recent_priority_puts_newest_ahead_of_catalog_order(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text(f'id,duration_ms\n{A},\n{B},\n')
    history = tmp_path / 'history.csv'
    history.write_text(f'track_id,timestamp\n{A},2024-01-01\n{B},2020-01-01\n'
                       f'{B},2026-01-01T00:00:00Z\n')
    sp = Spotify()
    dm.backfill_metadata(path, tmp_path / 'state.json', fields=['duration_ms'], sp=sp,
                         fetch_only=True, recent_first=True, history_path=history, limit=1)
    assert sp.calls == [B]


@pytest.mark.parametrize('contents', [None, 'track_id\n' + A + '\n', 'timestamp\n2026-01-01\n'])
def test_recent_priority_rejects_missing_or_malformed_history_before_requests(tmp_path, contents, monkeypatch):
    path = tmp_path / 'master.csv'
    path.write_text(f'id,duration_ms\n{A},\n')
    history = tmp_path / 'history.csv'
    if contents is not None:
        history.write_text(contents)
    checkpoint = tmp_path / 'state.json'
    def forbidden():
        pytest.fail('Invalid history must fail before authentication')
    monkeypatch.setattr(dm, 'get_spotify_client', forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        dm.backfill_metadata(path, checkpoint, fields=['duration_ms'], recent_first=True,
                             history_path=history, sp=None)
    assert not checkpoint.exists()


def test_cli_recent_priority_dry_run_uses_selected_history_without_writes(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    path = tmp_path / 'master.csv'
    path.write_text(f'id,duration_ms\n{A},\n')
    history = tmp_path / 'history.csv'
    history.write_text(f'track_id,timestamp\n{A},2026-01-01\n')
    checkpoint = tmp_path / 'state.json'
    before = path.read_bytes(), history.read_bytes()
    result = subprocess.run([sys.executable, str(Path(dm.__file__).resolve()), '--backfill',
                             '--master', str(path), '--checkpoint', str(checkpoint),
                             '--fields', 'duration_ms', '--fetch-only', '--recent-first',
                             '--history', str(history), '--dry-run'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'most recently listened' in result.stdout.lower()
    assert before == (path.read_bytes(), history.read_bytes())
    assert not checkpoint.exists()


def test_dry_run_has_no_auth_network_or_writes(catalog, tmp_path, monkeypatch):
    def forbidden():
        pytest.fail('dry run authenticated')
    monkeypatch.setattr(dm, 'get_spotify_client', forbidden)
    checkpoint = tmp_path / 'checkpoint.json'
    before = catalog.read_bytes()
    assert dm.backfill_metadata(catalog, checkpoint, dry_run=True)['eligible'] == 2
    assert not checkpoint.exists()
    assert catalog.read_bytes() == before


def test_local_aliases_and_year_are_filled_without_network(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text(f'id,name,artists,album,release_date\n{A},Title,Artist,Album,1999\n')
    sp = Spotify()
    dm.backfill_metadata(path, tmp_path / 'state.json', sp=sp,
                         fields=['artist_name', 'year', 'track_name', 'album_name'])
    assert sp.calls == []
    assert read(path).loc[A, 'year'] == '1999'


def test_fetch_only_queues_only_api_tracks_and_limit_counts_lookups(tmp_path, monkeypatch):
    from scripts import spotify_metadata as sm
    path, checkpoint = tmp_path / 'master.csv', tmp_path / 'state.json'
    path.write_text('id,name,artists,album,release_date,year,duration_ms,explicit\n'
                    f'{A},Title,Artist,Album,1999,,120000,False\n'
                    f'{B},Title,,,,,120000,False\n'
                    'invalid,Title,,,,,120000,False\n')
    before = path.read_bytes()
    def forbidden_iterrows(self):
        pytest.fail('Queue building must not iterate catalog rows in Python')
    monkeypatch.setattr(pd.DataFrame, 'iterrows', forbidden_iterrows)
    sp = Spotify()
    stats = dm.backfill_metadata(path, checkpoint, sp=sp, fetch_only=True, limit=1,
                                 fields=['track_name', 'artist_name', 'album_name', 'year'])
    assert stats['eligible'] == 1
    assert stats['local_only'] == 1
    assert stats['invalid_ids'] == 1
    assert sp.calls == [B]
    assert set(json.loads(checkpoint.read_text())['records']) == {B}
    assert path.read_bytes() == before


def test_fetch_only_dry_run_reports_api_queue_after_checkpoint_skips(tmp_path):
    path, checkpoint = tmp_path / 'master.csv', tmp_path / 'state.json'
    path.write_text(f'id,name,artists\n{A},Title,Artist\n{B},Title,\n')
    options = {'fetch_only': True, 'fields': ['artist_name']}
    sp = Spotify()
    stats = dm.backfill_metadata(path, checkpoint, sp=sp, **options)
    assert stats['eligible'] == 1
    before = checkpoint.read_bytes()
    stats = dm.backfill_metadata(path, checkpoint, dry_run=True, **options)
    assert stats['eligible'] == 0
    assert stats['resumed'] == 1
    assert checkpoint.read_bytes() == before


def test_fetch_only_with_only_local_fills_never_authenticates(tmp_path, monkeypatch):
    path, checkpoint = tmp_path / 'master.csv', tmp_path / 'state.json'
    path.write_text(f'id,name,artists,release_date\n{A},Title,Artist,1999\n')
    def forbidden():
        pytest.fail('Local-only rows must not authenticate')
    monkeypatch.setattr(dm, 'get_spotify_client', forbidden)
    stats = dm.backfill_metadata(path, checkpoint, fetch_only=True,
                                 fields=['artist_name', 'year'])
    assert stats['eligible'] == 0
    assert stats['local_only'] == 1
    assert json.loads(checkpoint.read_text())['records'] == {}


def test_unavailable_and_invalid_ids_are_reported(catalog, tmp_path):
    class Missing(Spotify):
        def track(self, tid):
            raise SpotifyException(404, -1, 'missing')
    with catalog.open('a') as output:
        output.write('spotify:local:example,,,,,,,,\n')
    stats = dm.backfill_metadata(catalog, tmp_path / 'state.json', sp=Missing())
    assert stats['unavailable'] == 2
    assert stats['invalid_ids'] == 1
    assert read(catalog).loc[A, 'artist_name'] == ''


def test_rate_limit_and_auth_error_stop(catalog, tmp_path, monkeypatch):
    from scripts import spotify_metadata as sm
    sleeps = []
    monkeypatch.setattr(sm.time, 'sleep', sleeps.append)
    class Limited(Spotify):
        def track(self, tid):
            if not self.calls:
                self.calls.append('limited')
                raise SpotifyException(429, -1, 'rate limit', headers={'Retry-After': '2'})
            return super().track(tid)
    with pytest.raises(sm.SpotifyCooldown):
        dm.backfill_metadata(catalog, tmp_path / 'state.json', sp=Limited(), fetch_only=True)
    assert not sleeps
    class Denied(Spotify):
        def track(self, tid):
            raise SpotifyException(403, -1, 'denied')
    with pytest.raises(SpotifyException):
        dm.backfill_metadata(catalog, tmp_path / 'denied.json', sp=Denied())


def test_optional_album_and_artist_metadata_is_cached(catalog, tmp_path):
    class Extra(Spotify):
        def album(self, aid):
            self.calls.append(('album', aid))
            return {'label': 'Label'}
        def artist(self, aid):
            self.calls.append(('artist', aid))
            return {'genres': ['rock', 'pop']}
    sp = Extra()
    dm.backfill_metadata(catalog, tmp_path / 'state.json', sp=sp, extra_metadata=True)
    assert sp.calls.count(('album', 'album')) == 1
    assert sp.calls.count(('artist', 'artist')) == 1
    assert read(catalog).loc[B, 'label'] == 'Label'
    assert set(json.loads(read(catalog).loc[B, 'artist_genres'])) == {'rock', 'pop'}
    assert 'genre' not in read(catalog).columns  # Artist genres are not track genres.


def test_checkpoint_cannot_overwrite_audio_or_insert_unknown_tracks(catalog, tmp_path):
    checkpoint = tmp_path / 'state.json'
    dm.backfill_metadata(catalog, checkpoint, sp=Spotify(), fetch_only=True)
    state = json.loads(checkpoint.read_text())
    state['records'][A]['tempo'] = '999'
    state['records']['C' * 22] = {'name': 'New'}
    checkpoint.write_text(json.dumps(state))
    dm.apply_metadata_checkpoint(catalog, checkpoint)
    assert read(catalog).loc[A, 'tempo'] == '-1'
    assert 'C' * 22 not in read(catalog).index


def test_sync_uses_shared_mapping_and_preserves_existing_rows(catalog, tmp_path, monkeypatch):
    history = tmp_path / 'history.csv'
    history.write_text(f'track_id\n{A}\n{"C" * 22}\n')
    monkeypatch.setattr(dm, 'MASTER_DB', catalog)
    monkeypatch.setattr(dm, 'HISTORY_DB', history)
    monkeypatch.setattr(dm, 'get_spotify_client', Spotify)
    dm.sync_metadata()
    rows = read(catalog)
    assert rows.loc['C' * 22, 'artist_name'] == 'Artist'
    assert rows.loc['C' * 22, 'year'] == '2020'
    assert rows.loc[A, 'label'] == '0007'
    assert rows.loc[A, 'country'] == 'NA'


def test_transient_failure_with_local_aliases_is_retried(catalog, tmp_path, monkeypatch):
    from scripts import spotify_metadata as sm
    monkeypatch.setattr(sm.time, 'sleep', lambda seconds: None)
    class Broken(Spotify):
        def track(self, tid):
            raise SpotifyException(503, -1, 'temporary')
    state = tmp_path / 'state.json'
    dm.backfill_metadata(catalog, state, sp=Broken(), fetch_only=True)
    dm.backfill_metadata(catalog, state, sp=Spotify())
    assert read(catalog).loc[A, 'artist_name'] == 'Artist'


def test_checkpoint_rejects_other_catalog_and_option_changes(catalog, tmp_path):
    state = tmp_path / 'state.json'
    dm.backfill_metadata(catalog, state, sp=Spotify(), fetch_only=True)
    other = tmp_path / 'other.csv'
    other.write_bytes(catalog.read_bytes())
    with pytest.raises(ValueError, match='different catalog'):
        dm.apply_metadata_checkpoint(other, state)
    with pytest.raises(ValueError, match='options differ'):
        dm.backfill_metadata(catalog, state, fields=['year'])


def test_failed_apply_does_not_change_master(catalog, tmp_path, monkeypatch):
    checkpoint = tmp_path / 'state.json'
    dm.backfill_metadata(catalog, checkpoint, sp=Spotify(), fetch_only=True)
    before = catalog.read_bytes()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail)
    with pytest.raises(OSError, match='disk full'):
        dm.apply_metadata_checkpoint(catalog, checkpoint)
    assert catalog.read_bytes() == before


def test_cli_preview_and_apply_use_selected_catalog(catalog, tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    state = tmp_path / 'state.json'
    base = [sys.executable, str(Path(dm.__file__).resolve()), '--master', str(catalog),
            '--checkpoint', str(state)]
    before = catalog.read_bytes()
    preview = subprocess.run([*base, '--backfill', '--dry-run'], capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert catalog.read_bytes() == before
    assert not state.exists()
    dm.backfill_metadata(catalog, state, sp=Spotify(), fetch_only=True)
    apply = subprocess.run([*base, '--apply-backfill'], capture_output=True, text=True)
    assert apply.returncode == 0, apply.stderr
    assert read(catalog).loc[A, 'year'] == '2020'


def test_sync_keeps_successful_tracks_when_later_request_times_out(catalog, tmp_path, monkeypatch):
    from requests.exceptions import Timeout
    from scripts import spotify_metadata as sm
    monkeypatch.setattr(sm.time, 'sleep', lambda seconds: None)
    class Partial(Spotify):
        def track(self, tid):
            if tid == 'D' * 22:
                raise Timeout('offline')
            return super().track(tid)
    history = tmp_path / 'history.csv'
    history.write_text(f'track_id\n{"C" * 22}\n{"D" * 22}\n')
    monkeypatch.setattr(dm, 'MASTER_DB', catalog)
    monkeypatch.setattr(dm, 'HISTORY_DB', history)
    monkeypatch.setattr(dm, 'get_spotify_client', Partial)
    dm.sync_metadata()
    assert read(catalog).loc['C' * 22, 'artist_name'] == 'Artist'
    assert 'D' * 22 not in read(catalog).index


def test_interrupt_saves_progress_without_writing_master(catalog, tmp_path):
    class Interrupted(Spotify):
        def track(self, tid):
            if tid == B:
                raise KeyboardInterrupt()
            return super().track(tid)
    checkpoint = tmp_path / 'state.json'
    before = catalog.read_bytes()
    with pytest.raises(KeyboardInterrupt):
        dm.backfill_metadata(catalog, checkpoint, sp=Interrupted(), fetch_only=True)
    assert catalog.read_bytes() == before
    assert json.loads(checkpoint.read_text())['records'][A]['artist_name'] == 'Artist'
    sp = Spotify()
    dm.backfill_metadata(catalog, checkpoint, sp=sp, fetch_only=True)
    assert sp.calls == [B]


def test_sparse_response_leaves_unknown_fields_empty_and_does_not_refetch(catalog, tmp_path):
    class Sparse(Spotify):
        def track(self, tid):
            self.calls.append(tid)
            return {'id': tid, 'name': 'Only a title'}
    checkpoint = tmp_path / 'state.json'
    sp = Sparse()
    dm.backfill_metadata(catalog, checkpoint, sp=sp)
    assert read(catalog).loc[B, 'name'] == 'Only a title'
    assert read(catalog).loc[B, 'artist_name'] == ''
    dm.backfill_metadata(catalog, checkpoint, sp=sp)
    assert sp.calls == [A, B]


def test_duplicate_ids_stop_before_requests(tmp_path):
    catalog = tmp_path / 'master.csv'
    catalog.write_text(f'id,name\n{A},One\n{A},Two\n')
    sp = Spotify()
    with pytest.raises(ValueError, match='duplicate'):
        dm.backfill_metadata(catalog, tmp_path / 'state.json', sp=sp)
    assert sp.calls == []
