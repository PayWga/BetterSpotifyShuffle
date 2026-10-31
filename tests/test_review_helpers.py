"""Regression checks for library dates and metadata recording identity."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest
from spotipy.exceptions import SpotifyException

from scripts import metadata_fetcher, spotify_metadata as sm


@pytest.mark.parametrize('added_at,expected', [
    ((datetime.now(timezone.utc) - timedelta(days=10)).isoformat(), 10),
    (None, None),
    ((datetime.now(timezone.utc) + timedelta(days=10)).isoformat(), 0),
])
def test_library_age_accessors_agree(added_at, expected, tmp_path):
    tracker = metadata_fetcher.MetadataTracker(save_file=tmp_path / 'absent-memory.json')
    tracker.memory['synthetic'] = {'play_count': 2, 'skip_count': 1, 'added_at': added_at}
    assert tracker.get_days_since_added('synthetic') == expected
    assert tracker.get_metadata('synthetic')['days_since_added'] == expected
    assert not tracker.save_file.exists()


@pytest.mark.usefixtures('metadata_clock')
@pytest.mark.parametrize('returned_id,linked_id,accepted', [
    ('A' * 22, None, True),
    ('B' * 22, 'A' * 22, True),
    ('B' * 22, None, False),
    ('B' * 22, 'C' * 22, False),
])
def test_master_metadata_checks_requested_identity(returned_id, linked_id, accepted):
    track = {'id': returned_id, 'name': 'Synthetic', 'duration_ms': 180000}
    if linked_id is not None:
        track['linked_from'] = {'id': linked_id}
    client = SimpleNamespace(track=lambda tid: track)
    if accepted:
        assert sm.fetch_track(client, 'A' * 22)['duration_ms'] == 180000
    else:
        with pytest.raises(SpotifyException, match='unrelated recording'):
            sm.fetch_track(client, 'A' * 22)


@pytest.mark.usefixtures('metadata_clock')
def test_unrelated_metadata_is_not_saved_and_remains_retryable(tmp_path):
    master = tmp_path / 'master.csv'
    checkpoint = tmp_path / 'checkpoint.json'
    master.write_text('id,duration_ms\n' + 'A' * 22 + ',\n', encoding='utf-8')
    client = SimpleNamespace(track=lambda tid: {'id': 'B' * 22, 'name': 'Unrelated', 'duration_ms': 999})
    def save(frame, path):
        frame.to_csv(path, index=False)
    stats = sm.backfill(master, checkpoint, lambda: client, save, fields=['duration_ms'])
    assert stats['failed'] == 1
    assert pd.isna(pd.read_csv(master)['duration_ms'].iloc[0])
    import json
    state = json.loads(checkpoint.read_text(encoding='utf-8'))
    assert 'A' * 22 in state['retry']
    assert not state['records']
