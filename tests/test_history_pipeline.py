"""History import and live appends share a schema without losing extra metadata."""
import csv
import json

import pandas as pd
import pytest

from scripts import import_etended_spotify_history as history
from scripts import polling


def write_history(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def event(track_id='track', timestamp='2020-01-01T00:00:00Z', **extra):
    return dict(timestamp=timestamp, track_id=track_id, ms_played=180000,
                duration_ms=180000, completion_rate=1., event_type='PLAY',
                is_play=1, is_ghost_skip=0, **extra)


def test_imported_history_can_be_reopened_and_appended_with_extra_columns(tmp_path):
    extended, live, merged = (tmp_path / name for name in ('extended.csv', 'live.csv', 'merged.csv'))
    write_history(extended, [event(shuffle=True, platform='synthetic', custom_label='0007')])
    polling.init_csv(csv_path=live)
    history.combine_and_deduplicate(extended, live, merged)
    polling.init_csv(csv_path=merged)
    polling.log_track({'id': 'next', 'name': 'NA', 'artists': [], 'duration_ms': 180000},
                      180000, 'playlist:test', 'playlist', 'track_change', csv_path=merged, quiet=True)
    with merged.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['custom_label'] == '0007'
    assert rows[1]['track_id'] == 'next' and rows[1]['track_name'] == 'NA'
    assert rows[1]['event_type'] == 'PLAY'
    assert rows[1]['custom_label'] == '' and rows[1]['platform'] == ''
    assert None not in rows[1]


def test_appends_follow_existing_header_order_and_record_duration_source(tmp_path):
    path = tmp_path / 'history.csv'
    columns = ['duration_ms_source', 'duration_ms_is_estimated', 'extra'] + list(reversed(polling.CSV_COLUMNS))
    with path.open('w', newline='', encoding='utf-8') as stream:
        csv.DictWriter(stream, fieldnames=columns).writeheader()
    polling.init_csv(csv_path=path)
    polling.log_track({'id': 'next', 'name': 'Synthetic', 'artists': [], 'duration_ms': 180000},
                      180000, None, None, 'track_change', csv_path=path, quiet=True)
    with path.open(newline='', encoding='utf-8') as stream:
        row = next(csv.DictReader(stream))
    assert row['track_id'] == 'next' and row['ms_played'] == '180000'
    assert row['duration_ms_source'] == 'spotify_polling'
    assert row['duration_ms_is_estimated'] == 'False'
    assert row['extra'] == '' and None not in row


def test_legacy_header_is_preserved_when_appending(tmp_path):
    path = tmp_path / 'legacy.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        csv.DictWriter(stream, fieldnames=polling.CSV_COLUMNS).writeheader()
    polling.log_track({'id': 'track', 'name': 'Synthetic', 'artists': [], 'duration_ms': 180000},
                      180000, None, None, 'track_change', csv_path=path, quiet=True)
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == polling.CSV_COLUMNS
        row = next(reader)
    assert row['track_id'] == 'track' and row['event_type'] == 'PLAY'
    assert None not in row


@pytest.mark.parametrize('columns', [polling.CSV_COLUMNS[:-1], polling.CSV_COLUMNS + ['track_id']])
def test_missing_or_duplicate_required_header_is_rejected_without_changes(tmp_path, columns):
    path = tmp_path / 'history.csv'
    path.write_text(','.join(columns) + '\n', encoding='utf-8')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='schema'):
        polling.init_csv(csv_path=path)
    assert path.read_bytes() == before


def test_ids_are_cleaned_before_dedup_and_cross_source_matching(tmp_path):
    extended, live, merged = (tmp_path / name for name in ('extended.csv', 'live.csv', 'merged.csv'))
    write_history(extended, [event(' track '), event(None, timestamp='2020-01-01T00:01:00Z'),
                             event(' ', timestamp='2020-01-01T00:02:00Z'),
                             event('NA', timestamp='2020-01-01T00:03:00Z')])
    write_history(live, [event('track', timestamp='2020-01-01T00:00:03Z')])
    history.combine_and_deduplicate(extended, live, merged)
    result = pd.read_csv(merged, dtype='string', keep_default_na=False)
    assert result.track_id.tolist() == ['track', 'NA']
    assert result.event_source.tolist() == ['extended_history', 'extended_history']


def test_history_merge_classifies_blank_labels_and_flags_from_text_csv(tmp_path):
    extended, live, merged = (tmp_path / name for name in ('extended.csv', 'live.csv', 'merged.csv'))
    row = event()
    row.update(event_type='', is_play='', is_ghost_skip='')
    write_history(extended, [row])
    polling.init_csv(csv_path=live)
    history.combine_and_deduplicate(extended, live, merged)
    result = pd.read_csv(merged)
    assert result.event_type.tolist() == ['PLAY']
    assert result.is_play.tolist() == [1] and result.is_ghost_skip.tolist() == [0]


@pytest.mark.parametrize('invalid_rows', [[], [event(None)], [event(' ')]])
def test_empty_or_all_invalid_history_merges_keep_a_valid_empty_schema(tmp_path, invalid_rows):
    extended, live, merged = (tmp_path / name for name in ('extended.csv', 'live.csv', 'merged.csv'))
    if invalid_rows:
        write_history(extended, invalid_rows)
    else:
        polling.init_csv(csv_path=extended)
    polling.init_csv(csv_path=live)
    history.combine_and_deduplicate(extended, live, merged)
    result = pd.read_csv(merged)
    assert result.empty
    assert set(polling.CSV_COLUMNS).issubset(result.columns)
    polling.init_csv(csv_path=merged)


@pytest.mark.parametrize('source_field', ['duration_ms_source', 'duration_ms_metadata_source'])
@pytest.mark.parametrize('flag', [None, False])
def test_estimate_source_prevents_completion_labels_without_reliable_flag(source_field, flag):
    row = dict(track_id='estimated', ms_played=1000, duration_ms=180000,
               reason_end='fwdbtn', **{source_field: 'history_completed_play_median'})
    if flag is not None:
        row['duration_ms_is_estimated'] = flag
    result = history.normalize_listening_history(pd.DataFrame([row]), 'extended_history')
    assert result.event_type.tolist() == ['NEUTRAL']
    assert result[source_field].tolist() == ['history_completed_play_median']
    if flag is None:
        assert 'duration_ms_is_estimated' not in result
    else:
        assert result.duration_ms_is_estimated.tolist() == [flag]


def import_events(tmp_path, rows, catalog=None):
    source = tmp_path / 'Streaming_History_Audio_synthetic.json'
    source.write_text(json.dumps(rows), encoding='utf-8')
    master = tmp_path / 'master.csv'
    if catalog is not None:
        write_history(master, catalog)
    output = tmp_path / 'imported.csv'
    history.process_extended_streaming_history(tmp_path, master, output)
    return pd.read_csv(output)


def raw_event(track, played, reason, minute=0, **extra):
    return dict(spotify_track_uri='spotify:track:' + track,
                ts=f'2020-01-01T00:{minute:02d}:00Z', ms_played=played,
                reason_end=reason, skipped=reason == 'fwdbtn', **extra)


def test_unknown_duration_stays_missing_and_does_not_invent_skip_evidence(tmp_path):
    result = import_events(tmp_path, [raw_event('unknown', 1000, 'fwdbtn')])
    row = result.iloc[0]
    assert pd.isna(row.duration_ms) and pd.isna(row.completion_rate)
    assert pd.isna(row.duration_ms_is_estimated) and pd.isna(row.duration_ms_source)
    assert row.event_type == 'NEUTRAL' and row.is_ghost_skip == 0


def test_completed_play_estimate_is_flagged_and_does_not_classify_later_skip(tmp_path):
    result = import_events(tmp_path, [raw_event('estimated', 180000, 'trackdone'),
                                      raw_event('estimated', 1000, 'fwdbtn', minute=1)])
    assert result.duration_ms.tolist() == [180000, 180000]
    assert result.duration_ms_is_estimated.tolist() == [True, True]
    assert result.duration_ms_source.tolist() == ['history_completed_play_median'] * 2
    assert result.event_type.tolist() == ['PLAY', 'NEUTRAL']


def test_catalog_duration_and_its_existing_estimate_provenance_take_priority(tmp_path):
    result = import_events(tmp_path, [raw_event('confirmed', 1000, 'fwdbtn'),
                                      raw_event('estimated', 1000, 'fwdbtn', minute=1),
                                      raw_event('estimated', 180000, 'trackdone', minute=2)],
        catalog=[dict(id='confirmed', duration_ms=200000, duration_ms_is_estimated=False,
                      duration_ms_source='spotify_track_api'),
                 dict(id='estimated', duration_ms=190000, duration_ms_is_estimated=True,
                      duration_ms_source='history_completed_play_median')])
    assert result.duration_ms.tolist() == [200000, 190000, 190000]
    assert result.duration_ms_is_estimated.tolist() == [False, True, True]
    assert result.duration_ms_source.tolist() == ['spotify_track_api', 'history_completed_play_median',
                                                 'history_completed_play_median']
    assert result.event_type.tolist() == ['GHOST_SKIP', 'NEUTRAL', 'PLAY']


def test_duration_lookup_keeps_numeric_return_values_for_existing_callers(tmp_path):
    master = tmp_path / 'master.csv'
    write_history(master, [dict(id='track', duration_ms=180000)])
    assert history.build_duration_lookup([], master) == {'track': 180000}


def test_normalizing_unlabelled_unknown_or_estimated_duration_stays_conservative():
    original = pd.DataFrame([
        dict(track_id='unknown', timestamp='2020-01-01T00:00:00Z', ms_played=1000,
             duration_ms=None, reason_end='fwdbtn'),
        dict(track_id='estimated', timestamp='2020-01-01T00:01:00Z', ms_played=1000,
             duration_ms=180000, duration_ms_is_estimated=True, reason_end='fwdbtn'),
        dict(track_id='done', timestamp='2020-01-01T00:02:00Z', ms_played=180000,
             duration_ms=None, reason_end='trackdone'),
    ])
    result = history.normalize_listening_history(original, 'extended_history')
    assert result.event_type.tolist() == ['NEUTRAL', 'NEUTRAL', 'PLAY']
