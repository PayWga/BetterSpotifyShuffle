import json

import numpy as np
import pandas as pd

from scripts.plot_readme import collect_snapshot, render_figures


def test_snapshot_uses_valid_values_and_history_matched_denominators(tmp_path):
    master = tmp_path / 'master.csv'
    history = tmp_path / 'history.csv'
    pd.DataFrame({'id': ['a', 'b', 'c'], 'tempo': [120, -1, 140],
                  'loudness': [-1, np.nan, -8], 'key': [-1, 0, 11],
                  'local_rms_db': [-1, np.nan, -20],
                  'duration_ms': [1000, 2000, 3000],
                  'duration_ms_is_estimated': [False, True, False]}).to_csv(master, index=False)
    pd.DataFrame({'track_id': ['a', 'a', 'b', 'outside-secret-track'],
                  'event_type': ['PLAY', 'GHOST_SKIP', 'NEUTRAL', None]}).to_csv(history, index=False)
    before = {p: p.read_bytes() for p in (master, history)}
    snapshot = collect_snapshot(master, history, chunksize=2)
    assert snapshot['catalog_rows'] == 3
    assert snapshot['history_matched_rows'] == 2
    assert snapshot['history_unique_tracks'] == 3
    assert snapshot['coverage']['catalog']['tempo'] == 2
    assert snapshot['coverage']['history_matched']['tempo'] == 1
    assert snapshot['coverage']['catalog']['key'] == 2
    assert snapshot['coverage']['catalog']['loudness'] == 2
    assert snapshot['coverage']['catalog']['duration_ms'] == 2
    assert snapshot['events']['Unclassified'] == 1
    assert sum(snapshot['histograms']['tempo']['counts']) == 1
    assert all(p.read_bytes() == content for p, content in before.items())
    exported = json.dumps(snapshot)
    assert 'outside-secret-track' not in exported
    assert 'track_id' not in exported
    assert str(tmp_path) not in exported


def test_empty_history_and_missing_optional_features(tmp_path):
    master, history = tmp_path / 'master.csv', tmp_path / 'history.csv'
    pd.DataFrame({'id': ['a']}).to_csv(master, index=False)
    pd.DataFrame(columns=['track_id', 'event_type']).to_csv(history, index=False)
    snapshot = collect_snapshot(master, history)
    assert snapshot['history_matched_rows'] == 0
    assert snapshot['coverage']['catalog']['tempo'] == 0
    assert sum(snapshot['histograms']['tempo']['counts']) == 0
    render_figures(snapshot, tmp_path / 'figures')
    assert len(list((tmp_path / 'figures').glob('*.png'))) == 3


def test_empty_category_lists_are_unavailable_and_artist_names_can_fill_missing_ids(tmp_path):
    master, history = tmp_path / 'master.csv', tmp_path / 'history.csv'
    pd.DataFrame({'id': ['a', 'b'], 'artists': ['[]', 'Available artist'],
                  'artist_ids': ['[]', '[]'], 'track_genre': ['[]', "['rock']"]}).to_csv(master, index=False)
    pd.DataFrame({'track_id': ['a', 'b'], 'event_type': ['PLAY', 'PLAY']}).to_csv(history, index=False)
    snapshot = collect_snapshot(master, history)
    assert snapshot['coverage']['catalog']['artists'] == 1
    assert snapshot['coverage']['catalog']['track_genre'] == 1


def test_render_comparison_keeps_validation_protocol_and_private_inputs(tmp_path):
    master, history = tmp_path / 'master.csv', tmp_path / 'history.csv'
    report = tmp_path / 'comparison.json'
    pd.DataFrame({'id': ['private-track'], 'tempo': [120]}).to_csv(master, index=False)
    pd.DataFrame({'track_id': ['private-track'], 'event_type': ['PLAY']}).to_csv(history, index=False)
    report.write_text(json.dumps({'test_evaluated': False, 'results': [{
        'variant': 'audio_baseline', 'parameters': {'total': 10},
        'validation': {'hit_at_k': .3, 'ndcg_at_k': .2, 'k': 10,
                       'examples': 5, 'mean_candidates': 4},
        'history': [{'private': 'not for export'}],
        'evaluation_fingerprint': 'same-plan', 'test_evaluated': False}]}))
    before = report.read_bytes()
    snapshot = collect_snapshot(master, history, comparison=report)
    render_figures(snapshot, tmp_path / 'figures')
    assert len(list((tmp_path / 'figures').glob('*.png'))) == 4
    assert snapshot['comparison'][0]['validation']['mean_candidates'] == 4
    assert report.read_bytes() == before
    assert 'private' not in json.dumps(snapshot)
