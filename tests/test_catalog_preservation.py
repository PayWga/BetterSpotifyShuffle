"""Catalog writers keep text metadata intact while cleaning numeric fields."""
import pandas as pd
import pytest

from scripts import data_merging as dm


@pytest.mark.parametrize('action', ['merge', 'repair', 'init'])
def test_catalog_writers_preserve_text_metadata(action, tmp_path, monkeypatch):
    master = tmp_path / 'master.csv'
    baseline = tmp_path / 'baseline.csv'
    incoming = tmp_path / 'incoming.csv'
    text = ('id,name,label,country,tempo,custom_value,tempo_source\n'
            '0007,NA,0007,NA,120,0012,local_acoustic\n'
            'other,Original,,US,80,NA,spotify\n')
    if action == 'init':
        baseline.write_text(text, encoding='utf-8')
    else:
        master.write_text(text, encoding='utf-8')
    incoming.write_text('id,name,label,country,tempo,custom_value,tempo_source\n'
                        '0007,Replacement,9999,XX,999,9999,replacement\n'
                        'other,Replacement,0008,,90,,replacement\n'
                        'new,NULL,0010,NA,100,0001,local_acoustic\n', encoding='utf-8')
    monkeypatch.setattr(dm, 'MASTER_DB', master)
    monkeypatch.setattr(dm, 'BASE_DATASET', baseline)
    if action == 'merge':
        dm.merge_csv(incoming)
    elif action == 'repair':
        dm.repair_database()
    else:
        dm.init_database()
    result = pd.read_csv(master, dtype='string', keep_default_na=False).set_index('id')
    row = result.loc['0007']
    assert row['name'] == 'NA'
    assert row['label'] == '0007'
    assert row['country'] == 'NA'
    assert row['custom_value'] == '0012'
    assert row['tempo_source'] == 'local_acoustic'
    assert float(row['tempo']) == 120
    assert result.loc['other', 'custom_value'] == 'NA'
    if action == 'merge':
        assert result.loc['other', 'label'] == '0008'
        assert result.loc['other', 'name'] == 'Original'
        assert result.loc['new', 'name'] == 'NULL'
        assert result.loc['new', 'custom_value'] == '0001'


def test_merge_fills_empty_numeric_cells_without_changing_existing_values(tmp_path, monkeypatch):
    master = tmp_path / 'master.csv'
    incoming = tmp_path / 'incoming.csv'
    master.write_text('id,name,tempo,loudness,custom_value\na,NA,,-1,0007\n', encoding='utf-8')
    incoming.write_text('id,name,tempo,loudness,custom_value\na,Replacement,120,-12,9999\n', encoding='utf-8')
    monkeypatch.setattr(dm, 'MASTER_DB', master)
    dm.merge_csv(incoming)
    row = pd.read_csv(master, dtype='string', keep_default_na=False).iloc[0]
    assert row['name'] == 'NA'
    assert float(row['tempo']) == 120
    assert float(row['loudness']) == -1
    assert row['custom_value'] == '0007'


def test_merge_fills_textual_nan_local_features_without_reformatting_valid_values(tmp_path, monkeypatch):
    master = tmp_path / 'master.csv'
    incoming = tmp_path / 'incoming.csv'
    master.write_text('id,local_rms_db,local_onset_strength,tempo_analysis_seconds,custom_value\n'
                      'a,nan,NaN,nan,NaN\nb,-20.00,4.00,60.00,0007\n', encoding='utf-8')
    incoming.write_text('id,local_rms_db,local_onset_strength,tempo_analysis_seconds,custom_value\n'
                        'a,-20,4,60,replacement\nb,-30,5,120,replacement\n', encoding='utf-8')
    monkeypatch.setattr(dm, 'MASTER_DB', master)
    dm.merge_csv(incoming)
    result = pd.read_csv(master, dtype='string', keep_default_na=False).set_index('id')
    assert result.loc['a', 'local_rms_db'] == '-20'
    assert result.loc['a', 'local_onset_strength'] == '4'
    assert result.loc['a', 'tempo_analysis_seconds'] == '60'
    assert result.loc['a', 'custom_value'] == 'NaN'
    assert result.loc['b', 'local_rms_db'] == '-20.00'
    assert result.loc['b', 'local_onset_strength'] == '4.00'
    assert result.loc['b', 'tempo_analysis_seconds'] == '60.00'
    assert result.loc['b', 'custom_value'] == '0007'
