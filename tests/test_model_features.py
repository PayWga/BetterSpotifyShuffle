"""Synthetic enriched catalogs: no private files or acquisition imports."""
import importlib

import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('shufflerec') is not None, 'model package is not implemented'
    return importlib.import_module('shufflerec.features')


def catalog():
    return pd.DataFrame([
        dict(id='a', artists="['Alice', 'Bob']", artist_ids="['id-a', 'id-b']",
             year=2000, duration_ms=100000, tempo=100, key=11, time_signature=4,
             track_genre='rock', artist_genres='["folk", "indie"]',
             local_rms_db=-20, local_rms_db_source='librosa_acoustics_v1',
             local_rms_db_analysis_seconds=60),
        dict(id='b', artists='Bob', artist_ids='id-b', year=2020,
             duration_ms=200000, tempo=200, key=0, time_signature=3,
             track_genre='jazz', local_rms_db=-10),
        dict(id='c', artists='Alice', year=2090, duration_ms=900000,
             tempo=900, key=-1, time_signature=-1,
             track_genre='unseen', history_duration_candidate_ms=900000),
        dict(id='d', artists='New Artist', history_duration_candidate_ms=120000),
    ])


def test_train_statistics_are_frozen_and_missing_candidates_not_durations():
    m=api(); frame=catalog(); original=frame.copy(deep=True)
    p=m.FeaturePreprocessor().fit(frame, ['a','b'])
    batch=p.transform(frame)
    names=p.numeric_names
    year=names.index('year'); duration=names.index('duration_ms')
    np.testing.assert_allclose(batch['numeric'][:3, year], [0,1,1])
    assert batch['numeric_mask'][3,duration] == 0
    assert batch['numeric'][3,duration] == 0
    assert batch['numeric_mask'][0,names.index('local_rms_db')] == 1
    pd.testing.assert_frame_equal(frame,original)


def test_artist_alias_collaboration_unknown_and_genre_namespaces():
    m=api(); p=m.FeaturePreprocessor().fit(catalog(), ['a','b'])
    batch=p.transform(catalog())
    assert set(batch['artist'][0]) - {0} == {p.vocabularies['artist']['id:id-a'],p.vocabularies['artist']['id:id-b']}
    assert batch['artist'][2,0] == p.vocabularies['artist']['id:id-a']
    assert batch['artist'][3,0] == 1  # unknown, distinct from padding
    assert batch['genre'][2,0] == 1
    assert batch['artist_genre'][0,0] >= 2
    assert 'folk' not in p.vocabularies['genre']
    assert m.parse_labels('Earth, Wind & Fire') == ['earth, wind & fire']


def test_circular_key_validates_missing_and_handles_wraparound():
    m=api(); p=m.FeaturePreprocessor().fit(catalog(), ['a','b'])
    batch=p.transform(catalog()); names=p.numeric_names
    ix=[names.index('key_sin'), names.index('key_cos')]
    np.testing.assert_allclose(batch['numeric'][0,ix],[-0.5,np.sqrt(3)/2],atol=1e-6)
    np.testing.assert_allclose(batch['numeric'][1,ix],[0,1],atol=1e-6)
    assert batch['numeric_mask'][2,ix].sum()==0
    assert batch['time_signature'][2,0] == 0


def test_provenance_and_analysis_scope_are_separate_from_audio_values():
    m=api(); p=m.FeaturePreprocessor().fit(catalog(), ['a','b'])
    batch=p.transform(catalog()); names=p.numeric_names
    i=names.index('local_rms_db_is_local')
    assert batch['numeric'][0,i] == 1
    assert batch['numeric'][1,i] == 0
    i=names.index('local_rms_db_analysis_seconds')
    assert batch['numeric_mask'][0,i] == 1
    assert batch['numeric_mask'][1,i] == 0


def test_state_roundtrip_and_optional_schema(tmp_path):
    m=api(); p=m.FeaturePreprocessor().fit(catalog(), ['a','b'])
    path=tmp_path/'preprocessor.json'; p.save(path)
    restored=m.FeaturePreprocessor.load(path)
    for key,values in p.transform(catalog()).items():
        np.testing.assert_array_equal(values,restored.transform(catalog())[key])
    sparse=pd.DataFrame([dict(id='z',artists='New Artist',duration_ms=-1,key=-1,tempo=np.inf)])
    transformed=restored.transform(sparse)
    assert np.isfinite(transformed['numeric']).all()
    assert transformed['numeric_mask'][0,restored.numeric_names.index('duration_ms')]==0


def test_ambiguous_aliases_do_not_merge_artists_and_absent_training_ids_fail():
    m=api(); f=pd.DataFrame([dict(id='a',artists='Same',artist_ids='x'),dict(id='b',artists='Same',artist_ids='y'),dict(id='c',artists='Same')])
    p=m.FeaturePreprocessor().fit(f,['a','b'])
    assert p.transform(f)['artist'][2,0] == 1
    with pytest.raises(ValueError,match='training'):
        m.FeaturePreprocessor().fit(f,['missing'])


def test_year_precision_dates_are_calendar_years_even_when_csv_infers_numbers(tmp_path):
    m=api(); from shufflerec.audit import read_catalog
    frame=pd.DataFrame([dict(id='old',release_date=1990),dict(id='new',release_date=2020)])
    path=tmp_path/'years.csv';frame.to_csv(path,index=False)
    for values in (frame,read_catalog(path)):
        p=m.FeaturePreprocessor().fit(values,['old','new'])
        assert p.scales['year']==[1990,2020]
        np.testing.assert_allclose(p.transform(values)['numeric'][:,p.numeric_names.index('year')],[0,1])


def test_invalid_year_can_use_valid_alias_but_invalid_calendar_date_stays_missing():
    m=api()
    frame=pd.DataFrame([dict(id='alias',year=-1,release_year=2001),
                        dict(id='invalid',release_date='2020-02-30'),
                        dict(id='month',release_date='2021-02')])
    p=m.FeaturePreprocessor().fit(frame,['alias','invalid','month'])
    assert p.scales['year']==[2001,2021]
    values=p.transform(frame);i=p.numeric_names.index('year')
    assert values['numeric_mask'][:,i].tolist()==[1,0,1]
