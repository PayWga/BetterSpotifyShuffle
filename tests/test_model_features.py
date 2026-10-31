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
    assert batch['artist'][3,0] == 1  # Unknown category ID stays separate from padding.
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


@pytest.mark.parametrize('flag',[True, 'True', 'true', 1, '1.0'])
def test_estimated_durations_are_unavailable_without_dropping_the_track(tmp_path, flag):
    m=api(); from shufflerec.audit import read_catalog
    frame=pd.DataFrame([dict(id='confirmed',duration_ms=100000,artists='A'),
                        dict(id='estimated',duration_ms=900000,duration_ms_is_estimated=flag,artists='B')])
    path=tmp_path/'catalog.csv';frame.to_csv(path,index=False)
    loaded=read_catalog(path)
    p=m.FeaturePreprocessor().fit(loaded,['confirmed','estimated'])
    i=p.numeric_names.index('duration_ms');batch=p.transform(loaded)
    assert batch['numeric_mask'][:,i].tolist()==[1,0]
    assert p.scales['duration_ms']==[100000,100000]
    assert batch['artist_mask'][1,0]==1
    assert len(loaded)==2


def test_duration_policy_is_saved_and_legacy_artifacts_keep_their_original_transform():
    m=api(); frame=pd.DataFrame([dict(id='a',duration_ms=100000),
                                dict(id='b',duration_ms=200000,duration_ms_is_estimated=True)])
    p=m.FeaturePreprocessor().fit(frame,['a','b'])
    state=p.state_dict()
    assert state['duration_policy']=='exclude_estimated'
    i=p.numeric_names.index('duration_ms')
    assert m.FeaturePreprocessor.from_state_dict(state).transform(frame)['numeric_mask'][1,i]==0
    legacy=dict(state);legacy.pop('duration_policy')
    legacy['scales']={**legacy['scales'],'duration_ms':[100000,200000]}
    values=m.FeaturePreprocessor.from_state_dict(legacy).transform(frame)
    assert values['numeric_mask'][1,i]==1
    assert values['numeric'][1,i]==1


@pytest.mark.parametrize('feature',['local_rms_db','local_onset_strength','local_spectral_centroid',
                                    'local_spectral_contrast','local_beat_regularity'])
def test_local_measurement_and_its_provenance_can_be_disabled_independently(feature):
    m=api()
    config=m.FeatureConfig(**{feature:False})
    frame=catalog()
    for name,value in [('local_rms_db',-20),('local_onset_strength',1),('local_spectral_centroid',1000),
                       ('local_spectral_contrast',10),('local_beat_regularity',0.5)]:
        frame[name]=value;frame[name+'_source']='librosa_acoustics_v1';frame[name+'_analysis_seconds']=60
    p=m.FeaturePreprocessor(config).fit(frame,['a','b'])
    assert feature not in p.numeric_names
    assert feature+'_is_local' not in p.numeric_names
    assert feature+'_analysis_seconds' not in p.numeric_names
    assert all(name in p.numeric_names for name in m.LOCAL if name!=feature)


def test_estimated_durations_can_be_included_with_an_explicit_unknown_aware_indicator(tmp_path):
    m=api()
    frame=pd.DataFrame([dict(id='confirmed',duration_ms=100000,duration_ms_is_estimated=False),
                        dict(id='estimated',duration_ms=200000,duration_ms_is_estimated=True),
                        dict(id='unknown',duration_ms=150000),
                        dict(id='missing',duration_ms_is_estimated=True)])
    config=m.FeatureConfig(include_estimated_durations=True)
    p=m.FeaturePreprocessor(config).fit(frame,['confirmed','estimated','unknown','missing'])
    batch=p.transform(frame);duration=p.numeric_names.index('duration_ms')
    flag=p.numeric_names.index('duration_ms_is_estimated')
    assert batch['numeric_mask'][:,duration].tolist()==[1,1,1,0]
    assert batch['numeric'][:,flag].tolist()==[0,1,0,0]
    assert batch['numeric_mask'][:,flag].tolist()==[1,1,0,0]
    path=tmp_path/'include-estimated.json';p.save(path)
    restored=m.FeaturePreprocessor.load(path)
    assert restored.duration_policy=='include_estimated'
    for name,values in batch.items():
        np.testing.assert_array_equal(values,restored.transform(frame)[name])


def test_estimate_indicator_is_not_minmax_scaled_when_only_estimates_are_observed():
    m=api();frame=pd.DataFrame([dict(id='a',duration_ms=100000,duration_ms_is_estimated=True)])
    p=m.FeaturePreprocessor(m.FeatureConfig(include_estimated_durations=True)).fit(frame,['a'])
    assert p.transform(frame)['numeric'][0,p.numeric_names.index('duration_ms_is_estimated')]==1
