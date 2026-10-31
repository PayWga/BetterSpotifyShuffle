import importlib

import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('shufflerec.data') is not None, 'sequence pipeline is not implemented'
    return importlib.import_module('shufflerec.data')


def test_complete_sessions_are_disjoint_chronological_and_small_inputs_fail():
    m=api()
    history=pd.DataFrame([dict(track_id=str(i),event_type='PLAY',timestamp=f'2020-01-{i+1:02d}T00:00:00Z') for i in range(10)])
    sessions=m.chunk_sessions(history.sample(frac=1,random_state=2))
    parts=m.split_sessions(sessions)
    assert [len(parts[k]) for k in ('train','validation','test')]==[8,1,1]
    assert parts['train'][-1].timestamp.max()<parts['validation'][0].timestamp.min()
    assert parts['validation'][-1].timestamp.max()<parts['test'][0].timestamp.min()
    with pytest.raises(ValueError,match='three'):
        m.split_sessions(sessions[:2])


def test_examples_keep_metadata_only_tracks_and_skips_share_past_context():
    m=api()
    history=pd.DataFrame(dict(track_id=['a','b','s','c','d','a'],event_type=['PLAY','PLAY','GHOST_SKIP','PLAY','NEUTRAL','PLAY'],
                             timestamp=pd.date_range('2020-01-01',periods=6,freq='min',tz='UTC')))
    nip,pairs=m.extract_examples([history], {'a','b','s','c','d'}, max_length=10)
    assert [s.positive for s in nip]==['b','c']
    assert pairs[0].context==('a','b')
    assert pairs[0].positive=='c' and pairs[0].negative=='s'


def test_missing_tracks_break_context_and_do_not_become_targets():
    m=api()
    history=pd.DataFrame(dict(track_id=['a','missing','b','c'],event_type=['PLAY']*4,
                             timestamp=pd.date_range('2020-01-01',periods=4,freq='min',tz='UTC')))
    nip,pairs=m.extract_examples([history], {'a','b','c'})
    assert len(nip)==1 and nip[0].context==('b',) and nip[0].positive=='c'


def test_store_and_dataset_padding_are_separate_from_feature_availability():
    m=api(); from shufflerec.features import FeaturePreprocessor
    catalog=pd.DataFrame([dict(id='a',artists='Alice'),dict(id='b',artists='Bob'),dict(id='n',tempo=120)])
    prep=FeaturePreprocessor().fit(catalog,['a','b','n']); store=m.TrackStore(catalog,prep)
    assert 'a' in store.usable_ids
    sample=m.Example(('a',),'b',pd.Timestamp('2020-01-01',tz='UTC'))
    ds=m.SequenceDataset([sample],store,max_length=3,seed=7)
    batch=ds[0]
    assert batch['padding_mask'].tolist()==[True,True,False]
    assert batch['context']['artist_mask'].squeeze(-1).tolist()==[0,0,1]
    assert not batch['context']['numeric_mask'].any()
    assert batch['negative']['numeric_mask'].any()  # Only track 'n' is eligible as a negative.


def test_negative_sampling_never_hangs_when_pool_is_exhausted():
    m=api(); from shufflerec.features import FeaturePreprocessor
    catalog=pd.DataFrame([dict(id='a',tempo=120),dict(id='b',tempo=130)])
    store=m.TrackStore(catalog,FeaturePreprocessor().fit(catalog,['a','b']))
    sample=m.Example(('a',),'b',pd.Timestamp('2020-01-01',tz='UTC'))
    with pytest.raises(ValueError,match='negative'):
        m.SequenceDataset([sample],store)[0]


def test_preparation_fits_only_training_session_tracks():
    m=api()
    rows=[]; catalog=[]
    for day in range(1,11):
        for j in range(2):
            identity=f'{day}-{j}'; catalog.append(dict(id=identity,tempo=100 if day<9 else 900,artists='train' if day<9 else 'future'))
            rows.append(dict(track_id=identity,event_type='PLAY',timestamp=f'2020-01-{day:02d}T00:0{j}:00Z'))
    prepared=m.prepare_data(pd.DataFrame(catalog),pd.DataFrame(rows))
    assert prepared.preprocessor.scales['tempo']==[100,100]
    assert 'name:future' not in prepared.preprocessor.vocabularies['artist']
    assert len(prepared.examples['train']['nip'])==8
    assert len(prepared.examples['test']['nip'])==1
