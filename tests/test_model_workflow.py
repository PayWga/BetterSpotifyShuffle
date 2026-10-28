import importlib
import json

import pandas as pd
import pytest

from shufflerec.features import FeatureConfig
from shufflerec.model import ModelConfig


def audit_api():
    assert importlib.util.find_spec('shufflerec.audit') is not None, 'read-only audit is not implemented'
    return importlib.import_module('shufflerec.audit')


def comparison_api():
    assert importlib.util.find_spec('shufflerec.experiments') is not None, 'controlled comparisons are not implemented'
    return importlib.import_module('shufflerec.experiments')


def test_audit_reads_chunks_counts_missing_inputs_and_never_changes_sources(tmp_path):
    m=audit_api()
    catalog=pd.DataFrame([dict(id='a',artists='Private Artist',key=-1,duration_ms=100000),
                          dict(id='b',tempo=120,history_duration_candidate_ms=123000)])
    history=pd.DataFrame(dict(track_id=['a','a','c'],event_type=['PLAY']*3,timestamp=['2020-01-01']*3))
    master=tmp_path/'master.csv'; hist=tmp_path/'history.csv'
    catalog.to_csv(master,index=False);history.to_csv(hist,index=False)
    before=(master.read_bytes(),hist.read_bytes())
    report=m.audit_paths(master,hist,chunksize=1)
    assert report['catalog_rows']==2
    assert report['matched_history_tracks']==1
    assert report['matched_history_events']==2
    assert report['valid_numeric_counts']['duration_ms']==1
    assert report['valid_numeric_counts']['key_sin']==0
    assert report['valid_numeric_counts']['local_rms_db']==0
    assert 'Private Artist' not in json.dumps(report)
    assert (master.read_bytes(),hist.read_bytes())==before
    selected=m.read_catalog(master,track_ids={'a'})
    assert selected.id.tolist()==['a']


def test_comparison_shares_validation_cases_and_leaves_test_untouched():
    m=comparison_api()
    catalog=pd.DataFrame([dict(id=t,tempo=100+i*20,artists=t,year=2000+i) for i,t in enumerate('abcd')])
    history=pd.DataFrame([dict(track_id=t,event_type='PLAY' if t!='d' else 'NEUTRAL',timestamp=f'2020-01-{day:02d}T00:0{i}:00Z')
                          for day in range(1,11) for i,t in enumerate('abcd')])
    variants={'audio':FeatureConfig(artist=False,year=False,duration=False,genre=False,artist_genre=False,local=False,provenance=False),
              'metadata':FeatureConfig(local=False,provenance=False)}
    result=m.run_comparison(catalog,history,variants=variants,model_config=ModelConfig(d_model=8,nhead=2,feedforward=16,dropout=0),
                            epochs=1,preference_epochs=0,num_negatives=1,max_examples=2,seed=8,
                            learning_rate=0.002,preference_learning_rate=0.0002,patience=2,cooldown_seconds=0)
    assert len(result)==2
    assert result[0]['evaluation_fingerprint']==result[1]['evaluation_fingerprint']
    assert all(r['test_evaluated'] is False for r in result)
    assert all(r['validation']['examples']==2 for r in result)
    assert all(r['training_options']['learning_rate']==0.002 for r in result)
    assert all(r['training_options']['patience']==2 for r in result)
    assert all(r['history'][0]['phase']=='next_item' for r in result)


def test_notebook_runs_synthetic_training_and_checkpoint_without_source_writes(tmp_path,monkeypatch):
    from pathlib import Path
    notebook=json.loads(Path('notebooks/ShuffleRec.ipynb').read_text(encoding='utf-8'))
    assert any('RUN_TRAINING' in ''.join(c.get('source',[])) for c in notebook['cells']), 'notebook is not integrated'
    project=tmp_path/'project';(project/'data').mkdir(parents=True);(project/'sensitive_info').mkdir()
    (project/'shufflerec').mkdir()  # root detection; imports use real project on sys.path
    catalog=pd.DataFrame([dict(id=t,tempo=100+i*10,artists=t) for i,t in enumerate('abcd')])
    history=pd.DataFrame([dict(track_id=t,event_type='PLAY' if t!='d' else 'NEUTRAL',timestamp=f'2020-01-{day:02d}T00:0{i}:00Z')
                          for day in range(1,11) for i,t in enumerate('abcd')])
    master=project/'data/master_track_table.csv';hist=project/'sensitive_info/spotify_listening_history.csv'
    catalog.to_csv(master,index=False);history.to_csv(hist,index=False)
    before=(master.read_bytes(),hist.read_bytes())
    monkeypatch.chdir(project)
    ns={}
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        exec(compile(source,'ShuffleRec.ipynb','exec'),ns)
        if 'RUN_TRAINING = False' in source:
            ns['RUN_TRAINING']=True
            ns['RUN_COMPARISON']=False
            ns['TRAINING_OPTIONS'].update(epochs=1,preference_epochs=0,batch_size=16)
            ns['MODEL_CONFIG']=ModelConfig(d_model=8,nhead=2,feedforward=16,dropout=0)
            ns['NUM_NEGATIVES']=1
    assert ns['training_result']['test_evaluated'] is False
    assert (master.read_bytes(),hist.read_bytes())==before
    assert list((project/'sensitive_info/model_runs').glob('*/model.pt'))


def test_notebook_comparison_uses_custom_variants_options_and_saves_results(tmp_path,monkeypatch):
    from pathlib import Path
    nb=json.loads(Path('notebooks/ShuffleRec.ipynb').read_text(encoding='utf-8'))
    project=tmp_path/'project';(project/'data').mkdir(parents=True)
    (project/'sensitive_info').mkdir();(project/'shufflerec').mkdir()
    frame=pd.DataFrame([dict(id=t,artists=t,tempo=100+i*10,duration_ms=100000+i*1000,
                             duration_ms_is_estimated=(t=='d')) for i,t in enumerate('abcd')])
    history=pd.DataFrame([dict(track_id=t,event_type='NEUTRAL' if t=='d' else 'PLAY',
                              timestamp=f'2020-01-{day:02d}T00:0{i}:00Z')
                          for day in range(1,11) for i,t in enumerate('abcd')])
    master=project/'data/master_track_table.csv';hist=project/'sensitive_info/spotify_listening_history.csv'
    frame.to_csv(master,index=False);history.to_csv(hist,index=False)
    before=(master.read_bytes(),hist.read_bytes());monkeypatch.chdir(project)
    ns={}
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source']);exec(compile(source,'notebook-comparison','exec'),ns)
        if 'RUN_TRAINING = False' in source:
            ns.update(RUN_TRAINING=False,RUN_COMPARISON=True,RUN_AUDIT=False,NUM_NEGATIVES=1,MAX_EVAL_EXAMPLES=2)
            ns['TRAINING_OPTIONS'].update(epochs=1,preference_epochs=0,batch_size=16,learning_rate=0.002,patience=2)
            ns['MODEL_CONFIG']=ModelConfig(d_model=8,nhead=2,feedforward=16,dropout=0)
            ns['COMPARISON_VARIANTS']={'artist_only':FeatureConfig(local=False,provenance=False)}
    assert [r['variant'] for r in ns['comparison_results']]==['artist_only']
    assert ns['comparison_results'][0]['training_options']['learning_rate']==0.002
    reports=list((project/'sensitive_info/model_runs').glob('*/comparison_results.json'))
    assert len(reports)==1
    saved=json.loads(reports[0].read_text(encoding='utf-8'))
    assert saved['results'][0]['test_evaluated'] is False
    assert saved['candidate_scope']=='history'
    assert (master.read_bytes(),hist.read_bytes())==before


def test_individual_local_variants_change_exactly_one_local_measurement():
    m=comparison_api()
    variants=m.individual_local_variants()
    assert len(variants)==6
    baseline=variants['no_local']
    assert baseline.enabled_local_features==()
    for feature in ('local_rms_db','local_onset_strength','local_spectral_centroid',
                    'local_spectral_contrast','local_beat_regularity'):
        config=variants['add_'+feature]
        assert config.enabled_local_features==(feature,)
        assert config.provenance==baseline.provenance


def test_estimated_duration_variants_make_policy_choice_explicit():
    m=comparison_api();variants=m.estimated_duration_variants()
    assert len(variants)==2
    assert variants['without_estimated_durations'].include_estimated_durations is False
    assert variants['with_estimated_durations'].include_estimated_durations is True
    assert all(config.duration for config in variants.values())


def test_comparison_pauses_only_between_completed_model_fits(monkeypatch):
    m=comparison_api();events=[]
    monkeypatch.setattr(m,'sleep',lambda seconds:events.append(('pause',seconds)),raising=False)
    catalog=pd.DataFrame([dict(id=t,tempo=100+i*20,artists=t) for i,t in enumerate('abcd')])
    history=pd.DataFrame([dict(track_id=t,event_type='NEUTRAL' if t=='d' else 'PLAY',
                              timestamp=f'2020-01-{day:02d}T00:0{i}:00Z')
                          for day in range(1,11) for i,t in enumerate('abcd')])
    config=FeatureConfig(local=False,provenance=False)
    m.run_comparison(catalog,history,variants={'first':config,'second':config},
        model_config=ModelConfig(d_model=8,nhead=2,feedforward=16,dropout=0),epochs=1,
        preference_epochs=0,num_negatives=1,max_examples=1,logger=lambda r:events.append(('log',r)))
    pauses=[i for i,event in enumerate(events) if event[0]=='pause']
    assert len(pauses)==1 and events[pauses[0]]==('pause',10)
    completed_first=next(i for i,event in enumerate(events) if event[0]=='log' and
                          event[1].get('variant')=='first' and event[1].get('epoch')==1)
    started_second=next(i for i,event in enumerate(events) if event[0]=='log' and
                          event[1].get('variant')=='second' and 'parameters' in event[1])
    assert completed_first<pauses[0]<started_second


@pytest.mark.parametrize('seconds',[-1,float('nan'),float('inf')])
def test_comparison_rejects_invalid_cooldown_before_loading_data(seconds):
    m=comparison_api()
    with pytest.raises(ValueError,match='cooldown'):
        m.run_comparison(None,None,cooldown_seconds=seconds)
