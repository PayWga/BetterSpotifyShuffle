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
                            epochs=1,preference_epochs=0,num_negatives=1,max_examples=2,seed=8)
    assert len(result)==2
    assert result[0]['evaluation_fingerprint']==result[1]['evaluation_fingerprint']
    assert all(r['test_evaluated'] is False for r in result)
    assert all(r['validation']['examples']==2 for r in result)


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
            ns['TRAINING_OPTIONS'].update(epochs=1,preference_epochs=0,batch_size=16)
            ns['MODEL_CONFIG']=ModelConfig(d_model=8,nhead=2,feedforward=16,dropout=0)
            ns['NUM_NEGATIVES']=1
    assert ns['training_result']['test_evaluated'] is False
    assert (master.read_bytes(),hist.read_bytes())==before
    assert list((project/'sensitive_info/model_runs').glob('*/model.pt'))
