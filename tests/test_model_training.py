import importlib
import math

import pandas as pd
import pytest
import torch

from shufflerec.data import Example, SequenceDataset, TrackStore, prepare_data
from shufflerec.features import FeaturePreprocessor
from shufflerec.model import ModelConfig, SpotifySASRec


def api():
    assert importlib.util.find_spec('shufflerec.training') is not None, 'training/evaluation is not implemented'
    return importlib.import_module('shufflerec.training')


def fixture_data():
    catalog=pd.DataFrame([dict(id=t,artists=t,tempo=80+i*20,key=i) for i,t in enumerate('abcd')])
    rows=[]
    for day in range(1,11):
        for minute,(track,event) in enumerate([('a','PLAY'),('d','GHOST_SKIP'),('b','PLAY'),('c','PLAY')]):
            rows.append(dict(track_id=track,event_type=event,timestamp=f'2020-01-{day:02d}T00:0{minute}:00Z'))
    return prepare_data(catalog,pd.DataFrame(rows),max_length=3)


def test_evaluation_plan_is_seeded_and_excludes_context_and_target_from_negatives():
    m=api(); examples=[Example(('a',),'b',pd.Timestamp('2020-01-01',tz='UTC'))]
    first=m.build_evaluation_plan(examples,list('abcdef'),num_negatives=3,seed=7)
    assert first==m.build_evaluation_plan(examples,list(reversed('abcdef')),num_negatives=3,seed=7)
    assert len(first.cases[0].candidates)==4
    assert set(first.cases[0].candidates) & {'a','b'}=={'b'}
    with pytest.raises(ValueError,match='negative'):
        m.build_evaluation_plan(examples,['a','b'],num_negatives=3)


def test_metrics_use_target_rank_not_its_position_in_candidates():
    m=api(); frame=pd.DataFrame([dict(id='a',tempo=80),dict(id='b',tempo=100),dict(id='c',tempo=120)])
    p=FeaturePreprocessor().fit(frame,list('abc')); store=TrackStore(frame,p)
    tempo=p.numeric_names.index('tempo')
    class TempoRanker(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.marker=torch.nn.Parameter(torch.zeros(1));self.max_length=2
        def forward(self,context,padding):
            return torch.zeros(1,1)
        def score_candidates(self,state,tracks):
            return tracks['numeric'][:,tempo].unsqueeze(0)
    plan=m.EvaluationPlan((m.EvaluationCase(('a',),'b',('b','c','a')),),7,2,'synthetic')
    model=TempoRanker().train()
    metrics=m.evaluate(model,store,plan,k=2)
    assert metrics['hit_at_k']==1
    assert metrics['ndcg_at_k']==pytest.approx(1/math.log2(3))
    assert metrics['mean_candidates']==3
    assert model.training  # evaluation must restore prior state


def test_training_and_preference_phase_use_training_only_and_roundtrip_checkpoint(tmp_path):
    m=api(); data=fixture_data()
    m.seed_everything(5)
    config=ModelConfig(d_model=16,nhead=2,feedforward=32,dropout=0)
    model=SpotifySASRec(data.preprocessor,max_length=3,config=config)
    before={k:v.clone() for k,v in model.state_dict().items()}
    plan=m.build_evaluation_plan(data.examples['validation']['nip'],data.store.usable_ids,num_negatives=1)
    result=m.fit(model,data,plan,epochs=1,preference_epochs=1,batch_size=16,seed=5)
    assert [row['phase'] for row in result['history']]==['next_item','preference']
    assert result['training_examples']=={'next_item':16,'preference':8}
    assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
    path=tmp_path/'model.pt';m.save_checkpoint(path,model,data.preprocessor,metadata={'validation':result['best_validation']})
    restored,prep,metadata=m.load_checkpoint(path)
    restored.eval();model.eval()
    ctx,pad=data.store.context(['a'],3);ctx={k:v.unsqueeze(0) for k,v in ctx.items()}
    with torch.no_grad():
        torch.testing.assert_close(model(ctx,pad.unsqueeze(0)),restored(ctx,pad.unsqueeze(0)))
    assert prep.state_dict()==data.preprocessor.state_dict()
    assert metadata['validation']==result['best_validation']


def test_invalid_training_and_overwriting_checkpoint_are_rejected(tmp_path):
    m=api(); data=fixture_data();model=SpotifySASRec(data.preprocessor,max_length=3)
    plan=m.build_evaluation_plan(data.examples['validation']['nip'],data.store.usable_ids,num_negatives=1)
    with pytest.raises(ValueError):
        m.fit(model,data,plan,epochs=0)
    path=tmp_path/'exists.pt';path.write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        m.save_checkpoint(path,model,data.preprocessor)
    assert path.read_bytes()==b'keep'


def test_sampler_uses_prebuilt_membership_for_large_catalogs():
    # Scanning every catalog row for each training example would dominate runtime.
    from shufflerec import data
    import random
    class IndexedOnly(tuple):
        def __iter__(self):
            raise AssertionError('per-example full-catalog scan')
    pool=IndexedOnly(range(1000));membership=frozenset(range(1000))
    picks=data.sample_negatives(pool,{0,1},1,random.Random(7),membership=membership)
    assert len(picks)==1 and picks[0] not in {0,1}


def test_test_partition_plan_cannot_select_weights_even_for_repeated_sequences():
    m=api(); data=fixture_data(); model=SpotifySASRec(data.preprocessor,max_length=3)
    test_plan=m.build_evaluation_plan(data.examples['test']['nip'],data.store.usable_ids,num_negatives=1)
    with pytest.raises(ValueError,match='validation'):
        m.fit(model,data,test_plan,epochs=1,preference_epochs=0)
