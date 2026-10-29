import importlib

import pandas as pd
import pytest
import torch

from shufflerec.data import TrackStore
from shufflerec.features import FeatureConfig, FeaturePreprocessor


def api():
    assert importlib.util.find_spec('shufflerec.model') is not None, 'shared model encoder is not implemented'
    return importlib.import_module('shufflerec.model')


def store_and_preprocessor():
    frame=pd.DataFrame([dict(id='a',artists='Alice',tempo=100,key=11,time_signature=4),
                        dict(id='b',artists='Bob',tempo=130,key=0,time_signature=3),
                        dict(id='c',artists='Carol',tempo=90,key=5)])
    prep=FeaturePreprocessor().fit(frame,['a','b','c'])
    return TrackStore(frame,prep),prep


@pytest.mark.parametrize('length',[1,2,10])
def test_short_padded_contexts_have_finite_scores_and_gradients(length):
    m=api(); store,prep=store_and_preprocessor()
    model=m.SpotifySASRec(prep,max_length=10)
    context,pad=store.context(['a']*length,10)
    batch={k:v.unsqueeze(0) for k,v in context.items()}
    state=model(batch,pad.unsqueeze(0))
    scores=model.score_candidates(state,store.take(['b','c']))
    loss=-torch.nn.functional.logsigmoid(scores[0,0]-scores[0,1]); loss.backward()
    assert torch.isfinite(scores).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_shared_track_encoder_scores_candidates_and_padding_is_ignored():
    m=api(); store,prep=store_and_preprocessor()
    model=m.SpotifySASRec(prep,max_length=3).eval()
    ctx,pad=store.context(['a'],3)
    ctx={k:v.unsqueeze(0) for k,v in ctx.items()}
    with torch.no_grad():
        expected=model(ctx,pad.unsqueeze(0))
        changed={k:v.clone() for k,v in ctx.items()}
        changed['numeric'][:,:2]=999
        changed['artist'][:,:2]=2
        actual=model(changed,pad.unsqueeze(0))
        torch.testing.assert_close(actual,expected)
        scores=model.score_candidates(expected,store.take(['b','c']))
        torch.testing.assert_close(scores,expected @ model.track_encoder(store.take(['b','c'])).T)
    with pytest.raises(ValueError,match='empty'):
        model(ctx,torch.ones(1,3,dtype=torch.bool))


def test_artist_pooling_does_not_depend_on_collaborator_order():
    m=api(); store,prep=store_and_preprocessor(); model=m.SpotifySASRec(prep)
    inputs=store.take(['a']); inputs['artist'][0,:2]=torch.tensor([2,3])
    reversed_inputs={k:v.clone() for k,v in inputs.items()}
    reversed_inputs['artist'][0,:2]=torch.tensor([3,2])
    torch.testing.assert_close(model.track_encoder(inputs),model.track_encoder(reversed_inputs))


def test_parameter_report_includes_actual_vocab_cost():
    m=api(); _,prep=store_and_preprocessor(); model=m.SpotifySASRec(prep)
    report=model.parameter_report()
    assert report['total']==sum(p.numel() for p in model.parameters())
    assert report['embeddings']['artist']==5*8
    assert report['numeric_inputs']==len(prep.numeric_names)*2
