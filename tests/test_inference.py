"""Inference must agree with direct model scoring without refitting features."""
import importlib

import pandas as pd
import pytest
import torch

from shufflerec.data import TrackStore
from shufflerec.features import FeaturePreprocessor
from shufflerec.model import ModelConfig, SpotifySASRec
from shufflerec.training import save_checkpoint


def api():
    assert importlib.util.find_spec('shufflerec.inference'), 'inference engine missing'
    return importlib.import_module('shufflerec.inference')


def fixture_model():
    frame = pd.DataFrame([
        dict(id='a', artists='Alice', tempo=80),
        dict(id='b', artists='Bob', tempo=100),
        dict(id='c', artists='Alice', tempo=120),
        dict(id='empty'),
    ])
    prep = FeaturePreprocessor().fit(frame, ['a', 'b', 'c'])
    torch.manual_seed(7)
    model = SpotifySASRec(prep, max_length=3,
                         config=ModelConfig(d_model=8, nhead=2, feedforward=16, dropout=0))
    return frame, prep, model


def test_cached_scores_match_direct_scoring_and_frozen_checkpoint(tmp_path):
    m = api()
    frame, prep, model = fixture_model()
    path = tmp_path / 'model.pt'
    save_checkpoint(path, model, prep)
    before = prep.state_dict()
    original = frame.copy(deep=True)
    engine = m.InferenceEngine.from_checkpoint(path, frame, batch_size=1)
    ranked = engine.rank(['a'], ['c', 'b', 'empty', 'unknown', 'b'])
    store = TrackStore(frame, prep)
    ctx, pad = store.context(['a'], 3)
    model.eval()
    with torch.inference_mode():
        state = model({k: v.unsqueeze(0) for k, v in ctx.items()}, pad.unsqueeze(0))
        direct = model.score_candidates(state, store.take(['b', 'c']))[0]
    assert {r.track_id for r in ranked} == {'b', 'c'}
    assert {r.track_id: r.score for r in ranked} == pytest.approx(dict(zip(['b', 'c'], direct.tolist())))
    assert engine.preprocessor.state_dict() == before
    pd.testing.assert_frame_equal(frame, original)
    assert not engine.model.training
    assert all(p.grad is None for p in engine.model.parameters())


def test_cache_batches_candidates_and_does_not_encode_them_again():
    m = api()
    frame, prep, model = fixture_model()
    sizes = []
    handle = model.track_encoder.register_forward_hook(
        lambda module, args, output: sizes.append(tuple(output.shape)))
    engine = m.InferenceEngine(model, prep, frame, batch_size=2)
    assert sizes == [(2, 8), (1, 8)]
    sizes.clear()
    first = engine.rank(['a'], ['b', 'c'])
    second = engine.rank(['a'], ['b', 'c'])
    assert first == second
    assert sizes == [(1, 3, 8), (1, 3, 8)]  # Only context is encoded; candidate vectors are already cached.
    handle.remove()


def test_unknown_or_unusable_context_resets_history_and_exclusions_are_applied():
    m = api()
    frame, prep, model = fixture_model()
    engine = m.InferenceEngine(model, prep, frame)
    assert engine.rank(['a', 'unknown', 'b'], ['a', 'c']) == engine.rank(['b'], ['a', 'c'])
    assert engine.rank(['a', 'empty', 'b'], ['a', 'c']) == engine.rank(['b'], ['a', 'c'])
    for context in ([], ['unknown'], ['a', 'empty']):
        assert engine.rank(context, ['b', 'c']) == []
    assert [r.track_id for r in engine.rank(['a'], ['b', 'c'], exclude=['b'])] == ['c']


def test_ties_use_ids_and_nonfinite_embeddings_are_rejected():
    m = api()
    frame, prep, model = fixture_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    engine = m.InferenceEngine(model, prep, frame)
    assert [r.track_id for r in engine.rank(['a'], ['c', 'b'])] == ['b', 'c']
    with torch.no_grad():
        model.track_encoder.projection.bias.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        m.InferenceEngine(model, prep, frame)


def test_invalid_catalog_and_batch_size_are_rejected():
    m = api()
    frame, prep, model = fixture_model()
    with pytest.raises(ValueError, match='batch_size'):
        m.InferenceEngine(model, prep, frame, batch_size=0)
    with pytest.raises(ValueError, match='duplicate'):
        m.InferenceEngine(model, prep, pd.concat([frame, frame.iloc[:1]]))
