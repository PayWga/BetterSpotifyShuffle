"""Cache checks use synthetic catalogs and count actual encoder work."""
from types import SimpleNamespace
import math

import pandas as pd
import pytest
import torch

from shufflerec import live_state, training
from shufflerec.data import TrackStore
from shufflerec.features import FeaturePreprocessor
from shufflerec.model import ModelConfig, SpotifySASRec


def inputs():
    frame = pd.DataFrame([dict(id=track, tempo=80 + index * 20, artists=track)
                          for index, track in enumerate('abcd')])
    preprocessor = FeaturePreprocessor().fit(frame, frame.id)
    torch.manual_seed(12)
    model = SpotifySASRec(preprocessor, max_length=2,
                         config=ModelConfig(d_model=8, nhead=2, feedforward=16, dropout=0))
    return frame, preprocessor, model


def provider(monkeypatch):
    frame, preprocessor, model = inputs()
    playlist = SimpleNamespace(version='v1', track_ids=('a', 'b'), catalog=pd.DataFrame([
        dict(id='a', duration_ms=180000), dict(id='b', duration_ms=180000)]))
    cache = SimpleNamespace(load=lambda *args, **kwargs: playlist)
    monkeypatch.setattr(live_state, 'read_catalog', lambda path: frame)
    monkeypatch.setattr(live_state, 'load_checkpoint', lambda *args, **kwargs: (model, preprocessor, {}))
    return live_state.EngineProvider('synthetic-model', 'synthetic-master', cache), playlist, model


def test_live_refresh_reuses_last_engine_for_identical_resolved_rows(monkeypatch):
    source, playlist, model = provider(monkeypatch)
    sizes = []
    hook = model.track_encoder.register_forward_hook(lambda module, args, output: sizes.append(output.shape[0]))
    first = source.load('playlist', None, [])
    assert sizes == [2]
    playlist.version = 'v2'
    refreshed = source.load('playlist', None, [])
    assert refreshed.version == 'v2'
    assert refreshed.engine is first.engine
    assert sizes == [2]
    hook.remove()


def test_live_refresh_rebuilds_for_metadata_context_and_weight_changes(monkeypatch):
    source, playlist, model = provider(monkeypatch)
    first = source.load('playlist', None, [])
    playlist.catalog.loc[0, 'duration_ms'] = 190000
    changed = source.load('playlist', None, [])
    assert changed.engine is not first.engine
    assert changed.engine.catalog.set_index('id').loc['a', 'duration_ms'] == 190000
    contextual = source.load('playlist', None, ['c'])
    assert contextual.engine is not changed.engine
    assert 'c' in contextual.engine.index
    with torch.no_grad():
        model.track_encoder.projection.bias.add_(1)
    reweighted = source.load('playlist', None, ['c'])
    assert reweighted.engine is not contextual.engine


def plan():
    return training.EvaluationPlan((
        training.EvaluationCase(('a',), 'b', ('b', 'c', 'd')),
        training.EvaluationCase(('a',), 'c', ('d', 'b', 'c')),
    ), 7, 2, 'synthetic')


def direct_metrics(model, store, evaluation, k):
    hits, ndcg = [], []
    model.eval()
    with torch.no_grad():
        for case in evaluation.cases:
            context, padding = store.context(case.context, model.max_length)
            state = model({name: value.unsqueeze(0) for name, value in context.items()}, padding.unsqueeze(0))
            scores = model.score_candidates(state, store.take(case.candidates))[0].tolist()
            target = case.candidates.index(case.positive)
            rank = 1 + sum(score > scores[target] or score == scores[target] and i < target
                           for i, score in enumerate(scores))
            hits.append(float(rank <= k))
            ndcg.append(1 / math.log2(rank + 1) if rank <= k else 0.)
    return sum(hits) / len(hits), sum(ndcg) / len(ndcg)


@pytest.mark.parametrize('ties', [False, True])
def test_evaluation_encodes_repeated_candidates_once_and_preserves_metrics(ties):
    frame, preprocessor, model = inputs()
    store = TrackStore(frame, preprocessor)
    if ties:
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
    evaluation = plan()
    expected = direct_metrics(model, store, evaluation, 2)
    model.train()
    candidate_sizes = []
    def record(module, args, output):
        if output.ndim == 2:
            candidate_sizes.append(output.shape[0])
    hook = model.track_encoder.register_forward_hook(record)
    metrics = training.evaluate(model, store, evaluation, k=2)
    assert (metrics['hit_at_k'], metrics['ndcg_at_k']) == pytest.approx(expected)
    assert candidate_sizes == [3]
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    hook.remove()


def test_evaluation_candidate_cache_is_rebuilt_after_weights_change():
    frame, preprocessor, model = inputs()
    store = TrackStore(frame, preprocessor)
    evaluation = plan()
    training.evaluate(model, store, evaluation, k=2)
    with torch.no_grad():
        model.track_encoder.projection.weight.neg_()
    expected = direct_metrics(model, store, evaluation, 2)
    sizes = []
    hook = model.track_encoder.register_forward_hook(
        lambda module, args, output: sizes.append(output.shape[0]) if output.ndim == 2 else None)
    metrics = training.evaluate(model, store, evaluation, k=2)
    assert sizes == [3]
    assert (metrics['hit_at_k'], metrics['ndcg_at_k']) == pytest.approx(expected)
    hook.remove()


def test_large_evaluation_uses_uncached_scoring_and_same_metrics(monkeypatch):
    frame, preprocessor, model = inputs()
    store = TrackStore(frame, preprocessor)
    evaluation = plan()
    expected = direct_metrics(model, store, evaluation, 2)
    monkeypatch.setattr(training, 'EVALUATION_CACHE_TRACKS', 2)
    sizes = []
    hook = model.track_encoder.register_forward_hook(
        lambda module, args, output: sizes.append(output.shape[0]) if output.ndim == 2 else None)
    metrics = training.evaluate(model, store, evaluation, k=2)
    assert sizes == [3, 3]
    assert (metrics['hit_at_k'], metrics['ndcg_at_k']) == pytest.approx(expected)
    hook.remove()


def test_custom_scoring_on_standard_model_is_not_bypassed(monkeypatch):
    frame, preprocessor, model = inputs()
    store = TrackStore(frame, preprocessor)
    calls = []
    def custom_scores(state, candidates):
        calls.append(len(candidates['numeric']))
        return torch.zeros(1, len(candidates['numeric']))
    monkeypatch.setattr(model, 'score_candidates', custom_scores)
    metrics = training.evaluate(model, store, plan(), k=2)
    assert calls == [3, 3]
    assert metrics['hit_at_k'] == .5


def test_evaluation_restores_training_mode_after_nonfinite_scores():
    frame, preprocessor, model = inputs()
    store = TrackStore(frame, preprocessor)
    with torch.no_grad():
        model.track_encoder.projection.bias.fill_(float('nan'))
    model.train()
    with pytest.raises(FloatingPointError, match='Non-finite'):
        training.evaluate(model, store, plan())
    assert model.training
