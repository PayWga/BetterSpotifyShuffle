"""Observable sampling and queue lifecycle contracts, using synthetic tracks."""
import importlib
import random

import pandas as pd
import pytest

from shufflerec.inference import RankedTrack


def api():
    assert importlib.util.find_spec('shufflerec.sequencing'), 'sequencer missing'
    return importlib.import_module('shufflerec.sequencing')


class RankedFixture:
    """A context-sensitive rank source with explicit missing-context behavior."""
    def __init__(self, size=20):
        self.catalog = pd.DataFrame([
            dict(id=str(i), artists='Artist ' + str(i % 3),
                 artist_ids='ID' + str(i % 3), genre='rock' if i % 2 else 'pop')
            for i in range(size)
        ])
        self.contexts = []

    def usable_context(self, context):
        usable = []
        for track in context:
            if track in set(self.catalog.id):
                usable.append(track)
            else:
                usable.clear()
        return usable

    def rank(self, context, candidates, *, exclude=()):
        self.contexts.append(tuple(context))
        if not self.usable_context(context):
            return []
        eligible = set(candidates) & set(self.catalog.id) - set(exclude)
        return [RankedTrack(t, 100. - int(t)) for t in sorted(eligible, key=int)]


def test_final_probabilities_respect_bounds_and_available_pool_fallback():
    m = api()
    policy = m.PoolPolicy(m.SequencerConfig())
    assert policy.probabilities(['random', 'top', 'filtered'], [0., 0., 0.]) == pytest.approx(
        {'random': 1 / 3, 'top': 1 / 3, 'filtered': 1 / 3})
    for _ in range(100):
        policy.observe('top', True)
        policy.observe('filtered', False)
    p = policy.probabilities(['random', 'top', 'filtered'], [0., 0., 0.])
    assert sum(p.values()) == pytest.approx(1)
    assert all(.05 <= v <= .75 for v in p.values())
    assert p['top'] == pytest.approx(.75)
    assert policy.probabilities(['random'], [100., 0., 0.]) == {'random': 1.}
    assert policy.probabilities([], [0., 0., 0.]) == {}
    p = policy.probabilities(['random', 'top'], [0., 100., 0.])
    assert sum(p.values()) == pytest.approx(1)
    assert all(.05 <= v <= .75 for v in p.values())


def test_feedback_preference_and_temporary_fatigue_are_independent():
    m = api()
    policy = m.PoolPolicy(m.SequencerConfig())
    before = policy.probabilities(['random', 'top', 'filtered'], [0., 0., 0.])
    fatigue = policy.advance([0., 0., 0.], 'top')
    after_use = policy.probabilities(['random', 'top', 'filtered'], fatigue)
    assert after_use['top'] < before['top']
    policy.observe('top', True)
    reinforced = policy.probabilities(['random', 'top', 'filtered'], [0., 0., 0.])
    assert reinforced['top'] > before['top']
    recovered = policy.advance(fatigue, 'random')
    assert recovered[1] < fatigue[1]


def test_expansion_preserves_artist_weights_and_gently_weights_genre_only():
    m = api()
    from shufflerec.pools import build_pools
    frame = pd.DataFrame([
        dict(id='anchor', artists='A', artist_ids='X', genre='rock'),
        dict(id='1', artists='Z', artist_ids='Z', genre='jazz'),
        dict(id='2', artists='A', artist_ids='X', genre='jazz'),
        dict(id='3', artists='G', artist_ids='G', genre='rock'),
        dict(id='4', artists='A', artist_ids='X', genre='rock'),
    ])
    config = m.SequencerConfig(initial_candidates=2, max_candidates=4, min_filtered=3, top_k=2)
    ranked = [RankedTrack(str(i), 10-i) for i in range(1, 5)]
    pools, depth = build_pools(frame, 'anchor', ['1', '2', '3', '4'], ranked, config)
    assert depth == 4
    assert pools['random'] == [('1', 1.), ('2', 1.), ('3', 1.), ('4', 1.)]
    assert pools['top'] == [('1', 1.), ('2', .5)]
    assert dict(pools['filtered']) == pytest.approx({'2': 1., '3': .5, '4': 1.})


def test_artist_ids_override_names_and_genre_types_do_not_cross_match():
    m = api()
    from shufflerec.pools import build_pools
    frame = pd.DataFrame([
        dict(id='a', artists='Same', artist_ids='Case', genre='rock', artist_genres='metal'),
        dict(id='b', artists='Same', artist_ids='case', genre='jazz'),
        dict(id='c', artists=' same ', genre='jazz'),
        dict(id='d', artists='Other', genre='metal'),
        dict(id='e', artists='Other', artist_genres='metal'),
    ])
    ranked = [RankedTrack(t, 0) for t in 'bcde']
    pools, _ = build_pools(frame, 'a', list('bcde'), ranked, m.SequencerConfig())
    assert dict(pools['filtered']) == pytest.approx({'c': 1., 'e': .375})


def test_plan_is_seeded_sequential_unique_and_has_no_committed_side_effects():
    m = api()
    engine = RankedFixture()
    seq = m.Sequencer(engine, [str(i) for i in range(20)], context_ids=['0'], seed=5)
    queue = seq.plan()
    assert len(queue) == 5
    assert len({item.track_id for item in queue}) == 5
    assert '0' not in {item.track_id for item in queue}
    assert len(engine.contexts) == 5
    assert engine.contexts[1] == ('0', queue[0].track_id)
    assert seq.context_ids == ['0']
    assert seq.fatigue == [0., 0., 0.]
    assert seq.policy.preferences == [0., 0., 0.]
    other = m.Sequencer(RankedFixture(), [str(i) for i in range(20)], context_ids=['0'], seed=5)
    assert queue == other.plan()
    seq.plan()
    assert seq.fatigue == [0., 0., 0.]
    assert seq.policy.preferences == [0., 0., 0.]


@pytest.mark.parametrize('completion,reason,outcome', [
    (.2499, 'track_change', 'rejected'), (.25, 'track_change', 'accepted'),
    (1., 'track_end', 'accepted'), (.01, 'interrupted', 'neutral'),
])
def test_feedback_boundary_attribution_and_replanning(completion, reason, outcome):
    m = api()
    seq = m.Sequencer(RankedFixture(), [str(i) for i in range(20)], context_ids=['0'])
    seq.plan()
    active = seq.start_next()
    discarded_ids = {q.selection_id for q in seq.queue}
    fatigue = seq.fatigue.copy()
    result = seq.feedback(active.selection_id, completion, reason_end=reason)
    assert result == outcome
    assert seq.active is None
    assert seq.fatigue == fatigue
    assert not discarded_ids & {q.selection_id for q in seq.queue}
    assert active.track_id not in {q.track_id for q in seq.queue}
    assert (active.track_id in seq.context_ids) == (outcome == 'accepted')
    changed = [v for v in seq.policy.preferences if v != 0]
    if outcome == 'neutral':
        assert changed == []
    else:
        assert len(changed) == 1
        assert (changed[0] > 0) == (outcome == 'accepted')
    with pytest.raises(ValueError, match='active'):
        seq.feedback(active.selection_id, completion, reason_end=reason)


def test_invalid_feedback_cannot_mutate_or_advance_active_selection():
    m = api()
    seq = m.Sequencer(RankedFixture(), ['0', '1', '2'], context_ids=['0'])
    seq.plan()
    active = seq.start_next()
    before = seq.status()
    for identity, completion in [(999, .5), (active.selection_id, float('nan')),
                                  (active.selection_id, 1.1), (active.selection_id, -.1)]:
        with pytest.raises(ValueError):
            seq.feedback(identity, completion, reason_end='track_change')
        assert seq.status() == before
    with pytest.raises(ValueError, match='active'):
        seq.start_next()
    with pytest.raises(ValueError, match='active'):
        seq.plan()


def test_no_context_or_inference_failure_reports_singleton_fallback():
    m = api()
    seq = m.Sequencer(RankedFixture(2), ['0', '1', 'missing'], seed=7)
    queue = seq.plan()
    assert len(queue) == 3
    assert queue[0].probabilities == {'random': 1.}
    assert 'single_available_pool' in queue[0].diagnostics
    assert 'no_model_context' in queue[0].diagnostics
    assert 'playlist_exhausted' in seq.diagnostics
    class Broken(RankedFixture):
        def rank(self, *args, **kwargs):
            raise FloatingPointError('bad scores')
    seq = m.Sequencer(Broken(), ['1', '2'], context_ids=['0'])
    assert all('inference_unavailable' in q.diagnostics for q in seq.plan())


@pytest.mark.parametrize('kwargs', [
    {'queue_size': 0}, {'queue_size': True}, {'min_probability': 0},
    {'max_probability': .9}, {'max_probability': .4},
    {'initial_candidates': 100, 'max_candidates': 50},
    {'genre_weight_low': 0}, {'genre_weight_high': 2},
    {'fatigue_decay': float('nan')}, {'base_weights': [1, 0, 1]},
    {'early_skip_threshold': -1}, {'recent_window': -1},
    {'fatigue_strength': 10 ** 400}, {'base_weights': [10 ** 400, 1, 1]},
])
def test_invalid_configuration_is_rejected(kwargs):
    m = api()
    with pytest.raises(ValueError):
        m.SequencerConfig(**kwargs)


def test_randomized_initial_weights_remain_seeded_and_bounded():
    m = api()
    config = m.SequencerConfig(randomize_initial_weights=True)
    a = m.PoolPolicy(config, random.Random(9))
    b = m.PoolPolicy(config, random.Random(9))
    p = a.probabilities(['random', 'top', 'filtered'], [0., 0., 0.])
    assert p == b.probabilities(['random', 'top', 'filtered'], [0., 0., 0.])
    assert len(set(p.values())) > 1
    assert all(.05 <= value <= .75 for value in p.values())


@pytest.mark.parametrize('options,fatigue', [
    ({'learning_rate': 1e30, 'preference_limit': 1e30}, [0., 0., 0.]),
    ({'fatigue_strength': 1e308}, [3., 0., 0.]),
    ({'fatigue_strength': 1e308}, [0., 3., 4.]),
    ({'base_weights': (1e308, 1e308, 1e308), 'randomize_initial_weights': True,
      'initial_weight_low': 2., 'initial_weight_high': 2.}, [0., 0., 0.]),
])
def test_extreme_finite_tuning_keeps_actual_sampling_probabilities_bounded(options, fatigue):
    m = api()
    policy = m.PoolPolicy(m.SequencerConfig(**options), random.Random(1))
    policy.observe('top', True)
    probabilities = policy.probabilities(['random', 'top', 'filtered'], fatigue)
    assert sum(probabilities.values()) == pytest.approx(1., abs=1e-12)
    assert all(.05 <= value <= .75 for value in probabilities.values())


def test_oversized_feedback_integer_is_rejected_without_losing_active_item():
    m = api()
    seq = m.Sequencer(RankedFixture(), ['0', '1', '2'], context_ids=['0'])
    seq.plan()
    active = seq.start_next()
    before = seq.status()
    with pytest.raises(ValueError, match='completion_rate'):
        seq.feedback(active.selection_id, 10 ** 400)
    assert seq.status() == before
