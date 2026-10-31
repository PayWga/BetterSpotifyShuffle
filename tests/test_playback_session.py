import importlib

import pandas as pd
import pytest

from shufflerec.inference import RankedTrack
from shufflerec.playback import Snapshot
from shufflerec.sequencing import Sequencer


def api():
    assert importlib.util.find_spec('shufflerec.playback_session'), 'playback coordinator missing'
    return importlib.import_module('shufflerec.playback_session')


class Engine:
    def __init__(self):
        self.catalog = pd.DataFrame([dict(id=str(i), duration_ms=180000, artists='A', genre='rock')
                                     for i in range(20)])
    def usable_context(self, ids):
        return list(ids)
    def rank(self, context, ids):
        return [RankedTrack(t, 0.) for t in sorted(ids)]


def setup_session():
    m = api()
    seq = Sequencer(Engine(), [str(i) for i in range(20)], context_ids=['0'])
    seq.plan()
    session = m.PlaybackSession(seq)
    submitted = session.submit_prefix(3)
    return seq, session, submitted


def observation(track, time, progress=0):
    return Snapshot(track, 180000, progress, True, time, 'pc', 'playlist:x')


def test_submitted_prefix_is_frozen_and_never_reselected_by_replanning():
    seq, session, submitted = setup_session()
    assert len(submitted) == 3
    seq.plan()
    assert not {t.track_id for t in submitted} & {t.track_id for t in seq.queue}
    assert seq.fatigue == [0., 0., 0.]
    assert len(session.submitted) == 3


def test_acceptance_is_rewarded_once_while_song_remains_active():
    seq, session, submitted = setup_session()
    a, b, _ = submitted
    session.observe(observation(a.track_id, 0))
    session.observe(observation(a.track_id, 45, 45000), continuity=True)
    assert session.tracker.active.track_id == a.track_id
    before = seq.policy.preferences.copy()
    assert sum(before) == pytest.approx(.15)
    assert seq.context_ids[-1] == a.track_id
    session.observe(observation(b.track_id, 75), continuity=True)
    assert seq.policy.preferences == before  # Track A was already accepted and cannot receive another penalty.


def test_inferred_weak_rejection_gets_half_penalty_and_full_skip_gets_full_penalty():
    seq, session, submitted = setup_session()
    a, b, c = submitted
    session.observe(observation(a.track_id, 0))
    session.observe(observation(a.track_id, 45, 45000), continuity=True)
    before = seq.policy.preferences.copy()
    events = session.observe(observation(c.track_id, 120), continuity=True)
    b_event = next(e for e in events if e.track_id == b.track_id)
    assert b_event.weak and b_event.inferred and b_event.weight == .5
    index = ('random', 'top', 'filtered').index(b.pool)
    assert seq.policy.preferences[index] - before[index] == pytest.approx(-.075)
    assert b.track_id not in seq.context_ids
    with pytest.raises(ValueError):
        seq.record_selection_feedback(b.selection_id, 'rejected', weight=.5)


def test_only_submitted_occurrences_can_earn_pool_feedback():
    seq, session, submitted = setup_session()
    before = seq.policy.preferences.copy()
    with pytest.raises(ValueError):
        seq.record_selection_feedback(seq.queue[0].selection_id, 'accepted')
    session.observe(observation('external', 0))
    session.observe(observation('external', 45, 45000))
    assert seq.policy.preferences == before
    assert seq.context_ids[-1] == 'external'


def test_tracker_uses_same_configured_threshold_as_sequencer():
    m = api()
    from shufflerec.pools import SequencerConfig
    seq = Sequencer(Engine(), ['0', '1', '2'], config=SequencerConfig(early_skip_threshold=.15))
    session = m.PlaybackSession(seq)
    assert session.tracker.config.acceptance_threshold == .15


def test_weighted_feedback_validates_input_before_mutation():
    api()
    from shufflerec.pools import PoolPolicy, SequencerConfig
    policy = PoolPolicy(SequencerConfig())
    for weight in (float('nan'), -.1, 1.1, True, 10 ** 400):
        with pytest.raises(ValueError):
            policy.observe('top', False, weight=weight)
    assert policy.preferences == [0., 0., 0.]
    policy.observe('top', False, weight=.5)
    assert policy.preferences == [0., -.075, 0.]


def test_offline_commands_can_acknowledge_queue_and_feed_playback_snapshots():
    m = api()
    from shufflerec.sequence import handle_request
    seq = Sequencer(Engine(), [str(i) for i in range(20)], context_ids=['0'])
    seq.plan()
    session = m.PlaybackSession(seq)
    event, result = handle_request(seq, {'command': 'submit', 'count': 3}, session=session)
    assert event == 'submitted'
    first = result['submitted'][0]['track_id']
    for time, progress in [(0, 0), (45, 45000)]:
        event, result = handle_request(seq, {'command': 'observe', 'continuity': True,
            'snapshot': dict(track_id=first, duration_ms=180000, progress_ms=progress,
                             is_playing=True, observed_at=time, device_id='pc', context_uri='x')},
            session=session)
    assert event == 'observed'
    assert result['playback_active']['track_id'] == first
    assert result['events'][0]['outcome'] == 'accepted'


def test_nonboolean_continuity_does_not_assert_verified_queue_order():
    seq, session, submitted = setup_session()
    session.observe(observation(submitted[0].track_id, 0))
    before = session.tracker.previous
    with pytest.raises(ValueError):
        session.observe(observation(submitted[2].track_id, 15), continuity='false')
    assert session.tracker.previous == before


def test_one_polling_loop_feeds_coordinator_and_recovers_submitted_skips():
    from scripts.playback_daemon import run_polling
    seq, session, submitted = setup_session()
    a, b, c = submitted
    now, events = [0.], []
    class Client:
        calls = 0
        def current_playback(self):
            self.calls += 1
            return {'item': {'id': a.track_id if self.calls == 1 else c.track_id, 'duration_ms': 180000},
                    'is_playing': True, 'progress_ms': 0, 'device': {'id': 'pc'},
                    'context': {'uri': 'x'}}
    def sleep(delay):
        now[0] += delay
    client = Client()
    run_polling(client, tracker=session, clock=lambda: now[0], sleeper=sleep, max_polls=2,
                event_sink=events.append, continuity_check=lambda snapshot: True)
    assert client.calls == 2
    inferred = next(e for e in events if e.track_id == b.track_id)
    assert inferred.inferred and inferred.weight == 1.
    assert sum(seq.policy.preferences) == pytest.approx(-.30)
    assert session.tracker.active is None
    assert session.submitted == []


def test_verified_prefix_can_anchor_skips_after_an_external_current_track():
    m = api()
    seq = Sequencer(Engine(), [str(i) for i in range(20)], context_ids=['0'])
    seq.plan()
    session = m.PlaybackSession(seq, auto_replan=False)
    session.observe(observation('external', 0))
    submitted = session.submit_prefix(3)
    session.confirm_order(follows_active=True)
    events = session.observe(observation(submitted[2].track_id, 15), continuity=True)
    inferred = [e for e in events if e.inferred]
    assert [e.selection_id for e in inferred] == [submitted[0].selection_id, submitted[1].selection_id]
    assert all(e.weight == 1. for e in inferred)


def test_abandonment_releases_unheard_submissions_without_penalties():
    seq, session, submitted = setup_session()
    before = seq.policy.preferences.copy()
    session.abandon([item.selection_id for item in submitted])
    assert session.submitted == []
    assert seq.policy.preferences == before
    assert seq.fatigue == [0., 0., 0.]
    seq.plan()
    assert not seq._submitted


def test_planning_snapshot_anchors_after_committed_tail_without_mutating_session():
    seq, session, submitted = setup_session()
    session.observe(observation('external', 0))
    before = seq.policy.preferences.copy(), seq.fatigue.copy(), seq.context_ids.copy()
    clone, ids, pools = session.planning_snapshot(exclude=['19'])
    clone.plan(provisional_ids=ids, provisional_pools=pools, exclude=['19'])
    assert ids == ['external'] + [item.track_id for item in submitted]
    assert pools == [item.pool for item in submitted]
    assert '19' not in {item.track_id for item in clone.queue}
    assert (seq.policy.preferences, seq.fatigue, seq.context_ids) == before
