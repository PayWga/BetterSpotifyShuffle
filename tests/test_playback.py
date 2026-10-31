"""Timing evidence must never manufacture precise listening durations."""
import importlib

import pytest


def api():
    assert importlib.util.find_spec('shufflerec.playback'), 'playback tracker missing'
    return importlib.import_module('shufflerec.playback')


def snap(m, track, when, progress=0, duration=180000, playing=True, device='pc', context='playlist:x'):
    return m.Snapshot(track, duration, progress, playing, when, device, context)


def test_acceptance_changes_schedule_without_finishing_or_rewarding_twice():
    m = api(); tracker = m.PlaybackTracker()
    assert tracker.observe(snap(m, 'a', 0))[-1].kind == 'started'
    assert tracker.next_delay() == 15
    tracker.observe(snap(m, 'a', 15, 15000))
    tracker.observe(snap(m, 'a', 30, 30000))
    events = tracker.observe(snap(m, 'a', 45, 45000))
    assert [(e.kind, e.outcome, e.weight) for e in events] == [('feedback', 'accepted', 1.)]
    assert tracker.active.track_id == 'a'
    assert tracker.next_delay() == 75
    assert tracker.observe(snap(m, 'a', 120, 120000)) == []
    assert tracker.next_delay() == 58  # Check just before the predicted ending.
    tracker.observe(snap(m, 'a', 179, 179000))
    assert tracker.next_delay() == 1
    ended = tracker.observe(snap(m, 'b', 180, 0))
    assert not any(e.kind == 'feedback' and e.track_id == 'a' for e in ended)
    assert [e for e in ended if e.track_id == 'a'][0].outcome == 'accepted'


@pytest.mark.parametrize('gap,duration,outcome,weight,weak', [
    (15, 180000, 'rejected', 1., False),
    (75, 180000, 'rejected', .5, True),
    (75, 360000, 'rejected', 1., False),
    (75, 60000, 'unresolved', 0., False),
    (15, None, 'unresolved', 0., False),
])
def test_intervening_submitted_tracks_use_conservative_gap_bounds(gap, duration, outcome, weight, weak):
    m = api(); tracker = m.PlaybackTracker()
    tracker.set_submitted([m.SubmittedTrack(1, 'a', 180000), m.SubmittedTrack(2, 'b', duration),
                           m.SubmittedTrack(3, 'c', 180000)], version='v1')
    tracker.observe(snap(m, 'a', 0))
    events = tracker.observe(snap(m, 'c', gap), continuity=True)
    event = next(e for e in events if e.track_id == 'b')
    assert (event.kind, event.outcome, event.weight, event.weak) == ('ended', outcome, weight, weak)
    assert event.inferred
    assert event.listened_ms == 0
    assert event.upper_bound_ms == gap * 1000
    assert event.gap_ms == gap * 1000
    assert event.selection_id == 2


@pytest.mark.parametrize('changes', [dict(device='phone'), dict(context='playlist:y')])
def test_device_or_context_change_prevents_queue_inference(changes):
    m = api(); tracker = m.PlaybackTracker()
    tracker.set_submitted([m.SubmittedTrack(i, t, 180000) for i, t in enumerate('abc', 1)], version='v')
    tracker.observe(snap(m, 'a', 0))
    events = tracker.observe(snap(m, 'c', 15, **changes), continuity=True)
    assert not any(e.track_id == 'b' for e in events)
    assert next(e for e in events if e.track_id == 'a').outcome == 'unresolved'


def test_unconfirmed_queue_duplicate_ids_and_outages_do_not_invent_skips():
    m = api()
    for mode in ('unconfirmed', 'duplicates', 'outage'):
        tracker = m.PlaybackTracker()
        tracks = 'abcc' if mode == 'duplicates' else 'abc'
        tracker.set_submitted([m.SubmittedTrack(i, t, 180000) for i, t in enumerate(tracks, 1)], version='v')
        tracker.observe(snap(m, 'a', 0))
        if mode == 'outage':
            tracker.interrupt()
        events = tracker.observe(snap(m, 'c', 15), continuity=mode != 'unconfirmed')
        assert not any(e.track_id == 'b' for e in events)


def test_seeking_does_not_count_as_listening_and_pause_uses_slow_schedule():
    m = api(); tracker = m.PlaybackTracker()
    tracker.observe(snap(m, 'a', 0))
    events = tracker.observe(snap(m, 'a', 15, 100000))
    assert not any(e.outcome == 'accepted' for e in events)
    assert tracker.active.listened_ms == 0
    tracker.observe(snap(m, 'a', 20, 100000, playing=False))
    assert tracker.next_delay() == 30
    events = tracker.observe(snap(m, 'a', 50, 100000, playing=True))
    assert not any(e.outcome == 'accepted' for e in events)
    assert tracker.active.listened_ms == 0


def test_same_track_replay_creates_new_occurrence_and_does_not_reuse_acceptance():
    m = api(); tracker = m.PlaybackTracker()
    tracker.observe(snap(m, 'a', 0))
    tracker.observe(snap(m, 'a', 170, 170000))
    events = tracker.observe(snap(m, 'a', 180, 0))
    assert [e.kind for e in events] == ['ended', 'started']
    assert not tracker.active.accepted
    assert tracker.active.listened_ms == 0


def test_out_of_order_observations_and_invalid_values_leave_state_unchanged():
    m = api(); tracker = m.PlaybackTracker()
    tracker.observe(snap(m, 'a', 15))
    for when, progress in [(14, 0), (float('nan'), 0), (16, -1)]:
        with pytest.raises(ValueError):
            tracker.observe(snap(m, 'b', when, progress))
    assert tracker.active.track_id == 'a'
    with pytest.raises(ValueError):
        m.PlaybackConfig(max_gap_seconds=0)


def test_first_upcoming_submission_is_attributed_after_external_current_song():
    m = api(); tracker = m.PlaybackTracker()
    tracker.observe(snap(m, 'external', 0))
    tracker.set_submitted([m.SubmittedTrack(1, 'a', 180000), m.SubmittedTrack(2, 'b', 180000)], version='v')
    events = tracker.observe(snap(m, 'a', 15), continuity=True)
    assert next(e for e in events if e.kind == 'started').selection_id == 1
    events = tracker.observe(snap(m, 'a', 60, 45000), continuity=True)
    assert events[0].outcome == 'accepted' and events[0].selection_id == 1


def test_consumed_occurrences_do_not_make_later_repeats_ambiguous():
    m = api(); tracker = m.PlaybackTracker()
    a, b, later = m.SubmittedTrack(1, 'a', 180000), m.SubmittedTrack(2, 'b', 180000), m.SubmittedTrack(3, 'a', 180000)
    tracker.set_submitted([a, b], version='v1')
    tracker.observe(snap(m, 'a', 0))
    tracker.observe(snap(m, 'b', 15), continuity=True)
    tracker.set_submitted([a, b, later], version='v2')
    events = tracker.observe(snap(m, 'a', 30), continuity=True)
    assert next(e for e in events if e.kind == 'started').selection_id == 3
