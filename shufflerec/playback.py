"""Backend-independent playback evidence, queue reconciliation and poll timing."""
from __future__ import annotations

from dataclasses import dataclass
import math


def finite(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class PlaybackConfig:
    acceptance_threshold: float = .25
    early_poll_seconds: float = 15.
    max_gap_seconds: float = 75.
    idle_poll_seconds: float = 30.
    inactive_timeout_seconds: float = 60.
    weak_feedback_weight: float = .5
    seek_tolerance_ms: float = 1500.
    end_guard_seconds: float = 2.

    def __post_init__(self):
        for name, value in vars(self).items():
            if not finite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.acceptance_threshold > 1 or self.weak_feedback_weight > 1:
            raise ValueError('threshold and feedback weight must be at most 1')


@dataclass(frozen=True)
class Snapshot:
    track_id: str | None
    duration_ms: float | None
    progress_ms: float
    is_playing: bool
    observed_at: float
    device_id: str | None = None
    context_uri: str | None = None

    def __post_init__(self):
        if not finite(self.observed_at) or not finite(self.progress_ms) or self.progress_ms < 0:
            raise ValueError('Snapshot time/progress must be finite and progress nonnegative')
        if self.duration_ms is not None and (not finite(self.duration_ms) or self.duration_ms <= 0):
            raise ValueError('Known duration must be finite and positive')
        if self.track_id is not None and (not isinstance(self.track_id, str) or not self.track_id.strip()):
            raise ValueError('track_id must be a nonempty string or None')
        if not isinstance(self.is_playing, bool):
            raise ValueError('is_playing must be boolean')


@dataclass(frozen=True)
class SubmittedTrack:
    selection_id: int
    track_id: str
    duration_ms: float | None

    def __post_init__(self):
        if isinstance(self.selection_id, bool) or not isinstance(self.selection_id, int) or self.selection_id < 1:
            raise ValueError('selection_id must be a positive integer')
        Snapshot(self.track_id, self.duration_ms, 0, False, 0)


@dataclass(frozen=True)
class PlaybackEvent:
    occurrence_id: int
    selection_id: int | None
    track_id: str
    kind: str
    outcome: str
    weight: float
    inferred: bool
    weak: bool
    listened_ms: float
    upper_bound_ms: float | None
    gap_ms: float
    reason: str


@dataclass
class ActivePlayback:
    occurrence_id: int
    selection_id: int | None
    track_id: str
    duration_ms: float | None
    listened_ms: float = 0.
    upper_bound_ms: float = 0.
    accepted: bool = False


class PlaybackTracker:
    def __init__(self, config=None):
        self.config = config or PlaybackConfig()
        self.active = None
        self.previous = None
        self.submitted = ()
        self.version = None
        self._index = None
        self._next_occurrence = 1
        self._broken = False
        self._order_valid = False
        self._inactive_since = None
        self._consumed = set()

    def set_submitted(self, tracks, *, version):
        tracks = tuple(tracks)
        if not isinstance(version, str) or not version or any(not isinstance(t, SubmittedTrack) for t in tracks):
            raise ValueError('Submitted order requires a nonempty version and SubmittedTrack entries')
        if len({t.selection_id for t in tracks}) != len(tracks):
            raise ValueError('Submitted selection IDs must be unique')
        tracks = tuple(t for t in tracks if t.selection_id not in self._consumed)
        self.submitted, self.version = tracks, version
        self._order_valid = True
        self._index = None
        if self.active and self.active.selection_id is not None:
            self._index = next((i for i, t in enumerate(tracks) if t.selection_id == self.active.selection_id), None)

    def interrupt(self):
        """An outage is not a pause, listen, skip, or proof of queue continuity."""
        self._broken = True
        self._order_valid = False

    def _new(self, track_id, duration, selection_id=None, upper=0.):
        item = ActivePlayback(self._next_occurrence, selection_id, track_id, duration, upper_bound_ms=upper)
        self._next_occurrence += 1
        return item

    @staticmethod
    def _event(item, kind, outcome='unresolved', weight=0., *, inferred=False,
               weak=False, gap=0., upper=None, reason=''):
        bound = item.upper_bound_ms if upper is None else upper
        bound = bound if finite(bound) else None
        return PlaybackEvent(item.occurrence_id, item.selection_id, item.track_id, kind,
                             outcome, weight, inferred, weak, item.listened_ms,
                             bound, gap, reason)

    def _classify(self, duration, upper, safe):
        if not safe or duration is None or upper >= duration:
            return 'unresolved', 0., False
        if upper < duration * self.config.acceptance_threshold:
            return 'rejected', 1., False
        return 'rejected', self.config.weak_feedback_weight, True

    def _end(self, gap, safe, reason):
        item = self.active
        upper = min(item.duration_ms, item.upper_bound_ms + gap) if item.duration_ms else item.upper_bound_ms + gap
        outcome, weight, weak = self._classify(item.duration_ms, upper, safe)
        if item.accepted:
            outcome, weight, weak = 'accepted', 0., False
        event = self._event(item, 'ended', outcome, weight, weak=weak, gap=gap, upper=upper, reason=reason)
        if item.selection_id is not None:
            self._consumed.add(item.selection_id)
        self.active = None
        return event

    def finish(self, reason='daemon_stopped'):
        if not self.active:
            return []
        events = [self._end(0, False, reason)]
        self.previous = None
        self._index = None
        return events

    def observe(self, snapshot, *, continuity=False):
        if not isinstance(snapshot, Snapshot):
            raise ValueError('Expected Snapshot')
        if not isinstance(continuity, bool):
            raise ValueError('Queue continuity must be explicitly boolean')
        if self.previous and snapshot.observed_at <= self.previous.observed_at:
            raise ValueError('Observations must have strictly increasing timestamps')
        old = self.previous
        gap = (snapshot.observed_at - old.observed_at) * 1000 if old else 0.
        safe = bool(old and not self._broken and old.device_id == snapshot.device_id
                    and old.context_uri == snapshot.context_uri)
        events = []
        if snapshot.track_id is None:
            if self._inactive_since is None:
                self._inactive_since = snapshot.observed_at
            self.interrupt()
            if self.active and snapshot.observed_at - self._inactive_since >= self.config.inactive_timeout_seconds:
                events.append(self._end(0, False, 'playback_inactive'))
            self.previous = snapshot
            return events
        self._inactive_since = None
        replay = bool(self.active and old and snapshot.track_id == self.active.track_id and old.duration_ms
                      and old.progress_ms > old.duration_ms * .85 and snapshot.progress_ms < old.duration_ms * .15)
        transition = self.active is not None and (snapshot.track_id != self.active.track_id or replay)
        next_matches = [i for i, t in enumerate(self.submitted)
                        if t.track_id == snapshot.track_id and t.selection_id not in self._consumed]
        next_index = next_matches[0] if len(next_matches) == 1 else None
        if transition:
            can_infer = (safe and continuity and self._order_valid and not replay and old.is_playing
                         and snapshot.is_playing and self._index is not None and next_index is not None
                         and next_index > self._index)
            events.append(self._end(gap, safe and not replay, 'replay_detected' if replay else 'track_change'))
            if can_infer:
                for submitted in self.submitted[self._index + 1:next_index]:
                    item = self._new(submitted.track_id, submitted.duration_ms, submitted.selection_id)
                    outcome, weight, weak = self._classify(item.duration_ms, gap, True)
                    events.append(self._event(item, 'ended', outcome, weight, inferred=True,
                                              weak=weak, gap=gap, upper=gap, reason='queue_progression'))
                    self._consumed.add(submitted.selection_id)
            elif not replay and not (safe and continuity and self._order_valid and next_index is not None):
                # Identifying the next acknowledged item does not require an
                # acknowledged predecessor (e.g. initial external playback).
                self._order_valid = False
        if self.active is None:
            # Ambiguous repeated track IDs never identify a submitted occurrence.
            identity = self.submitted[next_index].selection_id if next_index is not None and self._order_valid and not replay else None
            self.active = self._new(snapshot.track_id, snapshot.duration_ms, identity, snapshot.progress_ms)
            self._index = next_index if identity is not None else None
            events.append(self._event(self.active, 'started', reason='observed'))
        elif old:
            item = self.active
            delta = snapshot.progress_ms - old.progress_ms
            continuous = safe and old.is_playing and snapshot.is_playing and 0 <= delta <= gap + self.config.seek_tolerance_ms
            if continuous:
                item.listened_ms += min(delta, gap)
                item.upper_bound_ms += gap
            elif not safe or delta < 0 or delta > gap + self.config.seek_tolerance_ms:
                item.upper_bound_ms = item.duration_ms or float('inf')
                self._order_valid = False
            else:
                # Pause/resume endpoints don't locate the change within the gap.
                # Count no measured listening, but retain the maximum possible.
                item.upper_bound_ms += gap
            if item.duration_ms:
                item.listened_ms = min(item.listened_ms, item.duration_ms)
                item.upper_bound_ms = min(item.upper_bound_ms, item.duration_ms)
            if not item.accepted and item.duration_ms and item.listened_ms >= item.duration_ms * self.config.acceptance_threshold:
                item.accepted = True
                events.append(self._event(item, 'feedback', 'accepted', 1., gap=gap, reason='threshold_reached'))
        self.previous = snapshot
        self._broken = False
        return events

    def next_delay(self):
        snapshot = self.previous
        if not snapshot or not snapshot.track_id or not snapshot.is_playing:
            return self.config.idle_poll_seconds
        interval = self.config.max_gap_seconds if self.active and self.active.accepted else self.config.early_poll_seconds
        if snapshot.duration_ms:
            remaining = max(0., (snapshot.duration_ms - snapshot.progress_ms) / 1000)
            if remaining > self.config.end_guard_seconds:
                remaining -= self.config.end_guard_seconds
            # Avoid a busy loop when a player lingers at its reported end position.
            return min(interval, remaining) if remaining > 0 else min(interval, 1.)
        return min(interval, self.config.early_poll_seconds)
