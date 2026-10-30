"""Attribute backend-independent playback events to acknowledged submissions."""
from dataclasses import asdict, replace
import copy
import random

from .playback import PlaybackConfig, PlaybackTracker, SubmittedTrack, finite


class PlaybackSession:
    def __init__(self, sequencer, *, playback_config=None, tracker=None, auto_replan=True):
        self.sequencer = sequencer
        config = playback_config or PlaybackConfig()
        self.tracker = tracker or PlaybackTracker(replace(config, acceptance_threshold=sequencer.config.early_skip_threshold))
        self.auto_replan = auto_replan
        self.submitted = []
        self._version = 0
        rows = sequencer.engine.catalog
        self._durations = dict(zip(rows.id, rows.duration_ms)) if 'duration_ms' in rows else {}

    def submit_prefix(self, count):
        """Register an acknowledged prefix, not merely planned or sent requests."""
        items = self.sequencer.reserve_prefix(count)
        self.submitted.extend(items)
        self.confirm_order()
        return items

    def confirm_order(self, *, follows_active=False):
        entries = []
        for item in self.submitted:
            duration = self._durations.get(item.track_id)
            if not finite(duration) or duration <= 0:
                duration = None
            entries.append(SubmittedTrack(item.selection_id, item.track_id, duration))
        self._version += 1
        self.tracker.set_submitted(entries, version=str(self._version), follows_active=follows_active)

    def abandon(self, selection_ids):
        identities = set(selection_ids)
        for item in self.submitted:
            if item.selection_id in identities:
                self.sequencer.abandon_selection(item.selection_id)
        if self.tracker.active and self.tracker.active.selection_id in identities:
            self.tracker.active.selection_id = None
        self.submitted = [item for item in self.submitted if item.selection_id not in identities]
        self.confirm_order()

    def planning_snapshot(self, *, exclude=()):
        """Copy mutable state on the owner thread before background inference."""
        seq = self.sequencer
        clone = copy.copy(seq)
        clone.context_ids = seq.context_ids.copy()
        clone.recent = seq.recent.copy()
        clone.fatigue = seq.fatigue.copy()
        clone.policy = copy.copy(seq.policy)
        clone.policy.preferences = seq.policy.preferences.copy()
        clone._submitted = seq._submitted.copy()
        clone.rng = random.Random()
        clone.rng.setstate(seq.rng.getstate())
        active = self.tracker.active
        ids = [active.track_id] if active and not active.accepted else []
        pools = []
        for item in self.submitted:
            if not active or item.selection_id != active.selection_id:
                ids.append(item.track_id)
                if item.selection_id not in seq._started_submissions:
                    pools.append(item.pool)
        return clone, ids, pools

    def observe(self, snapshot, *, continuity=False):
        events = self.tracker.observe(snapshot, continuity=continuity)
        self._consume(events)
        return events

    def _consume(self, events):
        seq = self.sequencer
        for event in events:
            identity = event.selection_id
            if identity is not None:
                seq.observe_selection(identity)
                if event.kind in ('feedback', 'ended') and not seq.has_selection_feedback(identity):
                    seq.record_selection_feedback(identity, event.outcome, weight=event.weight)
                if event.kind == 'ended':
                    seq.resolve_selection(identity)
                    self.submitted = [item for item in self.submitted if item.selection_id != identity]
            elif event.kind == 'feedback' and event.outcome == 'accepted':
                seq.context_ids.append(event.track_id)
                seq.recent.append(event.track_id)
        if events and self.auto_replan:
            clone, ids, pools = self.planning_snapshot()
            seq.plan(provisional_ids=ids, provisional_pools=pools)

    @property
    def config(self):
        return self.tracker.config

    def next_delay(self):
        return self.tracker.next_delay()

    def finish(self, reason='daemon_stopped'):
        events = self.tracker.finish(reason)
        self._consume(events)
        return events

    def interrupt(self):
        self.tracker.interrupt()

    def status(self):
        active = asdict(self.tracker.active) if self.tracker.active else None
        if active and not finite(active['upper_bound_ms']):
            active['upper_bound_ms'] = None
        return {'submitted': [asdict(item) for item in self.submitted],
                'playback_active': active, 'next_poll_seconds': self.tracker.next_delay()}
