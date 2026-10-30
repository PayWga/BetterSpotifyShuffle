"""Adaptive five-track planning with separate committed and speculative state."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import random

from .pools import CatalogMatches, PoolPolicy, SequencerConfig, build_pools


@dataclass(frozen=True)
class Selection:
    selection_id: int
    track_id: str
    pool: str
    probabilities: dict[str, float]
    candidate_depth: int
    diagnostics: tuple[str, ...]


class Sequencer:
    """Single-session state. Playback adapters explicitly start and resolve items.

    Context input is an ordered sequence of accepted IDs, not raw history rows.
    Replanning changes speculative selections only. It never fabricates listens.
    """

    def __init__(self, engine, playlist_ids, *, context_ids=(), config=None, seed=42):
        self.engine = engine
        self.config = config or SequencerConfig()
        self.playlist_ids = sorted({str(t).strip() for t in playlist_ids if str(t).strip()})
        if not self.playlist_ids:
            raise ValueError('Playlist must contain at least one track')
        self.context_ids = [str(t).strip() for t in context_ids]
        self.matches = CatalogMatches(engine.catalog)
        self.rng = random.Random(seed)
        self.policy = PoolPolicy(self.config, self.rng)
        self.fatigue = [0., 0., 0.]
        self.recent = deque(self.context_ids[-self.config.recent_window:] if self.config.recent_window else [],
                            maxlen=self.config.recent_window)
        self.queue = []
        self.active = None
        self.diagnostics = []
        self._next_id = 1
        self._submitted = {}
        self._started_submissions = set()
        self._resolved_submissions = set()
        self._submission_feedback = set()

    def plan(self, *, provisional_ids=(), provisional_pools=(), exclude=(), relax_recent=False):
        if self.active is not None:
            raise ValueError('Resolve the active selection before replanning')
        queue, diagnostics = [], []
        context = self.context_ids + list(provisional_ids)
        fatigue = self.fatigue.copy()
        for pool in provisional_pools:
            fatigue = self.policy.advance(fatigue, pool)
        hard_excluded = set(exclude)
        excluded = set(self.recent) | hard_excluded
        excluded.update(item.track_id for identity, item in self._submitted.items()
                        if identity not in self._resolved_submissions)
        hard_excluded.update(item.track_id for identity, item in self._submitted.items()
                             if identity not in self._resolved_submissions)
        anchor = context[-1] if context else None
        if anchor:
            excluded.add(anchor)
            hard_excluded.add(anchor)
        for _ in range(self.config.queue_size):
            eligible = [track for track in self.playlist_ids if track not in excluded]
            if not eligible and relax_recent:
                eligible = [track for track in self.playlist_ids if track not in hard_excluded]
                if eligible and 'recent_window_relaxed' not in diagnostics:
                    diagnostics.append('recent_window_relaxed')
            if not eligible:
                diagnostics.append('playlist_exhausted')
                break
            notes = []
            try:
                ranked = self.engine.rank(context, eligible)
            except (RuntimeError, FloatingPointError):
                ranked = []
                notes.append('inference_unavailable')
            if not ranked and not notes:
                notes.append('no_model_context' if not self.engine.usable_context(context) else 'no_model_candidates')
            pools, depth = build_pools(self.matches, anchor, eligible, ranked, self.config)
            probabilities = self.policy.probabilities([name for name, pool in pools.items() if pool], fatigue)
            if len(probabilities) == 1:
                notes.append('single_available_pool')
            pool = self.rng.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
            items, weights = zip(*pools[pool])
            track = self.rng.choices(items, weights=weights, k=1)[0]
            queue.append(Selection(self._next_id, track, pool, probabilities, depth, tuple(notes)))
            self._next_id += 1
            context.append(track)
            anchor = track
            excluded.add(track)
            hard_excluded.add(track)
            fatigue = self.policy.advance(fatigue, pool)
        self.queue = queue
        self.diagnostics = diagnostics
        return list(queue)

    def reserve_prefix(self, count):
        """Acknowledge submitted order; call only AFTER a transport confirms it."""
        if self.active is not None:
            raise ValueError('Cannot mix offline active selections and submitted playback')
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(self.queue):
            raise ValueError('Submission count must identify a nonempty queue prefix')
        items = self.queue[:count]
        self.queue = self.queue[count:]
        self._submitted.update((item.selection_id, item) for item in items)
        return list(items)

    def observe_selection(self, selection_id):
        if selection_id not in self._submitted or selection_id in self._resolved_submissions:
            raise ValueError('Playback must identify an unresolved submitted selection')
        if selection_id not in self._started_submissions:
            item = self._submitted[selection_id]
            self.fatigue = self.policy.advance(self.fatigue, item.pool)
            self.recent.append(item.track_id)
            self._started_submissions.add(selection_id)

    def record_selection_feedback(self, selection_id, outcome, *, weight=1.):
        if selection_id not in self._started_submissions or selection_id in self._submission_feedback:
            raise ValueError('Feedback requires a started submitted selection without prior feedback')
        if outcome not in ('accepted', 'rejected', 'unresolved'):
            raise ValueError('Unknown playback outcome')
        if (isinstance(weight, bool) or not isinstance(weight, (int, float))
                or not 0 <= weight <= 1 or not math.isfinite(weight)):
            raise ValueError('feedback weight must be a finite fraction in [0, 1]')
        item = self._submitted[selection_id]
        if outcome != 'unresolved':
            self.policy.observe(item.pool, outcome == 'accepted', weight=weight)
        if outcome == 'accepted':
            self.context_ids.append(item.track_id)
        self._submission_feedback.add(selection_id)

    def resolve_selection(self, selection_id):
        if selection_id not in self._started_submissions:
            raise ValueError('Cannot resolve an unstarted selection')
        self._resolved_submissions.add(selection_id)
        self._submitted.pop(selection_id, None)
        self._started_submissions.discard(selection_id)

    def has_selection_feedback(self, selection_id):
        return selection_id in self._submission_feedback

    def abandon_selection(self, selection_id):
        """Release uncertain ownership without inventing a listen or feedback."""
        if selection_id not in self._submitted:
            raise ValueError('Unknown submitted selection')
        self._submitted.pop(selection_id)
        self._started_submissions.discard(selection_id)
        self._resolved_submissions.add(selection_id)
        self._submission_feedback.add(selection_id)

    def start_next(self):
        if self._submitted and self._submitted.keys() - self._resolved_submissions:
            raise ValueError('Cannot mix offline start commands with submitted playback')
        if self.active is not None:
            raise ValueError('Resolve the active selection before starting another')
        if not self.queue:
            self.plan()
        if not self.queue:
            raise ValueError('No eligible tracks remain')
        self.active = self.queue.pop(0)
        self.fatigue = self.policy.advance(self.fatigue, self.active.pool)
        self.recent.append(self.active.track_id)
        return self.active

    def feedback(self, selection_id, completion_rate, *, reason_end='track_change'):
        if self.active is None or isinstance(selection_id, bool) or selection_id != self.active.selection_id:
            raise ValueError('Feedback must identify the active selection')
        if (isinstance(completion_rate, bool) or not isinstance(completion_rate, (float, int))
                or not 0 <= completion_rate <= 1 or not math.isfinite(completion_rate)):
            raise ValueError('completion_rate must be a finite fraction in [0, 1]')
        if not isinstance(reason_end, str):
            raise ValueError('reason_end must be a string')
        if completion_rate >= self.config.early_skip_threshold:
            outcome = 'accepted'
            self.context_ids.append(self.active.track_id)
            self.policy.observe(self.active.pool, True)
        elif reason_end in ('track_change', 'skip'):
            outcome = 'rejected'
            self.policy.observe(self.active.pool, False)
        else:
            outcome = 'neutral'
        self.active = None
        self.plan()
        return outcome

    def status(self):
        return {'queue': [asdict(item) for item in self.queue],
                'active': asdict(self.active) if self.active else None,
                'preferences': dict(zip(('random', 'top', 'filtered'), self.policy.preferences)),
                'fatigue': self.fatigue.copy(), 'diagnostics': self.diagnostics.copy()}
