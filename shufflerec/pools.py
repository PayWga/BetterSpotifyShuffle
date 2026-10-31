"""Playlist pools, metadata matching, and bounded adaptive sampling policy."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
import random

import pandas as pd

from .features import clean_catalog, first_present, parse_labels, series


POOL_NAMES = ('random', 'top', 'filtered')


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class SequencerConfig:
    queue_size: int = 5
    base_weights: tuple[float, ...] = (1., 1., 1.)
    randomize_initial_weights: bool = False
    initial_weight_low: float = .8
    initial_weight_high: float = 1.2
    min_probability: float = .05
    max_probability: float = .75
    top_k: int = 20
    top_weight_low: float = .5
    initial_candidates: int = 50
    max_candidates: int = 200
    min_filtered: int = 10
    genre_weight_high: float = .75
    genre_weight_low: float = .375
    fatigue_decay: float = .65
    fatigue_strength: float = .8
    learning_rate: float = .15
    preference_limit: float = 2.
    early_skip_threshold: float = .25
    recent_window: int = 10

    def __post_init__(self):
        for name in ('queue_size', 'top_k', 'initial_candidates', 'max_candidates', 'min_filtered', 'recent_window'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'recent_window' else 1):
                raise ValueError(f'{name} must be an integer in its valid range')
        for name in ('initial_weight_low', 'initial_weight_high', 'min_probability', 'max_probability',
                     'top_weight_low', 'genre_weight_high', 'genre_weight_low', 'fatigue_decay',
                     'fatigue_strength', 'learning_rate', 'preference_limit', 'early_skip_threshold'):
            value = getattr(self, name)
            if not _finite_number(value):
                raise ValueError(f'{name} must be finite numeric data')
        if not isinstance(self.randomize_initial_weights, bool):
            raise ValueError('randomize_initial_weights must be boolean')
        if not isinstance(self.base_weights, (list, tuple)) or len(self.base_weights) != 3 or any(
            not _finite_number(v) or v <= 0
            for v in self.base_weights
        ):
            raise ValueError('base_weights must contain three positive finite weights')
        if not 0 < self.min_probability <= 1 / 3 or not .5 <= self.max_probability <= .75:
            raise ValueError('probability bounds must be feasible for two or three pools and capped at .75')
        if not 0 < self.initial_weight_low <= self.initial_weight_high:
            raise ValueError('initial weight limits must be positive and ordered')
        if self.initial_candidates > self.max_candidates:
            raise ValueError('initial_candidates cannot exceed max_candidates')
        if not 0 < self.genre_weight_low <= self.genre_weight_high <= 1:
            raise ValueError('genre weights must be positive and at most the artist weight of 1')
        if not 0 < self.top_weight_low <= 1:
            raise ValueError('top_weight_low must be in (0, 1]')
        if not 0 <= self.fatigue_decay < 1 or self.fatigue_strength < 0:
            raise ValueError('fatigue decay must be in [0, 1), strength must be nonnegative')
        if self.learning_rate < 0 or self.preference_limit <= 0:
            raise ValueError('learning_rate must be nonnegative and preference_limit positive')
        if not 0 < self.early_skip_threshold <= 1:
            raise ValueError('early_skip_threshold must be in (0, 1]')


class PoolPolicy:
    def __init__(self, config, rng=None):
        self.config = config
        rng = rng or random.Random()
        self._log_base_weights = [math.log(v) for v in config.base_weights]
        if config.randomize_initial_weights:
            self._log_base_weights = [v + math.log(rng.uniform(config.initial_weight_low, config.initial_weight_high))
                                      for v in self._log_base_weights]
        self.preferences = [0., 0., 0.]

    def probabilities(self, available, fatigue):
        available = [name for name in POOL_NAMES if name in available]
        if not available:
            return {}
        if len(available) == 1:
            return {available[0]: 1.}
        c = self.config
        active = [i for i, name in enumerate(POOL_NAMES) if name in available]
        fixed = {}
        while active:
            # Subtract shared fatigue before applying its strength.
            # This keeps at least one logit finite if another penalty overflows.
            baseline = min(fatigue[i] for i in active)
            logs = {i: self._log_base_weights[i] + self.preferences[i]
                    - c.fatigue_strength * (fatigue[i] - baseline) for i in active}
            maximum = max(logs.values())
            weights = {i: math.exp(value - maximum) for i, value in logs.items()}
            remaining = 1 - sum(fixed.values())
            total = sum(weights.values())
            proposed = {i: remaining * value / total for i, value in weights.items()}
            # Cap oversized probabilities first. Redistributing their weight can
            # bring smaller pools above the lower bound without another adjustment.
            over = [i for i in active if proposed[i] > c.max_probability]
            under = [i for i in active if proposed[i] < c.min_probability]
            clamped = over or under
            if not clamped:
                fixed.update(proposed)
                break
            for i in clamped:
                fixed[i] = c.max_probability if over else c.min_probability
                active.remove(i)
        return {name: fixed[i] for i, name in enumerate(POOL_NAMES) if name in available}

    def advance(self, fatigue, chosen_pool):
        return [value * self.config.fatigue_decay + float(name == chosen_pool)
                for name, value in zip(POOL_NAMES, fatigue)]

    def observe(self, pool, accepted, *, weight=1.):
        if not _finite_number(weight) or not 0 <= weight <= 1:
            raise ValueError('feedback weight must be a finite fraction in [0, 1]')
        i = POOL_NAMES.index(pool)
        value = self.preferences[i] + self.config.learning_rate * weight * (1 if accepted else -1)
        self.preferences[i] = max(-self.config.preference_limit, min(self.config.preference_limit, value))


def _artist_ids(value):
    if isinstance(value, (list, tuple)):
        values = value
    elif value is None or pd.isna(value):
        return frozenset()
    else:
        text = str(value).strip()
        try:
            values = ast.literal_eval(text) if text.startswith('[') else text.split(',')
        except (SyntaxError, ValueError):
            return frozenset()
    if not isinstance(values, (list, tuple)):
        return frozenset()
    return frozenset(str(v).strip() for v in values if isinstance(v, str) and v.strip())


class CatalogMatches:
    """Metadata-only index, independent of which model features were enabled."""
    def __init__(self, catalog):
        frame = clean_catalog(catalog)
        self.rows = {}
        names = first_present(frame, ['artists', 'artist_name'])
        genres = first_present(frame, ['track_genre', 'genre'])
        for track, name, ids, genre, artist_genres in zip(
            frame.id, names, series(frame, 'artist_ids'), genres, series(frame, 'artist_genres')
        ):
            self.rows[track] = (_artist_ids(ids), frozenset(parse_labels(name)),
                                frozenset(parse_labels(genre)), frozenset(parse_labels(artist_genres)))

    def match(self, anchor, candidate):
        a, b = self.rows.get(anchor), self.rows.get(candidate)
        if a is None or b is None:
            return False, False
        artist = bool(a[0] & b[0]) if a[0] and b[0] else bool(a[1] & b[1])
        genre = bool(a[2] & b[2] or a[3] & b[3])
        return artist, genre


def build_pools(catalog, anchor, eligible_ids, ranked, config):
    matches = catalog if isinstance(catalog, CatalogMatches) else CatalogMatches(catalog)
    pools = {'random': [(track, 1.) for track in eligible_ids], 'top': [], 'filtered': []}
    top = ranked[:config.top_k]
    pools['top'] = [(item.track_id, 1 - (1 - config.top_weight_low) * i / max(1, len(top) - 1))
                    for i, item in enumerate(top)]
    limit = min(len(ranked), config.max_candidates)
    depth = min(limit, config.initial_candidates)
    while depth:
        filtered = []
        for i, item in enumerate(ranked[:depth]):
            artist, genre = matches.match(anchor, item.track_id)
            if artist:
                filtered.append((item.track_id, 1.))
            elif genre:
                weight = config.genre_weight_high - (config.genre_weight_high - config.genre_weight_low) * i / max(1, depth - 1)
                filtered.append((item.track_id, weight))
        if len(filtered) >= config.min_filtered or depth == limit:
            pools['filtered'] = filtered
            break
        depth = min(limit, depth * 2)
    return pools, depth
