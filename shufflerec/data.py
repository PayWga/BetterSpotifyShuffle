"""Chronological partitions and tensor stores; never writes source data."""
from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FeaturePreprocessor, clean_catalog


def chunk_sessions(history, gap_minutes=20):
    required = {'timestamp', 'track_id', 'event_type'}
    if not required.issubset(history):
        raise ValueError('History requires timestamp, track_id, event_type')
    if gap_minutes <= 0:
        raise ValueError('Session gap must be positive')
    frame = history[list(sorted(required))].copy()
    frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='coerce', format='mixed')
    frame['track_id'] = frame.track_id.astype('string').str.strip()
    frame = frame.dropna(subset=['timestamp', 'track_id'])
    frame = frame[frame.track_id.ne('')].sort_values('timestamp', kind='stable').reset_index(drop=True)
    boundaries = frame.timestamp.diff().gt(pd.Timedelta(minutes=gap_minutes))
    return [group.reset_index(drop=True) for _, group in frame.groupby(boundaries.cumsum(), sort=False)]


def split_sessions(sessions, fractions=(0.8, 0.1, 0.1)):
    if len(fractions) != 3 or any(v <= 0 for v in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('Three positive split fractions must sum to one')
    if len(sessions) < 3:
        raise ValueError('Need at least three chronological sessions for train/validation/test')
    sessions = sorted(sessions, key=lambda group: group.timestamp.min())
    if any(a.timestamp.max() >= b.timestamp.min() for a, b in zip(sessions, sessions[1:])):
        raise ValueError('Sessions must not overlap in time')
    n = len(sessions)
    ntrain = min(n - 2, max(1, int(n * fractions[0])))
    nval = min(n - ntrain - 1, max(1, int(n * fractions[1])))
    return {'train': sessions[:ntrain], 'validation': sessions[ntrain:ntrain+nval],
            'test': sessions[ntrain+nval:]}


@dataclass(frozen=True)
class Example:
    context: tuple[str, ...]
    positive: str
    timestamp: pd.Timestamp
    negative: str | None = None


def extract_examples(sessions, usable_ids, max_length=10):
    if max_length < 1:
        raise ValueError('max_length must be positive')
    nip, pairs = [], []
    for session in sessions:
        context, pending = [], []
        for row in session.itertuples():
            track = str(row.track_id)
            if track not in usable_ids:
                context, pending = [], []
                continue
            if row.event_type == 'GHOST_SKIP':
                pending.append(track)
            elif row.event_type == 'PLAY':
                if context:
                    nip.append(Example(tuple(context), track, row.timestamp))
                    pairs.extend(Example(tuple(context), track, row.timestamp, negative)
                                 for negative in pending if negative != track)
                pending = []
                context = (context + [track])[-max_length:]
            else:
                # Events other than PLAY and GHOST_SKIP reset the context.
                # Other history signals are not used for training here.
                context, pending = [], []
    return nip, pairs


class TrackStore:
    """One contiguous tensor per input block; catalog metadata remains untouched."""
    def __init__(self, catalog, preprocessor):
        catalog = clean_catalog(catalog)
        self.ids = catalog.id.tolist()
        self.index = {track: i for i, track in enumerate(self.ids)}
        self.tensors = {key: torch.from_numpy(value) for key, value in preprocessor.transform(catalog).items()}
        observed = np.zeros(len(catalog), dtype=bool)
        for name, values in self.tensors.items():
            if name.endswith('_mask'):
                observed |= values.numpy().any(axis=-1)
        self.usable_ids = {track for track, known in zip(self.ids, observed) if known}

    def take(self, ids):
        indexes = torch.tensor([self.index[t] for t in ids], dtype=torch.long)
        return {key: values[indexes] for key, values in self.tensors.items()}

    def context(self, ids, max_length):
        ids = tuple(ids)[-max_length:]
        if not ids:
            raise ValueError('Context must contain a real track')
        result = {key: torch.zeros((max_length, values.shape[-1]), dtype=values.dtype)
                  for key, values in self.tensors.items()}
        for key, values in self.take(ids).items():
            result[key][-len(ids):] = values
        padding = torch.ones(max_length, dtype=torch.bool)
        padding[-len(ids):] = False
        return result, padding


def sample_negatives(pool, excluded, count, rng, *, membership=None):
    """Bounded rejection sampling, with a safe fallback for tiny/dense pools."""
    if count < 0:
        raise ValueError('Negative count must not be negative')
    if not isinstance(pool, (list, tuple)):
        pool = tuple(pool)
    excluded = set(excluded)
    membership = frozenset(pool) if membership is None else membership
    count = min(count, len(pool) - len(excluded & membership))
    if count == 0:
        return []
    if count * 4 >= len(pool):
        return rng.sample([t for t in pool if t not in excluded], count)
    picked, seen = [], set()
    for _ in range(max(100, count * 20)):
        track = rng.choice(pool)
        if track not in excluded and track not in seen:
            picked.append(track)
            seen.add(track)
            if len(picked) == count:
                return picked
    picked.extend(rng.sample([t for t in pool if t not in excluded and t not in seen], count - len(picked)))
    return picked


class SequenceDataset(Dataset):
    def __init__(self, examples, store, max_length=10, seed=42, candidate_ids=None):
        self.examples, self.store = list(examples), store
        self.max_length = max_length
        self.rng = random.Random(seed)
        self.candidate_ids = tuple(sorted(store.usable_ids if candidate_ids is None else
                                          set(candidate_ids) & store.usable_ids))
        self.candidate_membership = frozenset(self.candidate_ids)
        if max_length < 1:
            raise ValueError('max_length must be positive')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sample = self.examples[index]
        negative = sample.negative
        if negative is None:
            choices = sample_negatives(self.candidate_ids, set(sample.context) | {sample.positive}, 1,
                                       self.rng, membership=self.candidate_membership)
            if not choices:
                raise ValueError('No eligible negative track outside context and target')
            negative = choices[0]
        context, padding = self.store.context(sample.context, self.max_length)
        positive = {key: value[0] for key, value in self.store.take([sample.positive]).items()}
        negative = {key: value[0] for key, value in self.store.take([negative]).items()}
        return {'context': context, 'padding_mask': padding, 'positive': positive, 'negative': negative}


@dataclass
class PreparedData:
    preprocessor: FeaturePreprocessor
    store: TrackStore
    sessions: dict
    examples: dict
    max_length: int

    def summary(self):
        return {name: {'sessions': len(self.sessions[name]),
                       'events': sum(len(s) for s in self.sessions[name]),
                       'nip': len(groups['nip']), 'preference_pairs': len(groups['preference'])}
                for name, groups in self.examples.items()}


def prepare_data(catalog, history, config=None, max_length=10, fractions=(0.8, 0.1, 0.1)):
    catalog = clean_catalog(catalog)
    parts = split_sessions(chunk_sessions(history), fractions)
    training_ids = {str(track) for session in parts['train'] for track in session.track_id}
    preprocessor = FeaturePreprocessor(config).fit(catalog, training_ids)
    store = TrackStore(catalog, preprocessor)
    examples = {}
    for name, sessions in parts.items():
        nip, preference = extract_examples(sessions, store.usable_ids, max_length)
        examples[name] = {'nip': nip, 'preference': preference}
    if not examples['train']['nip']:
        raise ValueError('Training partition contains no next-item examples')
    return PreparedData(preprocessor, store, parts, examples, max_length)
