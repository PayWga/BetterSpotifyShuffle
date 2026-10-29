"""Frozen, missing-aware transformations for the enriched master-table schema."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

AUDIO = ('danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
         'instrumentalness', 'liveness', 'valence', 'tempo', 'mode')
BOUNDED = ('danceability', 'energy', 'speechiness', 'acousticness',
           'instrumentalness', 'liveness', 'valence')
LOCAL = ('local_rms_db', 'local_onset_strength', 'local_spectral_centroid',
         'local_spectral_contrast', 'local_beat_regularity')
EXTRACTED = ('tempo', 'key', 'mode') + LOCAL
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeatureConfig:
    artist: bool = True
    year: bool = True
    duration: bool = True
    genre: bool = True
    artist_genre: bool = True
    local: bool = True
    provenance: bool = True
    album: bool = False
    popularity: bool = False
    key_encoding: str = 'circular'
    time_signature_encoding: str = 'categorical'
    max_labels: int = 16
    local_rms_db: bool = True
    local_onset_strength: bool = True
    local_spectral_centroid: bool = True
    local_spectral_contrast: bool = True
    local_beat_regularity: bool = True
    include_estimated_durations: bool = False

    @property
    def enabled_local_features(self):
        # Keep the old aggregate gate for saved configurations and callers;
        # each measurement now also has its own independent switch.
        return tuple(name for name in LOCAL if self.local and getattr(self, name))

    def __post_init__(self):
        if self.key_encoding not in ('raw', 'circular'):
            raise ValueError('key_encoding must be raw or circular')
        if self.time_signature_encoding not in ('numeric', 'categorical'):
            raise ValueError('time_signature_encoding must be numeric or categorical')
        if self.max_labels < 1:
            raise ValueError('max_labels must be positive')


def parse_labels(value, *, split_commas=False):
    """Parse explicit lists safely; commas in plain display names are preserved."""
    if isinstance(value, (list, tuple)):
        values = value
    elif value is None or pd.isna(value):
        return []
    else:
        text = str(value).strip()
        if not text or text.casefold() in ('nan', 'none', 'null'):
            return []
        if text.startswith('['):
            try:
                values = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return []
            if not isinstance(values, (list, tuple)):
                return []
        else:
            values = text.split(',') if split_commas else [text]
    labels = [re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(v))).strip().casefold()
              for v in values if isinstance(v, (str, int))]
    return list(dict.fromkeys(v for v in labels if v))


def series(frame, name):
    return frame[name] if name in frame else pd.Series(np.nan, index=frame.index)


def first_present(frame, names):
    result = pd.Series(np.nan, index=frame.index, dtype=object)
    for name in names:
        candidate = series(frame, name)
        valid = candidate.notna() & candidate.astype(str).str.strip().ne('')
        result = result.where(result.notna(), candidate.where(valid))
    return result


def clean_catalog(frame):
    """Keep input immutable and reject ambiguous identities instead of choosing a row."""
    if 'id' not in frame:
        raise ValueError("Master table requires 'id'")
    result = frame.copy()
    result['id'] = result['id'].astype('string').str.strip()
    result = result[result.id.notna() & result.id.ne('')].reset_index(drop=True)
    if result.id.duplicated().any():
        raise ValueError('Master table contains duplicate IDs; resolve upstream')
    return result


class FeaturePreprocessor:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.vocabularies = {}
        self.scales = {}
        self.artist_aliases = {}
        self.numeric_names = []
        self.fitted = False
        self.duration_policy = 'include_estimated' if self.config.include_estimated_durations else 'exclude_estimated'

    def _numeric(self, frame):
        names = list(AUDIO)
        names += ['key'] if self.config.key_encoding == 'raw' else ['key_sin', 'key_cos']
        if self.config.time_signature_encoding == 'numeric':
            names.append('time_signature')
        names += [name for enabled, name in ((self.config.year, 'year'),
                  (self.config.duration, 'duration_ms'), (self.config.popularity, 'popularity')) if enabled]
        names.extend(self.config.enabled_local_features)
        values = pd.DataFrame(index=frame.index)
        for name in names:
            values[name] = pd.to_numeric(series(frame, name), errors='coerce')
        values = values.replace([np.inf, -np.inf], np.nan)
        for name in BOUNDED:
            values[name] = values[name].where(values[name].between(0, 1))
        values['mode'] = values['mode'].where(values['mode'].isin([0, 1]))
        values['tempo'] = values['tempo'].where(values['tempo'] >= 0)
        key = pd.to_numeric(series(frame, 'key'), errors='coerce')
        key = key.where(key.isin(range(12)))
        if self.config.key_encoding == 'raw':
            values['key'] = key
        else:
            values['key_sin'] = np.sin(2 * np.pi * key / 12)
            values['key_cos'] = np.cos(2 * np.pi * key / 12)
        if self.config.year:
            fallback = pd.Series(np.nan, index=frame.index)
            for name in ('year', 'release_year'):
                candidate = pd.to_numeric(series(frame, name), errors='coerce')
                candidate = candidate.where(candidate.between(1850, 2100) & candidate.mod(1).eq(0))
                fallback = fallback.fillna(candidate)
            # Spotify supports year/month/day precision. Numeric YYYY values
            # inferred by CSV readers are calendar years, never epoch nanoseconds.
            text = series(frame, 'release_date').astype('string').str.strip()
            text = text.str.replace(r'^(\d{4})\.0+$', r'\1', regex=True)
            text = text.where(text.str.fullmatch(r'\d{4}(?:-\d{2}(?:-\d{2})?)?', na=False))
            dates = pd.to_datetime(text, format='mixed', errors='coerce')
            fallback = fallback.fillna(dates.dt.year)
            values['year'] = fallback.where(fallback.between(1850, 2100) & fallback.mod(1).eq(0))
        for name in ('duration_ms', 'popularity', 'time_signature') + LOCAL:
            if name not in values:
                continue
            v = values[name]
            if name == 'local_rms_db':
                continue  # negative dB, including -1, are real measurements
            valid = v > 0 if name in ('duration_ms', 'time_signature') else v >= 0
            if name in ('popularity', 'local_beat_regularity'):
                valid &= v <= (100 if name == 'popularity' else 1)
            if name == 'time_signature':
                valid &= v.mod(1).eq(0)
            values[name] = v.where(valid)
        if self.config.duration:
            text = series(frame, 'duration_ms_is_estimated').astype('string').str.strip().str.casefold()
            estimated = text.map({'true': 1., '1': 1., '1.0': 1., 'false': 0., '0': 0., '0.0': 0.})
            if self.duration_policy == 'exclude_estimated':
                values['duration_ms'] = values['duration_ms'].mask(estimated.eq(1))
            elif self.duration_policy == 'include_estimated':
                # Unknown source status remains missing; unflagged never means
                # independently verified. Invalid/missing duration stays missing.
                values['duration_ms_is_estimated'] = estimated.where(values['duration_ms'].notna())
        if self.config.provenance:
            for name in EXTRACTED[:3] + self.config.enabled_local_features:
                raw = pd.to_numeric(series(frame, name), errors='coerce')
                observed = raw.notna() & np.isfinite(raw)
                if name in values:
                    observed &= values[name].notna()
                elif name == 'key':
                    observed &= key.notna()
                source = series(frame, name + '_source').astype('string')
                known_source = source.notna() & source.str.strip().ne('') & observed
                local = source.str.contains('librosa|local|acoustic', case=False, regex=True, na=False)
                values[name + '_is_local'] = local.astype(float).where(known_source)
                seconds = pd.to_numeric(series(frame, name + '_analysis_seconds'), errors='coerce')
                seconds = seconds.where((seconds > 0) & np.isfinite(seconds) & observed)
                values[name + '_analysis_seconds'] = np.log1p(seconds)
        return values

    def _artist_records(self, frame):
        names = first_present(frame, ['artists', 'artist_name'])
        ids = series(frame, 'artist_ids')
        # IDs are case-sensitive: normalize whitespace, never case-fold them.
        records = []
        for raw_names, raw_ids in zip(names, ids):
            labels = parse_labels(raw_names)
            if pd.isna(raw_ids) if not isinstance(raw_ids, (list, tuple)) else False:
                identities = []
            else:
                text = str(raw_ids).strip()
                try:
                    identities = ast.literal_eval(text) if text.startswith('[') else text.split(',')
                except (SyntaxError, ValueError):
                    identities = []
                if isinstance(raw_ids, (list, tuple)):
                    identities = raw_ids
                identities = list(dict.fromkeys(str(v).strip() for v in identities if str(v).strip()))
            records.append((labels, identities))
        return records

    def _categories(self, frame):
        result = {}
        if self.config.artist:
            result['artist'] = [(['id:' + v for v in ids] if ids else
                [self.artist_aliases.get(v, 'name:' + v) for v in names])
                for names, ids in self._artist_records(frame)]
        if self.config.genre:
            result['genre'] = [parse_labels(v) for v in first_present(frame, ['track_genre', 'genre'])]
        if self.config.artist_genre:
            result['artist_genre'] = [parse_labels(v) for v in series(frame, 'artist_genres')]
        if self.config.album:
            # Album titles alone are ambiguous across artists; require a stable ID.
            result['album'] = [[str(v).strip()] if pd.notna(v) and str(v).strip() else []
                               for v in series(frame, 'album_id')]
        if self.config.time_signature_encoding == 'categorical':
            result['time_signature'] = [[str(int(v))] if pd.notna(v) and np.isfinite(v) and v > 0 and v % 1 == 0 else []
                for v in pd.to_numeric(series(frame, 'time_signature'), errors='coerce')]
        return result

    def fit(self, frame, training_ids):
        frame = clean_catalog(frame)
        train = frame[frame.id.isin(set(map(str, training_ids)))]
        if train.empty:
            raise ValueError('No catalog tracks match training IDs')
        self.artist_aliases = {}
        if self.config.artist:
            aliases = {}
            for names, ids in self._artist_records(train):
                if len(names) == len(ids):
                    for name, identity in zip(names, ids):
                        aliases.setdefault(name, set()).add('id:' + identity)
            self.artist_aliases = {name: next(iter(ids)) for name, ids in aliases.items() if len(ids) == 1}
        values = self._numeric(train)
        self.numeric_names = list(values)
        self.scales = {}
        fixed = set(BOUNDED) | {'mode', 'key', 'key_sin', 'key_cos', 'duration_ms_is_estimated'}
        for name in values:
            if name not in fixed and not name.endswith('_is_local') and values[name].notna().any():
                self.scales[name] = [float(values[name].min()), float(values[name].max())]
        self.vocabularies = {name: {label: i + 2 for i, label in enumerate(sorted({v for row in rows for v in row}))}
                             for name, rows in self._categories(train).items()}
        self.fitted = True
        return self

    def transform(self, frame):
        if not self.fitted:
            raise ValueError('Fit preprocessing on training tracks first')
        frame = clean_catalog(frame)
        values = self._numeric(frame)[self.numeric_names]
        mask = values.notna().to_numpy(dtype=np.float32)
        fixed = set(BOUNDED) | {'mode', 'key', 'key_sin', 'key_cos', 'duration_ms_is_estimated'}
        for name in values:
            if name in self.scales:
                lo, hi = self.scales[name]
                values[name] = ((values[name] - lo) / (hi - lo if hi > lo else 1)).clip(0, 1)
            elif name not in fixed and not name.endswith('_is_local'):
                # No training observations: do not introduce uncalibrated magnitudes.
                values[name] = np.nan
                mask[:, self.numeric_names.index(name)] = 0
        result = {'numeric': values.fillna(0).to_numpy(dtype=np.float32), 'numeric_mask': mask}
        for name, rows in self._categories(frame).items():
            encoded = np.zeros((len(frame), self.config.max_labels), dtype=np.int64)
            observed = np.zeros((len(frame), 1), dtype=np.float32)
            for i, labels in enumerate(rows):
                observed[i] = bool(labels)
                indexes = list(dict.fromkeys(self.vocabularies[name].get(v, 1) for v in labels))
                encoded[i, :min(len(indexes), self.config.max_labels)] = indexes[:self.config.max_labels]
            result[name] = encoded
            result[name + '_mask'] = observed
        return result

    def state_dict(self):
        if not self.fitted:
            raise ValueError('Preprocessor has not been fitted')
        return dict(version=SCHEMA_VERSION, config=asdict(self.config), numeric_names=self.numeric_names,
                    vocabularies=self.vocabularies, scales=self.scales, artist_aliases=self.artist_aliases,
                    duration_policy=self.duration_policy)

    @classmethod
    def from_state_dict(cls, state):
        if state.get('version') != SCHEMA_VERSION:
            raise ValueError('Unsupported preprocessing schema version')
        obj = cls(FeatureConfig(**state['config']))
        obj.duration_policy = state.get('duration_policy', 'legacy')
        if obj.duration_policy not in ('exclude_estimated', 'include_estimated', 'legacy'):
            raise ValueError('Unsupported duration policy')
        for name in ('numeric_names', 'vocabularies', 'scales', 'artist_aliases'):
            setattr(obj, name, state[name])
        obj.fitted = True
        return obj

    def save(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), ensure_ascii=False, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding='utf-8')))
