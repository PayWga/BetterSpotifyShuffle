"""Offline, missing-only metadata recovery from explicitly supported local files."""
from datetime import date
from decimal import Decimal, InvalidOperation
import json
import re
from pathlib import Path

import pandas as pd

if __package__:
    from . import spotify_metadata as sm
else:
    import spotify_metadata as sm


GROUPS = [('name', 'track_name'), ('artists', 'artist_name'), ('album', 'album_name')]
ALIASES = {field: group for group in GROUPS for field in group}
FIELDS = set(sm.TRACK_FIELDS + sm.EXTRA_FIELDS + ('release_year',))
INTEGERS = {'year', 'release_year', 'duration_ms', 'track_number', 'disc_number', 'popularity',
            'history_duration_candidate_ms'}
MATCHING_FIELDS = {'spotify_track_name', 'spotify_artist_names'}


def missing_mask(df, field):
    if field not in df:
        return pd.Series(True, index=df.index)
    values = df[field]
    result = values.isna() | values.str.strip().eq('')
    if field in INTEGERS:
        result = result | values.str.strip().str.casefold().eq('nan')
    return result


def put(df, field, values, source, counts):
    """Vectorized assignment; provenance changes only for newly filled cells."""
    values = values.reindex(df.index).astype('string')
    fill = missing_mask(df, field) & values.notna() & values.str.strip().ne('')
    n = int(fill.sum())
    if not n:
        return
    if field not in df:
        df[field] = pd.Series('', index=df.index, dtype='string')
    df.loc[fill, field] = values[fill]
    provenance = f'{field}_metadata_source'
    if provenance not in df:
        df[provenance] = pd.Series('', index=df.index, dtype='string')
    if isinstance(source, pd.Series):
        df.loc[fill, provenance] = source.reindex(df.index)[fill]
    else:
        df.loc[fill, provenance] = source
    counts[field] = counts.get(field, 0) + n


def fill_aliases(df, counts, origin='master'):
    for group in GROUPS:
        for target in group:
            for source in group:
                if source != target and source in df:
                    put(df, target, df[source], f'{origin}:{source}', counts)
    if 'release_date' not in df:
        return
    dates = df['release_date'].str.strip()
    # Validate complete dates/months before interpreting their detail or year.
    unique = {}
    for text in dates.dropna().unique():
        if not re.fullmatch(r'\d{4}(?:-\d{2})?(?:-\d{2})?', text):
            continue
        try:
            if len(text) == 4 and text.isdigit():
                date(int(text), 1, 1)
                unique[text] = (text, 'year')
            elif len(text) == 7 and text[4] == '-':
                date.fromisoformat(text + '-01')
                unique[text] = (text[:4], 'month')
            elif len(text) == 10:
                date.fromisoformat(text)
                unique[text] = (text[:4], 'day')
        except ValueError:
            continue
    years = dates.map({text: value[0] for text, value in unique.items()})
    put(df, 'year', years, f'{origin}:release_date', counts)
    if 'release_year' in df:
        put(df, 'release_year', years, f'{origin}:release_date', counts)
    precision = dates.map({text: value[1] for text, value in unique.items()})
    put(df, 'release_date_precision', precision, f'{origin}:release_date', counts)


def normalize(field, value):
    if field == 'spotify_artist_names':
        try:
            names = json.loads(value) if isinstance(value, str) else value
        except ValueError:
            return None
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
            return None
        return json.dumps([n.strip() for n in names], ensure_ascii=False)
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if field in INTEGERS:
        try:
            number = Decimal(str(value).strip())
        except InvalidOperation:
            return None
        if not number.is_finite() or number != number.to_integral_value() or number < 0:
            return None
        if field in ('duration_ms', 'history_duration_candidate_ms', 'track_number', 'disc_number') and number <= 0:
            return None
        if field in ('year', 'release_year') and not 1850 <= number <= 2100:
            return None
        if field == 'popularity' and number > 100:
            return None
        return str(int(number))
    if field == 'explicit':
        return {'true': 'True', '1': 'True', 'false': 'False', '0': 'False'}.get(str(value).strip().casefold())
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False) if field == 'artist_genres' else None
    text = str(value).strip()
    return text or None


class Candidates:
    def __init__(self, df, allow_matching=False):
        self.df = df
        self.gaps = {}
        self.values = {}
        self.conflicts = {}
        self.fields = FIELDS | (MATCHING_FIELDS if allow_matching else set())

    def add(self, tid, field, value, source):
        if field not in self.fields and field != 'history_duration_candidate_ms':
            return
        if tid not in self.df.index:
            return
        for target in ALIASES.get(field, (field,)):
            if target not in self.gaps:
                self.gaps[target] = self.df.index[missing_mask(self.df, target)]
            if tid not in self.gaps[target]:
                continue
            text = normalize(target, value)
            if text is None:
                continue
            records = self.values.setdefault(target, {})
            if tid not in records:
                records[tid] = (text, source)
            elif records[tid] is not None and records[tid][0] != text:
                records[tid] = None
                self.conflicts[target] = self.conflicts.get(target, 0) + 1

    def apply(self, counts):
        for field, records in self.values.items():
            valid = {tid: record for tid, record in records.items() if record is not None}
            if valid:
                put(self.df, field, pd.Series({tid: record[0] for tid, record in valid.items()}),
                    pd.Series({tid: record[1] for tid, record in valid.items()}), counts)


def confirmed_duration(row):
    return (row.get('event_source') in ('polling', 'live_polling')
            or row.get('duration_ms_source') in ('spotify', 'spotify_track_api', 'spotify_polling')
            or str(row.get('duration_ms_metadata_source', '')).startswith(
                ('spotify_cache:', 'history_csv:', 'master_csv:')))


def history_csv(path, candidates):
    columns = {'track_id', 'id', 'track_name', 'name', 'artist_name', 'artists', 'artist',
               'album', 'album_name', 'duration_ms', 'event_source', 'duration_ms_source',
               'duration_ms_metadata_source'} | MATCHING_FIELDS
    for chunk in pd.read_csv(path, dtype='string', keep_default_na=False, chunksize=100000,
                             usecols=lambda col: col in columns):
        if 'track_id' not in chunk and 'id' not in chunk:
            raise ValueError(f'History CSV needs track_id or id: {path.name}')
        for row in chunk.drop_duplicates().to_dict('records'):
            tid = str(row.get('track_id', row.get('id', ''))).strip()
            source = f'history_csv:{path.name}'
            for field in ('name', 'track_name', 'artists', 'artist_name', 'album', 'album_name'):
                candidates.add(tid, field, row.get(field), source)
            candidates.add(tid, 'artist_name', row.get('artist'), source)
            for field in MATCHING_FIELDS:
                candidates.add(tid, field, row.get(field), source)
            verified = confirmed_duration(row)
            duration_field = 'duration_ms' if verified else 'history_duration_candidate_ms'
            # ms_played/listened_ms never enter the recording duration column.
            candidates.add(tid, duration_field, row.get('duration_ms'), source)


def history_json(path, candidates):
    with path.open(encoding='utf-8-sig') as source:
        rows = json.load(source)
    if not isinstance(rows, list):
        raise ValueError(f'Audio history JSON must contain a list: {path.name}')
    for row in rows:
        uri = row.get('spotify_track_uri')
        if not isinstance(uri, str) or not uri.startswith('spotify:track:'):
            continue
        tid = uri.removeprefix('spotify:track:')
        for field, key in [('track_name', 'master_metadata_track_name'),
                           ('artist_name', 'master_metadata_album_artist_name'),
                           ('album_name', 'master_metadata_album_album_name')]:
            candidates.add(tid, field, row.get(key), f'history_json:{path.name}')


def spotify_cache(path, candidates):
    with path.open(encoding='utf-8') as source:
        state = json.load(source)
    if state.get('version') != 1 or not isinstance(state.get('records'), dict):
        raise ValueError(f'Unsupported Spotify checkpoint: {path.name}')
    fields = set(state.get('fields', [])) & candidates.fields
    for tid, row in state['records'].items():
        if not isinstance(row, dict):
            raise ValueError(f'Invalid Spotify checkpoint record: {path.name}')
        for field in fields:
            candidates.add(tid, field, row.get(field), f'spotify_cache:{path.name}')


def spotify_backlog(path, candidates):
    df = pd.read_csv(path, dtype='string', keep_default_na=False)
    if 'track_id' not in df:
        raise ValueError('Backlog CSV needs track_id')
    for row in df.to_dict('records'):
        if row.get('spotify_metadata_source') != 'spotify_track_api' or row.get('spotify_metadata_error'):
            continue
        tid = str(row['track_id']).strip()
        verified_duration = row.get('duration_ms_source') in ('spotify', 'spotify_track_api', 'spotify_polling')
        duration_field = 'duration_ms' if verified_duration else 'history_duration_candidate_ms'
        candidates.add(tid, duration_field, row.get('duration_ms'), 'spotify_backlog')
        candidates.add(tid, 'isrc', row.get('isrc'), 'spotify_backlog')
        candidates.add(tid, 'track_name', row.get('spotify_track_name'), 'spotify_backlog')
        try:
            artists = json.loads(row.get('spotify_artist_names') or '[]')
        except ValueError:
            artists = []
        if isinstance(artists, list) and artists and all(isinstance(a, str) for a in artists):
            candidates.add(tid, 'artist_name', ', '.join(artists), 'spotify_backlog')


def master_csv(path, candidates):
    columns = candidates.fields | {'id', 'duration_ms_source', 'duration_ms_metadata_source'}
    for chunk in pd.read_csv(path, dtype='string', keep_default_na=False, chunksize=100000,
                             usecols=lambda col: col in columns):
        if 'id' not in chunk:
            raise ValueError('Master source needs id')
        # The master can dwarf the backlog; inspect only IDs relevant to this job.
        chunk = chunk[chunk['id'].str.strip().isin(candidates.df.index)]
        for row in chunk.to_dict('records'):
            tid = str(row['id']).strip()
            source = f'master_csv:{path.name}'
            for field in candidates.fields - {'duration_ms'}:
                candidates.add(tid, field, row.get(field), source)
            duration = 'duration_ms' if confirmed_duration(row) else 'history_duration_candidate_ms'
            candidates.add(tid, duration, row.get('duration_ms'), source)


def run(master_path, source_root, save, dry_run=False, *, target='master', include_checkpoints=True):
    path, root = Path(master_path), Path(source_root)
    if target not in ('master', 'backlog'):
        raise ValueError('Unknown local backfill target')
    signature = lambda: (path.stat().st_mtime_ns, path.stat().st_size)
    original = signature()
    print(f'Reading {target} catalog for offline backfill...', flush=True)
    if target == 'master':
        df = sm.read_catalog(path)
    else:
        df = pd.read_csv(path, dtype='string', keep_default_na=False)
        if ('track_id' not in df or df['track_id'].str.strip().eq('').any()
                or df['track_id'].duplicated().any()):
            raise ValueError('Backlog needs unique nonempty track_id values')
        df = df.set_index('track_id')
    counts = {}
    print(f'Filling {target} aliases and deriving date metadata...', flush=True)
    if target == 'master':
        fill_aliases(df, counts)
    else:
        fill_aliases(df, counts, origin=target)
    candidates = Candidates(df, allow_matching=target == 'backlog')
    private, data = root / 'sensitive_info', root / 'data'
    sources = []
    sources += [(p, history_csv) for p in sorted(private.rglob('*.csv'))
                if 'histor' in p.name.casefold() or 'listening' in p.name.casefold()]
    history_dir = private / 'history'
    sources += [(p, history_json) for p in sorted(history_dir.rglob('*.json'))
                if 'audio' in p.name.casefold() or 'streaminghistory' in p.name.casefold()]
    if include_checkpoints:
        sources += [(p, spotify_cache) for p in sorted(data.glob('*.spotify-metadata.json'))]
    if target == 'backlog' and (data / 'master_track_table.csv').exists():
        if path.resolve() == (data / 'master_track_table.csv').resolve():
            raise ValueError('Backlog target must be separate from the master source')
        sources.append((data / 'master_track_table.csv', master_csv))
    if target == 'master' and (data / 'dsp_backlog.csv').exists():
        sources.append((data / 'dsp_backlog.csv', spotify_backlog))
    for number, (source_path, reader) in enumerate(sources, 1):
        print(f'Reading local metadata source {number}/{len(sources)} ({reader.__name__})...', flush=True)
        reader(source_path, candidates)
    candidates.apply(counts)
    # External metadata can complete aliases or supply a release_date.
    if target == 'master':
        fill_aliases(df, counts)
    else:
        fill_aliases(df, counts, origin=target)
    total = sum(counts.values())
    if not dry_run and total:
        if signature() != original:
            raise RuntimeError(f'{target.title()} CSV changed during local backfill; stop other catalog writers')
        save(df.reset_index(), path)
    result = {'updated_cells': total, 'columns': counts, 'conflicts': candidates.conflicts,
              'sources_read': len(sources)}
    print(f"{'DRY RUN: would fill' if dry_run else 'Filled'} {total} local metadata cells: {counts}")
    print(f'Skipped conflicting candidates by column: {candidates.conflicts}')
    print('Offline only: no Spotify authentication or requests.')
    return result
