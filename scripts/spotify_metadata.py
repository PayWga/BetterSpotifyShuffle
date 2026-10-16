"""Missing-only catalog enrichment; independent of audio extraction dependencies."""
import json
import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd
from requests.exceptions import RequestException
from spotipy.exceptions import SpotifyException


TRACK_FIELDS = (
    'name', 'track_name', 'artists', 'artist_name', 'artist_ids', 'album',
    'album_name', 'album_id', 'release_date', 'release_date_precision', 'year',
    'duration_ms', 'explicit', 'track_number', 'disc_number', 'popularity', 'isrc',
)
EXTRA_FIELDS = ('label', 'artist_genres')
NUMERIC_FIELDS = {'year', 'release_year', 'duration_ms', 'track_number', 'disc_number', 'popularity'}


def missing(value, field=''):
    return (pd.isna(value) or not str(value).strip()
            or (field in NUMERIC_FIELDS and str(value).strip().casefold() == 'nan'))


def read_catalog(path):
    df = pd.read_csv(path, dtype='string', keep_default_na=False)
    if 'id' not in df:
        raise ValueError('Master CSV must have an id column')
    if df['id'].str.strip().eq('').any() or df['id'].duplicated().any():
        raise ValueError('Invalid/duplicate master IDs: repair the database first')
    return df.set_index('id')


def local_metadata(row):
    """Use only equivalent metadata, without cleaning or rewriting existing cells."""
    result = {}
    for aliases in [('name', 'track_name'), ('artists', 'artist_name'), ('album', 'album_name')]:
        value = next((row.get(key) for key in aliases if not missing(row.get(key))), None)
        if value is not None:
            result.update(dict.fromkeys(aliases, value))
    date = str(row.get('release_date', '')).strip()
    if re.fullmatch(r'\d{4}(?:-\d{2})?(?:-\d{2})?', date):
        result['year'] = int(date[:4])
        result['release_year'] = int(date[:4])
    return result


def track_metadata(track):
    album = track.get('album') or {}
    artists = track.get('artists') or []
    result = {key: track.get(key) for key in
              ('name', 'duration_ms', 'explicit', 'track_number', 'disc_number', 'popularity')}
    result.update({
        'track_name': track.get('name'),
        'artists': ', '.join(a['name'] for a in artists if a.get('name')) or None,
        'artist_ids': ', '.join(a['id'] for a in artists if a.get('id')) or None,
        'album': album.get('name'), 'album_name': album.get('name'),
        'album_id': album.get('id'), 'release_date': album.get('release_date'),
        'release_date_precision': album.get('release_date_precision'),
        'isrc': (track.get('external_ids') or {}).get('isrc'),
    })
    result['artist_name'] = result['artists']
    result.update({k: v for k, v in local_metadata(result).items() if k in ('year', 'release_year')})
    return {key: value for key, value in result.items() if not missing(value, key)}


def request(call):
    """Retry transient failures; do not hide authentication/permission failures."""
    for attempt in range(4):
        try:
            result = call()
            time.sleep(0.1)
            return result
        except SpotifyException as exc:
            if attempt == 3 or (exc.http_status != 429 and exc.http_status < 500):
                raise
            delay = float((exc.headers or {}).get('Retry-After', 2 ** attempt))
        except RequestException:
            if attempt == 3:
                raise
            delay = 2 ** attempt
        time.sleep(max(0, delay))


def fetch_track(sp, track_id, extra_metadata=False, cache=None):
    track = request(lambda: sp.track(track_id))
    if not track or not track.get('id'):
        return {}
    # Spotify may return a relinked recording. The caller retains the requested
    # catalog ID; never rename IDs or attach metadata by response order.
    result = track_metadata(track)
    if not extra_metadata:
        return result
    cache = {} if cache is None else cache
    album_id = (track.get('album') or {}).get('id')
    endpoints = [('album', album_id)] if album_id else []
    endpoints += [('artist', a['id']) for a in track.get('artists', []) if a.get('id')]
    genres = set()
    for kind, item_id in endpoints:
        key = (kind, item_id)
        if key not in cache:
            try:
                cache[key] = request(lambda: getattr(sp, kind)(item_id)) or {}
            except SpotifyException as exc:
                if exc.http_status != 404:
                    raise
                cache[key] = {}
        data = cache[key]
        if kind == 'album' and data.get('label'):
            result['label'] = data['label']
        if kind == 'artist':
            genres.update(data.get('genres') or [])
    if genres:
        result['artist_genres'] = json.dumps(sorted(genres), ensure_ascii=False)
    return result


def write_checkpoint(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.metadata_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(state, output, ensure_ascii=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_checkpoint(path, master_path):
    with path.open(encoding='utf-8') as source:
        state = json.load(source)
    if state.get('version') != 1 or state.get('master') != str(master_path.resolve()):
        raise ValueError('Checkpoint belongs to a different catalog or version; select another --checkpoint')
    if not isinstance(state.get('records'), dict) or not isinstance(state.get('fields'), list):
        raise ValueError('Invalid metadata checkpoint')
    if set(state['fields']) - set(TRACK_FIELDS + EXTRA_FIELDS + ('release_year',)):
        raise ValueError('Unsupported checkpoint fields')
    return state


def fill_records(df, records, fields, allow_new=False):
    counts = {}
    for tid, record in records.items():
        if tid not in df.index:
            if not allow_new:
                continue
            df.loc[tid] = dict.fromkeys(df.columns, '')
        for field in fields:
            value = record.get(field)
            if missing(value, field):
                continue
            if field not in df:
                df[field] = pd.Series(pd.NA, index=df.index, dtype='string')
            if missing(df.at[tid, field], field):
                df.at[tid, field] = str(value)
                counts[field] = counts.get(field, 0) + 1
    return counts


def apply_checkpoint(master_path, checkpoint, save, dry_run=False):
    master_path, checkpoint = Path(master_path), Path(checkpoint)
    state = load_checkpoint(checkpoint, master_path)
    # Reload here: extraction may have added measurements since the fetch began.
    df = read_catalog(master_path)
    counts = fill_records(df, state['records'], state['fields'])
    if counts and not dry_run:
        save(df.reset_index(), master_path)
    stats = {'updated_cells': sum(counts.values()), 'columns': counts}
    print(f"{'DRY RUN: would fill' if dry_run else 'Filled'} {stats['updated_cells']} metadata cells: {counts}")
    return stats


def backfill(master_path, checkpoint, client_factory, save, *, sp=None, fields=None,
             dry_run=False, fetch_only=False, limit=None, extra_metadata=False):
    master_path, checkpoint = Path(master_path), Path(checkpoint)
    if master_path.resolve() == checkpoint.resolve():
        raise ValueError('Checkpoint must be separate from the master CSV')
    if limit is not None and limit <= 0:
        raise ValueError('limit must be positive')
    fields = list(fields if fields is not None else TRACK_FIELDS + (EXTRA_FIELDS if extra_metadata else ()))
    if not fields or set(fields) - set(TRACK_FIELDS + EXTRA_FIELDS + ('release_year',)):
        raise ValueError('Select supported metadata fields only')
    if set(fields) & set(EXTRA_FIELDS) and not extra_metadata:
        raise ValueError('label and artist_genres require --extra-metadata')
    df = read_catalog(master_path)
    if checkpoint.exists():
        state = load_checkpoint(checkpoint, master_path)
        if state['fields'] != fields or state['extra_metadata'] != extra_metadata:
            raise ValueError('Checkpoint options differ; select another --checkpoint')
    else:
        state = {'version': 1, 'master': str(master_path.resolve()), 'fields': fields,
                 'extra_metadata': extra_metadata, 'records': {}, 'unavailable': [], 'retry': []}
    state.setdefault('retry', [])
    eligible = [tid for tid, row in df.iterrows()
                if any(missing(row.get(field), field) for field in fields)]
    stats = {'eligible': len(eligible), 'fetched': 0, 'resumed': 0,
             'unavailable': 0, 'invalid_ids': 0, 'failed': 0, 'updated_cells': 0}
    print(f"{len(eligible)} tracks have missing requested metadata.")
    if dry_run:
        print('DRY RUN: no authentication, requests, or writes.')
        return stats
    cache = {}
    processed = 0
    dirty = False
    try:
        for tid in eligible:
            if tid in state['unavailable']:
                stats['unavailable'] += 1
                continue
            if tid in state['records'] and tid not in state['retry']:
                stats['resumed'] += 1
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1
            row = df.loc[tid]
            record = {key: value for key, value in local_metadata(row).items() if key in fields}
            needed = [f for f in fields if missing(row.get(f), f) and f not in record]
            if needed and not re.fullmatch(r'[A-Za-z0-9]{22}', tid):
                stats['invalid_ids'] += 1
            elif needed:
                if sp is None:
                    sp = client_factory()
                try:
                    fetched = fetch_track(sp, tid, extra_metadata, cache)
                    record.update({k: v for k, v in fetched.items() if k in fields and k not in record})
                    stats['fetched'] += 1
                    if not fetched:
                        state['unavailable'].append(tid)
                        stats['unavailable'] += 1
                except (SpotifyException, RequestException) as exc:
                    if isinstance(exc, SpotifyException) and exc.http_status in (401, 403):
                        raise
                    if isinstance(exc, SpotifyException) and exc.http_status == 404:
                        state['unavailable'].append(tid)
                        stats['unavailable'] += 1
                    else:
                        stats['failed'] += 1
                        if tid not in state['retry']:
                            state['retry'].append(tid)
                        print(f'Request failed (HTTP {getattr(exc, "http_status", "network")}); rerun to retry.')
                    # Preserve useful local aliases even if Spotify failed, but
                    # leave transient failures eligible for the next run.
                    if record:
                        state['records'][tid] = record
                    dirty = True
                    if processed % 50 == 0:
                        write_checkpoint(state, checkpoint)
                        dirty = False
                        print(f'Checked {processed} tracks; fetched {stats["fetched"]}.')
                    continue
            state['records'][tid] = record
            if tid in state['retry']:
                state['retry'].remove(tid)
            dirty = True
            if processed % 50 == 0:
                write_checkpoint(state, checkpoint)
                dirty = False
                print(f'Checked {processed} tracks; fetched {stats["fetched"]}.')
    finally:
        if dirty or not checkpoint.exists():
            write_checkpoint(state, checkpoint)
    if not fetch_only:
        stats.update(apply_checkpoint(master_path, checkpoint, save))
    print(f'Backfill: {stats}. Checkpoint: {checkpoint}')
    return stats
