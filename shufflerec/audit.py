"""Aggregate-only, read-only catalog audit and selective model input loading."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

from .features import AUDIO, EXTRACTED, LOCAL, FeaturePreprocessor, first_present, series

CATALOG_COLUMNS = set(AUDIO + ('id', 'key', 'time_signature', 'artists', 'artist_name',
    'artist_ids', 'year', 'release_year', 'release_date', 'duration_ms', 'track_genre',
    'genre', 'artist_genres', 'album_id', 'popularity') + LOCAL)
CATALOG_COLUMNS.update(name + suffix for name in EXTRACTED for suffix in ('_source', '_analysis_seconds'))


def signature(path):
    info = Path(path).stat()
    return info.st_size, info.st_mtime_ns


def read_catalog(path, *, track_ids=None, chunksize=100000):
    before = signature(path)
    selected_ids = None if track_ids is None else set(map(str, track_ids))
    frames = []
    for chunk in pd.read_csv(path, usecols=lambda name: name in CATALOG_COLUMNS,
                             dtype={'id': 'string', 'artist_ids': 'string', 'album_id': 'string'},
                             chunksize=chunksize, low_memory=False):
        if 'id' not in chunk:
            raise ValueError("Master table requires 'id'")
        chunk['id'] = chunk.id.str.strip()
        if selected_ids is not None:
            chunk = chunk[chunk.id.isin(selected_ids)]
        frames.append(chunk)
    if before != signature(path):
        raise RuntimeError('Catalog changed during loading; retry against a stable snapshot')
    if not frames:
        raise ValueError('Catalog is empty')
    return pd.concat(frames, ignore_index=True)


def read_history(path):
    before = signature(path)
    frame = pd.read_csv(path, usecols=['timestamp', 'track_id', 'event_type'],
                        dtype={'track_id': 'string'}, low_memory=False)
    if before != signature(path):
        raise RuntimeError('History changed during loading; retry against a stable snapshot')
    return frame


def audit_paths(master_path, history_path, *, chunksize=100000):
    before = signature(master_path)
    history = read_history(history_path)
    history_ids = set(history.track_id.dropna().str.strip()) - {''}
    rows, duplicate_ids = 0, 0
    seen, matched = set(), set()
    counts, raw_counts = Counter(), Counter()
    preprocessor = FeaturePreprocessor()
    present_columns = set()
    for chunk in pd.read_csv(master_path, usecols=lambda name: name in CATALOG_COLUMNS,
                             dtype={'id': 'string'}, chunksize=chunksize, low_memory=False):
        if 'id' not in chunk:
            raise ValueError("Master table requires 'id'")
        rows += len(chunk)
        present_columns.update(chunk.columns)
        ids = chunk.id.dropna().str.strip()
        duplicate_ids += int(ids.isin(seen).sum() + ids[~ids.isin(seen)].duplicated().sum())
        seen.update(ids)
        matched.update(set(ids) & history_ids)
        counts.update(preprocessor._numeric(chunk).notna().sum().to_dict())
        for name in ('artists', 'artist_ids', 'year', 'release_date', 'duration_ms',
                     'track_genre', 'genre', 'artist_genres', 'album_id', 'popularity'):
            values = first_present(chunk, ['artists', 'artist_name']) if name == 'artists' else series(chunk, name)
            raw_counts[name] += int((values.notna() & values.astype(str).str.strip().ne('')).sum())
    if before != signature(master_path):
        raise RuntimeError('Catalog changed during audit; retry against a stable snapshot')
    return {'catalog_rows': rows, 'duplicate_ids': duplicate_ids, 'history_events': len(history),
            'unique_history_tracks': len(history_ids), 'matched_history_tracks': len(matched),
            'missing_history_tracks': len(history_ids - matched),
            'matched_history_events': int(history.track_id.str.strip().isin(matched).sum()),
            'valid_numeric_counts': dict(counts), 'populated_metadata_counts': dict(raw_counts),
            'missing_extended_columns': sorted(set(LOCAL + ('artist_genres',)) - present_columns),
            'note': 'Counts describe this catalog snapshot. Metadata counts are populated cells, not verified labels. '
                    'No private rows or names are included. Duration candidates are excluded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', required=True, type=Path)
    parser.add_argument('--history', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_paths(args.master, args.history), indent=2))


if __name__ == '__main__':
    main()
