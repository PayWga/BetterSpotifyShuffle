"""Shared required history fields and optional duration provenance."""
import csv


REQUIRED_COLUMNS = (
    'timestamp', 'track_id', 'track_name', 'artist_name', 'ms_played',
    'duration_ms', 'completion_rate', 'reason_start', 'reason_end',
    'event_type', 'is_play', 'is_ghost_skip', 'context_uri', 'context_type',
    'event_source',
)
DURATION_COLUMNS = ('duration_ms_is_estimated', 'duration_ms_source')
NEW_FILE_COLUMNS = REQUIRED_COLUMNS + DURATION_COLUMNS


def read_header(path):
    """Validate required fields while retaining the file's actual column order."""
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        columns = next(csv.reader(stream), [])
    if (not set(REQUIRED_COLUMNS).issubset(columns)
            or len(columns) != len(set(columns))
            or any(not name.strip() for name in columns)):
        raise ValueError('Existing listening history has an incompatible schema. '
                         'Required columns must be present and column names unique.')
    return columns
