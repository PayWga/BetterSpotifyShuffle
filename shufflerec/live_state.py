"""Read-only model preparation and a durable, private queue-command journal."""
from dataclasses import dataclass
from concurrent.futures import CancelledError
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import pandas as pd

from .audit import read_catalog
from .features import clean_catalog
from .inference import InferenceEngine
from .training import load_checkpoint


@dataclass(frozen=True)
class SourceBundle:
    playlist_id: str
    version: str
    track_ids: tuple
    engine: object


class EngineProvider:
    """Called only by the single worker; no master catalog or checkpoint writes."""
    def __init__(self, checkpoint, master, cache, *, device='cpu', batch_size=1024, clock=time.monotonic):
        self.checkpoint, self.master, self.cache = checkpoint, master, cache
        self.device, self.batch_size, self.clock = device, batch_size, clock
        self._model = self._preprocessor = self._master = None

    def __call__(self, source, playback, context):
        return self.load(source, playback, context)

    def load(self, source, playback, context, *, cancelled=None):
        def check_cancelled():
            if cancelled is not None and cancelled():
                raise CancelledError()
        check_cancelled()
        options = {'cancelled': cancelled} if cancelled is not None else {}
        playlist = self.cache.load(source, now=self.clock(), **options)
        if self._model is None:
            check_cancelled()
            model, preprocessor, _ = load_checkpoint(self.checkpoint, device=self.device)
            master = clean_catalog(read_catalog(self.master)).set_index('id')
            self._model, self._preprocessor, self._master = model, preprocessor, master
        check_cancelled()
        current = (playback or {}).get('item') or {}
        identities = set(playlist.track_ids) | set(context)
        if current.get('id'):
            identities.add(current['id'])
        frame = self._master.reindex(sorted(identities)).astype(object)
        extra = playlist.catalog.copy()
        if current.get('id'):
            artists, album = current.get('artists') or [], current.get('album') or {}
            row = dict(id=current['id'], artists=str([a['name'] for a in artists if a.get('name')]),
                       artist_ids=str([a['id'] for a in artists if a.get('id')]),
                       duration_ms=current.get('duration_ms'), album_id=album.get('id'),
                       release_date=album.get('release_date'))
            extra = pd.concat([extra, pd.DataFrame([row])], ignore_index=True)
        for row in extra.to_dict('records'):
            identity = row.pop('id')
            for name, value in row.items():
                if name not in frame:
                    frame[name] = pd.Series(index=frame.index, dtype=object)
                previous = frame.at[identity, name]
                absent = pd.isna(previous) or isinstance(previous, str) and not previous.strip()
                if absent and pd.notna(value):
                    frame.at[identity, name] = value
        frame['id'] = frame.index
        check_cancelled()
        engine = InferenceEngine(self._model, self._preprocessor, frame.reset_index(drop=True),
                                 batch_size=self.batch_size)
        return SourceBundle(source, playlist.version, playlist.track_ids, engine)


class CommandJournal:
    """Intent is persisted before POST; uncertain operations are never replayed."""
    def __init__(self, path):
        self.path = Path(path)

    def unresolved(self):
        if not self.path.exists():
            return []
        latest = {}
        try:
            with self.path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get('command_id'), str) or not isinstance(row.get('track_id'), str):
                        raise ValueError()
                    if row.get('status') not in ('intent', 'queued', 'finished', 'abandoned'):
                        raise ValueError()
                    latest[row['command_id']] = row
        except (OSError, ValueError, TypeError):
            raise ValueError('Command journal is unreadable; refusing unverified queue writes') from None
        return [row for row in latest.values() if row['status'] in ('intent', 'queued')]

    def record(self, status, command):
        row = {**command, 'status': status, 'timestamp': datetime.now(timezone.utc).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
