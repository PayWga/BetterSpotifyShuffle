"""Serialized Spotify operations and versioned, paginated playlist caching."""
from dataclasses import dataclass
from concurrent.futures import CancelledError
import threading
import time

import pandas as pd
from spotipy.exceptions import SpotifyException


LIVE_SCOPE = ('user-read-playback-state user-read-currently-playing '
              'user-modify-playback-state playlist-read-private playlist-read-collaborative')


class PlaylistUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class QueueSnapshot:
    current_id: str | None
    track_ids: tuple


@dataclass(frozen=True)
class PlaylistSnapshot:
    playlist_id: str
    version: str
    track_ids: tuple[str, ...]
    catalog: pd.DataFrame


class SpotifyQueueGateway:
    def __init__(self, client, *, min_interval=0., clock=time.monotonic, sleeper=time.sleep):
        self.client = client
        self._lock = threading.RLock()
        self._clock, self._sleep = clock, sleeper
        self._interval = min_interval
        self._last = None

    def _call(self, method, *args, **kwargs):
        with self._lock:
            if self._last is not None:
                delay = self._interval - (self._clock() - self._last)
                if delay > 0:
                    self._sleep(delay)
            self._last = self._clock()
            return getattr(self.client, method)(*args, **kwargs)

    def playback(self):
        return self._call('current_playback')

    def queue(self):
        response = self._call('queue') or {}
        if not isinstance(response.get('queue'), list):
            raise ValueError('Spotify returned an invalid queue snapshot')
        current = response.get('currently_playing') or {}
        # Keep unknown items and episodes in their queue positions.
        # Dropping them could make unrelated queue entries look like a match.
        ids = tuple(item.get('id') if isinstance(item, dict) and item.get('type', 'track') == 'track'
                    else None for item in response['queue'])
        return QueueSnapshot(current.get('id'), ids)

    def add(self, track_id, *, device_id):
        return self._call('add_to_queue', f'spotify:track:{track_id}', device_id=device_id)

    def playlist_header(self, identity):
        return self._call('playlist', identity, fields='id,snapshot_id')

    def playlist_page(self, identity, offset):
        return self._call('playlist_items', identity, limit=50, offset=offset, additional_types=('track',))


class PlaylistCache:
    def __init__(self, gateway, *, refresh_seconds=300):
        self.gateway = gateway
        self.refresh_seconds = refresh_seconds
        self._cache = {}
        self._checked = {}

    def load(self, identity, *, now, cancelled=None):
        def check_cancelled():
            if cancelled is not None and cancelled():
                raise CancelledError()
        check_cancelled()
        previous = self._cache.get(identity)
        if previous is not None and now - self._checked[identity] < self.refresh_seconds:
            return previous
        try:
            header = self.gateway.playlist_header(identity) or {}
            version = header.get('snapshot_id')
            if not isinstance(version, str) or not version:
                raise PlaylistUnavailable('Playlist version is unavailable')
            if previous is not None and previous.version == version:
                self._checked[identity] = now
                return previous
            rows, offset = {}, 0
            for _ in range(1000):
                check_cancelled()
                page = self.gateway.playlist_page(identity, offset) or {}
                items = page.get('items')
                if not isinstance(items, list):
                    raise PlaylistUnavailable('Playlist items are unavailable')
                for entry in items:
                    if not isinstance(entry, dict) or entry.get('is_local'):
                        continue
                    item = entry.get('item', entry.get('track'))
                    if not isinstance(item, dict) or item.get('type', 'track') != 'track':
                        continue
                    tid = item.get('id')
                    if not isinstance(tid, str) or not tid or item.get('is_local') or item.get('is_playable') is False:
                        continue
                    artists = item.get('artists') or []
                    album = item.get('album') or {}
                    rows[tid] = dict(id=tid, artists=str([a['name'] for a in artists if a.get('name')]),
                                     artist_ids=str([a['id'] for a in artists if a.get('id')]),
                                     duration_ms=item.get('duration_ms'), album_id=album.get('id'),
                                     release_date=album.get('release_date'))
                if not page.get('next'):
                    break
                if not items:
                    raise PlaylistUnavailable('Playlist pagination did not advance')
                offset += len(items)
            else:
                raise PlaylistUnavailable('Playlist pagination exceeded its safety bound')
            check_cancelled()
            after = self.gateway.playlist_header(identity) or {}
            if after.get('snapshot_id') != version:
                raise PlaylistUnavailable('Playlist changed while loading; retry on the next refresh')
            if not rows:
                raise PlaylistUnavailable('Playlist has no accessible playable tracks')
        except SpotifyException as exc:
            if exc.http_status in (403, 404):
                raise PlaylistUnavailable('Playlist is inaccessible; ownership/collaboration permission may be required') from None
            raise
        snapshot = PlaylistSnapshot(identity, version, tuple(rows), pd.DataFrame(rows.values()))
        self._cache[identity], self._checked[identity] = snapshot, now
        return snapshot
