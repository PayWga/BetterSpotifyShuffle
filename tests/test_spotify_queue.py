import importlib

import pytest


def api():
    assert importlib.util.find_spec('scripts.spotify_queue'), 'live Spotify gateway missing'
    return importlib.import_module('scripts.spotify_queue')


def track(identity, **extra):
    return dict(id=identity, type='track', duration_ms=180000,
                artists=[dict(id='artist', name='Artist')], **extra)


class Client:
    def __init__(self):
        self.calls = []
        self.version = 'v1'
    def playlist(self, identity, fields=None):
        self.calls.append(('playlist', identity))
        return dict(id=identity, snapshot_id=self.version)
    def playlist_items(self, identity, limit=50, offset=0, **kwargs):
        self.calls.append(('items', identity, offset, limit))
        if offset == 0:
            return dict(items=[{'item': track('a')}, {'track': track('b')}, {'item': None}],
                        next='more', offset=0)
        return dict(items=[{'item': track('episode', type_override='episode')},
                           {'item': track('local', is_local=True)},
                           {'item': track('unavailable', is_playable=False)},
                           {'item': track('c')}], next=None, offset=3)


def test_playlist_pages_handle_new_legacy_and_unavailable_items_without_overwriting_cache():
    m = api(); client = Client()
    # Set the episode type on a copy so other fixture records stay unchanged.
    original = client.playlist_items
    def pages(*args, **kwargs):
        page = original(*args, **kwargs)
        for row in page['items']:
            if row.get('item') and row['item'].pop('type_override', None):
                row['item']['type'] = 'episode'
        return page
    client.playlist_items = pages
    gateway = m.SpotifyQueueGateway(client)
    cache = m.PlaylistCache(gateway, refresh_seconds=300)
    first = cache.load('playlist', now=0)
    assert first.track_ids == ('a', 'b', 'c')
    assert first.catalog.set_index('id').loc['a', 'duration_ms'] == 180000
    calls = len(client.calls)
    assert cache.load('playlist', now=100) is first
    assert len(client.calls) == calls
    assert cache.load('playlist', now=301) is first
    assert len(client.calls) == calls + 1  # An unchanged snapshot needs no item-page requests.


def test_empty_or_unavailable_playlist_is_not_a_successful_candidate_source():
    m = api()
    client = Client()
    client.playlist_items = lambda *args, **kwargs: {'items': [], 'next': None}
    with pytest.raises(m.PlaylistUnavailable):
        m.PlaylistCache(m.SpotifyQueueGateway(client)).load('playlist', now=0)


def test_stalled_pagination_and_changing_snapshots_reject_partial_catalogs():
    m = api(); client = Client()
    client.playlist_items = lambda *args, **kwargs: {'items': [], 'next': 'more'}
    with pytest.raises(m.PlaylistUnavailable):
        m.PlaylistCache(m.SpotifyQueueGateway(client)).load('playlist', now=0)
    client = Client()
    def metadata(*args, **kwargs):
        client.version += 'x'
        return {'snapshot_id': client.version}
    client.playlist = metadata
    with pytest.raises(m.PlaylistUnavailable):
        m.PlaylistCache(m.SpotifyQueueGateway(client)).load('playlist', now=0)


def test_gateway_serializes_additions_and_uses_explicit_active_device():
    m = api()
    class Commands:
        def add_to_queue(self, uri, device_id=None):
            assert uri == 'spotify:track:a' and device_id == 'phone'
            return None
        def current_playback(self):
            return {'is_playing': True}
        def queue(self):
            return {'currently_playing': track('a'), 'queue': [track('b')]}
    gateway = m.SpotifyQueueGateway(Commands())
    assert gateway.add('a', device_id='phone') is None
    assert gateway.playback()['is_playing']
    assert gateway.queue().track_ids == ('b',)


def test_cancelled_source_scan_stops_between_pages_without_caching_partial_result():
    from concurrent.futures import CancelledError
    m = api(); client = Client(); cancelled = [False]
    original = client.playlist_items
    def page(*args, **kwargs):
        value = original(*args, **kwargs); cancelled[0] = True; return value
    client.playlist_items = page
    cache = m.PlaylistCache(m.SpotifyQueueGateway(client))
    with pytest.raises(CancelledError):
        cache.load('playlist', now=0, cancelled=lambda: cancelled[0])
    assert len([c for c in client.calls if c[0] == 'items']) == 1
    assert not cache._cache
