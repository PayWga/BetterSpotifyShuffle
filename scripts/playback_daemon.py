"""One playback read feeds adaptive scheduling, history and evidence consumers."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from requests.exceptions import RequestException
from spotipy.exceptions import SpotifyException

from shufflerec.playback import PlaybackTracker, Snapshot, finite
from .spotify_budget import SpotifyBudgetBlocked


EVIDENCE_PATH = Path(__file__).resolve().parents[1] / 'sensitive_info' / 'playback_evidence.jsonl'


def write_evidence(event, *, path=EVIDENCE_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {'timestamp': datetime.now(timezone.utc).isoformat(), **asdict(event)}
    with path.open('a', encoding='utf-8') as output:
        output.write(json.dumps(row, allow_nan=False) + '\n')


def snapshot_from_playback(playback, observed_at):
    playback = playback or {}
    item = playback.get('item') or {}
    progress = playback.get('progress_ms')
    valid_track = item.get('id') and item.get('type', 'track') == 'track' and finite(progress) and progress >= 0
    duration = item.get('duration_ms')
    if not finite(duration) or duration <= 0:
        duration = None
    return Snapshot(str(item['id']) if valid_track else None, duration,
                    progress if valid_track else 0, bool(playback.get('is_playing')) if valid_track else False,
                    observed_at, (playback.get('device') or {}).get('id'),
                    (playback.get('context') or {}).get('uri'))


class PlaybackRecorder:
    def __init__(self, *, event_sink=None, history_sink=None, historical_skip_threshold=.15):
        self.event_sink, self.history_sink = event_sink, history_sink
        self.historical_skip_threshold = historical_skip_threshold
        self.metadata = {}

    def consume(self, events, item=None, context=None):
        for event in events:
            if event.kind == 'started' and item:
                self.metadata[event.occurrence_id] = (item, context or {})
            if self.event_sink is not None:
                self.event_sink(event)
            if event.kind == 'ended':
                record = self.metadata.pop(event.occurrence_id, None)
                if self.history_sink is not None and record and not event.inferred:
                    track, source = record
                    reason = event.reason
                    duration = track.get('duration_ms')
                    # Zero measured listening alone does not prove an early skip.
                    # Outages, device changes, and uncertain timing need to stay
                    # out of the historical GHOST_SKIP label.
                    if reason == 'track_change' and (not finite(duration) or duration <= 0
                            or event.listened_ms < duration * self.historical_skip_threshold):
                        proven_skip = (event.outcome == 'rejected' and not event.weak
                                       and finite(duration) and duration > 0 and finite(event.upper_bound_ms)
                                       and event.upper_bound_ms < duration * self.historical_skip_threshold)
                        if not proven_skip:
                            reason = 'uncertain_transition'
                    self.history_sink(track, int(event.listened_ms), source.get('uri'), source.get('type'), reason)



def run_polling(client, *, tracker=None, clock=time.monotonic, sleeper=time.sleep,
                event_sink=None, history_sink=None, max_polls=None, historical_skip_threshold=.15,
                continuity_check=None):
    """Read-only playback loop. No metadata lookups or queue reads/writes.

    Tests inject client/time/sinks. This standalone observer has no acknowledged
    queue, so only an integrated PlaybackSession may assert queue continuity.
    """
    tracker = tracker or PlaybackTracker()
    polls = 0

    recorder = PlaybackRecorder(event_sink=event_sink, history_sink=history_sink,
                                historical_skip_threshold=historical_skip_threshold)
    dispatch = recorder.consume

    try:
        while max_polls is None or polls < max_polls:
            polls += 1
            try:
                playback = client.current_playback()
                snapshot = snapshot_from_playback(playback, clock())
                continuity = continuity_check(snapshot) if continuity_check is not None else False
                events = tracker.observe(snapshot, continuity=continuity)
                dispatch(events, (playback or {}).get('item'), (playback or {}).get('context'))
                delay = tracker.next_delay()
            except SpotifyBudgetBlocked:
                tracker.interrupt()
                print('Spotify observation stopped: shared request budget/cooldown is active.')
                break
            except (SpotifyException, RequestException):
                tracker.interrupt()
                print('Spotify observation unavailable; timing continuity has been reset.')
                delay = tracker.config.idle_poll_seconds
            if max_polls is None or polls < max_polls:
                sleeper(delay)
    except KeyboardInterrupt:
        pass
    finally:
        dispatch(tracker.finish())
