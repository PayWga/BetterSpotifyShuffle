"""Single owner of live playback state, tagged planning jobs and queue writes."""
from dataclasses import dataclass
from concurrent.futures import CancelledError, ThreadPoolExecutor
import time
import uuid

from requests.exceptions import RequestException
from spotipy.exceptions import SpotifyException

from scripts.playback_daemon import snapshot_from_playback
from scripts.spotify_budget import SpotifyBudgetBlocked
from .playback import PlaybackConfig, PlaybackTracker, finite
from .playback_session import PlaybackSession
from .pools import CatalogMatches, SequencerConfig
from .sequencing import Sequencer


@dataclass(frozen=True)
class LiveConfig:
    buffer_size: int = 5
    refresh_seconds: float = 300.
    follow_playlist: bool = True
    dry_run: bool = False

    def __post_init__(self):
        if isinstance(self.buffer_size, bool) or not isinstance(self.buffer_size, int) or not 1 <= self.buffer_size <= 20:
            raise ValueError('buffer_size must be an integer from 1 to 20')
        if not finite(self.refresh_seconds) or self.refresh_seconds <= 0:
            raise ValueError('refresh_seconds must be positive and finite')
        if not isinstance(self.follow_playlist, bool) or not isinstance(self.dry_run, bool):
            raise ValueError('follow_playlist and dry_run must be boolean')


def _plan(clone, identities, pools, exclusions=()):
    clone.plan(provisional_ids=identities, provisional_pools=pools, relax_recent=True, exclude=exclusions)
    return clone


class LiveController:
    def __init__(self, gateway, source_loader, *, journal, config=None, sequencer_config=None,
                 playback_config=None, clock=time.monotonic, executor=None, recorder=None, seed=42):
        self.gateway, self.loader, self.journal = gateway, source_loader, journal
        self.config = config or LiveConfig()
        self.seq_config = sequencer_config or SequencerConfig()
        self.playback_config = playback_config or PlaybackConfig(acceptance_threshold=self.seq_config.early_skip_threshold)
        self.clock = clock
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='shufflerec-model')
        self._owns_executor = executor is None
        self.recorder, self.seed = recorder, seed
        self.tracker = PlaybackTracker(self.playback_config)
        self.session = None
        self._context = []
        self._source = self._ready_source = None
        self._revision = 0
        self._source_generation = 0
        self._future = self._job = None
        self._next_poll = self._next_refresh = 0.
        self._last = self._raw = self._queue = None
        self._queue_valid = self._trusted = self._anchored = False
        self._needs_queue = True
        self._uncertain = None
        self._commands = {}
        self._orphans = {row['command_id']: row for row in journal.unresolved()}
        self._state = 'starting'
        self._playback_source = None
        self._write_error = None
        self._blocked_track_ids = set()
        self._natural_tail = ()
        self._manual_tail = ()
        self._closed = False
        self._next_plan = 0.

    def _record(self, events, raw=None):
        if self.recorder is not None:
            self.recorder.consume(events, item=(raw or {}).get('item'), context=(raw or {}).get('context'))
        for event in events:
            command = self._commands.get(event.selection_id)
            if event.kind == 'ended' and command:
                self.journal.record('finished', command)
                self._commands.pop(event.selection_id, None)

    def _abandon(self, items):
        if not items:
            return
        for item in items:
            command = self._commands.pop(item.selection_id, None)
            if command:
                self._orphans[command['command_id']] = command
        self.session.abandon([item.selection_id for item in items])
        self._trusted = self._anchored = False
        self._revision += 1

    def _switch_source(self, source):
        if self._source and not self.config.follow_playlist:
            self._state = 'playlist_changed_paused'
            return False
        if self.session:
            self._record(self.session.finish('context_changed'))
            self._abandon(list(self.session.submitted))
        self.tracker.interrupt()
        self._supersede_uncertain()
        self._write_error = None
        self._source = source
        self._source_generation += 1
        self._natural_tail = self._manual_tail = ()
        self._next_refresh = 0.
        self._revision += 1
        self._queue_valid = False
        self._needs_queue = True
        return True

    def _supersede_uncertain(self):
        if self._uncertain:
            command = self._uncertain
            self._blocked_track_ids.add(command['track_id'])
            self._orphans[command['command_id']] = {**command, 'status': 'intent'}
            self._uncertain = None

    def acknowledge_uncertain(self):
        """Operator accepts unknown outcomes; never resend those commands."""
        unknown = {key: row for key, row in self._orphans.items() if row['status'] == 'intent'}
        if self._uncertain:
            unknown[self._uncertain['command_id']] = self._uncertain
        visible = set(self._queue.track_ids) | {self._queue.current_id} if self._queue else set()
        for key, command in unknown.items():
            self._blocked_track_ids.add(command['track_id'])
            if command['track_id'] in visible:
                row = {**command, 'status': 'queued', 'disposition': 'acknowledged_unattributed'}
                self._orphans[key] = row
                self.journal.record('queued', row)
            else:
                self.journal.record('abandoned', command)
                self._orphans.pop(key, None)
        self._uncertain = None
        self._queue_valid = self._trusted = self._anchored = False
        self._needs_queue = True
        self._next_poll = self.clock() + .05
        self._revision += 1
        if self.session:
            self.session.sequencer.queue = []
        self._state = 'reconciling'

    @staticmethod
    def _prefix(queue, items):
        ids = tuple(item.track_id for item in items)
        return queue[:len(ids)] == ids

    def _reconcile(self, snapshot):
        queue = self.gateway.queue()
        self._queue = queue
        if queue.current_id != snapshot.track_id:
            self._queue_valid = self._trusted = False
            self._needs_queue = True
            self._next_poll = min(self._next_poll, self.clock() + 2.)
            return False
        visible = set(queue.track_ids) | {queue.current_id}
        for key, command in list(self._orphans.items()):
            if (command['status'] == 'intent' and command['track_id'] in visible
                    and command['track_id'] not in command.get('before_ids', [])):
                command = {**command, 'status': 'queued'}
                self._orphans[key] = command
                if not self.config.dry_run:
                    self.journal.record('queued', command)
            if command['track_id'] not in visible and command['status'] == 'queued':
                if not self.config.dry_run:
                    self.journal.record('finished', command)
                self._orphans.pop(key)
        if any(command['status'] == 'intent' for command in self._orphans.values()):
            self._state = 'submission_uncertain'
            self._queue_valid = False
            return False
        if self._uncertain:
            command = self._uncertain
            if command['track_id'] not in visible or command['track_id'] in command['before_ids']:
                self._state = 'submission_uncertain'
                self._queue_valid = False
                return False
            if self.session and self.session.sequencer.queue and self.session.sequencer.queue[0].selection_id == command['selection_id']:
                self.session.submit_prefix(1)
                self._commands[command['selection_id']] = {**command, 'status': 'queued'}
                self.journal.record('queued', command)
                self._uncertain = None
        continuity = False
        if self.session and self.session.submitted:
            pending = list(self.session.submitted)
            positions = [i for i, item in enumerate(pending) if item.track_id == snapshot.track_id]
            if len(positions) == 1:
                index = positions[0]
                if self._prefix(queue.track_ids, pending[index + 1:]):
                    if not self._trusted and index:
                        self._abandon(pending[:index])
                    self.session.confirm_order(follows_active=self._anchored)
                    continuity = True
                    self._trusted = True
                else:
                    self._abandon([item for item in pending if item.track_id != snapshot.track_id])
                    self.session.confirm_order()
                    continuity = True
            else:
                ids = tuple(item.track_id for item in pending)
                contiguous = any(queue.track_ids[i:i + len(ids)] == ids
                                 for i in range(max(0, len(queue.track_ids) - len(ids) + 1)))
                if contiguous:
                    self._anchored = self._prefix(queue.track_ids, pending)
                    self.session.confirm_order(follows_active=self._anchored)
                    continuity = True
                    self._trusted = True
                elif self._trusted:
                    self._abandon(pending)
                else:
                    self._state = 'awaiting_queue_confirmation'
                    self._queue_valid = False
                    self._needs_queue = True
                    self._next_poll = min(self._next_poll, self.clock() + self.playback_config.early_poll_seconds)
                    return False
        if self.session and self.session.submitted:
            pending_ids = {item.track_id for item in self.session.submitted}
            positions = [i for i, identity in enumerate(queue.track_ids) if identity in pending_ids]
            if positions:
                tail = queue.track_ids[max(positions) + 1:]
                self._manual_tail = ()
                for count in range(len(tail) + 1):
                    suffix = tail[count:]
                    if suffix == self._natural_tail[:len(suffix)]:
                        self._manual_tail = tail[:count]
                        self._natural_tail = suffix
                        break
                else:
                    self._natural_tail = tail
        else:
            self._natural_tail = queue.track_ids
            self._manual_tail = ()
        self._queue_valid = True
        self._needs_queue = False
        if self._state == 'awaiting_queue_confirmation':
            self._state = 'running'
        return continuity

    def _upcoming(self):
        active_id = self.tracker.active.selection_id if self.tracker.active else None
        own = sum(item.selection_id != active_id for item in self.session.submitted) if self.session else 0
        visible = self._queue.track_ids if self._queue else ()
        orphaned = sum(command['track_id'] in visible for command in self._orphans.values())
        return own + orphaned

    def _collect(self):
        if self._future is None or not self._future.done():
            return False
        future, job = self._future, self._job
        self._future = self._job = None
        try:
            result = future.result()
        except CancelledError:
            return True
        except SpotifyBudgetBlocked:
            raise
        except Exception:
            self._state = 'source_unavailable' if job[0] == 'source' else 'inference_unavailable'
            if job[0] == 'source':
                self._ready_source = None
                if self.session:
                    self.session.sequencer.queue = []
            self._next_refresh = self.clock() + self.config.refresh_seconds
            return True
        if job[1] != self._source:
            return True
        if job[0] == 'source':
            if job[2] != self._source_generation:
                return True
            if self.session is None:
                seq = Sequencer(result.engine, result.track_ids, context_ids=self._context,
                                config=self.seq_config, seed=self.seed)
                self.session = PlaybackSession(seq, tracker=self.tracker, auto_replan=False)
            else:
                seq = self.session.sequencer
                seq.engine, seq.playlist_ids = result.engine, list(result.track_ids)
                seq.matches = CatalogMatches(result.engine.catalog)
                seq.queue = []
                self.session._durations = dict(zip(result.engine.catalog.id, result.engine.catalog.duration_ms)) if 'duration_ms' in result.engine.catalog else {}
            self._ready_source = result.playlist_id
            self._revision += 1
            self._state = 'running'
        elif job[2] == self._revision:
            seq = self.session.sequencer
            seq.queue, seq._next_id = result.queue, result._next_id
            seq.rng.setstate(result.rng.getstate())
            seq.diagnostics = result.diagnostics
            self._state = 'preview' if self.config.dry_run else 'running'
            if not result.queue:
                self._state = 'no_eligible_tracks'
                self._next_plan = self.clock() + self.playback_config.early_poll_seconds
        return True

    def _load_source(self, source, playback, context, generation):
        cancelled = lambda: self._closed or self._source_generation != generation or self._source != source
        if cancelled():
            raise CancelledError()
        if hasattr(self.loader, 'load'):
            return self.loader.load(source, playback, context, cancelled=cancelled)
        return self.loader(source, playback, context)

    def _schedule(self):
        if self._future is not None or not self._source or self._playback_source != self._source:
            return
        now = self.clock()
        if now >= self._next_refresh:
            context = self.session.sequencer.context_ids[-20:] if self.session else self._context[-20:]
            context = context + [row['track_id'] for row in self._orphans.values()]
            self._job = ('source', self._source, self._source_generation)
            self._future = self.executor.submit(self._load_source, self._source, self._raw,
                                                context.copy(), self._source_generation)
            self._next_refresh = now + self.config.refresh_seconds
            self._state = 'loading_source'
        elif (self.session and self._ready_source == self._source and self._queue_valid
              and not self.session.sequencer.queue and self._upcoming() < self.config.buffer_size
              and now >= self._next_plan):
            clone, ids, pools = self.session.planning_snapshot()
            pending = {item.track_id for item in self.session.submitted}
            positions = [i for i, identity in enumerate(self._queue.track_ids) if identity in pending]
            manual_ids = set()
            if positions:
                future = self._queue.track_ids[:max(positions) + 1] + self._manual_tail
                active = self.tracker.active
                ids = ([active.track_id] if active and not active.accepted else []) + [
                    identity or 'unmanaged:queue-item' for identity in future]
                manual_ids = set(future) - pending - {None}
            self._job = ('plan', self._source, self._revision)
            exclusions = self._blocked_track_ids | manual_ids | {row['track_id'] for row in self._orphans.values()}
            self._future = self.executor.submit(_plan, clone, ids, pools, exclusions)

    def _fill(self):
        if (not self.session or self._ready_source != self._source or not self._queue_valid
                or self._uncertain or not self._last or not self._last.is_playing
                or not self._last.device_id or self.config.dry_run or self._write_error
                or self._playback_source != self._source):
            return
        if self.clock() - self._last.observed_at >= self.playback_config.early_poll_seconds:
            self._next_poll = min(self._next_poll, self.clock())
            return
        if ((self._raw.get('device') or {}).get('is_restricted')):
            self._state = 'restricted_device'
            return
        seq = self.session.sequencer
        added = False
        while seq.queue and self._upcoming() < self.config.buffer_size:
            if self.clock() - self._last.observed_at >= self.playback_config.early_poll_seconds:
                self._next_poll = min(self._next_poll, self.clock())
                break
            item = seq.queue[0]
            command = dict(command_id=uuid.uuid4().hex, selection_id=item.selection_id,
                           track_id=item.track_id, playlist_id=self._source,
                           before_ids=list(self._queue.track_ids if self._queue else ()))
            self.journal.record('intent', command)
            try:
                self.gateway.add(item.track_id, device_id=self._last.device_id)
            except SpotifyBudgetBlocked:
                self.journal.record('abandoned', command)
                raise
            except SpotifyException as exc:
                if exc.http_status in (400, 401, 403, 404):
                    self.journal.record('abandoned', command)
                    self._write_error = 'playback_permission_required' if exc.http_status in (401, 403) else 'playback_command_rejected'
                    self._state = self._write_error
                    break
                self._uncertain = command
                self._queue_valid = self._trusted = False
                self._needs_queue = True
                self._state = 'submission_uncertain'
                break
            except RequestException:
                self._uncertain = command
                self._queue_valid = self._trusted = False
                self._needs_queue = True
                self._state = 'submission_uncertain'
                break
            self.journal.record('queued', command)
            self.session.submit_prefix(1)
            self._commands[item.selection_id] = {**command, 'status': 'queued'}
            self._revision += 1
            added = True
        if added:
            self._trusted = False
            self._reconcile(self._last)

    def step(self):
        """Process ready work; only read playback when its deadline is due."""
        events = []
        now = self.clock()
        if now >= self._next_poll:
            raw = self.gateway.playback()
            snapshot = snapshot_from_playback(raw, self.clock())
            self._raw = raw or {}
            source = snapshot.context_uri.split(':')[-1] if snapshot.context_uri and snapshot.context_uri.startswith('spotify:playlist:') else None
            self._playback_source = source
            if source and source != self._source:
                if not self._switch_source(source):
                    self._next_poll = now + self.playback_config.early_poll_seconds
                    return []
            self._next_poll = self.clock() + self.playback_config.early_poll_seconds
            changed = (not self._last or snapshot.track_id != self._last.track_id
                       or snapshot.device_id != self._last.device_id or snapshot.context_uri != self._last.context_uri)
            if self._last and snapshot.device_id != self._last.device_id:
                self._supersede_uncertain()
                self._write_error = None
                self._trusted = self._anchored = False
            continuity = False
            if snapshot.track_id and (changed or self._needs_queue or self._uncertain):
                continuity = self._reconcile(snapshot)
            elif self._trusted:
                continuity = True
            observer = self.session or self.tracker
            events = observer.observe(snapshot, continuity=continuity)
            if events:
                self._revision += 1
                if self.session:
                    self.session.sequencer.queue = []
            if self.session is None:
                self._context.extend(e.track_id for e in events if e.kind == 'feedback' and e.outcome == 'accepted')
            self._record(events, self._raw)
            self._last = snapshot
            if self.session and self._queue_valid and self.session.submitted:
                # Re-anchor an external current track only after it was observed.
                self._anchored = self._prefix(self._queue.track_ids, self.session.submitted)
                self.session.confirm_order(follows_active=self._anchored)
            self._next_poll = min(self._next_poll if self._needs_queue else float('inf'),
                                  self.clock() + observer.next_delay())
            if not source:
                self._state = 'waiting_for_playlist'
                return events
        for _ in range(3):
            self._collect()
            self._schedule()
            if self._future is None or not self._future.done():
                break
        self._fill()
        return events

    def next_delay(self):
        remaining = max(.05, self._next_poll - self.clock())
        return min(remaining, .25) if self._future else remaining

    def interrupt(self):
        self.tracker.interrupt()
        self._queue_valid = self._trusted = self._anchored = False
        self._needs_queue = True
        self._revision += 1
        if self.session:
            self.session.sequencer.queue = []
        self._next_poll = self.clock() + self.playback_config.idle_poll_seconds
        self._state = 'connection_unavailable'

    def status(self):
        visible = self._queue.track_ids if self._queue else ()
        state = self._write_error or ('submission_uncertain' if self._uncertain or any(
            row['status'] == 'intent' for row in self._orphans.values()) else self._state)
        return {'state': state, 'playlist_id': self._source,
                'upcoming': self._upcoming(), 'orphaned_upcoming': sum(r['track_id'] in visible for r in self._orphans.values()),
                'worker_busy': self._future is not None,
                **(self.session.sequencer.status() if self.session else {}),
                **(self.session.status() if self.session else {})}

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            observer = self.session or self.tracker
            self._record(observer.finish())
        finally:
            if self._owns_executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
