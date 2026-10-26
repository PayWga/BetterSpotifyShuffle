"""Persistent local safeguards, not an estimate of Spotify's undocumented quota."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time


WINDOW = 86400
DEFAULT_LIMIT = None
_thread_lock = threading.Lock()


class SpotifyBudgetBlocked(RuntimeError):
    """Stop the whole metadata job, allowing its checkpoint finalizer to run."""


def utc_text(timestamp):
    try:
        return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except (OverflowError, OSError, ValueError):
        return 'a time beyond the supported date range'


class SpotifyRequestBudget:
    def __init__(self, path, limit=DEFAULT_LIMIT, clock=None, *, interval=0,
                 rest_every=0, rest_seconds=300, sleeper=None):
        if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
            raise ValueError('Request budget must be a positive integer')
        if (not math.isfinite(interval) or interval < 0 or not isinstance(rest_every, int)
                or rest_every < 0 or not math.isfinite(rest_seconds) or rest_seconds < 0):
            raise ValueError('Invalid request pacing configuration')
        self.path = Path(path)
        self.limit = limit
        self.clock = time.time if clock is None else clock
        self.interval, self.rest_every, self.rest_seconds = interval, rest_every, rest_seconds
        self.sleeper = time.sleep if sleeper is None else sleeper

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _thread_lock, self.path.with_suffix('.lock').open('a+b') as lock:
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b'0')
                lock.flush()
            lock.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                lock.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def read(self):
        if not self.path.exists():
            return {'version': 1, 'attempts': [], 'blocked_until': 0, 'reason': ''}
        try:
            state = json.loads(self.path.read_text(encoding='utf-8'))
            def timestamp(value):
                return (isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) and value >= 0)
            if (state.get('version') != 1 or not isinstance(state.get('attempts'), list)
                    or not all(timestamp(t) for t in state['attempts'])
                    or not timestamp(state.get('blocked_until'))
                    or not isinstance(state.get('reason'), str)):
                raise ValueError('Invalid budget schema')
            if (not timestamp(state.get('rest_until', 0))
                    or not isinstance(state.get('burst_count', 0), int)
                    or state.get('burst_count', 0) < 0):
                raise ValueError('Invalid pacing state')
            return state
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise SpotifyBudgetBlocked('Request budget is unreadable; stopped without making a request.') from exc

    def write(self, state):
        fd, temp = tempfile.mkstemp(prefix='.spotify_budget_', suffix='.json', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as output:
                json.dump(state, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def check(self, state, now):
        if state['blocked_until'] > now:
            raise SpotifyBudgetBlocked(f"Spotify metadata paused ({state['reason']}). "
                                       f"Retry no earlier than {utc_text(state['blocked_until'])}.")
        recent = sorted(t for t in state['attempts'] if t > now - WINDOW)
        if self.limit is not None and len(recent) >= self.limit:
            # If the configured budget is lowered, enough entries must expire.
            retry_at = recent[len(recent) - self.limit] + WINDOW
            raise SpotifyBudgetBlocked(f'Local metadata budget of {self.limit} HTTP attempts per rolling '
                                       f'24 hours reached. Retry no earlier than {utc_text(retry_at)}.')
        return recent

    def check_allowed(self):
        with self.locked():
            self.check(self.read(), self.clock())

    def reserve(self):
        """Pace and count actual HTTP attempts across restarts/processes.

        Release file locks while resting so other processes can inspect status
        or record a Spotify cooldown. Recheck all state before reserving.
        """
        announced = False
        while True:
            with self.locked():
                state, now = self.read(), self.clock()
                recent = self.check(state, now)
                deadline = max(recent[-1] + self.interval if recent else 0,
                               state.get('rest_until', 0))
                delay = deadline - now
                if delay <= 0:
                    count = state.get('burst_count', 0)
                    if not recent or now - recent[-1] >= self.rest_seconds:
                        count = 0
                    state['attempts'] = recent + [now]
                    count += 1
                    if self.rest_every and count >= self.rest_every:
                        state['rest_until'] = now + self.rest_seconds
                        count = 0
                    state['burst_count'] = count
                    self.write(state)
                    return
            if delay > self.interval and not announced:
                print(f'Metadata pacing: resting until {utc_text(deadline)} before the next HTTP attempt.')
                announced = True
            self.sleeper(min(delay, 30))

    def pause(self, seconds, reason='reported cooldown'):
        if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(self.clock() + seconds):
            raise ValueError('Cooldown must be a finite, nonnegative number of seconds')
        with self.locked():
            state = self.read()
            until = self.clock() + seconds
            if until > state['blocked_until']:
                state['blocked_until'], state['reason'] = until, reason
                self.write(state)
            elif reason == 'QUOTA_EXCEEDED' and state['reason'] != reason:
                state['reason'] = reason
                self.write(state)

    def summary(self):
        with self.locked():
            state, now = self.read(), self.clock()
            used = sum(t > now - WINDOW for t in state['attempts'])
            return {'limit_per_24h': self.limit, 'attempts_in_last_24h': used,
                    'remaining': max(0, self.limit - used) if self.limit is not None else None,
                    'request_interval_seconds': self.interval,
                    'rest_every_attempts': self.rest_every,
                    'rest_seconds': self.rest_seconds,
                    'rest_until_utc': utc_text(state['rest_until']) if state.get('rest_until', 0) > now else None,
                    'attempts_in_last_30s': sum(t > now - 30 for t in state['attempts']),
                    'attempts_in_last_15min': sum(t > now - 900 for t in state['attempts']),
                    'cooldown_until_utc': utc_text(state['blocked_until']) if state['blocked_until'] > now else None,
                    'cooldown_reason': state['reason'] if state['blocked_until'] > now else None}
