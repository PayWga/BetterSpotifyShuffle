import importlib
from concurrent.futures import Future

import pandas as pd
import pytest
from requests.exceptions import ConnectionError

from shufflerec.inference import RankedTrack


def api():
    assert importlib.util.find_spec('shufflerec.live_controller'), 'live controller missing'
    return importlib.import_module('shufflerec.live_controller')


class Immediate:
    def submit(self, function, *args):
        result = Future()
        try:
            result.set_result(function(*args))
        except Exception as exc:
            result.set_exception(exc)
        return result


class Clock:
    now = 0.
    def __call__(self):
        return self.now
    def advance(self, seconds=15):
        self.now += seconds


def song(identity):
    return {'id': identity, 'type': 'track', 'duration_ms': 180000,
            'artists': [{'name': 'Artist', 'id': 'artist'}]}


class Spotify:
    def __init__(self):
        self.current = '0'
        self.source = 'p1'
        self.device = 'pc'
        self.progress = 0
        self.is_playing = True
        self.queued = []
        self.writes = []
        self.reads = 0
        self.queue_reads = 0
        self.fail_add = False
        self.accept_before_failure = False
    def current_playback(self):
        self.reads += 1
        return {'item': song(self.current), 'progress_ms': self.progress, 'is_playing': self.is_playing,
                'device': {'id': self.device}, 'context': {'type': 'playlist', 'uri': 'spotify:playlist:' + self.source}}
    def queue(self):
        self.queue_reads += 1
        return {'currently_playing': song(self.current), 'queue': [song(t) for t in self.queued]}
    def add_to_queue(self, uri, device_id=None):
        self.writes.append((uri, device_id))
        identity = uri.rsplit(':', 1)[-1]
        if self.fail_add:
            if self.accept_before_failure:
                self.queued.append(identity)
            self.fail_add = False
            raise ConnectionError('synthetic lost acknowledgement')
        self.queued.append(identity)
    def advance_to(self, index=0):
        self.current = self.queued[index]
        self.queued = self.queued[index + 1:]
        self.progress = 0


class Engine:
    def __init__(self, source):
        self.contexts = []
        self.catalog = pd.DataFrame([dict(id=str(i), artists='Artist', duration_ms=180000, genre='rock')
                                     for i in range(40)])
    def rank(self, context, ids):
        self.contexts.append(tuple(context))
        return [RankedTrack(t, 1.) for t in sorted(ids)]
    def usable_context(self, ids):
        return list(ids)


def controller(tmp_path, *, dry_run=False, executor=None, spotify=None):
    m = api()
    from scripts.spotify_queue import SpotifyQueueGateway
    from shufflerec.live_state import SourceBundle, CommandJournal
    clock = Clock()
    sp = spotify or Spotify()
    def loader(source, playback, context):
        identities = tuple(str(i) for i in (range(20) if source == 'p1' else range(20, 40)))
        return SourceBundle(source, 'v1', identities, Engine(source))
    journal = CommandJournal(tmp_path / 'commands.jsonl')
    app = m.LiveController(SpotifyQueueGateway(sp), loader, clock=clock,
                          executor=executor or Immediate(), journal=journal,
                          config=m.LiveConfig(dry_run=dry_run))
    return app, sp, clock, journal


def test_inference_to_five_queue_additions_then_skip_feedback_and_refill(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    app.step()
    assert len(sp.writes) == 5 and len(sp.queued) == 5
    assert all(device == 'pc' for _, device in sp.writes)
    assert len(set(sp.queued)) == 5 and '0' not in sp.queued
    before_reads = sp.reads
    app.step()
    assert sp.reads == before_reads  # Background wakeups leave the playback read count unchanged.
    sp.advance_to(2)
    clock.advance()
    events = app.step()
    assert len([e for e in events if e.inferred]) == 2
    assert sum(app.session.sequencer.policy.preferences) == pytest.approx(-.30)
    assert len(sp.queued) == 5


def test_manual_queue_is_preserved_and_dry_run_performs_no_additions(tmp_path):
    sp = Spotify(); sp.queued = ['manual']
    app, sp, _, journal = controller(tmp_path, dry_run=True, spotify=sp)
    app.step()
    assert sp.queued == ['manual'] and sp.writes == []
    assert len(app.session.sequencer.queue) == 5
    assert not journal.path.exists()


def test_ambiguous_post_is_never_blindly_retried(tmp_path):
    sp = Spotify(); sp.fail_add = True
    app, sp, clock, journal = controller(tmp_path, spotify=sp)
    app.step()
    assert len(sp.writes) == 1
    clock.advance()
    app.step()
    assert len(sp.writes) == 1
    assert app.status()['state'] == 'submission_uncertain'
    assert journal.unresolved()[0]['status'] == 'intent'


def test_lost_acknowledgement_can_be_recovered_from_queue_without_duplicate(tmp_path):
    sp = Spotify(); sp.fail_add = True; sp.accept_before_failure = True
    app, sp, clock, _ = controller(tmp_path, spotify=sp)
    app.step()
    first = sp.writes[0][0]
    clock.advance()
    app.step()
    assert [uri for uri, _ in sp.writes].count(first) == 1
    assert len(sp.queued) == 5


def test_playlist_change_follows_new_source_and_preserves_learned_weights(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    app.step()
    old_preferences = app.session.sequencer.policy.preferences.copy()
    sp.source = 'p2'; sp.current = '20'; sp.queued = []
    clock.advance()
    app.step()
    assert len(sp.queued) == 5 and all(int(t) >= 20 for t in sp.queued)
    assert app.session.sequencer.policy.preferences == old_preferences


def test_paused_playback_does_not_send_queue_commands(tmp_path):
    sp = Spotify(); sp.is_playing = False
    app, sp, _, _ = controller(tmp_path, spotify=sp)
    app.step()
    assert sp.writes == []


def test_journaled_queued_tracks_survive_restart_as_unowned_buffer_entries(tmp_path):
    app, sp, clock, journal = controller(tmp_path)
    app.step()
    assert len(journal.unresolved()) == 5
    restarted, _, _, _ = controller(tmp_path, spotify=sp)
    restarted.step()
    assert len(sp.writes) == 5
    assert restarted.status()['orphaned_upcoming'] == 5


def test_stale_background_source_is_not_installed_after_playlist_switch(tmp_path):
    class Deferred:
        def __init__(self):
            self.calls = []
        def submit(self, function, *args):
            future = Future(); self.calls.append((future, function, args)); return future
    executor = Deferred()
    app, sp, clock, _ = controller(tmp_path, executor=executor)
    app.step()
    future, fn, args = executor.calls[0]
    completed = fn(*args)
    sp.source = 'p2'; sp.current = '20'; clock.advance()
    app.step()
    future.set_result(completed)
    app.step()
    assert app.session is None and sp.writes == []


def test_feedback_discards_unsubmitted_plan_while_preserving_committed_songs(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    app.step(); sp.advance_to(); clock.advance(); app.step()
    assert app.session.sequencer.queue
    committed = sp.queued.copy()
    sp.progress = 45000; clock.advance(45); app.step()
    assert app.session.sequencer.queue == []
    assert sp.queued == committed


def test_nonplaylist_context_cannot_trigger_old_source_additions_on_worker_wakeup(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    app.step(); initial = len(sp.writes)
    sp.advance_to(); clock.advance()
    original = sp.current_playback
    def album():
        value = original(); value['context'] = {'type': 'album', 'uri': 'spotify:album:x'}; return value
    sp.current_playback = album
    app.step(); app.step()
    assert len(sp.writes) == initial


def test_recovered_command_remains_safe_across_playlist_change(tmp_path):
    sp = Spotify(); sp.fail_add = True; sp.accept_before_failure = True
    app, sp, clock, _ = controller(tmp_path, spotify=sp)
    app.step(); clock.advance(); app.step()
    sp.source = 'p2'; sp.current = '20'; sp.queued = []; clock.advance()
    app.step()
    assert len(sp.queued) == 5


def test_explicit_permission_rejection_is_not_journaled_as_uncertain(tmp_path):
    from spotipy.exceptions import SpotifyException
    sp = Spotify()
    def forbidden(*args, **kwargs):
        raise SpotifyException(403, -1, 'synthetic permission denial')
    sp.add_to_queue = forbidden
    app, sp, _, journal = controller(tmp_path, spotify=sp)
    app.step()
    assert app.status()['state'] == 'playback_permission_required'
    assert journal.unresolved() == []


def test_restart_recovers_visible_intent_without_replaying_it(tmp_path):
    sp = Spotify(); sp.fail_add = True; sp.accept_before_failure = True
    app, sp, _, _ = controller(tmp_path, spotify=sp)
    app.step()
    first = sp.writes[0][0]
    restarted, _, _, journal = controller(tmp_path, spotify=sp)
    restarted.step()
    assert [uri for uri, _ in sp.writes].count(first) == 1
    assert len(sp.queued) == 5
    assert not any(row['status'] == 'intent' for row in journal.unresolved())


def test_queue_confirmation_is_retried_after_eventually_consistent_response(tmp_path):
    sp = Spotify()
    original = sp.queue
    def queue():
        value = original()
        if sp.queue_reads == 2:
            value['queue'] = []
        return value
    sp.queue = queue
    app, sp, clock, _ = controller(tmp_path, spotify=sp)
    app.step()
    assert app.status()['state'] == 'awaiting_queue_confirmation'
    clock.advance(); app.step()
    assert sp.queue_reads == 3
    assert app.status()['state'] == 'running'
    assert len(sp.writes) == 5


def test_source_aba_change_rejects_original_generation(tmp_path):
    class Deferred:
        def __init__(self): self.calls = []
        def submit(self, function, *args):
            future = Future(); self.calls.append((future, function, args)); return future
    executor = Deferred()
    app, sp, clock, _ = controller(tmp_path, executor=executor)
    app.step()
    future, function, args = executor.calls[0]
    completed = function(*args)
    sp.source = 'p2'; sp.current = '20'; clock.advance(); app.step()
    sp.source = 'p1'; sp.current = '8'; clock.advance(); app.step()
    future.set_result(completed)
    app.step()
    assert app.session is None and sp.writes == []


def test_new_manual_tail_is_the_anchor_for_appended_recommendations(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    app.step()
    sp.advance_to(); sp.queued.append('39'); clock.advance()
    engine = app.session.sequencer.engine
    engine.contexts.clear()
    app.step()
    assert engine.contexts[0][-1] == '39'
    assert sp.queued[-2] == '39'


def test_explicit_acknowledgement_resumes_without_replaying_uncertain_song(tmp_path):
    sp = Spotify(); sp.fail_add = True
    app, sp, clock, journal = controller(tmp_path, spotify=sp)
    app.step()
    uncertain = sp.writes[0][0]
    app.acknowledge_uncertain()
    clock.advance(); app.step()
    assert len(sp.queued) == 5
    assert [uri for uri, _ in sp.writes].count(uncertain) == 1
    assert not any(row['status'] == 'intent' for row in journal.unresolved())


def test_unknown_write_stays_paused_after_source_switch_until_acknowledged(tmp_path):
    sp = Spotify(); sp.fail_add = True
    app, sp, clock, _ = controller(tmp_path, spotify=sp)
    app.step()
    sp.source = 'p2'; sp.current = '20'; clock.advance(); app.step()
    assert app.status()['state'] == 'submission_uncertain' and len(sp.writes) == 1


def test_recorder_failure_still_shuts_down_owned_executor(tmp_path, monkeypatch):
    m = api()
    from scripts.spotify_queue import SpotifyQueueGateway
    from shufflerec.live_state import CommandJournal
    class Executor:
        closed = False
        def shutdown(self, **kwargs): self.closed = True
    executor = Executor()
    monkeypatch.setattr(m, 'ThreadPoolExecutor', lambda **kwargs: executor)
    class Recorder:
        def consume(self, *args, **kwargs): raise OSError('synthetic disk failure')
    app = m.LiveController(SpotifyQueueGateway(Spotify()), lambda *args: None,
                           journal=CommandJournal(tmp_path / 'commands.jsonl'), recorder=Recorder())
    with pytest.raises(OSError): app.close()
    assert executor.closed


def test_failed_playlist_refresh_stops_using_outdated_candidates(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    app.step()
    def unavailable(*args): raise ValueError('synthetic source unavailable')
    app.loader = unavailable
    clock.advance(300); app.step()
    initial = len(sp.writes)
    sp.advance_to(); clock.advance(); app.step()
    assert len(sp.writes) == initial
    assert app.status()['state'] == 'source_unavailable'


def test_slow_queue_writes_pause_batch_for_fresh_playback_observation(tmp_path):
    app, sp, clock, _ = controller(tmp_path)
    original = sp.add_to_queue
    def slow(*args, **kwargs):
        original(*args, **kwargs); clock.advance(8)
    sp.add_to_queue = slow
    app.step()
    assert len(sp.writes) == 2
    assert app.next_delay() <= .1


def test_engine_provider_preserves_existing_values_and_uses_frozen_checkpoint(tmp_path):
    api()
    from shufflerec.live_state import EngineProvider
    from scripts.spotify_queue import PlaylistSnapshot
    from shufflerec.features import FeaturePreprocessor
    from shufflerec.model import SpotifySASRec, ModelConfig
    from shufflerec.training import save_checkpoint
    frame = pd.DataFrame([dict(id='a', artists='Original', tempo=100, duration_ms=180000),
                          dict(id='b', artists='Other', tempo=120, duration_ms=180000)])
    prep = FeaturePreprocessor().fit(frame, frame.id)
    model = SpotifySASRec(prep, max_length=3, config=ModelConfig(d_model=8, nhead=2, feedforward=16, dropout=0))
    checkpoint, master = tmp_path / 'model.pt', tmp_path / 'master.csv'
    save_checkpoint(checkpoint, model, prep); frame.to_csv(master, index=False)
    original = master.read_bytes()
    class Cache:
        def load(self, identity, *, now):
            return PlaylistSnapshot(identity, 'v1', ('a', 'b'),
                                    pd.DataFrame([dict(id='a', artists='Replacement', duration_ms=999999)]))
    provider = EngineProvider(checkpoint, master, Cache(), clock=lambda: 0)
    bundle = provider('p1', {'item': song('a')}, [])
    assert bundle.engine.catalog.set_index('id').loc['a', 'artists'] == 'Original'
    assert bundle.engine.catalog.set_index('id').loc['a', 'duration_ms'] == 180000
    assert bundle.engine.preprocessor.state_dict() == prep.state_dict()
    assert bundle.engine.rank(['a'], ['b'])[0].track_id == 'b'
    assert master.read_bytes() == original
