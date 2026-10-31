"""Run the complete Spotify shuffler: python -m shufflerec.live --help."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from requests.exceptions import RequestException
from spotipy.exceptions import SpotifyException, SpotifyOauthError

from scripts.playback_daemon import PlaybackRecorder, write_evidence
from scripts.playback_lease import PlaybackLease
from scripts.spotify_budget import SpotifyBudgetBlocked
from scripts.spotify_gateway import create_client
from scripts.spotify_queue import LIVE_SCOPE, PlaylistCache, SpotifyQueueGateway
from scripts import polling
from .live_controller import LiveConfig, LiveController
from .live_state import CommandJournal, EngineProvider
from .playback import PlaybackConfig, finite
from .pools import SequencerConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = PROJECT_ROOT / 'sensitive_info'
DEFAULT_SETTINGS = PRIVATE_DIR / 'spotify_pipeline.json'


def _settings(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise ValueError('Could not read private pipeline settings') from None
    allowed = {'checkpoint', 'catalog', 'buffer_size', 'follow_playlist', 'playlist_refresh_seconds',
               'request_interval_seconds', 'seed', 'sequencer', 'polling'}
    if not isinstance(result, dict) or set(result) - allowed:
        raise ValueError('Pipeline settings contain unknown fields or are not an object')
    return result


def run_pipeline(*, settings_path=DEFAULT_SETTINGS, private_dir=PRIVATE_DIR, checkpoint=None,
                 catalog=None, buffer_size=None, use_proxy=None, dry_run=False, once=False,
                 acknowledge_uncertain=False, client=None, executor=None, clock=time.monotonic,
                 sleeper=time.sleep, on_status=None):
    settings_path, private_dir = Path(settings_path), Path(private_dir)
    if dry_run and acknowledge_uncertain:
        raise ValueError('Acknowledging uncertain commands cannot be combined with dry-run')
    settings = _settings(settings_path)
    checkpoint = checkpoint or settings.get('checkpoint')
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint):
        raise ValueError('Select a trained checkpoint with --checkpoint or private pipeline settings')
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = settings_path.parent / checkpoint
    master = Path(catalog or settings.get('catalog') or PROJECT_ROOT / 'data' / 'master_track_table.csv')
    if not master.is_absolute():
        master = settings_path.parent / master
    if not checkpoint.is_file() or not master.is_file():
        raise ValueError('Selected checkpoint or catalog file does not exist')
    live_config = LiveConfig(buffer_size=buffer_size if buffer_size is not None else settings.get('buffer_size', 5),
                             refresh_seconds=settings.get('playlist_refresh_seconds', 300.),
                             follow_playlist=settings.get('follow_playlist', True), dry_run=dry_run)
    seq_config = SequencerConfig(**settings.get('sequencer', {}))
    playback_config = replace(PlaybackConfig(**settings.get('polling', {})),
                              acceptance_threshold=seq_config.early_skip_threshold)
    interval, seed = settings.get('request_interval_seconds', 1.), settings.get('seed', 42)
    if not finite(interval) or interval < 0 or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('Request interval must be nonnegative and seed must be an integer')
    app = None
    owns_client = client is None
    with PlaybackLease(private_dir / 'playback_owner.lock'):
        try:
            if client is None:
                client = create_client(credentials_path=private_dir / 'spotify_credentials.json',
                                        cache_path=private_dir / '.spotify_cache',
                                        budget_path=private_dir / 'spotify_request_budget.json',
                                        proxy_config_path=private_dir / 'spotify_proxy.json',
                                        scope=LIVE_SCOPE, use_proxy=use_proxy)
            gateway = SpotifyQueueGateway(client, min_interval=interval, clock=clock, sleeper=sleeper)
            cache = PlaylistCache(gateway, refresh_seconds=live_config.refresh_seconds)
            provider = EngineProvider(checkpoint, master, cache, clock=clock)
            recorder = None
            if not dry_run:
                history = private_dir / 'spotify_listening_history.csv'
                polling.init_csv(csv_path=history)
                recorder = PlaybackRecorder(
                    event_sink=lambda event: write_evidence(event, path=private_dir / 'playback_evidence.jsonl'),
                    history_sink=lambda *args: polling.log_track(*args, csv_path=history, quiet=True))
            app = LiveController(gateway, provider, journal=CommandJournal(private_dir / 'queue_commands.jsonl'),
                                 config=live_config, sequencer_config=seq_config, playback_config=playback_config,
                                 executor=executor, clock=clock, recorder=recorder, seed=seed)
            previous = None
            while True:
                try:
                    app.step()
                except SpotifyBudgetBlocked:
                    app.interrupt()
                    app._state = 'spotify_cooldown'
                    return app.status()
                except (RequestException, SpotifyException):
                    app.interrupt()
                    if once:
                        return app.status()
                status = app.status()
                if acknowledge_uncertain and status['state'] == 'submission_uncertain':
                    app.acknowledge_uncertain()
                    acknowledge_uncertain = False
                    sleeper(.05)
                    continue
                summary = (status['state'], status['playlist_id'], status['upcoming'])
                if on_status is not None and summary != previous:
                    on_status(status)
                    previous = summary
                terminal = status['state'] in ('source_unavailable', 'inference_unavailable', 'waiting_for_playlist',
                    'submission_uncertain', 'playback_permission_required', 'playback_command_rejected',
                    'restricted_device', 'no_eligible_tracks', 'playlist_changed_paused')
                prepared = app.session is not None and (status.get('queue') or status['upcoming'] >= live_config.buffer_size)
                if once and not status['worker_busy'] and (terminal or prepared):
                    return status
                sleeper(app.next_delay())
        except KeyboardInterrupt:
            return app.status() if app else {'state': 'stopped'}
        finally:
            try:
                if app is not None:
                    app.close()
            finally:
                if owns_client and client is not None and hasattr(client, '_session'):
                    client._session.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', default=str(DEFAULT_SETTINGS))
    parser.add_argument('--private-dir', default=str(PRIVATE_DIR))
    parser.add_argument('--checkpoint')
    parser.add_argument('--catalog')
    parser.add_argument('--buffer-size', type=int)
    parser.add_argument('--dry-run', action='store_true', help='Read/plan without Spotify queue writes or listening logs')
    parser.add_argument('--once', action='store_true', help='Finish initialization/preview and exit')
    parser.add_argument('--acknowledge-uncertain', action='store_true',
                        help='Accept unknown prior command outcomes and resume with different songs')
    proxy = parser.add_mutually_exclusive_group()
    proxy.add_argument('--proxy', dest='use_proxy', action='store_true')
    proxy.add_argument('--no-proxy', dest='use_proxy', action='store_false')
    parser.set_defaults(use_proxy=None)
    args = parser.parse_args(argv)
    def report(status):
        print(json.dumps({key: status[key] for key in ('state', 'playlist_id', 'upcoming') if key in status}), flush=True)
    try:
        status = run_pipeline(**vars(args), on_status=report)
    except SpotifyBudgetBlocked:
        print(json.dumps({'state': 'spotify_cooldown', 'message': 'Shared Spotify cooldown is active; wait before restarting.'}))
        return 2
    except SpotifyOauthError:
        print(json.dumps({'state': 'authorization_required', 'message': 'Spotify authorization failed; authorize playback and playlist access.'}))
        return 2
    except (ValueError, TypeError, OSError, RuntimeError):
        print(json.dumps({'state': 'startup_failed', 'message': 'Check settings, checkpoint compatibility, and the playback-owner lease.'}))
        return 2
    if args.dry_run and args.once:
        print(json.dumps(status, allow_nan=False))
    elif status.get('state') in ('spotify_cooldown', 'connection_unavailable'):
        report(status)
    return 0 if status.get('state') in ('running', 'preview', 'waiting_for_playlist', 'no_eligible_tracks', 'stopped') else 2


if __name__ == '__main__':
    raise SystemExit(main())
