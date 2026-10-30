import argparse
import csv
from pathlib import Path
from datetime import datetime, timezone
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ''):
    sys.path.insert(0, str(PROJECT_ROOT))
PRIVATE_DIR = PROJECT_ROOT / "sensitive_info"

CSV_FILE = PRIVATE_DIR / "spotify_listening_history.csv"
CREDS_PATH = PRIVATE_DIR / "spotify_credentials.json"
CACHE_PATH = PRIVATE_DIR / ".spotify_cache"

SCOPE = (
    "user-read-playback-state "
    "user-read-currently-playing"
)

ACTIVE_POLL = 15
PAUSED_POLL = 30
INACTIVE_TIMEOUT = 60

PLAY_THRESHOLD = 0.70
SKIP_THRESHOLD = 0.15


CSV_COLUMNS = [
    "timestamp",
    "track_id",
    "track_name",
    "artist_name",
    "ms_played",
    "duration_ms",
    "completion_rate",
    "reason_start",
    "reason_end",
    "event_type",
    "is_play",
    "is_ghost_skip",
    "context_uri",
    "context_type",
    "event_source"
]


def get_spotify_client():
    """Use the shared budgeted transport and existing private OAuth files."""
    from scripts.spotify_gateway import create_client
    return create_client(credentials_path=CREDS_PATH, cache_path=CACHE_PATH,
                         budget_path=PRIVATE_DIR / 'spotify_request_budget.json', scope=SCOPE)


def init_csv():
    """Initialize the unified listening history CSV."""

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    if CSV_FILE.exists():
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])

        if existing_header != CSV_COLUMNS:
            raise ValueError(
                "Existing listening history has a different schema. "
                "Migrate or normalize it before starting the daemon."
            )

        return

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def classify_live_event(completion_rate, reason_end):
    """Classify observed playback using the shared thresholds."""

    if completion_rate >= PLAY_THRESHOLD:
        return "PLAY", 1, 0

    if (
        reason_end == "track_change"
        and completion_rate < SKIP_THRESHOLD
    ):
        # Inferred early skip; not a confirmed dislike.
        return "GHOST_SKIP", 0, 1

    return "NEUTRAL", 0, 0


def log_track(track_data, listened_ms, context_uri, context_type, reason_end):
    from shufflerec.playback import finite
    raw_duration = track_data.get('duration_ms')
    duration_ms = int(raw_duration) if finite(raw_duration) and raw_duration > 0 else None

    listened_ms = max(0, int(listened_ms))

    completion_rate = min(
        listened_ms / max(duration_ms, 1),
        1.0
    ) if duration_ms is not None else None

    event_type, is_play, is_ghost_skip = classify_live_event(
        completion_rate,
        reason_end
    ) if completion_rate is not None else ('NEUTRAL', 0, 0)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track_id": track_data["id"],
        "track_name": track_data["name"],
        "artist_name": ", ".join(
            a["name"] for a in track_data.get("artists", [])
        ),
        "ms_played": listened_ms,
        "duration_ms": duration_ms,
        "completion_rate": round(completion_rate, 4) if completion_rate is not None else None,
        "reason_start": "unknown",
        "reason_end": reason_end,
        "event_type": event_type,
        "is_play": is_play,
        "is_ghost_skip": is_ghost_skip,
        "context_uri": context_uri,
        "context_type": context_type,
        "event_source": "live_polling"
    }

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(record)

    completion_text = f'{completion_rate:.1%}' if completion_rate is not None else 'unknown'
    print(
        f"Logged: {track_data['name']} | "
        f"Completion: {completion_text} | "
        f"Event: {event_type}"
    )


def start_daemon(config=None):
    from shufflerec.playback import PlaybackConfig, PlaybackTracker
    from scripts.playback_daemon import run_polling, write_evidence
    from scripts.spotify_budget import SpotifyBudgetBlocked

    config = config or PlaybackConfig()
    init_csv()
    try:
        client = get_spotify_client()
    except SpotifyBudgetBlocked:
        print('Playback observation blocked by the shared Spotify cooldown.')
        return
    print('Adaptive listening daemon started...')
    run_polling(client, tracker=PlaybackTracker(config), event_sink=write_evidence,
                history_sink=log_track)


def main(argv=None):
    from shufflerec.playback import PlaybackConfig
    parser = argparse.ArgumentParser(description='Adaptive read-only Spotify playback tracker')
    parser.add_argument('--early-poll', type=float, default=15.)
    parser.add_argument('--max-gap', type=float, default=75.)
    parser.add_argument('--idle-poll', type=float, default=30.)
    parser.add_argument('--skip-threshold', type=float, default=.25)
    args = parser.parse_args(argv)
    try:
        config = PlaybackConfig(acceptance_threshold=args.skip_threshold,
                                early_poll_seconds=args.early_poll,
                                max_gap_seconds=args.max_gap, idle_poll_seconds=args.idle_poll)
    except ValueError as exc:
        parser.error(str(exc))
    start_daemon(config)


if __name__ == '__main__':
    main()
