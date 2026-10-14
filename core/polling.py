import os
import time
import csv
import json
from pathlib import Path
from datetime import datetime, timezone

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
from spotipy.exceptions import SpotifyException


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIVATE_DIR = PROJECT_ROOT / "sensitive_info"

CSV_FILE = PRIVATE_DIR / "spotify_listening_history.csv"
CREDS_PATH = PRIVATE_DIR / "spotify_credentials.json"
CACHE_PATH = PRIVATE_DIR / ".spotify_cache"

SCOPE = (
    "user-read-playback-state "
    "user-read-currently-playing"
)

ACTIVE_POLL = 5
PAUSED_POLL = 15
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
    """Authenticate using the project's Spotify credentials."""

    with open(CREDS_PATH, "r", encoding="utf-8") as f:
        creds = json.load(f)

    return spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
            redirect_uri=creds["redirect_uri"],
            scope=SCOPE,
            cache_handler=CacheFileHandler(
                cache_path=str(CACHE_PATH)
            )
        )
    )


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

    duration_ms = int(track_data["duration_ms"])

    listened_ms = max(0, int(listened_ms))

    completion_rate = min(
        listened_ms / max(duration_ms, 1),
        1.0
    )

    event_type, is_play, is_ghost_skip = classify_live_event(
        completion_rate,
        reason_end
    )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track_id": track_data["id"],
        "track_name": track_data["name"],
        "artist_name": ", ".join(
            a["name"] for a in track_data.get("artists", [])
        ),
        "ms_played": listened_ms,
        "duration_ms": duration_ms,
        "completion_rate": round(completion_rate, 4),
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

    print(
        f"Logged: {track_data['name']} | "
        f"Completion: {completion_rate:.1%} | "
        f"Event: {event_type}"
    )


def start_daemon():
    init_csv()
    sp = get_spotify_client()

    print("Listening daemon started...")

    current_track_id = None
    current_track_data = None
    current_context_uri = None
    current_context_type = None

    listened_ms = 0
    previous_progress_ms = None
    previous_poll_time = None
    previous_was_playing = False
    inactive_since = None

    def reset_state():
        nonlocal current_track_id, current_track_data
        nonlocal current_context_uri, current_context_type
        nonlocal listened_ms, previous_progress_ms
        nonlocal previous_poll_time, previous_was_playing
        nonlocal inactive_since

        current_track_id = None
        current_track_data = None
        current_context_uri = None
        current_context_type = None

        listened_ms = 0
        previous_progress_ms = None
        previous_poll_time = None
        previous_was_playing = False
        inactive_since = None

    def finish_track(reason):
        if current_track_data is not None:
            log_track(
                current_track_data,
                listened_ms,
                current_context_uri,
                current_context_type,
                reason
            )

        reset_state()

    try:
        while True:
            try:
                playback = sp.current_playback()
                poll_time = time.monotonic()

                active_item = (
                    playback.get("item")
                    if playback else None
                )

                if not active_item:
                    if current_track_data is not None:
                        if inactive_since is None:
                            inactive_since = poll_time

                        if poll_time - inactive_since >= INACTIVE_TIMEOUT:
                            finish_track("playback_inactive")

                    time.sleep(PAUSED_POLL)
                    continue

                track_id = active_item.get("id")
                progress_ms = playback.get("progress_ms")
                is_playing = playback.get("is_playing", False)

                if track_id is None or progress_ms is None:
                    time.sleep(ACTIVE_POLL)
                    continue

                inactive_since = None

                context = playback.get("context") or {}
                context_uri = context.get("uri")
                context_type = context.get("type")

                # Detect a transition to another track.
                if track_id != current_track_id:
                    if current_track_data is not None:
                        finish_track("track_change")

                    current_track_id = track_id
                    current_track_data = active_item
                    current_context_uri = context_uri
                    current_context_type = context_type

                    listened_ms = 0
                    previous_progress_ms = progress_ms
                    previous_poll_time = poll_time
                    previous_was_playing = is_playing

                    time.sleep(ACTIVE_POLL if is_playing else PAUSED_POLL)
                    continue

                # Detect an apparent replay of the same track.
                # Only treat this as a replay if the old position
                # was near the end and the new one is near the start.
                duration_ms = active_item.get("duration_ms") or 0

                if (
                    previous_progress_ms is not None
                    and duration_ms > 0
                    and previous_progress_ms > duration_ms * 0.85
                    and progress_ms < duration_ms * 0.15
                ):
                    finish_track("replay_detected")

                    current_track_id = track_id
                    current_track_data = active_item
                    current_context_uri = context_uri
                    current_context_type = context_type

                    previous_progress_ms = progress_ms
                    previous_poll_time = poll_time
                    previous_was_playing = is_playing

                    time.sleep(ACTIVE_POLL if is_playing else PAUSED_POLL)
                    continue

                # Accumulate observed playback progress.
                if (
                    previous_progress_ms is not None
                    and previous_poll_time is not None
                    and previous_was_playing
                    and is_playing
                ):
                    progress_delta = (
                        progress_ms - previous_progress_ms
                    )

                    elapsed_ms = (
                        poll_time - previous_poll_time
                    ) * 1000

                    # A large jump suggests seeking rather
                    # than continuous listening.
                    if 0 < progress_delta <= elapsed_ms + 1500:
                        listened_ms += progress_delta

                previous_progress_ms = progress_ms
                previous_poll_time = poll_time
                previous_was_playing = is_playing

                time.sleep(
                    ACTIVE_POLL if is_playing else PAUSED_POLL
                )

            except SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(
                        getattr(e, "headers", {}).get(
                            "Retry-After", 10
                        )
                        if getattr(e, "headers", None)
                        else 10
                    )
                    print(f"Rate limited. Retrying in {retry_after}s.")
                    time.sleep(retry_after)
                else:
                    print(f"Spotify error: {e}")
                    time.sleep(5)

            except Exception as e:
                print(f"Polling error: {e}")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\nStopping playback tracker...")

        # Save the unfinished listening event before exiting.
        if current_track_data is not None:
            finish_track("daemon_stopped")


if __name__ == "__main__":
    start_daemon()