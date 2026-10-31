
import os
import json
import glob
import math
from pathlib import Path
from typing import NamedTuple

import pandas as pd

if __package__:
    from .history_schema import REQUIRED_COLUMNS
else:
    from history_schema import REQUIRED_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MASTER_TRACK_TABLE = PROJECT_ROOT / "data" / "master_track_table.csv"

PRIVATE_DIR = PROJECT_ROOT / "sensitive_info"

DAEMON_MEMORY_FILE = PRIVATE_DIR / "daemon_memory.json"

TEMP_EXTENDED_CSV = PRIVATE_DIR / "extended_history_temp.csv"

LIVE_LOGGED_CSV = PRIVATE_DIR / "spotify_listening_history.csv"

UNZIPPED_SPOTIFY_FOLDER = PRIVATE_DIR / "history"

SKIP_THRESHOLD = 0.15
PLAY_THRESHOLD = 0.70

EVENT_COLUMNS = list(REQUIRED_COLUMNS)


class DurationEvidence(NamedTuple):
    duration_ms: float
    duration_ms_is_estimated: bool | None
    duration_ms_source: str | None


UNKNOWN_DURATION = DurationEvidence(float('nan'), None, None)


def estimate_status(value):
    """Read known estimate flags without treating an absent flag as confirmation."""
    return {'true': True, '1': True, '1.0': True,
            'false': False, '0': False, '0.0': False}.get(str(value).strip().casefold())


def duration_is_estimated(flag, *sources):
    return estimate_status(flag) is True or any(
        str(source).strip().casefold() == 'history_completed_play_median' for source in sources)


def build_duration_lookup(history_files, master_db_path, *, include_provenance=False):
    """Build a track duration lookup from the master database and history."""

    duration_lookup = {}

    if os.path.exists(master_db_path):
        fields = {'id', 'duration_ms', 'duration_ms_is_estimated',
                  'duration_ms_source', 'duration_ms_metadata_source'}
        db_df = pd.read_csv(master_db_path, usecols=lambda name: name in fields,
                            dtype='string', keep_default_na=False, low_memory=False)
        if not {'id', 'duration_ms'}.issubset(db_df):
            raise ValueError('Master duration lookup requires id and duration_ms')
        db_df["id"] = db_df["id"].str.strip()
        db_df["duration_ms"] = pd.to_numeric(
            db_df["duration_ms"],
            errors="coerce"
        )

        db_df = db_df.dropna(subset=["id", "duration_ms"])
        db_df = db_df[(db_df["duration_ms"] > 0) & db_df['duration_ms'].map(math.isfinite)
                        & db_df['id'].ne('')]
        db_df = db_df.drop_duplicates(subset=["id"])

        for row in db_df.itertuples(index=False):
            source = (getattr(row, 'duration_ms_source', '')
                      or getattr(row, 'duration_ms_metadata_source', '') or 'master_catalog')
            estimated = estimate_status(getattr(row, 'duration_ms_is_estimated', None))
            if estimated is None and source == 'history_completed_play_median':
                estimated = True
            duration_lookup[row.id] = DurationEvidence(row.duration_ms, estimated, source)

    # Estimate duration from completed plays only when the catalog has no value.
    # These estimates are approximate, not independently confirmed durations.
    estimated_durations = {}

    for filepath in history_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            uri = item.get("spotify_track_uri")

            if not isinstance(uri, str) or not uri.startswith("spotify:track:"):
                continue

            track_id = uri.split(":")[-1].strip()
            if not track_id:
                continue

            if track_id in duration_lookup:
                continue

            ms_played = pd.to_numeric(item.get("ms_played"), errors='coerce')

            if (
                item.get("reason_end") == "trackdone"
                and pd.notna(ms_played) and math.isfinite(ms_played) and ms_played > 0
            ):
                estimated_durations.setdefault(
                    track_id, []
                ).append(ms_played)

    for track_id, durations in estimated_durations.items():
        duration_lookup[track_id] = DurationEvidence(float(pd.Series(durations).median()),
            True, 'history_completed_play_median')

    return duration_lookup if include_provenance else {
        track: record.duration_ms for track, record in duration_lookup.items()}


def classify_listening_event(
    completion_rate,
    reason_end,
    skipped=False
):
    """Classify an event using completion and playback information."""

    if reason_end == "trackdone":
        return "PLAY", 1, 0

    if pd.isna(completion_rate) or not math.isfinite(completion_rate):
        return "NEUTRAL", 0, 0

    if completion_rate >= PLAY_THRESHOLD:
        return "PLAY", 1, 0

    confirmed_skip = (
        skipped is True
        or reason_end in ["fwdbtn", "backbtn"]
    )

    if confirmed_skip and completion_rate < SKIP_THRESHOLD:
        return "GHOST_SKIP", 0, 1

    return "NEUTRAL", 0, 0


def process_extended_streaming_history(history_folder, master_db_path, output_csv):
    """Convert Spotify Extended Streaming History into normalized events."""

    history_files = sorted(
        glob.glob(
            os.path.join(
                history_folder,
                "Streaming_History_Audio_*.json"
            )
        )
    )

    if not history_files:
        raise FileNotFoundError(
            f"No Spotify history JSON files found in {history_folder}"
        )

    duration_lookup = build_duration_lookup(
        history_files,
        master_db_path,
        include_provenance=True,
    )

    parsed_records = []

    for filepath in history_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for row in data:
            uri = row.get("spotify_track_uri")

            if not isinstance(uri, str) or not uri.startswith("spotify:track:"):
                continue

            track_id = uri.split(":")[-1].strip()
            if not track_id:
                continue

            timestamp = row.get("ts")
            if not timestamp:
                continue

            ms_played = max(
                0, int(row.get("ms_played") or 0)
            )

            evidence = duration_lookup.get(track_id, UNKNOWN_DURATION)
            duration_ms = evidence.duration_ms

            completion_rate = min(ms_played / duration_ms, 1.) if pd.notna(duration_ms) else float('nan')

            reason_end = row.get("reason_end") or ""
            reason_start = row.get("reason_start") or ""
            skipped = row.get("skipped", False)

            event_type, is_play, is_ghost_skip = (
                classify_listening_event(
                    float('nan') if duration_is_estimated(evidence.duration_ms_is_estimated,
                        evidence.duration_ms_source) else completion_rate,
                    reason_end,
                    skipped
                )
            )

            parsed_records.append({
                "timestamp": timestamp,
                "track_id": track_id,

                "track_name": row.get(
                    "master_metadata_track_name"
                ),

                "artist_name": row.get(
                    "master_metadata_album_artist_name"
                ),

                "ms_played": ms_played,
                "duration_ms": duration_ms,
                "duration_ms_is_estimated": evidence.duration_ms_is_estimated,
                "duration_ms_source": evidence.duration_ms_source,
                "completion_rate": round(completion_rate, 4),

                "reason_start": reason_start,
                "reason_end": reason_end,

                "event_type": event_type,
                "is_play": is_play,
                "is_ghost_skip": is_ghost_skip,

                "context_uri": None,
                "context_type": None,

                "event_source": "extended_history",

                # Keep extra Spotify fields available for later experiments.
                "shuffle": row.get("shuffle"),
                "skipped": skipped,
                "offline": row.get("offline"),
                "platform": row.get("platform"),
                "incognito_mode": row.get("incognito_mode")
            })

    df = pd.DataFrame(parsed_records)

    if df.empty:
        raise ValueError("No valid track events found in history.")

    # Extended Spotify history timestamps are UTC.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    df = df.sort_values(
        "timestamp",
        kind="stable"
    ).reset_index(drop=True)

    # Write an explicit UTC timestamp so timezone information survives CSV storage.
    df["timestamp"] = df["timestamp"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_csv, index=False)

    print(
        f"Processed {len(df)} historical listening events. "
        f"Saved to {output_csv}."
    )

    return output_csv


def normalize_listening_history(df, source):
    """Normalize historical and live listening records."""

    df = df.copy()

    if 'track_id' not in df:
        raise ValueError("Listening data must contain 'track_id'.")
    df['track_id'] = df['track_id'].astype('string').str.strip()
    df = df[df['track_id'].notna() & df['track_id'].ne('')].copy()

    # Rename columns from the original polling format.
    df = df.rename(columns={
        "listened_ms": "ms_played",
        "artist": "artist_name",
        "percent_completed": "completion_rate"
    })

    if "completion_rate" not in df.columns:
        df["completion_rate"] = float("nan")

    if "ms_played" not in df.columns:
        df["ms_played"] = 0

    if "duration_ms" not in df.columns:
        df["duration_ms"] = float("nan")

    df["ms_played"] = pd.to_numeric(
        df["ms_played"], errors="coerce"
    ).fillna(0)

    df["duration_ms"] = pd.to_numeric(
        df["duration_ms"], errors="coerce"
    )

    df["completion_rate"] = pd.to_numeric(
        df["completion_rate"], errors="coerce"
    )

    missing_completion = df["completion_rate"].isna()

    valid_duration = df["duration_ms"] > 0

    can_calculate = missing_completion & valid_duration

    df.loc[can_calculate, "completion_rate"] = (
        df.loc[can_calculate, "ms_played"]
        / df.loc[can_calculate, "duration_ms"]
    ).clip(0, 1)

    # Keep an existing event source; fill it only when missing.
    if "event_source" not in df.columns:
        df["event_source"] = source
    else:
        df["event_source"] = df["event_source"].fillna(source)

    for col in [
        "reason_start", "reason_end",
        "context_uri", "context_type",
        "track_name", "artist_name"
    ]:
        if col not in df.columns:
            df[col] = None

    for col in ["event_type", "is_play", "is_ghost_skip"]:
        if col not in df.columns:
            df[col] = None

    # Classify records that do not already have labels.
    missing_labels = df["event_type"].isna() | df['event_type'].astype('string').str.strip().eq('')

    for idx in df.index[missing_labels]:
        completion = df.at[idx, "completion_rate"]

        reason_end = df.at[idx, "reason_end"]
        reason_end = '' if pd.isna(reason_end) else str(reason_end)
        skipped = (
            df.at[idx, "skipped"]
            if "skipped" in df.columns
            else False
        )
        skipped = str(skipped).strip().casefold() in ('true', '1', '1.0')
        estimated = df.at[idx, 'duration_ms_is_estimated'] if 'duration_ms_is_estimated' in df else None
        sources = (df.at[idx, name] for name in ('duration_ms_source', 'duration_ms_metadata_source') if name in df)
        if duration_is_estimated(estimated, *sources):
            completion = float('nan')

        # Older live logs may lack a skip reason.
        # An early ending provides inferred feedback, not a confirmed dislike.
        if source == "live_polling" and not reason_end and pd.notna(completion):
            if completion >= PLAY_THRESHOLD:
                labels = ("PLAY", 1, 0)
            elif completion < SKIP_THRESHOLD:
                labels = ("GHOST_SKIP", 0, 1)
            else:
                labels = ("NEUTRAL", 0, 0)
        else:
            labels = classify_listening_event(
                completion,
                reason_end or "",
                skipped
            )

        for name, value in zip(('event_type', 'is_play', 'is_ghost_skip'), labels):
            df.at[idx, name] = str(value) if isinstance(df[name].dtype, pd.StringDtype) else value

    return df


def normalize_timestamps(df, legacy_timezone=None):
    """Convert timestamps to timezone-aware UTC datetimes."""

    df = df.copy()

    def parse_timestamp(value, source):
        if pd.isna(value):
            return pd.NaT

        ts = pd.Timestamp(value)

        if ts.tzinfo is not None:
            return ts.tz_convert("UTC")

        if source == "live_polling":
            if legacy_timezone is None:
                raise ValueError(
                    "Legacy live timestamps need an explicit timezone."
                )

            return ts.tz_localize(
                legacy_timezone,
                ambiguous="raise",
                nonexistent="raise"
            ).tz_convert("UTC")

        # Older extended-history CSV exports were stored
        # as timezone-naive UTC timestamps.
        return ts.tz_localize("UTC")

    timestamps = [
        parse_timestamp(value, source)
        for value, source in zip(
            df["timestamp"],
            df["event_source"]
        )
    ]
    df['timestamp'] = pd.Series(timestamps, index=df.index, dtype='datetime64[ns, UTC]')

    return df.dropna(subset=["timestamp"])


def combine_and_deduplicate(extended_csv, logged_csv, output_csv, legacy_timezone=None):
    """Merge listening histories and remove duplicate events."""

    print("Loading historical listening datasets...")

    options = dict(dtype='string', keep_default_na=False, na_values=[''], low_memory=False)
    df_ext = pd.read_csv(extended_csv, **options)
    df_log = pd.read_csv(logged_csv, **options)

    df_ext = normalize_listening_history(
        df_ext,
        source="extended_history"
    )

    df_log = normalize_listening_history(
        df_log,
        source="live_polling"
    )

    df_ext = normalize_timestamps(
        df_ext,
        legacy_timezone=legacy_timezone
    )

    df_log = normalize_timestamps(
        df_log,
        legacy_timezone=legacy_timezone
    )

    df_combined = pd.concat(
        [df_ext, df_log],
        ignore_index=True
    )

    # Preserve extended history when duplicate events overlap.
    df_combined["_priority"] = (
        df_combined["event_source"]
        .map({
            "extended_history": 0,
            "live_polling": 1
        })
        .fillna(2)
    )

    df_combined = df_combined.sort_values(
        ["_priority", "timestamp"],
        kind="stable"
    )

    # Remove exact duplicate event timestamps and track IDs.
    df_combined = df_combined.drop_duplicates(
        subset=["track_id", "timestamp"],
        keep="first"
    )

    # Check live events against imported history for nearby duplicates.
    # Keep events without a matching track and timestamp within the tolerance.
    extended = df_combined[
        df_combined["event_source"] == "extended_history"
    ].sort_values("timestamp")

    other = df_combined[
        df_combined["event_source"] != "extended_history"
    ].sort_values("timestamp")

    
    if not extended.empty and not other.empty:
        other = other.copy()
        other["_row_id"] = range(len(other))

        reference = extended[
            ["track_id", "timestamp"]
        ].rename(columns={
            "timestamp": "matched_timestamp"
        })

        matched = pd.merge_asof(
            other.sort_values("timestamp"),
            reference.sort_values("matched_timestamp"),
            left_on="timestamp",
            right_on="matched_timestamp",
            by="track_id",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=5)
        )

        unmatched_ids = set(
            matched.loc[
                matched["matched_timestamp"].isna(),
                "_row_id"
            ]
        )

        other = other[
            other["_row_id"].isin(unmatched_ids)
        ].drop(columns=["_row_id"])

    df_clean = pd.concat(
        [extended, other],
        ignore_index=True
    )

    df_clean = df_clean.sort_values(
        "timestamp",
        kind="stable"
    ).reset_index(drop=True)

    df_clean = df_clean.drop(columns=["_priority"])
    df_clean = df_clean.reindex(columns=EVENT_COLUMNS + [
        name for name in df_clean.columns if name not in EVENT_COLUMNS])

    df_clean["timestamp"] = (
        df_clean["timestamp"]
        .dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )

    output_csv = Path(output_csv)

    temp_path = output_csv.with_suffix(".tmp.csv")

    try:
        df_clean.to_csv(temp_path, index=False)
        os.replace(temp_path, output_csv)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(
        f"Removed {len(df_combined) - len(df_clean)} "
        f"additional cross-source overlaps."
    )
    print(
        f"Unified dataset contains {len(df_clean)} events."
    )

    return output_csv


def rebuild_preference_counts(history_csv, memory_file):
    """Recalculate play and skip counts from unified history."""

    history = pd.read_csv(history_csv, low_memory=False)

    required = ["track_id", "is_play", "is_ghost_skip"]
    missing = [c for c in required if c not in history.columns]

    if missing:
        raise ValueError(f"Missing history columns: {missing}")

    history = history.dropna(subset=["track_id"]).copy()
    history["track_id"] = history["track_id"].astype(str).str.strip()
    history = history[history["track_id"] != ""]

    for col in ["is_play", "is_ghost_skip"]:
        history[col] = pd.to_numeric(
            history[col], errors="coerce"
        ).fillna(0).astype(int)

    aggregates = history.groupby("track_id").agg(
        play_count=("is_play", "sum"),
        skip_count=("is_ghost_skip", "sum")
    )

    memory_file = Path(memory_file)

    if memory_file.exists():
        with open(memory_file, "r", encoding="utf-8") as f:
            memory = json.load(f)
    else:
        memory = {}

    if not isinstance(memory, dict):
        raise ValueError("Invalid daemon memory format.")

    # Rebuild counts from unified history while keeping other per-track metadata.
    for record in memory.values():
        record["play_count"] = 0
        record["skip_count"] = 0

    for track_id, row in aggregates.iterrows():
        record = memory.setdefault(
            track_id,
            {
                "play_count": 0,
                "skip_count": 0,
                "added_at": None
            }
        )

        record["play_count"] = int(row["play_count"])
        record["skip_count"] = int(row["skip_count"])

    memory_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = memory_file.with_suffix(".tmp.json")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=4)

        os.replace(temp_path, memory_file)

    finally:
        if temp_path.exists():
            temp_path.unlink()

    print("Preference counts rebuilt from unified history.")


if __name__ == "__main__":
    import shutil

    print("Step 1: Parsing Spotify Extended History...")

    process_extended_streaming_history(
        history_folder=UNZIPPED_SPOTIFY_FOLDER,
        master_db_path=MASTER_TRACK_TABLE,
        output_csv=TEMP_EXTENDED_CSV
    )

    # Preserve the existing listening history before modifying it.
    if LIVE_LOGGED_CSV.exists():
        backup_path = LIVE_LOGGED_CSV.with_suffix(".backup.csv")
        shutil.copy2(LIVE_LOGGED_CSV, backup_path)

        print(f"Backup saved to {backup_path}")

        print("\nStep 2: Merging and deduplicating history...")

        combine_and_deduplicate(
            extended_csv=TEMP_EXTENDED_CSV,
            logged_csv=LIVE_LOGGED_CSV,
            output_csv=LIVE_LOGGED_CSV,

            # Legacy live logs without an offset need the original polling
            # machine's timezone here.
            legacy_timezone=None
        )

    else:
        print("No existing live history found.")

        os.replace(
            TEMP_EXTENDED_CSV,
            LIVE_LOGGED_CSV
        )

    if TEMP_EXTENDED_CSV.exists():
        TEMP_EXTENDED_CSV.unlink()

    print("\nStep 3: Rebuilding preference counts...")

    rebuild_preference_counts(
        history_csv=LIVE_LOGGED_CSV,
        memory_file=DAEMON_MEMORY_FILE
    )

    print("\nHistory import completed successfully.")
