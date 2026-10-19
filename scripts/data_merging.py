
import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import spotipy
from requests.exceptions import RequestException
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler

if __package__:
    from . import spotify_metadata
    from . import local_backfill
else:
    import spotify_metadata
    import local_backfill

# Paths: assumes this file is in the project's scripts/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PRIVATE_DIR = PROJECT_ROOT / "sensitive_info"

MASTER_DB = DATA_DIR / "master_track_table.csv"
BASE_DATASET = DATA_DIR / "kaggle_track_table.csv"
HISTORY_DB = PRIVATE_DIR / "spotify_listening_history.csv"

SCOPE = "user-read-private"


# --- Audio features ---

# Continuous audio characteristics
CONTINUOUS_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo"
]

# Discrete audio characteristics
DISCRETE_FEATURES = [
    "key",
    "mode",
    "time_signature"
]

# Audio features expected to be between 0 and 1
BOUNDED_FEATURES = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]


# --- Additional numerical features ---

NUMERICAL_FEATURES = [
    "duration_ms",
    "track_number",
    "disc_number",
    "year",
    "popularity",
    "stream_count"
]

# --- Boolean features ---

BOOLEAN_FEATURES = [
    "explicit"
]

# --- Categorical features ---

CATEGORICAL_FEATURES = [
    "album_id",
    "artist_ids",
    "track_genre",
    "genre",
    "country",
    "label"
]

# --- Track identity and descriptive metadata ---

METADATA_COLUMNS = [
    "id",
    "name",
    "album",
    "artists",
    "release_date",
    "album_name",
    "track_name",
    "artist_name"
]


# A measured -1 is impossible in these known nonnegative fields. For key,
# Spotify's -1 means "no key detected", normalized to this project's NaN.
# Loudness and local_rms_db are deliberately excluded: -1 dB can be real.
# Unknown/custom columns and provenance are never interpreted as placeholders.
PLACEHOLDER_COLUMNS = (
    BOUNDED_FEATURES + ["tempo"] + DISCRETE_FEATURES
    + NUMERICAL_FEATURES + BOOLEAN_FEATURES
    + ["release_year", "local_onset_strength", "local_spectral_centroid",
       "local_spectral_contrast", "local_beat_regularity"]
)


def clear_feature_placeholders(df):
    """Return a copy with exact numeric -1 sentinels cleared, plus column counts.

    Numeric conversion is only used for comparison; all other cells keep their
    original values and representation. No range repair or row pruning occurs.
    """
    cleaned = df.copy()
    counts = {}
    for column in PLACEHOLDER_COLUMNS:
        if column not in cleaned:
            continue
        missing = pd.to_numeric(cleaned[column], errors="coerce").eq(-1).fillna(False)
        count = int(missing.sum())
        if count:
            cleaned[column] = cleaned[column].mask(missing)
            counts[column] = count
    return cleaned, counts


def clear_placeholders(master_path=None, dry_run=False):
    """Clear known -1 placeholders in a master CSV, or report a read-only preview.

    Run only when other catalog writers (including extraction) have stopped.
    Returns counts of changed cells per column; an unchanged file is not rewritten.
    """
    path = Path(MASTER_DB if master_path is None else master_path)
    # Read text verbatim so labels like 0007 and names like NA survive the rewrite.
    df = pd.read_csv(path, dtype="string", keep_default_na=False, low_memory=False)
    if "id" not in df.columns:
        raise ValueError("Master CSV must have an id column")
    cleaned, counts = clear_feature_placeholders(df)
    total = sum(counts.values())
    if not dry_run and total:
        save_database(cleaned, path=path)
    action = "DRY RUN: would clear" if dry_run else "Cleared"
    print(f"{action} {total} placeholder cells across {len(counts)} columns.")
    for column, count in counts.items():
        print(f"  {column}: {count}")
    return counts



def normalize_track_ids(df):
    """Normalize the track ID column to 'id'."""

    df = df.copy()

    if "id" not in df.columns:
        if "track_id" in df.columns:
            df = df.rename(columns={"track_id": "id"})
        else:
            raise ValueError("Dataset must contain 'id' or 'track_id'.")

    elif "track_id" in df.columns:
        # If both columns exist, check that they don't disagree.
        left = df["id"].astype("string").str.strip()
        right = df["track_id"].astype("string").str.strip()

        conflicts = (
            left.notna() & right.notna() & (left != right)
        ).fillna(False)

        if conflicts.any():
            raise ValueError(
                "Conflicting values found in 'id' and 'track_id'."
            )

        df["id"] = left.fillna(right)
        df = df.drop(columns=["track_id"])

    df["id"] = df["id"].astype("string").str.strip()

    df = df.dropna(subset=["id"])
    df = df[df["id"] != ""]

    return df



def clean_track_features(df):
    """Clean numerical features while preserving valid values."""

    df, _ = clear_feature_placeholders(df)

    audio_features = CONTINUOUS_FEATURES + DISCRETE_FEATURES
    numeric_columns = audio_features + NUMERICAL_FEATURES

    # Convert numerical columns to numeric types
    for col in numeric_columns:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate audio features bounded between 0 and 1
    for col in BOUNDED_FEATURES:
        if col in df.columns:
            invalid = (
                df[col].notna()
                & ~df[col].between(0.0, 1.0)
            )
            df.loc[invalid, col] = float("nan")

    # Tempo must not be negative.
    # Zero is retained as a possible no-beat estimate.
    if "tempo" in df.columns:
        df.loc[df["tempo"] < 0, "tempo"] = float("nan")

    # Musical key: C=0, C#=1, ..., B=11
    if "key" in df.columns:
        invalid = (
            df["key"].notna()
            & ~df["key"].isin(range(12))
        )
        df.loc[invalid, "key"] = float("nan")

    # Musical mode: minor=0, major=1
    if "mode" in df.columns:
        invalid = (
            df["mode"].notna()
            & ~df["mode"].isin([0, 1])
        )
        df.loc[invalid, "mode"] = float("nan")

    # Time signature must be a positive integer
    if "time_signature" in df.columns:
        values = df["time_signature"]
        invalid = (
            values.notna()
            & ((values <= 0) | (values % 1 != 0))
        )
        df.loc[invalid, "time_signature"] = float("nan")

    # Nonnegative numerical metadata
    nonnegative_features = [
        "duration_ms",
        "track_number",
        "disc_number",
        "stream_count"
    ]

    for col in nonnegative_features:
        if col in df.columns:
            df.loc[df[col] < 0, col] = float("nan")

    # Track popularity, when represented on Spotify's
    # historical 0–100 scale
    if "popularity" in df.columns:
        invalid = (
            df["popularity"].notna()
            & ~df["popularity"].between(0, 100)
        )
        df.loc[invalid, "popularity"] = float("nan")

    # Validate release year conservatively
    if "year" in df.columns:
        invalid = (
            df["year"].notna()
            & (
                (df["year"] < 1850)
                | (df["year"] > 2100)
                | (df["year"] % 1 != 0)
            )
        )
        df.loc[invalid, "year"] = float("nan")

    return df



def merge_track_data(master_df, new_df):
    """Merge tracks, filling missing values from incoming records."""

    master_df = normalize_track_ids(master_df)
    new_df = normalize_track_ids(new_df)

    master_df = clean_track_features(master_df)
    new_df = clean_track_features(new_df)

    # Preserve one record per Spotify ID.
    master_df = master_df.drop_duplicates(
        subset=["id"], keep="first"
    )
    new_df = new_df.drop_duplicates(
        subset=["id"], keep="first"
    )

    master_df = master_df.set_index("id")
    new_df = new_df.set_index("id")

    # Existing valid values take priority.
    # Incoming values fill missing fields.
    combined_df = master_df.combine_first(new_df)

    return combined_df.reset_index()


def save_database(df, path=None):
    """Write the master database using an atomic file replacement."""

    path = Path(MASTER_DB if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".csv", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as output:
            df.to_csv(output, index=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, path)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_metadata_guard():
    return spotify_metadata.SpotifyRequestBudget(
        PRIVATE_DIR / 'spotify_request_budget.json',
        interval=spotify_metadata.REQUEST_INTERVAL,
        rest_every=spotify_metadata.REQUEST_REST_EVERY,
        rest_seconds=spotify_metadata.REQUEST_REST_SECONDS,
    )


def get_spotify_client():
    """Use the same private credentials/cache as the project's playback tools."""
    budget = get_metadata_guard()
    budget.check_allowed()
    with (PRIVATE_DIR / 'spotify_credentials.json').open(encoding='utf-8') as source:
        creds = json.load(source)
    return spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=creds['client_id'], client_secret=creds['client_secret'],
        redirect_uri=creds['redirect_uri'], scope=SCOPE,
        cache_handler=CacheFileHandler(cache_path=str(PRIVATE_DIR / '.spotify_cache')),
    ), requests_session=spotify_metadata.metadata_session(budget),
        retries=0, status_retries=0, requests_timeout=30)


def run_metadata_action(action, *args, **kwargs):
    """Keep rate-limit exits readable after the action's checkpoint has run."""
    try:
        return action(*args, **kwargs)
    except spotify_metadata.SpotifyBudgetBlocked as exc:
        print(str(exc))
        print('Completed checkpoints are retained. Rerun the same command after the cooldown.')
        raise SystemExit(2) from None


def backfill_local_metadata(master_path=None, source_root=None, dry_run=False):
    return local_backfill.run(MASTER_DB if master_path is None else master_path,
                              PROJECT_ROOT if source_root is None else source_root,
                              save_database, dry_run)


def backfill_metadata(master_path=None, checkpoint=None, **options):
    """Fill existing catalog gaps, or stage metadata while extraction runs."""
    master_path = Path(MASTER_DB if master_path is None else master_path)
    checkpoint = Path(checkpoint) if checkpoint is not None else master_path.with_suffix('.spotify-metadata.json')
    return spotify_metadata.backfill(master_path, checkpoint, get_spotify_client,
                                     save_database, **options)


def backfill_backlog_metadata(backlog_path=None, **options):
    """Enrich backlog-only tracks without adding or rewriting master rows."""
    path = Path(DATA_DIR / 'dsp_backlog.csv' if backlog_path is None else backlog_path)
    return spotify_metadata.backfill_backlog(path, get_spotify_client, save_database, **options)


def apply_metadata_checkpoint(master_path=None, checkpoint=None, dry_run=False):
    """Run after all other catalog writers have stopped."""
    master_path = Path(MASTER_DB if master_path is None else master_path)
    checkpoint = Path(checkpoint) if checkpoint is not None else master_path.with_suffix('.spotify-metadata.json')
    return spotify_metadata.apply_checkpoint(master_path, checkpoint, save_database, dry_run)


def init_database():
    """Initialize the master database from the Kaggle dataset."""

    print("--- Initializing Master Database ---")

    if MASTER_DB.exists():
        print(f"Error: {MASTER_DB} already exists.")
        return

    if not BASE_DATASET.exists():
        print(f"Error: Could not find {BASE_DATASET}.")
        return

    print("Loading baseline dataset...")
    tracks_db = pd.read_csv(BASE_DATASET, low_memory=False)

    tracks_db = normalize_track_ids(tracks_db)
    tracks_db = clean_track_features(tracks_db)

    tracks_db = tracks_db.drop_duplicates(
        subset=["id"], keep="first"
    )

    save_database(tracks_db)

    print(
        f"SUCCESS: Master database created with "
        f"{len(tracks_db)} tracks!"
    )


def merge_csv(new_csv_filename):
    """Merge a third-party CSV into the master database."""

    print(f"--- Merging external dataset: {new_csv_filename} ---")

    if not MASTER_DB.exists():
        print("Error: Master database not found. Run --init first.")
        return

    target_path = Path(new_csv_filename)

    if not target_path.is_absolute():
        target_path = PROJECT_ROOT / target_path

    if not target_path.exists():
        print(f"Error: File not found: {target_path}")
        return

    master_df = pd.read_csv(MASTER_DB, low_memory=False)
    new_df = pd.read_csv(target_path, low_memory=False)

    initial_count = len(normalize_track_ids(master_df)["id"].unique())

    print(f"Loaded {len(new_df)} incoming records.")

    combined_df = merge_track_data(master_df, new_df)

    final_count = len(combined_df)
    added_count = final_count - initial_count

    save_database(combined_df)

    print(f"SUCCESS: Added {added_count} new tracks.")
    print(f"Master database now contains {final_count} tracks.")



def sync_metadata():
    """Add unknown history tracks using the same mapping as metadata backfill.

    Run only when extraction and other catalog writers have stopped.
    """
    if not MASTER_DB.exists() or not HISTORY_DB.exists():
        print("Error: Master DB or listening history not found.")
        return
    master = spotify_metadata.read_catalog(MASTER_DB)
    history = pd.read_csv(HISTORY_DB, dtype="string", keep_default_na=False)
    if "track_id" not in history:
        raise ValueError("Listening history must contain 'track_id'.")
    missing_ids = sorted(set(history['track_id'].str.strip()) - set(master.index) - {''})
    if not missing_ids:
        print("All tracks in history exist in the master database.")
        return
    import re
    records = {}
    sp = get_spotify_client()

    def save_records():
        if records:
            # Reload before filling gaps to preserve updates made during fetching.
            current = spotify_metadata.read_catalog(MASTER_DB)
            spotify_metadata.fill_records(current, records, spotify_metadata.TRACK_FIELDS, allow_new=True)
            save_database(current.reset_index())

    for tid in missing_ids:
        if not re.fullmatch(r'[A-Za-z0-9]{22}', tid):
            continue
        try:
            record = spotify_metadata.fetch_track(sp, tid)
            if record:
                records[tid] = record
        except spotify_metadata.SpotifyBudgetBlocked:
            save_records()
            raise
        except spotipy.SpotifyException as exc:
            if exc.http_status in (401, 403):
                raise
            print(f"Track metadata unavailable (HTTP {exc.http_status}).")
        except RequestException:
            print('Track metadata request failed after retries; rerun --sync to retry.')
    if not records:
        print("No metadata was retrieved.")
        return
    save_records()
    print(f"SUCCESS: Retrieved metadata for {len(records)} tracks.")


def repair_database():
    """Clean track identifiers, duplicates, and audio features."""

    print("--- Repairing Master Database ---")

    if not MASTER_DB.exists():
        print("Error: Master database not found.")
        return

    df = pd.read_csv(MASTER_DB, low_memory=False)

    initial_count = len(df)

    df = normalize_track_ids(df)
    df = clean_track_features(df)

    df = df.drop_duplicates(
        subset=["id"], keep="first"
    )

    removed_count = initial_count - len(df)

    save_database(df)

    print(f"Removed {removed_count} invalid/duplicate records.")
    print(f"Master database now contains {len(df)} tracks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Offline Track Features Database Manager")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--init", action="store_true", help="Initialize the Master DB from the Kaggle dataset")
    actions.add_argument("--merge", type=str, metavar="FILENAME", help="Merge an external CSV dataset into the Master DB")
    actions.add_argument("--sync", action="store_true", help="Fetch basic metadata for unknown tracks in your history")
    actions.add_argument("--backfill", action="store_true", help="Fill missing Spotify metadata for existing master tracks")
    actions.add_argument("--backfill-local", action="store_true", help="Fill missing master metadata from local aliases, history, and saved Spotify data; no API calls")
    actions.add_argument("--backfill-backlog", action="store_true", help="Fill missing duration/recording metadata directly in the audio backlog")
    actions.add_argument("--apply-backfill", action="store_true", help="Apply a metadata checkpoint after other catalog writers stop")
    actions.add_argument("--repair", action="store_true", help="Clean identifiers, duplicates, and invalid feature values")
    actions.add_argument("--clear-placeholders", action="store_true", help="Convert exact -1 sentinels in known nonnegative columns to NaN")
    actions.add_argument("--spotify-budget-status", action="store_true", help="Show local request-budget/cooldown status without calling Spotify")
    actions.add_argument("--set-spotify-cooldown", type=float, metavar="SECONDS", help="Persist a known cooldown without making API calls; never shortens an existing pause")
    parser.add_argument("--master", type=Path, help="Master CSV for placeholder cleanup or metadata backfill")
    parser.add_argument("--local-source-root", type=Path, help="Project root containing data/ and sensitive_info/ for --backfill-local")
    parser.add_argument("--backlog", type=Path, help="Backlog CSV for --backfill-backlog")
    parser.add_argument("--dry-run", action="store_true", help="Preview cleanup/backfill/apply without writing or API requests")
    parser.add_argument("--checkpoint", type=Path, help="Resumable metadata JSON (default: beside the selected master)")
    parser.add_argument("--fetch-only", action="store_true", help="Fetch to checkpoint only; safe while extraction runs")
    parser.add_argument("--limit", type=int, help="Maximum new tracks to check in this backfill run")
    parser.add_argument("--fields", nargs='+', help="Only backfill these supported metadata columns")
    parser.add_argument("--extra-metadata", action="store_true", help="Also fetch album label and artist_genres, with cached album/artist requests")
    
    args = parser.parse_args()
    enrichment = args.backfill or args.apply_backfill
    if args.local_source_root is not None and not args.backfill_local:
        parser.error('--local-source-root requires --backfill-local')
    if args.master is not None and not (args.clear_placeholders or enrichment or args.backfill_local):
        parser.error("--master and --dry-run require --clear-placeholders, --backfill, or --apply-backfill")
    if args.dry_run and not (args.clear_placeholders or enrichment or args.backfill_backlog or args.backfill_local):
        parser.error("--dry-run requires a cleanup or metadata action")
    if args.backlog is not None and not args.backfill_backlog:
        parser.error("--backlog requires --backfill-backlog")
    if args.checkpoint is not None and not enrichment:
        parser.error("--checkpoint requires --backfill or --apply-backfill")
    if (args.fetch_only or args.fields or args.extra_metadata) and not args.backfill:
        parser.error("--fetch-only, --limit, --fields, and --extra-metadata require --backfill")
    if args.limit is not None and not (args.backfill or args.backfill_backlog):
        parser.error("--limit requires --backfill or --backfill-backlog")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    
    if args.spotify_budget_status:
        print(json.dumps(get_metadata_guard().summary(), indent=2))
    elif args.set_spotify_cooldown is not None:
        budget = get_metadata_guard()
        budget.pause(args.set_spotify_cooldown)
        print(json.dumps(budget.summary(), indent=2))
    elif args.init:
        init_database()
    elif args.merge:
        merge_csv(args.merge)
    elif args.sync:
        run_metadata_action(sync_metadata)
    elif args.backfill:
        run_metadata_action(backfill_metadata, args.master, args.checkpoint, dry_run=args.dry_run,
                          fetch_only=args.fetch_only, limit=args.limit,
                          fields=args.fields, extra_metadata=args.extra_metadata)
    elif args.backfill_local:
        backfill_local_metadata(args.master, args.local_source_root, args.dry_run)
    elif args.backfill_backlog:
        run_metadata_action(backfill_backlog_metadata, args.backlog, dry_run=args.dry_run, limit=args.limit)
    elif args.apply_backfill:
        apply_metadata_checkpoint(args.master, args.checkpoint, args.dry_run)
    elif args.repair:
        repair_database()
    elif args.clear_placeholders:
        clear_placeholders(args.master, args.dry_run)
    else:
        print("Please specify an action. Usage examples:")
        print("  python.exe data_manager.py --init")
        print("  python.exe data_manager.py --merge workout_songs.csv")
        print("  python data_manager.py --sync")
        print("  python data_manager.py --repair")
        print("  python scripts/data_merging.py --clear-placeholders --dry-run")
        print("  python scripts/data_merging.py --backfill --fetch-only")
        print("  python scripts/data_merging.py --apply-backfill")
