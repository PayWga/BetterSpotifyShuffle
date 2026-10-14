
import argparse
import os
import time
from pathlib import Path

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth

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

    df = df.copy()

    audio_features = CONTINUOUS_FEATURES + DISCRETE_FEATURES
    numeric_columns = audio_features + NUMERICAL_FEATURES

    # Convert numerical columns to numeric types
    for col in numeric_columns:
        if col not in df.columns:
            continue

        df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert legacy unknown indicators for audio
        # features into NaN.
        if col in audio_features:
            df[col] = df[col].replace(-1, float("nan"))

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


def save_database(df):
    """Write the master database using an atomic file replacement."""

    MASTER_DB.parent.mkdir(parents=True, exist_ok=True)

    temp_path = MASTER_DB.with_suffix(".tmp.csv")

    try:
        df.to_csv(temp_path, index=False)
        os.replace(temp_path, MASTER_DB)

    finally:
        if temp_path.exists():
            temp_path.unlink()


def get_spotify_client():
    return spotipy.Spotify(auth_manager=SpotifyOAuth(scope=SCOPE))


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
    """Fetch Spotify metadata for tracks missing from the master DB."""

    print("--- Syncing Missing Tracks from Spotify ---")

    if not MASTER_DB.exists() or not HISTORY_DB.exists():
        print("Error: Master DB or listening history not found.")
        return

    master_df = pd.read_csv(MASTER_DB, low_memory=False)
    history_df = pd.read_csv(HISTORY_DB, low_memory=False)

    master_df = normalize_track_ids(master_df)

    if "track_id" not in history_df.columns:
        raise ValueError("Listening history must contain 'track_id'.")

    history_ids = set(
        history_df["track_id"]
        .dropna()
        .astype("string")
        .str.strip()
    )

    history_ids.discard("")

    known_ids = set(master_df["id"])
    missing_ids = sorted(history_ids - known_ids)

    if not missing_ids:
        print("All tracks in history exist in the master database.")
        return

    print(f"Found {len(missing_ids)} missing tracks.")

    sp = get_spotify_client()
    new_tracks_data = []

    chunk_size = 50

    for i in range(0, len(missing_ids), chunk_size):
        batch = missing_ids[i:i + chunk_size]

        try:
            results = sp.tracks(batch)

            for track in results.get("tracks", []):
                if not track or not track.get("id"):
                    continue

                album = track.get("album") or {}
                artists = track.get("artists") or []

                release_date = album.get("release_date") or ""
                year_text = release_date.split("-")[0]

                release_year = (
                    int(year_text)
                    if year_text.isdigit()
                    else None
                )

                track_data = {
                    "id": track["id"],
                    "name": track.get("name"),

                    "artists": ", ".join(
                        artist["name"]
                        for artist in artists
                        if artist.get("name")
                    ),

                    "artist_ids": ", ".join(
                        artist["id"]
                        for artist in artists
                        if artist.get("id")
                    ),

                    "album": album.get("name"),
                    "album_id": album.get("id"),

                    "release_date": release_date or None,
                    "year": release_year,

                    "duration_ms": track.get("duration_ms"),
                    "explicit": track.get("explicit"),

                    "track_number": track.get("track_number"),
                    "disc_number": track.get("disc_number")
                }

                new_tracks_data.append(track_data)

            print(
                f"Fetched batch {i // chunk_size + 1}: "
                f"{min(i + chunk_size, len(missing_ids))}"
                f"/{len(missing_ids)}"
            )

            time.sleep(0.5)

        except Exception as e:
            print(f"API error on batch {i // chunk_size + 1}: {e}")
            time.sleep(5)

    if not new_tracks_data:
        print("No metadata was retrieved.")
        return

    new_df = pd.DataFrame(new_tracks_data)

    combined_df = merge_track_data(master_df, new_df)

    save_database(combined_df)

    print(f"SUCCESS: Added metadata for {len(new_tracks_data)} tracks.")
    print(f"Master database contains {len(combined_df)} tracks.")



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
    parser.add_argument("--init", action="store_true", help="Initialize the Master DB from the Kaggle dataset")
    parser.add_argument("--merge", type=str, metavar="FILENAME", help="Merge an external CSV dataset into the Master DB")
    parser.add_argument("--sync", action="store_true", help="Fetch basic metadata for unknown tracks in your history")
    parser.add_argument("--repair", action="store_true", help="Format missing data for PyTorch <UNK> processing")
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    elif args.merge:
        merge_csv(args.merge)
    elif args.sync:
        sync_metadata()
    elif args.repair:
        repair_database()
    else:
        print("Please specify an action. Usage examples:")
        print("  python.exe data_manager.py --init")
        print("  python.exe data_manager.py --merge workout_songs.csv")
        print("  python data_manager.py --sync")
        print("  python data_manager.py --repair")