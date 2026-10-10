import os
import json
import glob
import pandas as pd

def build_duration_lookup(history_files, master_db_path):
    """Indexes true track durations using master DB and JSON completions."""
    duration_lookup = {}
    if os.path.exists(master_db_path):
        try:
            db_df = pd.read_csv(master_db_path, usecols=['track_id', 'duration_ms'], low_memory=False)
            db_df = db_df.dropna().drop_duplicates(subset=['track_id'])
            duration_lookup = dict(zip(db_df['track_id'], db_df['duration_ms'].astype(float)))
        except Exception:
            pass

    for filepath in history_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                uri = item.get('spotify_track_uri')
                if not uri or not uri.startswith('spotify:track:'):
                    continue
                track_id = uri.split(':')[-1]
                ms_played = item.get('ms_played', 0)
                # If reason_end is 'trackdone', the ms_played is the track's true length
                if item.get('reason_end', '') == 'trackdone' and ms_played > 0:
                    if ms_played > duration_lookup.get(track_id, 0):
                        duration_lookup[track_id] = float(ms_played)
    return duration_lookup

def process_extended_streaming_history(history_folder, master_db_path, output_csv, memory_file):
    """Parses Spotify JSONs into a sequence-ready CSV."""
    history_files = glob.glob(os.path.join(history_folder, "Streaming_History_Audio_*.json"))
    if not history_files:
        raise FileNotFoundError(f"No history JSONs found in {history_folder}")

    duration_lookup = build_duration_lookup(history_files, master_db_path)
    SKIP_THRESHOLD, PLAY_THRESHOLD, DEFAULT_DURATION_MS = 0.15, 0.70, 210000.0
    parsed_records, lifetime_plays, lifetime_skips = [], {}, {}

    for filepath in history_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for row in data:
            uri = row.get('spotify_track_uri')
            if not uri or not uri.startswith('spotify:track:'):
                continue
                
            track_id = uri.split(':')[-1]
            ms_played = row.get('ms_played', 0)
            reason_end = row.get('reason_end', '')
            skipped = row.get('skipped', False)
            
            total_duration = duration_lookup.get(track_id, DEFAULT_DURATION_MS)
            percent_completed = ms_played / max(total_duration, 1.0)

            is_play, is_ghost_skip, event_type = False, False, 'NEUTRAL'

            if reason_end == 'trackdone' or percent_completed >= PLAY_THRESHOLD:
                is_play, event_type = True, 'PLAY'
                lifetime_plays[track_id] = lifetime_plays.get(track_id, 0) + 1
            elif (skipped or reason_end in ['fwdbtn', 'backbtn', 'endplay']) and percent_completed < SKIP_THRESHOLD:
                is_ghost_skip, event_type = True, 'GHOST_SKIP'
                lifetime_skips[track_id] = lifetime_skips.get(track_id, 0) + 1

            parsed_records.append({
                'timestamp': row.get('ts'),
                'track_id': track_id,
                'track_name': row.get('master_metadata_track_name'),
                'artist_name': row.get('master_metadata_album_artist_name'),
                'ms_played': ms_played,
                'duration_ms': total_duration,
                'percent_completed': round(percent_completed, 4),
                'reason_start': row.get('reason_start'),
                'reason_end': reason_end,
                'event_type': event_type,
                'is_play': int(is_play),
                'is_ghost_skip': int(is_ghost_skip)
            })

    df = pd.DataFrame(parsed_records)
    # Ensure UTC timezone alignment for safe deduplication later
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    df = df.sort_values('timestamp').reset_index(drop=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(df)} historical streams. Saved to {output_csv}.")

    # Apply historical skips and plays to the live daemon memory JSON
    if os.path.exists(memory_file):
        with open(memory_file, 'r') as f:
            memory = json.load(f)
        for t_id, count in lifetime_plays.items():
            memory.setdefault(t_id, {'play_count': 0, 'skip_count': 0, 'days_since_added': 365})['play_count'] += count
        for t_id, count in lifetime_skips.items():
            memory.setdefault(t_id, {'play_count': 0, 'skip_count': 0, 'days_since_added': 365})['skip_count'] += count
        with open(memory_file, 'w') as f:
            json.dump(memory, f, indent=4)
        print("Live Daemon memory updated with lifetime interaction counts.")
            
    return output_csv

def combine_and_deduplicate(extended_csv, logged_csv, output_csv):
    print("Loading historical datasets...")
    
    df_ext = pd.read_csv(extended_csv, low_memory=False)
    df_log = pd.read_csv(logged_csv, low_memory=False)
    
    # NORMALIZATION: Force both formats to parse, then round down to the nearest second
    df_ext['timestamp'] = pd.to_datetime(df_ext['timestamp'], format='mixed', utc=True).dt.tz_localize(None).dt.floor('s')
    df_log['timestamp'] = pd.to_datetime(df_log['timestamp'], format='mixed', utc=True).dt.tz_localize(None).dt.floor('s')
    
    df_combined = pd.concat([df_ext, df_log], ignore_index=True)
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    
    # SCHEMA ENFORCEMENT: Safely fill missing string columns
    string_cols = ['episode_name', 'episode_show_name', 'reason_start', 'reason_end', 'track_name', 'artist_name']
    existing_str_cols = [c for c in string_cols if c in df_combined.columns]
    
    if existing_str_cols:
        df_combined[existing_str_cols] = df_combined[existing_str_cols].fillna("")
    
    # Safely force boolean columns to strict integers (0 or 1)
    bool_cols = ['shuffle', 'skipped', 'offline', 'incognito_mode', 'is_play', 'is_ghost_skip']
    for col in bool_cols:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].fillna(False).astype(int)
    
    # TIME-WINDOW DEDUPLICATION
    df_combined['time_diff'] = df_combined.groupby('track_id')['timestamp'].diff()
    is_duplicate = df_combined['time_diff'] < pd.Timedelta(seconds=60)
    
    df_clean = df_combined[~is_duplicate].drop(columns=['time_diff']).reset_index(drop=True)
    
    # FORMATTING: Convert the cleaned datetimes back into uniform strings
    df_clean['timestamp'] = df_clean['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    df_clean.to_csv(output_csv, index=False)
    print(f"Removed {len(df_combined) - len(df_clean)} overlaps. Unified dataset size: {len(df_clean)} streams.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    MASTER_TRACK_TABLE = os.path.join(BASE_DIR, "master_track_table.csv")
    DAEMON_MEMORY_FILE = os.path.join(BASE_DIR, "sensitive_info","daemon_memory.json")
    
    TEMP_EXTENDED_CSV = os.path.join(BASE_DIR, "sensitive_info", "extended_history_temp.csv")
    LIVE_LOGGED_CSV   = os.path.join(BASE_DIR, "sensitive_info", "spotify_listening_history.csv")
    UNZIPPED_SPOTIFY_FOLDER = os.path.join(BASE_DIR, "sensitive_info", "history")
    FINAL_UNIFIED_CSV = LIVE_LOGGED_CSV


    print("Step 1: Parsing Spotify Extended History JSONs...")
    process_extended_streaming_history(
        history_folder=UNZIPPED_SPOTIFY_FOLDER,
        master_db_path=MASTER_TRACK_TABLE,
        output_csv=TEMP_EXTENDED_CSV,
        memory_file=DAEMON_MEMORY_FILE
    )
    
    print("\nStep 2: Recombining and Deduplicating into Live Log...")
    if os.path.exists(LIVE_LOGGED_CSV):
        combine_and_deduplicate(
            extended_csv=TEMP_EXTENDED_CSV,
            logged_csv=LIVE_LOGGED_CSV,
            output_csv=FINAL_UNIFIED_CSV
        )
    else:
        print(f"No existing log found at {LIVE_LOGGED_CSV}. Promoting extended history to primary log.")
        os.rename(TEMP_EXTENDED_CSV, FINAL_UNIFIED_CSV)
        
    if os.path.exists(TEMP_EXTENDED_CSV):
        os.remove(TEMP_EXTENDED_CSV)
        
    print(f"\nSuccess. Unified history saved directly to: {LIVE_LOGGED_CSV}")