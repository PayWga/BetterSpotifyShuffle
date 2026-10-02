import argparse
import pandas as pd
import os

# --- Setup Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DB = os.path.join(SCRIPT_DIR, "master_track_table.csv")
BASE_DATASET = os.path.join(SCRIPT_DIR, "kaggle_track_table.csv")  

def init_database():
    """Run ONCE to create the Master DB from the heavy Kaggle dataset."""
    print("--- Initializing Master Database ---")
    
    if os.path.exists(MASTER_DB):
        print(f"Error: {MASTER_DB} already exists. Delete it if you want to re-initialize.")
        return
        
    if not os.path.exists(BASE_DATASET):
        print(f"Error: Could not find {BASE_DATASET}. Please download the Kaggle dataset first.")
        return
        
    print(f"Loading heavy baseline dataset ({BASE_DATASET}). This may take a moment...")
    tracks_db = pd.read_csv(BASE_DATASET)
    tracks_db.to_csv(MASTER_DB, index=False)
    print(f"SUCCESS: Master Database created with {len(tracks_db)} tracks!")


def merge_csv(new_csv_filename):
    """Run ON DEMAND. Merges a third-party CSV into the Master DB."""
    print(f"--- Merging external dataset: {new_csv_filename} ---")
    
    if not os.path.exists(MASTER_DB):
        print("Error: Master Database not found. Please run with --init first.")
        return
        
    target_path = new_csv_filename if os.path.isabs(new_csv_filename) else os.path.join(SCRIPT_DIR, new_csv_filename)
    
    if not os.path.exists(target_path):
        print(f"Error: Could not find the file {target_path}.")
        return
        
    print("Loading Master Database...")
    master_df = pd.read_csv(MASTER_DB)
    initial_count = len(master_df)
    
    print("Loading new dataset...")
    try:
        new_df = pd.read_csv(target_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
        
    # Smart column mapping (fixes common Kaggle naming differences)
    if 'track_id' in new_df.columns and 'id' not in new_df.columns:
        print("Auto-detected 'track_id' column. Renaming to 'id' to match Master DB...")
        new_df = new_df.rename(columns={'track_id': 'id'})
        
    if 'id' not in new_df.columns:
        print("Error: The new CSV must contain an 'id' column representing the Spotify Track ID.")
        return
        
    print(f"Found {len(new_df)} tracks in the new CSV. Merging and deduplicating...")
    
    # Concat and drop duplicates (keep='first' ensures we don't overwrite our good Master DB rows)
    combined_df = pd.concat([master_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['id'], keep='first')
    
    final_count = len(combined_df)
    added_count = final_count - initial_count
    duplicates_found = len(new_df) - added_count
    
    combined_df.to_csv(MASTER_DB, index=False)
    print(f"\nSUCCESS: Added {added_count} brand new tracks! (Ignored {duplicates_found} duplicates).")
    print(f"Master DB now has {final_count} tracks.")


def repair_database():
    """Prepares the database for a PyTorch Transformer by flagging missing data 
    with specific <UNK> indicator values instead of guessing them."""
    print("--- Formatting Missing Data for PyTorch ---")
    
    if not os.path.exists(MASTER_DB):
        print("Error: Master Database not found.")
        return
        
    df = pd.read_csv(MASTER_DB)
    
    # 1. STRICT PRUNING (The only thing we MUST have is the ID)
    initial_count = len(df)
    df = df.dropna(subset=['id'])
    if len(df) < initial_count:
        print(f"Dropped {initial_count - len(df)} tracks missing a Spotify ID.")
        
    # 2. PYTORCH INDICATORS (Flagging unknowns)
    # Continuous features get flagged with -1.0
    continuous_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    for col in continuous_features:
        if col in df.columns:
            df[col] = df[col].fillna(-1.0)
            
    # Discrete features get flagged with -1 (integer)
    discrete_features = ['key', 'mode', 'time_signature']
    for col in discrete_features:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype(int)
            
    df.to_csv(MASTER_DB, index=False)
    print(f"SUCCESS: Database formatted. Master DB has {len(df)} tracks ready for the Transformer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spotify Offline Track Features Database Manager")
    parser.add_argument("--init", action="store_true", help="Initialize the Master DB from the Kaggle dataset")
    parser.add_argument("--merge", type=str, metavar="FILENAME", help="Merge an external CSV dataset into the Master DB")
    parser.add_argument("--repair", action="store_true", help="Format missing data for PyTorch <UNK> processing")
    
    args = parser.parse_args()
    
    if args.init:
        init_database()
    elif args.merge:
        merge_csv(args.merge)
    elif args.repair:
        repair_database()
    else:
        print("Please specify an action. Usage examples:")
        print("  python.exe data_manager.py --init")
        print("  python.exe data_manager.py --merge workout_songs.csv")
        print("  python.exe data_manager.py --repair")