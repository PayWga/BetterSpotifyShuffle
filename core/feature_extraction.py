import os
import time
import random
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import librosa

# Robust path management using pathlib
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

BACKLOG_PATH = PROJECT_ROOT / 'data' / 'dsp_backlog.csv'
MASTER_PATH = PROJECT_ROOT / 'data' / 'master_track_table.csv'
TEMP_AUDIO_DIR = PROJECT_ROOT / 'data' / 'temp_audio'

MASTER_COLUMNS = [
    'id', 'name', 'album', 'album_id', 'artists', 'artist_ids', 
    'track_number', 'disc_number', 'explicit', 'danceability', 
    'energy', 'key', 'loudness', 'mode', 'speechiness', 
    'acousticness', 'instrumentalness', 'liveness', 'valence', 
    'tempo', 'duration_ms', 'time_signature', 'year', 'release_date', 
    'Unnamed: 0', 'album_name', 'track_name', 'popularity', 
    'track_genre', 'artist_name', 'genre', 'stream_count', 'country', 'label'
]

def fetch_audio_from_youtube(track_name, artist_name, output_filepath):
    """Searches and downloads studio audio from YouTube safely using yt-dlp."""
    query = f"ytsearch1:{artist_name} {track_name} official audio"
    base_output = output_filepath.rsplit('.', 1)[0]
    
    cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'mp3',
        '--default-search', 'ytsearch',
        '--output', f"{base_output}.%(ext)s",
        '--quiet',
        query
    ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download audio for {track_name} by {artist_name}: {e}")
        return False

def estimate_key_and_mode(y, sr):
    """Estimates musical key (0-11) and mode (1 for major, 0 for minor) using librosa chroma."""
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        
        # Standard Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_corrs = [np.corrcoef(np.roll(major_profile, i), chroma_avg)[0, 1] for i in range(12)]
        minor_corrs = [np.corrcoef(np.roll(minor_profile, i), chroma_avg)[0, 1] for i in range(12)]
        
        best_major_key = np.argmax(major_corrs)
        best_minor_key = np.argmax(minor_corrs)
        
        if max(major_corrs) >= max(minor_corrs):
            return int(best_major_key), 1
        else:
            return int(best_minor_key), 0
    except Exception:
        return -1, None

def estimate_danceability(y, sr):
    """Estimates danceability via beat consistency and pulse clarity."""
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) < 4:
            return 0.0
        # Compute beat interval variance (regularity of rhythm)
        beat_intervals = np.diff(beats)
        regularity = 1.0 / (1.0 + np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-5))
        return float(np.clip(regularity, 0.0, 1.0))
    except Exception:
        return 0.0

def estimate_instrumentalness(y):
    """Computes a lightweight heuristic proxy for instrumentalness."""
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harmonic**2)
        perc_energy = np.sum(y_percussive**2)
        total_energy = harm_energy + perc_energy
        
        if total_energy == 0:
            return 0.0
            
        flatness = librosa.feature.spectral_flatness(y=y)
        mean_flatness = np.mean(flatness)
        
        proxy = float(np.clip((harm_energy / total_energy) * (1.0 - mean_flatness), 0.0, 1.0))
        return round(proxy, 4)
    except Exception:
        return 0.0

def extract_audio_features(audio_path):
    """Extracts local audio features using pure Librosa."""
    features = {
        'danceability': None,
        'energy': None,
        'key': None,
        'mode': None,
        'tempo': None,
        'loudness': None,
        'instrumentalness': None
    }
    
    try:
        # Load audio slice (first 30 seconds for speed optimization)
        y, sr = librosa.load(audio_path, sr=22050, duration=30.0)
        
        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(np.atleast_1d(tempo)[0])
        
        # Energy (RMS normalized)
        rms = librosa.feature.rms(y=y)
        features['energy'] = float(np.clip(np.mean(rms) * 10, 0.0, 1.0))
        
        # Loudness (RMS converted to dB)
        features['loudness'] = float(20 * np.log10(np.maximum(np.mean(rms), 1e-5)))
        
        # Advanced Librosa heuristics
        features['danceability'] = estimate_danceability(y, sr)
        features['instrumentalness'] = estimate_instrumentalness(y)
        key, mode = estimate_key_and_mode(y, sr)
        features['key'] = key
        features['mode'] = mode
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        
    return features

def main():
    if not BACKLOG_PATH.exists():
        print(f"Backlog file not found at {BACKLOG_PATH}")
        return
        
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    backlog_df = pd.read_csv(BACKLOG_PATH)
    
    if MASTER_PATH.exists():
        master_df = pd.read_csv(MASTER_PATH)
    else:
        master_df = pd.DataFrame(columns=MASTER_COLUMNS)
        
    new_rows = []
    
    for _, row in backlog_df.iterrows():
        track_id = row.get('track_id')
        track_name = row.get('track_name')
        artist_name = row.get('artist_name')
        
        print(f"Processing: {track_name} by {artist_name}")
        temp_audio_path = TEMP_AUDIO_DIR / f"{track_id}.mp3"
        
        extracted = {}
        if not temp_audio_path.exists():
            success = fetch_audio_from_youtube(track_name, artist_name, str(temp_audio_path))
            if not success:
                continue
            time.sleep(random.uniform(2.0, 5.0))
                
        if temp_audio_path.exists():
            extracted = extract_audio_features(temp_audio_path)
            
        # Build master record mapping backlog data and extracted features
        track_record = {col: row.get(col, None) for col in MASTER_COLUMNS}
        track_record['id'] = track_id
        track_record['track_name'] = track_name
        track_record['name'] = track_name
        track_record['artist_name'] = artist_name
        track_record['artists'] = artist_name
        
        for key, val in extracted.items():
            if key in track_record:
                track_record[key] = val
                
        new_rows.append(track_record)
        
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        master_df = pd.concat([master_df, new_df], ignore_index=True)
        MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        master_df.to_csv(MASTER_PATH, index=False)
        print(f"Successfully updated {MASTER_PATH} with {len(new_rows)} tracks.")

if __name__ == '__main__':
    main()