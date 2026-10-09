import os
import time
import csv
import json
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import spotipy.exceptions

SCOPE = 'user-read-playback-state user-read-currently-playing user-modify-playback-state playlist-read-private'
CREDS_FILE = "sensitive_info/spotify_credentials.json"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "sensitive_info/spotify_listening_history.csv")
CREDS_PATH = os.path.join(SCRIPT_DIR, CREDS_FILE)

try:
    with open(CREDS_PATH, 'r') as f:
        creds = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Missing credentials file at {CREDS_PATH}. Check your sensitive_info folder.")
            
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=creds["client_id"],
    client_secret=creds["client_secret"],
    redirect_uri=creds["redirect_uri"],
    scope=SCOPE 
    ))



def init_csv():
    """Create the CSV and header if it doesn't exist. Now includes context fields."""
    try:
        with open(CSV_FILE, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'track_id', 'track_name', 'artist', 
                             'duration_ms', 'listened_ms', 'completion_rate',
                             'context_uri', 'context_type'])
    except FileExistsError:
        pass

def log_track(track_data, listened_ms, context_uri, context_type):
    """Calculate completion rate and append to CSV along with session context."""
    duration_ms = track_data['duration_ms']
    
    # Cap listened_ms at duration_ms (just in case of API lag at the end of a track)
    listened_ms = min(listened_ms, duration_ms)
    completion_rate = round(listened_ms / duration_ms, 4)
    
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            track_data['id'],
            track_data['name'],
            track_data['artists'][0]['name'],
            duration_ms,
            listened_ms,
            completion_rate,
            context_uri,
            context_type
        ])
    
    print(f"Logged: {track_data['name']} | Completion: {completion_rate * 100:.1f}% | Context: {context_type}")

def start_daemon():
    init_csv()
    print("Listening daemon started. Polling Spotify API...")
    
    current_track_id = None
    current_track_data = None
    current_context_uri = None
    current_context_type = None
    current_queue = []  # Tracks the upcoming queue to catch blind-spot skips
    max_progress_ms = 0
    
    ACTIVE_POLL = 5      
    PAUSED_POLL = 60

    while True:
        try:
            playback = sp.current_playback()
            
            # Fetch the queue and cap it at 10 to save memory/processing
            queue_data = sp.queue()
            upcoming_queue = queue_data.get('queue', [])[:10] if queue_data else []
            
            # Dynamic Polling: Backoff to 60s if nothing is playing
            if playback is None or not playback.get('is_playing') or playback.get('item') is None:
                print(f"Nothing active. Sleeping for {PAUSED_POLL}s...")
                time.sleep(PAUSED_POLL)
                continue
                
            item = playback['item']
            track_id = item['id']
            progress_ms = playback['progress_ms']
            duration_ms = item['duration_ms']
            is_playing = playback.get('is_playing', False)
            context = playback.get('context')
            
            if context:
                context_uri = context.get('uri', 'unknown_context')
                context_type = context.get('type', 'unknown_type')
            else:
                context_uri = 'no_context'
                context_type = 'no_context'
            
            # Scenario A: tracking a new song
            if track_id != current_track_id:
                if current_track_id is not None:
                    # 1. Log the song that just finished/skipped
                    log_track(current_track_data, max_progress_ms, current_context_uri, current_context_type)
                    
                    # 2. Queue Differencing (Ghost Skip Catching)
                    previous_queue_ids = [t['id'] for t in current_queue if t and 'id' in t]
                    
                    if track_id in previous_queue_ids:
                        idx = previous_queue_ids.index(track_id)
                        skipped_ghosts = current_queue[:idx]
                        
                        for ghost_track in skipped_ghosts:
                            if not ghost_track or 'id' not in ghost_track:
                                continue
                            # Log skipped tracks with a nominal 500ms so completion_rate is essentially 0
                            log_track(ghost_track, 500, current_context_uri, current_context_type)
                            print(f"👻 Ghost Skip Caught: {ghost_track.get('name', 'Unknown')}")
                
                # Reset state for the new song
                current_track_id = track_id
                current_track_data = item
                max_progress_ms = progress_ms
                current_context_uri = context_uri
                current_context_type = context_type
                current_queue = upcoming_queue
                
            # Scenario B: same song is still playing
            else:
                # Handle scrubbing/seeking backward by tracking the maximum progress seen
                if progress_ms > max_progress_ms:
                    max_progress_ms = progress_ms
                current_queue = upcoming_queue
                    
            if is_playing:
                # 5-second heartbeat to catch rapid manual skips
                time.sleep(ACTIVE_POLL)
            else:
                # If paused, calculate remaining time
                remaining_sec = (duration_ms - progress_ms) / 1000.0
                
                # Sleep for 60s, OR the remaining time minus a 2s buffer, whichever is smaller.
                sleep_time = min(PAUSED_POLL, remaining_sec - 2)
                
                # Floor it at 5 seconds so we don't spam the API if paused at the very end
                sleep_time = max(ACTIVE_POLL, sleep_time)
                
                print(f"Track paused with {remaining_sec:.1f}s left. Sleeping for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                
        except spotipy.exceptions.SpotifyException as e:
            if e.http_status == 429:
                print("Hit API Limit! Checking headers...")
                if "QUOTA_EXCEEDED" in str(e):
                    print("Developer Quota Exceeded. Sleeping for 1 hour...")
                    time.sleep(3600)
                # Standard Rate Limit Backoff
                else:
                    retry_after = int(e.headers.get('Retry-After', 10)) if hasattr(e, 'headers') and e.headers else 10
                    print(f"Rate limited. Pausing for {retry_after} seconds...")
                    time.sleep(retry_after)
            else:
                print(f"Spotify API Error: {e}")
                time.sleep(5)
                
        except Exception as e:
            print(f"General Error: {e}. Retrying in 5s...")
            time.sleep(5)

if __name__ == "__main__":
    start_daemon()