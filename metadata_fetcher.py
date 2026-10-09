import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
import json
import os
from datetime import datetime, timezone

class MetadataTracker:
    def __init__(self, save_file="daemon_memory.json", creds_file="sensitive_info/spotify_credentials.json", connect_to_spotify=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_file = os.path.join(base_dir, save_file)
        self.creds_path = os.path.join(base_dir, creds_file)
        self.cache_path = os.path.join(base_dir, "sensitive_info", ".spotify_cache")
        
        self.memory = self._load_memory()
        self.sp = None

        if connect_to_spotify:
            self.authenticate()

    def authenticate(self):
            """Wakes up the Spotify connection only when needed."""
            try:
                with open(self.creds_path, 'r') as f:
                    creds = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"\nCRITICAL: Cannot find file at:\n{self.creds_path}\n")
                
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=creds["client_id"],
                client_secret=creds["client_secret"],
                redirect_uri=creds["redirect_uri"],
                scope="user-library-read user-read-playback-state",
                cache_handler=CacheFileHandler(cache_path=self.cache_path)
            ))

    def _load_memory(self):
        """Loads the skip/play history from the hard drive."""
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as f:
                return json.load(f)
        return {}

    def _load_memory(self):
            """Loads the skip/play history from the hard drive."""
            if os.path.exists(self.save_file):
                with open(self.save_file, 'r') as f:
                    return json.load(f)
            return {}

    def save(self):
        """Saves the current memory to the hard drive."""
        with open(self.save_file, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def ensure_track(self, track_id):
        """Initializes a track in memory if it hasn't been seen before."""
        if track_id not in self.memory:
            self.memory[track_id] = {
                'play_count': 0,
                'skip_count': 0,
                'days_since_added': 365 # Default to 1 year if not in library
            }

    def register_play(self, track_id):
        """Call this when a song successfully finishes."""
        self.ensure_track(track_id)
        self.memory[track_id]['play_count'] += 1
        self.save()

    def register_skip(self, track_id):
        """Call this when you skip a song. Banishes it to the void."""
        self.ensure_track(track_id)
        self.memory[track_id]['skip_count'] += 1
        self.save()

    def sync_library_dates(self):
        """
        Pulls your entire Liked Songs library from Spotify, 
        extracts the 'added_at' timestamp, and calculates the age in days.
        """
        print("Syncing library dates from Spotify...")
        results = self.sp.current_user_saved_tracks(limit=50)
        tracks = results['items']
        
        # Handle pagination to get all songs
        while results['next']:
            results = self.sp.next(results)
            tracks.extend(results['items'])
            
        now = datetime.now(timezone.utc)
        
        for item in tracks:
            track_id = item['track']['id']
            # Spotify formats timestamps as '2023-10-15T14:30:00Z'
            added_str = item['added_at']
            
            # Convert string to timezone-aware datetime object
            added_date = datetime.fromisoformat(added_str.replace('Z', '+00:00'))
            
            # Calculate exact age in days
            days_old = (now - added_date).days
            
            self.ensure_track(track_id)
            self.memory[track_id]['days_since_added'] = days_old
            
        self.save()
        print(f"Successfully synced dates for {len(tracks)} tracks.")
        
    def get_metadata(self, track_id):
        """Returns the metadata for the PyTorch heuristic function."""
        self.ensure_track(track_id)
        return self.memory[track_id]


#tracker = MetadataTracker()

#tracker.sync_library_dates()