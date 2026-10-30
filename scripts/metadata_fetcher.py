import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import CacheFileHandler
from pathlib import Path
import json
import os
from datetime import datetime, timezone

class MetadataTracker:
    
    def __init__(self, save_file=None, creds_file=None, connect_to_spotify=False):
        project_root = Path(__file__).resolve().parent.parent
        private_dir = project_root / "sensitive_info"

        self.save_file = (
            Path(save_file)
            if save_file is not None
            else private_dir / "daemon_memory.json"
        )

        self.creds_path = (
            Path(creds_file)
            if creds_file is not None
            else private_dir / "spotify_credentials.json"
        )

        self.cache_path = private_dir / ".spotify_cache"

        self.memory = self._load_memory()
        self.sp = None

        if connect_to_spotify:
            self.authenticate()


    def authenticate(self, *, use_proxy=None):
        """Read private credentials and reuse budgeted API/OAuth routing."""
        try:
            from .spotify_metadata import metadata_session
            from .spotify_budget import SpotifyRequestBudget
        except ImportError:
            from spotify_metadata import metadata_session
            from spotify_budget import SpotifyRequestBudget
        budget = SpotifyRequestBudget(self.creds_path.parent / 'spotify_request_budget.json')
        budget.check_allowed()
        try:
            with self.creds_path.open(encoding='utf-8') as stream:
                creds = json.load(stream)
            values = {key: creds[key] for key in ('client_id', 'client_secret', 'redirect_uri')}
        except (OSError, ValueError, TypeError, KeyError):
            raise ValueError('Could not load Spotify credentials from the configured private file') from None
        session = metadata_session(budget, use_proxy=use_proxy)
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                **values, scope='user-library-read user-read-playback-state',
                cache_handler=CacheFileHandler(cache_path=str(self.cache_path)), requests_session=session),
                requests_session=session, retries=0, status_retries=0, requests_timeout=20)
        except Exception:
            session.close()
            raise


    def _load_memory(self):
        """Load previously stored per-track preference data."""

        if not self.save_file.exists():
            return {}

        with open(self.save_file, "r", encoding="utf-8") as f:
            memory = json.load(f)

        if not isinstance(memory, dict):
            raise ValueError(
                "Invalid daemon memory format: expected a JSON object."
            )

        return memory



    def save(self):
        """Safely persist per-track preference data."""

        self.save_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temp_path = self.save_file.with_suffix(".tmp.json")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=4)

            os.replace(temp_path, self.save_file)

        finally:
            if temp_path.exists():
                temp_path.unlink()


    def ensure_track(self, track_id):
        """Initialize a track's preference data if necessary."""

        if track_id is None or not str(track_id).strip():
            raise ValueError("A valid Spotify track ID is required.")

        track_id = str(track_id).strip()

        if track_id not in self.memory:
            self.memory[track_id] = {
                "play_count": 0,
                "skip_count": 0,
                "added_at": None
            }

        return track_id
    

    def get_days_since_added(self, track_id):
        """Return library age in days, or None if unknown."""

        metadata = self.get_metadata(track_id)
        added_at = metadata.get("added_at")

        if not added_at:
            return None

        added_date = datetime.fromisoformat(
            added_at.replace("Z", "+00:00")
        )

        now = datetime.now(timezone.utc)

        return max(0, (now - added_date).days)


    def register_play(self, track_id):
        """Record one completed or accepted playback event."""

        track_id = self.ensure_track(track_id)

        current_count = self.memory[track_id].get(
            "play_count", 0
        )

        self.memory[track_id]["play_count"] = (
            int(current_count) + 1
        )

        self.save()


    def register_skip(self, track_id):
        """Record one confirmed skip event."""

        track_id = self.ensure_track(track_id)

        current_count = self.memory[track_id].get(
            "skip_count", 0
        )

        self.memory[track_id]["skip_count"] = (
            int(current_count) + 1
        )

        self.save()


    def sync_library_dates(self):
        """Synchronize Liked Songs addition timestamps."""

        if self.sp is None:
            self.authenticate()

        print("Syncing Liked Songs library dates...")

        results = self.sp.current_user_saved_tracks(limit=50)
        synced_count = 0
        saved_ids = set()

        while True:
            for item in results.get("items", []):
                track = item.get("track") or {}
                track_id = track.get("id")
                added_at = item.get("added_at")

                if not track_id or not added_at:
                    continue

                track_id = self.ensure_track(track_id)

                self.memory[track_id]["added_at"] = added_at
                saved_ids.add(track_id)

                synced_count += 1

            if not results.get("next"):
                break

            results = self.sp.next(results)

        # Records absent from Liked Songs shouldn't retain
        # an outdated current-library addition timestamp.
        for track_id, metadata in self.memory.items():
            if track_id not in saved_ids:
                metadata["added_at"] = None

        self.save()

        print(
            f"Successfully synchronized "
            f"{synced_count} Liked Songs."
        )
 

    def get_metadata(self, track_id):
        """Return a track's preference metadata."""

        track_id = self.ensure_track(track_id)
        record = self.memory[track_id]

        return {
            "play_count": int(record.get("play_count", 0)),
            "skip_count": int(record.get("skip_count", 0)),
            "days_since_added": self._calculate_library_age(
                record.get("added_at")
            )
        }


    @staticmethod
    def _calculate_library_age(added_at):
        """Calculate library age from an ISO timestamp."""

        if not added_at:
            return None

        added_date = datetime.fromisoformat(
            added_at.replace("Z", "+00:00")
        )

        now = datetime.now(timezone.utc)

        return max(0, (now - added_date).days)


#tracker = MetadataTracker()

#tracker.sync_library_dates()