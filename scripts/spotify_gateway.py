"""Credential-backed Spotify transport sharing the existing persistent budget.

No authentication or network activity occurs on import. Private SOCKS settings
are loaded when the budgeted session is constructed.
"""
import json
from pathlib import Path

import spotipy
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

from .spotify_budget import SpotifyRequestBudget
from .spotify_metadata import metadata_session


PRIVATE_DIR = Path(__file__).resolve().parents[1] / 'sensitive_info'
CREDS_PATH = PRIVATE_DIR / 'spotify_credentials.json'
CACHE_PATH = PRIVATE_DIR / '.spotify_cache'
BUDGET_PATH = PRIVATE_DIR / 'spotify_request_budget.json'
SCOPE = 'user-read-playback-state user-read-currently-playing'


def create_client(*, credentials_path=CREDS_PATH, cache_path=CACHE_PATH,
                  budget_path=BUDGET_PATH, scope=SCOPE, use_proxy=None, proxy_config_path=None):
    budget = SpotifyRequestBudget(budget_path)
    budget.check_allowed()
    try:
        credentials = json.loads(Path(credentials_path).read_text(encoding='utf-8'))
        values = {key: credentials[key] for key in ('client_id', 'client_secret', 'redirect_uri')}
        if not all(isinstance(value, str) and value.strip() for value in values.values()):
            raise ValueError('Invalid credential fields')
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError('Could not load Spotify credentials from the configured private file') from None
    session = metadata_session(budget, use_proxy=use_proxy, proxy_config_path=proxy_config_path)
    try:
        auth = SpotifyOAuth(**values, scope=scope, cache_handler=CacheFileHandler(cache_path=str(cache_path)),
                            requests_session=session, requests_timeout=20)
        return spotipy.Spotify(auth_manager=auth, requests_session=session,
                               retries=0, status_retries=0, requests_timeout=20)
    except Exception:
        session.close()
        raise
