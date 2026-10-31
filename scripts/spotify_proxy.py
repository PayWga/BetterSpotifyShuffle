"""Private, per-session SOCKS routing for Spotify API and OAuth HTTP traffic."""
import importlib.util
import ipaddress
import json
from pathlib import Path
import re
from urllib.parse import quote

import requests


PROXY_CONFIG_PATH = Path(__file__).resolve().parents[1] / 'sensitive_info' / 'spotify_proxy.json'


class SpotifyProxyError(requests.exceptions.ConnectionError):
    """Sanitized connection failure without proxy URL, credentials or fallback."""


def _proxy_url(config_path, use_proxy):
    if use_proxy is not None and not isinstance(use_proxy, bool):
        raise ValueError('use_proxy must be boolean or None')
    if use_proxy is False:
        return None
    path = Path(config_path) if config_path is not None else PROXY_CONFIG_PATH
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        if use_proxy is not True:
            return None
        raise ValueError('Spotify proxy configuration is required when proxy routing is enabled') from None
    except (OSError, ValueError, TypeError):
        raise ValueError('Could not read Spotify proxy configuration') from None
    if not isinstance(config, dict) or not isinstance(config.get('enabled'), bool):
        raise ValueError('Spotify proxy configuration requires a boolean enabled field')
    if use_proxy is None and not config['enabled']:
        return None
    host, port = config.get('address'), config.get('port')
    username, password = config.get('username'), config.get('password')
    kind = config.get('type')
    if not isinstance(kind, str) or kind.casefold() not in ('socks5', 'socks5h'):
        raise ValueError('Spotify proxy configuration requires type SOCKS5')
    if not isinstance(host, str) or not host or any(c.isspace() for c in host):
        raise ValueError('Spotify proxy configuration has an invalid address')
    try:
        address = ipaddress.ip_address(host)
        host = f'[{address}]' if address.version == 6 else str(address)
    except ValueError:
        if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?', host):
            raise ValueError('Spotify proxy configuration has an invalid address') from None
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError('Spotify proxy configuration requires a port from 1 to 65535')
    if not all(isinstance(v, str) and v for v in (username, password)):
        raise ValueError('Spotify proxy configuration requires a username and password')
    if importlib.util.find_spec('socks') is None:
        raise ValueError('Spotify SOCKS routing requires PySocks; install requirements-spotify.txt')
    return f'socks5h://{quote(username, safe="")}:{quote(password, safe="")}@{host}:{port}'


def configure_session(session, *, config_path=None, use_proxy=None):
    """JSON enabled flag by default, explicit override when supplied.

    Ignore process/system proxy settings in both modes, so off really means
    direct. Enabled routing resolves destination DNS through the SOCKS proxy.
    Configuration is snapshotted at construction; restart to apply edits.
    """
    url = _proxy_url(config_path, use_proxy)
    session.trust_env = False
    session.proxies.clear()
    if url is not None:
        session.proxies.update({'http': url, 'https': url})
    session.spotify_proxy_enabled = url is not None
    return session
