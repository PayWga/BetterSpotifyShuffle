"""Proxy configuration and routing tests use synthetic credentials only."""
import importlib
import json
from types import SimpleNamespace

import pytest
import requests


def api():
    assert importlib.util.find_spec('scripts.spotify_proxy'), 'Spotify proxy routing missing'
    return importlib.import_module('scripts.spotify_proxy')


def config(tmp_path, **overrides):
    values = dict(enabled=True, type='SOCKS5', address='127.0.0.1', port=1080,
                  username='user@name', password='p:a/ss?#%')
    values.update(overrides)
    path = tmp_path / 'proxy.json'
    path.write_text(json.dumps(values), encoding='utf-8')
    return path


def test_enabled_proxy_uses_remote_dns_and_percent_encoded_credentials(tmp_path):
    m = api()
    path = config(tmp_path)
    before = path.read_bytes()
    session = m.configure_session(requests.Session(), config_path=path)
    assert session.proxies == {
        'http': 'socks5h://user%40name:p%3Aa%2Fss%3F%23%25@127.0.0.1:1080',
        'https': 'socks5h://user%40name:p%3Aa%2Fss%3F%23%25@127.0.0.1:1080'}
    assert not session.trust_env
    assert path.read_bytes() == before
    session.close()


def test_off_switch_ignores_file_and_environment_proxies(tmp_path, monkeypatch):
    m = api()
    path = tmp_path / 'broken.json'
    path.write_text('not json', encoding='utf-8')
    monkeypatch.setenv('HTTPS_PROXY', 'http://environment.invalid:123')
    session = m.configure_session(requests.Session(), config_path=path, use_proxy=False)
    settings = session.merge_environment_settings('https://api.spotify.com/v1/me/player', {}, False, True, None)
    assert not settings['proxies']
    assert not session.trust_env
    session.close()


def test_missing_optional_or_disabled_config_is_direct_but_forced_proxy_fails(tmp_path):
    m = api()
    missing = tmp_path / 'missing.json'
    with requests.Session() as session:
        assert m.configure_session(session, config_path=missing).proxies == {}
    with pytest.raises(ValueError, match='proxy configuration'):
        m.configure_session(requests.Session(), config_path=missing, use_proxy=True)
    path = config(tmp_path, enabled=False)
    with requests.Session() as session:
        assert m.configure_session(session, config_path=path).proxies == {}
        assert m.configure_session(session, config_path=path, use_proxy=True).proxies


@pytest.mark.parametrize('overrides', [
    {'enabled': 'false'}, {'port': True}, {'port': 0}, {'port': 65536},
    {'address': 'http://wrong/host'}, {'type': 'HTTP'}, {'username': ''},
    {'password': None}, {'address': 'bad\r\nhost'},
])
def test_invalid_config_fails_without_echoing_secret_values(tmp_path, overrides):
    m = api()
    path = config(tmp_path, **overrides)
    with pytest.raises(ValueError) as caught:
        m.configure_session(requests.Session(), config_path=path)
    assert 'p:a/ss?#%' not in str(caught.value)
    assert 'user@name' not in str(caught.value)


def test_shared_metadata_session_routes_actual_http_attempts(tmp_path, monkeypatch):
    api()
    from scripts.spotify_metadata import metadata_session
    seen = []
    def send(adapter, request, **kwargs):
        seen.append((request.url, kwargs['proxies']))
        response = requests.Response()
        response.status_code = 200
        response._content = b'{}'
        return response
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', send)
    session = metadata_session(proxy_config_path=config(tmp_path))
    try:
        session.get('https://api.spotify.com/v1/me/player')
        session.post('https://accounts.spotify.com/api/token', data={'grant_type': 'synthetic'})
        assert len(seen) == 2
        assert all(proxies['https'].startswith('socks5h://') for _, proxies in seen)
    finally:
        session.close()


def test_proxy_failure_never_leaks_url_or_falls_back_to_direct(tmp_path, monkeypatch):
    m = api()
    from scripts.spotify_metadata import metadata_session
    seen = []
    def send(adapter, request, **kwargs):
        seen.append(kwargs['proxies'])
        raise requests.exceptions.ProxyError('synthetic error containing ' + kwargs['proxies']['https'])
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', send)
    session = metadata_session(proxy_config_path=config(tmp_path))
    try:
        with pytest.raises(requests.exceptions.ConnectionError) as caught:
            session.get('https://api.spotify.com/v1/me/player')
        assert len(seen) == 1 and seen[0]
        assert 'user%40name' not in str(caught.value)
        assert 'p%3Aa' not in str(caught.value)
        assert caught.value.__context__ is None
    finally:
        session.close()


def test_gateway_off_override_applies_to_oauth_and_spotify_api(tmp_path, monkeypatch):
    api()
    from scripts import spotify_gateway as gateway
    path = tmp_path / 'credentials.json'
    path.write_text(json.dumps(dict(client_id='synthetic', client_secret='synthetic', redirect_uri='http://localhost')))
    captured = []
    monkeypatch.setattr(gateway, 'SpotifyOAuth', lambda **kwargs: captured.append(kwargs) or object())
    monkeypatch.setattr(gateway.spotipy, 'Spotify', lambda **kwargs: captured.append(kwargs) or object())
    gateway.create_client(credentials_path=path, cache_path=tmp_path / 'cache', budget_path=tmp_path / 'budget.json',
                          use_proxy=False, proxy_config_path=config(tmp_path))
    assert captured[0]['requests_session'] is captured[1]['requests_session']
    assert not captured[0]['requests_session'].proxies
    captured[0]['requests_session'].close()


def test_metadata_and_library_factories_share_routed_oauth_session(tmp_path, monkeypatch):
    api()
    from scripts import spotify_proxy, data_merging, metadata_fetcher
    monkeypatch.setattr(spotify_proxy, 'PROXY_CONFIG_PATH', config(tmp_path))
    credentials = tmp_path / 'spotify_credentials.json'
    credentials.write_text(json.dumps(dict(client_id='synthetic', client_secret='synthetic', redirect_uri='http://localhost')))
    monkeypatch.setattr(data_merging, 'PRIVATE_DIR', tmp_path)
    for module, build in [
        (data_merging, data_merging.get_spotify_client),
        (metadata_fetcher, lambda: metadata_fetcher.MetadataTracker(save_file=tmp_path / 'memory.json',
            creds_file=credentials, connect_to_spotify=True).sp),
    ]:
        captured = []
        monkeypatch.setattr(module, 'SpotifyOAuth', lambda **kwargs: captured.append(kwargs) or object())
        monkeypatch.setattr(module.spotipy, 'Spotify', lambda **kwargs: captured.append(kwargs) or object())
        build()
        oauth, api_call = captured
        assert oauth['requests_session'] is api_call['requests_session']
        assert oauth['requests_session'].proxies['https'].startswith('socks5h://')
        oauth['requests_session'].close()


@pytest.mark.parametrize('arguments,expected', [([], None), (['--proxy'], True), (['--no-proxy'], False)])
def test_polling_cli_override_reaches_daemon_without_authentication(arguments, expected, monkeypatch):
    api()
    from scripts import polling
    captured = []
    monkeypatch.setattr(polling, 'start_daemon', lambda config, **kwargs: captured.append(kwargs['use_proxy']))
    polling.main(arguments)
    assert captured == [expected]


def test_missing_socks_dependency_fails_closed_but_direct_override_still_works(tmp_path, monkeypatch):
    m = api()
    original = m.importlib.util.find_spec
    monkeypatch.setattr(m.importlib.util, 'find_spec', lambda name: None if name == 'socks' else original(name))
    path = config(tmp_path)
    with requests.Session() as session:
        with pytest.raises(ValueError, match='PySocks'):
            m.configure_session(session, config_path=path)
        assert m.configure_session(session, config_path=path, use_proxy=False).proxies == {}
