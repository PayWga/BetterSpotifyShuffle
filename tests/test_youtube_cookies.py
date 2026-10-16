"""Cookie authentication uses synthetic secrets, never a real browser/profile."""
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import feature_extraction as fe


def cookie_file(tmp_path):
    path = tmp_path / 'cookies.txt'
    path.write_text('# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tsynthetic-secret\n', encoding='utf-8')
    return path


def test_cookie_file_authenticates_search_and_download_without_rewriting_source(tmp_path):
    path = cookie_file(tmp_path)
    before = path.read_bytes()
    jar = fe.load_youtube_cookies(path, None)
    seen = []

    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            seen.append(next(iter(self.cookiejar)).value)
            # Response cookies belong to this request context, not other workers.
            next(iter(self.cookiejar)).value = 'changed-by-response'
            if not download:
                return {'entries': [{'title': 'Artist - Song', 'url': 'https://example.org/video'}]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'fixture')
            return {'title': 'Artist - Song', 'webpage_url': url}

    with patch('yt_dlp.YoutubeDL', Downloader):
        audio, _ = fe.acquire_audio({'track_id': 'track', 'track_name': 'Song',
                                     'artist_name': 'Artist'}, tmp_path, cookiejar=jar)
    assert audio.read_bytes() == b'fixture'
    assert seen == ['synthetic-secret', 'synthetic-secret']
    assert next(iter(jar)).value == 'synthetic-secret'
    assert path.read_bytes() == before


def test_browser_cookies_load_once_and_reach_worker(tmp_path):
    from yt_dlp.cookies import load_cookies
    jar = load_cookies(str(cookie_file(tmp_path)), None, None)
    master, backlog = tmp_path / 'master.csv', tmp_path / 'backlog.csv'
    pd.DataFrame([{'id': 'existing'}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': str(i), 'track_name': 'Song', 'artist_name': 'Artist'}
                  for i in range(3)]).to_csv(backlog, index=False)
    received = []

    def acquire(item, directory, *, cookiejar=None):
        received.append(next(iter(cookiejar)).value)
        return tmp_path / 'audio.wav', 'synthetic'

    with patch('yt_dlp.cookies.load_cookies', return_value=jar) as load, patch.object(
        fe, 'acquire_audio', side_effect=acquire), patch.object(
        fe, 'extract_audio_features', return_value={'tempo': 120.}):
        stats = fe.process(backlog, master, tmp_path / 'audio', workers=2,
                           cookies_from_browser='firefox')
    assert load.call_count == 1
    assert load.call_args.args[1] == ('firefox',)
    assert received == ['synthetic-secret'] * 3
    assert stats['saved'] == 3


@pytest.mark.parametrize('mode', ['dry_run', 'local_only'])
def test_offline_modes_do_not_read_cookie_file_or_browser(tmp_path, mode):
    master, backlog = tmp_path / 'master.csv', tmp_path / 'backlog.csv'
    pd.DataFrame([{'id': 'existing'}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': 'new'}]).to_csv(backlog, index=False)
    with patch('yt_dlp.cookies.load_cookies', side_effect=AssertionError('private access')):
        stats = fe.process(backlog, master, tmp_path / 'audio',
                           cookies=tmp_path / 'nonexistent.txt', **{mode: True})
    assert stats['saved'] == 0


def test_missing_cookie_file_is_configuration_error(tmp_path):
    with pytest.raises(fe.CookieConfigurationError, match='cookie file'):
        fe.load_youtube_cookies(tmp_path / 'missing.txt', None)


def test_cli_accepts_browser_and_passes_it_to_processing():
    with patch('sys.argv', ['feature_extraction.py', '--cookies-from-browser', 'edge']), patch.object(
        fe, 'process', return_value={'saved': 0}) as process:
        fe.main()
    assert process.call_args.kwargs['cookies_from_browser'] == 'edge'


def test_cli_rejects_two_cookie_sources(capsys):
    with patch('sys.argv', ['feature_extraction.py', '--cookies', 'file.txt',
                            '--cookies-from-browser', 'chrome']), pytest.raises(SystemExit) as exc:
        fe.main()
    assert exc.value.code == 2
    assert 'not allowed with argument' in capsys.readouterr().err


def test_real_yt_dlp_context_keeps_cookie_source_unchanged(tmp_path):
    path = cookie_file(tmp_path)
    before = path.read_bytes()
    jar = fe.load_youtube_cookies(path, None)
    with fe.youtube_session({'quiet': True}, jar) as session:
        next(iter(session.cookiejar)).value = 'response-cookie'
    assert next(iter(jar)).value == 'synthetic-secret'
    assert path.read_bytes() == before


def test_browser_load_failure_stops_before_any_backlog_updates(tmp_path):
    master, backlog = tmp_path / 'master.csv', tmp_path / 'backlog.csv'
    master.write_text('id\nexisting\n', encoding='utf-8')
    backlog.write_text('track_id\nnew\n', encoding='utf-8')
    before = master.read_bytes(), backlog.read_bytes()
    with patch('yt_dlp.cookies.load_cookies', side_effect=OSError('synthetic browser lock')):
        with pytest.raises(fe.CookieConfigurationError, match='Could not load cookies'):
            fe.process(backlog, master, tmp_path / 'audio', cookies_from_browser='chrome')
    assert (master.read_bytes(), backlog.read_bytes()) == before


def test_malformed_cookie_values_are_not_printed(tmp_path, capsys):
    path = tmp_path / 'malformed-cookies.txt'
    path.write_text('# Netscape HTTP Cookie File\n.youtube.com\tPRIVATE_COOKIE_VALUE\n', encoding='utf-8')
    with pytest.raises(fe.CookieConfigurationError):
        fe.load_youtube_cookies(path, None)
    captured = capsys.readouterr()
    assert 'PRIVATE_COOKIE_VALUE' not in captured.out + captured.err
