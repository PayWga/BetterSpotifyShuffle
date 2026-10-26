"""Real reported title shapes, with negative artist/language/version controls."""
import pytest

from scripts import feature_extraction as fe


@pytest.mark.parametrize('requested,artist,upload,uploader,expected', [
    ('PLAYING WITH FIRE - Japanese Version', 'BLACKPINK', 'BLACKPINK - Playing With Fire Japanese ver. (Lyric Video)', 'revelmixes', True),
    ('PLAYING WITH FIRE - Japanese Version', 'BLACKPINK', 'BLACKPINK - PLAYING WITH FIRE (Japanese) [Audio]', 'KiKi', True),
    ('PLAYING WITH FIRE - Japanese Version', 'BLACKPINK', 'PLAYING WITH FIRE', 'BLACKPINK - Topic', False),
    ('PLAYING WITH FIRE - Japanese Version', 'BLACKPINK', 'BLACKPINK - PLAYING WITH FIRE (Korean Ver.)', 'BLACKPINK', False),
    ('Double Fantasy (with Future)', 'The Weeknd', 'Double Fantasy - The Weeknd, Future | Lyrical Video | Thesoulofmusic', 'Thesoulofmusic', True),
    ('BOOMBAYAH (Live)', 'BLACKPINK', '붐바야 (Boombayah) (Live)', 'BLACKPINK', True),
    ('BOOMBAYAH (Live)', 'BLACKPINK', "BLACKPINK - '붐바야 (BOOMBAYAH)' M/V", 'BLACKPINK', False),
    ('BOOMBAYAH', 'BLACKPINK', "BLACKPINK - '붐바야 (BOOMBAYAH)' M/V", 'BLACKPINK', True),
    ('Dreamer - Stripped Down', 'Mokita', 'Mokita - Dreamer (Stripped back version)', 'Mokita', True),
    ('Dreamer - Stripped Down', 'Mokita', 'Mokita - Dreamer (feat. Kaptan)', 'UNCVRD', False),
    ('Dreamer - Stripped Down', 'Mokita', 'Dreamer (Stripped Down)', 'Kaptan - Topic', False),
    ('Red Sun In The Sky', 'Gippeul', 'Red Sun in the Sky', 'Mao Ze Dong - Topic', False),
    ('Song', 'Artist', 'Artist - Another Song (Lyrics)', 'Artist', False),
    ('Song', 'Artist', 'Artist - Song | Another Song', 'Artist', False),
    ('Song (with Guest) - DJ Max Remix', 'Artist', 'Artist - Song (feat. Guest) - DJ Other Remix', 'Artist', False),
    ('Song (with Guest) - DJ Max Remix', 'Artist', 'Artist - Song (feat. Guest) - DJ Max Remix', 'Artist', True),
])
def test_structured_recording_identity(requested, artist, upload, uploader, expected):
    item = {'track_name': requested, 'artist_name': artist, 'duration_ms': 180000}
    score, _ = fe.match_result(item, {'title': upload, 'uploader': uploader, 'duration': 180})
    assert (score is not None) is expected


def test_spotify_full_artist_metadata_is_used_without_changing_original_identity():
    item = {'track_name': 'Song (with Guest)', 'artist_name': 'Alias Main',
            'spotify_artist_names': '["Main", "Guest"]', 'duration_ms': 180000}
    score, _ = fe.match_result(item, {'title': 'Main feat. Guest - Song', 'duration': 180})
    assert score is not None
    assert item['artist_name'] == 'Alias Main'


def test_initial_search_uses_structured_base_and_credits(tmp_path):
    from unittest.mock import patch
    queries = []

    class Downloader:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            queries.append(url)
            return {'entries': []}

    with patch('yt_dlp.YoutubeDL', Downloader), pytest.raises(RuntimeError):
        fe.acquire_audio({'track_id': 'track', 'track_name': 'Double Fantasy (with Future)',
                          'artist_name': 'The Weeknd'}, tmp_path)
    assert any('double fantasy' in query.casefold() and 'with future' not in query.casefold() for query in queries)


@pytest.mark.parametrize('title,label', [('Song (Live)', 'live'), ('Song - Japanese Version', 'japanese')])
def test_fallback_searches_keep_recording_version_constraints(tmp_path, title, label):
    from unittest.mock import patch
    queries = []

    class Downloader:
        def __init__(self, options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            queries.append(url.casefold())
            return {'entries': []}

    with patch('yt_dlp.YoutubeDL', Downloader), pytest.raises(RuntimeError):
        fe.acquire_audio({'track_id': 'track', 'track_name': title, 'artist_name': 'Artist'}, tmp_path)
    assert all(label in query for query in queries)


@pytest.mark.parametrize('missing_duration', ['', 'NaN', 'nan'])
def test_enriched_metadata_survives_saving_existing_master_track(tmp_path, missing_duration):
    import pandas as pd
    path = tmp_path / 'master.csv'
    path.write_text(f'id,name,tempo,duration_ms\nexisting,Keep me,130,{missing_duration}\n', encoding='utf-8')
    fe.save_batch([{'track_id': 'existing', 'duration_ms': 180000, 'isrc': 'TEST123',
                    'spotify_artist_names': '["Main", "Guest"]', 'spotify_metadata_source': 'spotify_track_api'}], path)
    row = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[0]
    assert row['duration_ms'] == '180000'
    assert row['isrc'] == 'TEST123'
    assert row['spotify_artist_names'] == '["Main", "Guest"]'
    assert row['name'] == 'Keep me'
    assert row['tempo'] == '130'
