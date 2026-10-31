"""Interrupted cache publication never exposes a mixed audio/provenance pair."""
import json
from pathlib import Path

import pytest

from scripts import feature_extraction as fe


ITEM = {'track_id': 'track', 'track_name': 'Song', 'artist_name': 'Artist'}
SOURCE = 'https://example.org/new'


@pytest.fixture
def downloader(monkeypatch):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            entry = {'title': 'Artist - Song', 'url': SOURCE}
            if not download:
                return {'entries': [entry]}
            Path(self.options['outtmpl'].replace('%(ext)s', 'wav')).write_bytes(b'new recording')
            return entry

    monkeypatch.setattr('yt_dlp.YoutubeDL', Downloader)
    monkeypatch.setattr(fe.shutil, 'which', lambda name: 'synthetic-runtime')


def legacy_pair(directory):
    audio = directory / 'track.wav'
    audio.write_bytes(b'old recording')
    audio.with_suffix('.source.json').write_text(json.dumps({
        'title': 'Artist - Song (Remix)', 'url': 'https://example.org/old'}), encoding='utf-8')
    return audio


@pytest.mark.parametrize('failure', [PermissionError, KeyboardInterrupt])
def test_partial_publication_is_rejected_after_sidecar_failure(tmp_path, monkeypatch, downloader, failure):
    audio = legacy_pair(tmp_path)
    sidecar = audio.with_suffix('.source.json')
    replace = fe.os.replace

    def blocked(source, destination):
        if Path(destination) == sidecar:
            raise failure('synthetic sidecar lock or interruption')
        return replace(source, destination)

    monkeypatch.setattr(fe.os, 'replace', blocked)
    with pytest.raises(RuntimeError if failure is PermissionError else KeyboardInterrupt):
        fe.acquire_audio(ITEM, tmp_path)

    assert audio.read_bytes() == b'new recording'
    assert json.loads(sidecar.read_text(encoding='utf-8'))['url'].endswith('/old')
    assert fe.find_local_audio('track', tmp_path) is None
    prepared, error, _ = fe.prepare_track(ITEM, tmp_path, local_only=True, dry_run=False)
    assert prepared is None and error == 'audio_not_found'
    with pytest.raises(RuntimeError, match='publication'):
        fe.audio_source(audio)
    assert any(path.read_bytes() == b'old recording' for path in tmp_path.glob('.track_previous_*/track.wav'))
    assert any(path.read_bytes() == b'new recording' for path in tmp_path.glob('.track_*/track.wav'))


def test_successful_retry_repairs_interrupted_publication(tmp_path, monkeypatch, downloader):
    audio = legacy_pair(tmp_path)
    replace = fe.os.replace

    def blocked(source, destination):
        if Path(destination) == audio.with_suffix('.source.json'):
            raise PermissionError('synthetic sidecar lock')
        return replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(fe.os, 'replace', blocked)
        with pytest.raises(RuntimeError):
            fe.acquire_audio(ITEM, tmp_path)
    published, source = fe.acquire_audio(ITEM, tmp_path)
    assert published == fe.find_local_audio('track', tmp_path) == audio
    assert published.read_bytes() == b'new recording'
    assert source == fe.audio_source(published) == SOURCE


def test_untracked_local_audio_keeps_legacy_cache_behavior(tmp_path):
    audio = tmp_path / 'track.flac'
    audio.write_bytes(b'local recording')
    assert fe.find_local_audio('track', tmp_path) == audio
    assert fe.audio_source(audio) == str(audio.resolve())


def test_committed_cache_rejects_audio_replaced_outside_publication(tmp_path, downloader):
    audio, _ = fe.acquire_audio(ITEM, tmp_path)
    audio.write_bytes(b'unrelated recording with another size')
    assert fe.find_local_audio('track', tmp_path) is None
    with pytest.raises(RuntimeError, match='publication'):
        fe.audio_source(audio)


def test_failed_pending_marker_keeps_previous_cache_reusable(tmp_path, monkeypatch, downloader):
    audio = legacy_pair(tmp_path)
    original = audio.read_bytes(), audio.with_suffix('.source.json').read_bytes()
    replace = fe.os.replace

    def blocked(source, destination):
        if Path(destination) == audio.with_suffix('.publication.json'):
            raise PermissionError('synthetic publication marker lock')
        return replace(source, destination)

    monkeypatch.setattr(fe.os, 'replace', blocked)
    with pytest.raises(RuntimeError):
        fe.acquire_audio(ITEM, tmp_path)
    assert (audio.read_bytes(), audio.with_suffix('.source.json').read_bytes()) == original
    assert fe.find_local_audio('track', tmp_path) == audio


def test_failed_commit_marker_leaves_pair_unavailable_until_republished(tmp_path, monkeypatch, downloader):
    replace = fe.os.replace

    def blocked(source, destination):
        if Path(source).name == 'publication.committed.json':
            raise PermissionError('synthetic commit marker lock')
        return replace(source, destination)

    with monkeypatch.context() as context:
        context.setattr(fe.os, 'replace', blocked)
        with pytest.raises(RuntimeError):
            fe.acquire_audio(ITEM, tmp_path)
    assert fe.find_local_audio('track', tmp_path) is None
    assert json.loads((tmp_path / 'track.publication.json').read_text(encoding='utf-8'))['status'] == 'pending'
    audio, _ = fe.acquire_audio(ITEM, tmp_path)
    assert fe.find_local_audio('track', tmp_path) == audio


@pytest.mark.parametrize('text', ['{', '[]', '{"version":99,"status":"committed"}'])
def test_unreadable_or_unsupported_marker_cannot_fall_back_to_legacy_cache(tmp_path, text):
    audio = legacy_pair(tmp_path)
    marker = audio.with_suffix('.publication.json')
    marker.write_text(text, encoding='utf-8')
    before = {path: path.read_bytes() for path in (audio, audio.with_suffix('.source.json'), marker)}
    prepared, error, _ = fe.prepare_track(ITEM, tmp_path, local_only=True, dry_run=True)
    assert prepared is None and error == 'audio_not_found'
    assert {path: path.read_bytes() for path in before} == before


def test_committed_cache_rejects_sidecar_replaced_outside_publication(tmp_path, downloader):
    audio, _ = fe.acquire_audio(ITEM, tmp_path)
    audio.with_suffix('.source.json').write_text('{"url":"unrelated"}', encoding='utf-8')
    assert fe.find_local_audio('track', tmp_path) is None
