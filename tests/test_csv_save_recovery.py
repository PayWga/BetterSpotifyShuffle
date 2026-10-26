"""Reproduce blocked replacements using synthetic files only."""
import os
from unittest.mock import patch

import pandas as pd
import pytest

from scripts import feature_extraction as fe


def test_transient_permission_error_retries_without_serializing_again(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,tempo\nold,100\n', encoding='utf-8')
    real_replace = os.replace
    tries = []
    real_to_csv = pd.DataFrame.to_csv
    serializations = []

    def serialize(frame, *args, **kwargs):
        serializations.append(len(frame))
        return real_to_csv(frame, *args, **kwargs)

    def replace(source, destination):
        tries.append(source)
        if len(tries) < 3:
            raise PermissionError('synthetic temporary sharing violation')
        real_replace(source, destination)

    with patch.object(fe.os, 'replace', side_effect=replace), patch.object(fe, 'sleep', create=True), patch.object(
        pd.DataFrame, 'to_csv', serialize
    ):
        fe.atomic_csv(pd.DataFrame([{'id': 'new', 'tempo': 120.}]), path)
    assert pd.read_csv(path).iloc[0]['id'] == 'new'
    assert len(tries) == 3
    assert len(set(tries)) == 1
    assert serializations == [1]


def test_persistent_permission_error_preserves_original_and_prepared_snapshot(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,tempo\nold,100\n', encoding='utf-8')
    before = path.read_bytes()
    with patch.object(fe.os, 'replace', side_effect=PermissionError('locked')), patch.object(fe, 'sleep', create=True):
        with pytest.raises(PermissionError) as error:
            fe.atomic_csv(pd.DataFrame([{'id': 'new', 'tempo': 120.}]), path)
    assert path.read_bytes() == before
    snapshots = list(tmp_path.glob('.master_*.csv'))
    assert len(snapshots) == 1
    assert pd.read_csv(snapshots[0]).iloc[0]['id'] == 'new'
    assert 'retained' in str(error.value).casefold()


def test_retry_never_overwrites_an_external_update(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,tempo\nold,100\n', encoding='utf-8')

    def changed(source, destination):
        path.write_text('id,tempo\nexternal_update,150\n', encoding='utf-8')
        raise PermissionError('temporary block')

    with patch.object(fe.os, 'replace', side_effect=changed) as replace, patch.object(fe, 'sleep', create=True):
        with pytest.raises(RuntimeError, match='changed'):
            fe.atomic_csv(pd.DataFrame([{'id': 'new', 'tempo': 120.}]), path)
    assert replace.call_count == 1
    assert pd.read_csv(path).iloc[0]['id'] == 'external_update'
    assert len(list(tmp_path.glob('.master_*.csv'))) == 1


def test_serialization_failure_preserves_original(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,tempo\nold,100\n', encoding='utf-8')
    before = path.read_bytes()
    with patch.object(pd.DataFrame, 'to_csv', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            fe.atomic_csv(pd.DataFrame([{'id': 'new'}]), path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('.master_*.csv'))


@pytest.mark.skipif(os.name != 'nt', reason='Windows denies replacement while this file handle is open')
def test_real_windows_read_handle_can_release_during_retry(tmp_path):
    path = tmp_path / 'master.csv'
    path.write_text('id,tempo\nold,100\n', encoding='utf-8')
    held = path.open('rb')
    try:
        with patch.object(fe, 'sleep', side_effect=lambda delay: held.close(), create=True):
            fe.atomic_csv(pd.DataFrame([{'id': 'new', 'tempo': 120.}]), path)
    finally:
        held.close()
    assert pd.read_csv(path).iloc[0]['id'] == 'new'


def test_blocked_master_preserves_backlog_and_computed_results(tmp_path):
    master, backlog, audio = tmp_path / 'master.csv', tmp_path / 'backlog.csv', tmp_path / 'audio'
    master.write_text('id,tempo\nold,100\n', encoding='utf-8')
    backlog.write_text('track_id,track_name,artist_name\nnew,Example,Artist\n', encoding='utf-8')
    audio.mkdir()
    (audio / 'new.wav').write_bytes(b'fixture')
    before = master.read_bytes(), backlog.read_bytes()
    real_replace = os.replace

    def replace(source, destination):
        if destination == master:
            raise PermissionError('synthetic sharing violation')
        return real_replace(source, destination)

    with patch.object(fe.os, 'replace', side_effect=replace), patch.object(fe, 'sleep'), patch.object(
        fe, 'extract_audio_features', return_value={'tempo': 120., '_analysis_seconds': 60.}
    ), pytest.raises(fe.CsvSaveBlocked):
        fe.process(backlog, master, audio, workers=1)
    assert (master.read_bytes(), backlog.read_bytes()) == before
    snapshot = next(tmp_path.glob('.master_*.csv'))
    saved = pd.read_csv(snapshot).set_index('id')
    assert saved.loc['new', 'tempo'] == 120.
    assert saved.loc['new', 'tempo_analysis_seconds'] == 60.


def test_cli_reports_blocked_save_with_nonzero_exit_and_recovery_path(tmp_path, caplog):
    snapshot = tmp_path / '.master_prepared.csv'
    with patch('sys.argv', ['feature_extraction.py']), patch.object(
        fe, 'process', side_effect=fe.CsvSaveBlocked(tmp_path / 'master.csv', snapshot)
    ), pytest.raises(SystemExit) as result:
        fe.main()
    assert result.value.code == 1
    assert str(snapshot) in caplog.text
    assert 'retained' in caplog.text
