"""Performance contracts and concurrency safety, using synthetic catalogs only."""
import threading
import warnings
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from scripts import feature_extraction as fe


def make_catalog(tmp_path, count=6):
    master = tmp_path / 'master.csv'
    backlog = tmp_path / 'backlog.csv'
    audio = tmp_path / 'audio'
    audio.mkdir()
    pd.DataFrame([{'id': 'existing', 'name': 'NA', 'label': '0007', 'tempo': 130.}]).to_csv(master, index=False)
    pd.DataFrame([{'track_id': f'new{i}', 'track_name': f'Song {i}', 'artist_name': 'Artist'}
                  for i in range(count)]).to_csv(backlog, index=False)
    for i in range(count):
        (audio / f'new{i}.wav').write_bytes(b'fixture')
    return master, backlog, audio


def test_multi_checkpoint_run_reads_master_only_once(tmp_path):
    master, backlog, audio = make_catalog(tmp_path)
    with patch.object(fe, 'read_master', wraps=fe.read_master) as read, patch.object(
        fe, 'extract_audio_features', return_value={'tempo': 120., 'local_rms_db': -12.}
    ):
        stats = fe.process(backlog, master, audio, batch_size=2, workers=2)
    assert stats == {'eligible': 6, 'saved': 6, 'failed': 0}
    assert read.call_count == 1
    saved = pd.read_csv(master, dtype=str, keep_default_na=False).set_index('id')
    assert len(saved) == 7
    assert saved.loc['existing', 'label'] == '0007'
    assert float(saved.loc['new5', 'tempo']) == 120.
    assert pd.read_csv(backlog).empty


def test_default_checkpoint_batches_new_rows_together(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=12)
    real_concat = pd.concat
    with patch.object(fe.pd, 'concat', wraps=real_concat) as concat, patch.object(
        fe, 'save_batch', wraps=fe.save_batch
    ) as save, patch.object(fe, 'extract_audio_features', return_value={'tempo': 120.}):
        fe.process(backlog, master, audio, workers=1)
    assert len(pd.read_csv(master)) == 13
    assert concat.call_count == 1
    assert save.call_count == 1


def test_workers_overlap_and_only_main_thread_writes_csv(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=2)
    barrier = threading.Barrier(2)
    writer_threads = []
    original_write = fe.atomic_csv

    def analyze(path, **kwargs):
        barrier.wait(timeout=5)
        return {'tempo': 100. if path.stem == 'new0' else 140.}

    def write(df, path):
        writer_threads.append(threading.get_ident())
        original_write(df, path)

    with patch.object(fe, 'extract_audio_features', side_effect=analyze), patch.object(fe, 'atomic_csv', side_effect=write):
        stats = fe.process(backlog, master, audio, workers=2, batch_size=1)
    assert stats['saved'] == 2
    saved = pd.read_csv(master).set_index('id')
    assert saved.loc['new0', 'tempo'] == 100.
    assert saved.loc['new1', 'tempo'] == 140.
    assert set(writer_threads) == {threading.get_ident()}


def test_slow_first_track_does_not_block_checkpoint_of_finished_track(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=2)
    released = threading.Event()
    original_write = fe.atomic_csv

    def analyze(path, **kwargs):
        if path.stem == 'new0':
            assert released.wait(timeout=5), 'fast track could not checkpoint while slow track was active'
        return {'tempo': 120.}

    def write(df, path):
        original_write(df, path)
        if path == master:
            released.set()

    with patch.object(fe, 'extract_audio_features', side_effect=analyze), patch.object(fe, 'atomic_csv', side_effect=write):
        assert fe.process(backlog, master, audio, workers=2, batch_size=1)['saved'] == 2


def test_changed_catalog_is_not_overwritten_by_in_memory_snapshot(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=1)
    before = backlog.read_bytes()

    def analyze(path, **kwargs):
        master.write_text('id,name\nexternal,External update\n', encoding='utf-8')
        return {'tempo': 120.}

    with patch.object(fe, 'extract_audio_features', side_effect=analyze), pytest.raises(RuntimeError, match='changed'):
        fe.process(backlog, master, audio, workers=1)
    assert 'External update' in master.read_text()
    assert backlog.read_bytes() == before


def test_full_song_mode_analyzes_audio_beyond_first_minute(tmp_path):
    # The first minute is silent. Full mode must reach the later signal.
    path = tmp_path / 'long.wav'
    sr = fe.SAMPLE_RATE
    signal = np.concatenate([np.zeros(sr * 60), .2 * np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr)])
    sf.write(path, signal, sr)
    assert fe.extract_audio_features(path, seconds=60) is None
    with patch.object(fe, 'key_mode', return_value=(4., 1.)), patch.object(fe, 'rhythm', return_value=(120., .9)):
        result = fe.extract_audio_features(path, seconds=None)
    assert result is not None
    assert result['_analysis_seconds'] == pytest.approx(62)


def test_analysis_duration_is_saved_with_each_new_measurement(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=1)
    with patch.object(fe, 'extract_audio_features', return_value={
        'tempo': 120., '_analysis_seconds': 180., '_extractor_version': 'test_full'
    }) as analyze:
        fe.process(backlog, master, audio, workers=1, analysis_seconds=None)
    assert analyze.call_args.kwargs['seconds'] is None
    row = pd.read_csv(master).set_index('id').loc['new0']
    assert row['tempo_analysis_seconds'] == 180.
    assert row['tempo_source'] == 'test_full'


def test_existing_measurement_keeps_its_duration_and_provenance(tmp_path):
    master, _, _ = make_catalog(tmp_path, count=0)
    frame = pd.read_csv(master)
    frame['tempo_analysis_seconds'] = 60.
    frame['tempo_source'] = 'original'
    frame.to_csv(master, index=False)
    fe.save_batch([{'track_id': 'existing', 'tempo': 100., '_analysis_seconds': 180.,
                    '_extractor_version': 'different'}], master)
    row = pd.read_csv(master).iloc[0]
    assert row['tempo'] == 130.
    assert row['tempo_analysis_seconds'] == 60.
    assert row['tempo_source'] == 'original'


@pytest.mark.parametrize('kwargs', [{'workers': 0}, {'analysis_seconds': 0}, {'analysis_seconds': -1}])
def test_invalid_performance_options_fail_before_writing(tmp_path, kwargs):
    master, backlog, audio = make_catalog(tmp_path, count=1)
    before = master.read_bytes(), backlog.read_bytes()
    with pytest.raises(ValueError):
        fe.process(backlog, master, audio, **kwargs)
    assert (master.read_bytes(), backlog.read_bytes()) == before


@pytest.mark.parametrize('when', ['before_replace', 'after_replace'])
def test_interrupted_checkpoint_never_counts_the_same_attempt_twice(tmp_path, when):
    master, backlog, audio = make_catalog(tmp_path, count=1)
    original = fe.atomic_csv
    writes = []

    def interrupt(df, path):
        writes.append(path)
        if when == 'after_replace':
            original(df, path)
        raise KeyboardInterrupt()

    with patch.object(fe, 'extract_audio_features', return_value=None), patch.object(
        fe, 'atomic_csv', side_effect=interrupt
    ), pytest.raises(KeyboardInterrupt):
        fe.process(backlog, master, audio, workers=1, batch_size=1)
    assert writes == [backlog]
    row = pd.read_csv(backlog).iloc[0]
    assert row.get('attempts', 0) == (1 if when == 'after_replace' else 0)


def test_new_rows_without_optional_metadata_do_not_change_catalog_dtypes(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=1)
    pd.DataFrame([{'track_id': 'new0'}]).to_csv(backlog, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        with patch.object(fe, 'extract_audio_features', return_value={'tempo': 120.}):
            assert fe.process(backlog, master, audio, workers=1)['saved'] == 1
    row = pd.read_csv(master, dtype=str, keep_default_na=False).set_index('id')
    assert row.loc['existing', 'name'] == 'NA'
    assert row.loc['existing', 'label'] == '0007'


def test_rhythm_preserves_original_median_onsets_while_descriptor_uses_mean(tmp_path):
    sr = fe.SAMPLE_RATE
    t = np.arange(sr * 4) / sr
    signal = (.1 * np.sin(2 * np.pi * 440 * t) + np.random.default_rng(0).normal(0, .003, len(t))).astype(np.float32)
    signal += fe.librosa.clicks(times=np.arange(.25, 4, .5), sr=sr, length=len(t))
    path = tmp_path / 'varied.wav'
    sf.write(path, signal, sr, subtype='FLOAT')
    median = fe.librosa.onset.onset_strength(y=signal, sr=sr, aggregate=np.median)
    mean = fe.librosa.onset.onset_strength(y=signal, sr=sr, aggregate=np.mean)
    assert not np.allclose(median, mean)
    with patch.object(fe, 'rhythm', wraps=fe.rhythm) as rhythm, patch.object(fe, 'key_mode', return_value=(4., 1.)):
        result = fe.extract_audio_features(path)
    np.testing.assert_allclose(rhythm.call_args.kwargs['onset_envelope'], median, atol=1e-5)
    assert result['local_onset_strength'] == pytest.approx(float(mean.mean()), abs=1e-5)


def test_cli_parallel_full_song_mode_on_real_generated_audio(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=2)
    sr = fe.SAMPLE_RATE
    t = np.arange(sr * 3) / sr
    for i in range(2):
        sf.write(audio / f'new{i}.wav', .1 * np.sin(2 * np.pi * (220 + 110 * i) * t), sr)
    command = [sys.executable, str(Path(fe.__file__).resolve()), '--master', str(master),
               '--backlog', str(backlog), '--audio-dir', str(audio), '--local-only',
               '--workers', '2', '--analysis-seconds', 'full']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'eligible': 2, 'saved': 2, 'failed': 0}
    saved = pd.read_csv(master).set_index('id')
    assert saved.loc['new0', 'local_rms_db_analysis_seconds'] == pytest.approx(3)
    assert saved.loc['new1', 'local_rms_db_analysis_seconds'] == pytest.approx(3)
    assert 'Timing:' in result.stderr
    assert pd.read_csv(backlog).empty


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'invalid'])
def test_cli_analysis_duration_rejects_invalid_values(value):
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        fe.parse_analysis_seconds(value)


def test_independent_acquisition_pool_can_fill_one_analysis_worker(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=3)
    barrier = threading.Barrier(3)
    analysis_threads = set()
    original_find = fe.find_local_audio

    def find(tid, directory):
        barrier.wait(timeout=5)
        return original_find(tid, directory)

    def analyze(path, **kwargs):
        analysis_threads.add(threading.get_ident())
        return {'tempo': 120.}

    with patch.object(fe, 'find_local_audio', side_effect=find), patch.object(fe, 'extract_audio_features', side_effect=analyze):
        stats = fe.process(backlog, master, audio, acquisition_workers=3, analysis_workers=1, queue_size=3)
    assert stats['saved'] == 3
    assert len(analysis_threads) == 1


def test_ready_queue_and_reserved_acquisitions_are_bounded(tmp_path):
    master, backlog, audio = make_catalog(tmp_path, count=8)
    analyzing = threading.Event()
    lock = threading.Lock()
    acquisitions = []
    original_find = fe.find_local_audio

    def find(tid, directory):
        with lock:
            acquisitions.append(tid)
        return original_find(tid, directory)

    def analyze(path, **kwargs):
        if not analyzing.is_set():
            # At most one active analysis plus two reserved/ready paths.
            with lock:
                assert len(acquisitions) <= 3
            analyzing.set()
        return {'tempo': 120.}

    with patch.object(fe, 'find_local_audio', side_effect=find), patch.object(fe, 'extract_audio_features', side_effect=analyze):
        stats = fe.process(backlog, master, audio, acquisition_workers=8, analysis_workers=1, queue_size=2)
    assert stats['saved'] == 8


@pytest.mark.parametrize('kwargs', [{'acquisition_workers': 0}, {'analysis_workers': 0}, {'queue_size': 0}])
def test_invalid_stage_sizes_fail_before_csv_changes(tmp_path, kwargs):
    master, backlog, audio = make_catalog(tmp_path, count=1)
    before = master.read_bytes(), backlog.read_bytes()
    with pytest.raises(ValueError):
        fe.process(backlog, master, audio, **kwargs)
    assert (master.read_bytes(), backlog.read_bytes()) == before
