"""Run the actual offline CLI with a synthetic checkpoint and input files."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from shufflerec.features import FeaturePreprocessor
from shufflerec.model import ModelConfig, SpotifySASRec
from shufflerec.training import save_checkpoint


def files(tmp_path):
    frame = pd.DataFrame([dict(id=str(i), artists='Artist ' + str(i % 2),
                               genre='rock', tempo=80 + i * 5) for i in range(12)])
    prep = FeaturePreprocessor().fit(frame, frame.id)
    model = SpotifySASRec(prep, max_length=3,
                         config=ModelConfig(d_model=8, nhead=2, feedforward=16, dropout=0))
    save_checkpoint(tmp_path / 'model.pt', model, prep)
    frame.to_csv(tmp_path / 'catalog.csv', index=False)
    pd.DataFrame({'track_id': frame.id}).to_csv(tmp_path / 'playlist.csv', index=False)
    pd.DataFrame({'track_id': ['0']}).to_csv(tmp_path / 'context.csv', index=False)
    return ['--checkpoint', str(tmp_path / 'model.pt'), '--catalog', str(tmp_path / 'catalog.csv'),
            '--playlist', str(tmp_path / 'playlist.csv'), '--context', str(tmp_path / 'context.csv')]


def test_offline_json_lines_session_loads_once_accepts_feedback_and_preserves_inputs(tmp_path):
    assert importlib.util.find_spec('shufflerec.sequence'), 'local sequencing CLI missing'
    args = files(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    requests = [dict(command='start'),
                dict(command='feedback', selection_id=999, completion_rate=.8),
                dict(command='feedback', selection_id=1, completion_rate=10 ** 400),
                dict(command='feedback', selection_id=1, completion_rate=.25),
                dict(command='status')]
    result = subprocess.run([sys.executable, '-m', 'shufflerec.sequence', *args],
                            input='not json\n' + '\n'.join(json.dumps(r) for r in requests) + '\n',
                            text=True, capture_output=True, timeout=45,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(responses) == 7
    assert responses[0]['event'] == 'ready'
    assert len(responses[0]['queue']) == 5
    assert responses[1]['event'] == 'error'
    assert responses[2]['active']['selection_id'] == 1
    assert responses[3]['event'] == 'error'
    assert responses[4]['event'] == 'error'
    assert responses[5]['outcome'] == 'accepted'
    assert responses[5]['active'] is None
    assert responses[6]['queue'] == responses[5]['queue']
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_invalid_configuration_fails_before_session_without_creating_outputs(tmp_path):
    assert importlib.util.find_spec('shufflerec.sequence'), 'local sequencing CLI missing'
    args = files(tmp_path)
    config = tmp_path / 'config.json'
    config.write_text('{"max_probability": 0.99}', encoding='utf-8')
    result = subprocess.run([sys.executable, '-m', 'shufflerec.sequence', *args, '--config', str(config)],
                            input='', text=True, capture_output=True, timeout=45,
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode != 0
    assert not result.stdout
    assert 'probability' in result.stderr
