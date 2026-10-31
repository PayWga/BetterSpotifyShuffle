"""Notebook summaries describe the loaded input snapshot, not later file edits."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from shufflerec.audit import signature
from shufflerec.features import FeatureConfig


def cell_with(marker):
    notebook = json.loads(Path('notebooks/ShuffleRec.ipynb').read_text(encoding='utf-8'))
    return next(''.join(cell['source']) for cell in notebook['cells']
                if cell['cell_type'] == 'code' and marker in ''.join(cell['source']))


def test_preparation_rejects_source_changes_between_history_and_catalog_loading(tmp_path):
    master, history = tmp_path / 'master.csv', tmp_path / 'history.csv'
    master.write_text('id\na\n', encoding='utf-8')
    history.write_text('track_id\na\n', encoding='utf-8')
    def read_history(path):
        loaded = pd.DataFrame({'track_id': ['a']})
        path.write_text('track_id\na\nb\n', encoding='utf-8')
        return loaded
    namespace = dict(CANDIDATE_SCOPE='history', RUN_AUDIT=False,
                     RUN_TRAINING=False, RUN_COMPARISON=True,
                     MASTER_DB=master, HISTORY_DB=history, signature=signature,
                     read_history=read_history,
                     read_catalog=lambda *args, **kwargs: pd.DataFrame({'id': ['a']}))
    with pytest.raises(RuntimeError, match='changed during preparation'):
        exec(compile(cell_with('history = read_history('), 'snapshot-preparation', 'exec'), namespace)


def test_training_summary_keeps_loaded_signatures_and_reports_later_changes(tmp_path):
    master, history = tmp_path / 'master.csv', tmp_path / 'history.csv'
    master.write_text('id\na\n', encoding='utf-8')
    history.write_text('track_id\na\n', encoding='utf-8')
    initial = dict(master=signature(master), history=signature(history))
    config = FeatureConfig()
    data = SimpleNamespace(examples={'validation': {'nip': ['synthetic']}},
                           store=SimpleNamespace(usable_ids={'a', 'b'}),
                           preprocessor=SimpleNamespace(config=config), summary=lambda: {})
    model = SimpleNamespace(parameter_report=lambda: {})
    def fit(*args, **kwargs):
        history.write_text('track_id\na\nb\n', encoding='utf-8')
        return {'best_validation': {}, 'test_evaluated': False}
    captured = []
    namespace = dict(RUN_TRAINING=True, data=data, model=model, NUM_NEGATIVES=1,
                     MAX_EVAL_EXAMPLES=1, SEED=42, CANDIDATE_SCOPE='history',
                     TRAINING_OPTIONS={}, TRAINING_VARIANT='custom', PROJECT_ROOT=tmp_path,
                     MASTER_DB=master, HISTORY_DB=history, input_signatures=initial,
                     signature=signature, json=json, asdict=asdict, fit=fit,
                     build_evaluation_plan=lambda *args, **kwargs: object(),
                     save_checkpoint=lambda *args, **kwargs: captured.append(kwargs['metadata']))
    exec(compile(cell_with('training_result = None'), 'snapshot-training', 'exec'), namespace)
    assert captured[0]['master_signature'] == initial['master']
    assert captured[0]['history_signature'] == initial['history']
    assert captured[0]['input_signatures'] == initial
    assert captured[0]['source_files_changed_during_run'] is True
