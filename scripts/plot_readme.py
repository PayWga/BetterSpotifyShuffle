"""Build README figures from read-only inputs; export aggregate statistics only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shufflerec.audit import CATALOG_COLUMNS, signature
from shufflerec.features import FeaturePreprocessor, first_present


FEATURES = {
    'danceability': 'Danceability', 'energy': 'Energy', 'loudness': 'Loudness',
    'tempo': 'Tempo', 'key': 'Key', 'mode': 'Mode', 'duration_ms': 'Duration¹',
    'year': 'Release year', 'artists': 'Artist metadata', 'track_genre': 'Track genre',
    'local_rms_db': 'Local RMS level', 'local_onset_strength': 'Local onset strength',
    'local_spectral_centroid': 'Local spectral centroid',
    'local_spectral_contrast': 'Local spectral contrast',
    'local_beat_regularity': 'Local beat regularity',
}
HISTOGRAMS = {
    'tempo': ('Tempo (BPM)', np.linspace(0, 300, 41)),
    'local_rms_db': ('Local RMS level (dB)', np.linspace(-80, 0, 41)),
    'local_spectral_centroid': ('Local spectral centroid (Hz)', np.linspace(0, 10000, 41)),
    'local_beat_regularity': ('Local beat regularity', np.linspace(0, 1, 31)),
}


def collect_snapshot(master, history, *, comparison=None, chunksize=100000):
    """Count valid supplied values, not fitted/scaled model inputs or private rows."""
    inputs = [Path(master), Path(history)] + ([Path(comparison)] if comparison else [])
    before = {p: signature(p) for p in inputs}
    events = pd.read_csv(history, usecols=['track_id', 'event_type'], dtype='string')
    history_ids = set(events.track_id.dropna().str.strip()) - {''}
    event_counts = events.event_type.fillna('Unclassified').replace('', 'Unclassified').value_counts()
    counts = {'catalog': Counter(), 'history_matched': Counter()}
    histograms = {name: {'edges': bins.tolist(), 'counts': np.zeros(len(bins)-1, dtype=int),
                         'valid_count': 0, 'outside_range': 0} for name, (_, bins) in HISTOGRAMS.items()}
    rows = matched_rows = 0
    preprocessor = FeaturePreprocessor()
    for chunk in pd.read_csv(master, usecols=lambda c: c in CATALOG_COLUMNS,
                             dtype={'id': 'string'}, chunksize=chunksize, low_memory=False):
        if 'id' not in chunk:
            raise ValueError("Master table requires 'id'")
        matched = chunk.id.str.strip().isin(history_ids)
        rows += len(chunk)
        matched_rows += int(matched.sum())
        numeric = preprocessor._numeric(chunk)
        categories = preprocessor._categories(chunk)
        key = pd.to_numeric(first_present(chunk, ['key']), errors='coerce')
        valid = {name: numeric[name].notna() for name in FEATURES if name in numeric}
        valid['key'] = key.isin(range(12))
        valid['artists'] = pd.Series([bool(labels) for labels in categories['artist']], index=chunk.index)
        valid['track_genre'] = pd.Series([bool(labels) for labels in categories['genre']], index=chunk.index)
        for name in FEATURES:
            available = valid.get(name, pd.Series(False, index=chunk.index))
            counts['catalog'][name] += int(available.sum())
            counts['history_matched'][name] += int((available & matched).sum())
        for name, (_, bins) in HISTOGRAMS.items():
            values = numeric.loc[matched, name].dropna().to_numpy()
            binned, _ = np.histogram(values, bins=bins)
            histograms[name]['counts'] += binned
            histograms[name]['valid_count'] += len(values)
            histograms[name]['outside_range'] += int(len(values) - binned.sum())
    for item in histograms.values():
        item['counts'] = item['counts'].tolist()
    comparisons = []
    if comparison:
        report = json.loads(Path(comparison).read_text(encoding='utf-8'))
        for result in report['results']:
            metrics = result['validation']
            comparisons.append({
                'variant': result['variant'],
                'parameters': result['parameters']['total'],
                'validation': {k: metrics[k] for k in (
                    'hit_at_k', 'ndcg_at_k', 'k', 'examples', 'mean_candidates')},
                'test_evaluated': result.get('test_evaluated', report.get('test_evaluated')),
            })
    if any(signature(p) != old for p, old in before.items()):
        raise RuntimeError('An input changed during plotting; retry with stable source files')
    return {'snapshot_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'catalog_rows': rows, 'history_rows': len(events),
            'history_unique_tracks': len(history_ids), 'history_matched_rows': matched_rows,
            'events': {str(k): int(v) for k, v in event_counts.items()},
            'coverage': {group: dict(values) for group, values in counts.items()},
            'histograms': histograms, 'comparison': comparisons}


def render_figures(snapshot, output):
    """Render fixed, labeled figures without exporting track-level observations."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.axisbelow': True})
    blue, teal = '#35618f', '#188578'

    def save(fig, name):
        fig.savefig(output / name, dpi=170, facecolor='white')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8), layout='constrained')
    y = np.arange(len(FEATURES))
    for offset, group, denominator, color, label in (
        (-.19, 'catalog', snapshot['catalog_rows'], blue, f"Full catalog (n={snapshot['catalog_rows']:,})"),
        (.19, 'history_matched', snapshot['history_matched_rows'], teal,
         f"History-matched catalog (n={snapshot['history_matched_rows']:,})")):
        values = [100 * snapshot['coverage'][group].get(name, 0) / max(denominator, 1) for name in FEATURES]
        bars = ax.barh(y + offset, values, .36, color=color, label=label)
        ax.bar_label(bars, labels=[f'{v:.1f}%' for v in values], padding=3, fontsize=8)
    ax.set(yticks=y, yticklabels=list(FEATURES.values()), xlim=(0, 116),
           xlabel='Rows with a valid supplied value (%)', title='Feature availability')
    ax.invert_yaxis()
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower right', fontsize=9)
    fig.supxlabel('¹ Durations explicitly marked as estimates are excluded. Availability precedes training-only calibration.', fontsize=9)
    save(fig, 'feature_availability.png')

    fig, ax = plt.subplots(figsize=(9, 4.3), layout='constrained')
    events = snapshot['events']
    labels = list(events)
    values = list(events.values())
    bars = ax.barh(labels, values, color=[blue, teal, '#8293a5', '#b6bdc5'][:len(labels)])
    total = max(sum(values), 1)
    ax.bar_label(bars, labels=[f'{n:,}  ({100*n/total:.1f}%)' for n in values], padding=5, fontsize=9)
    ax.set(xlabel='Listening events', title=f"Listening-history event labels (n={snapshot['history_rows']:,})")
    ax.set_xlim(0, max(values, default=1) * 1.38)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.15)
    save(fig, 'listening_events.png')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    for ax, (name, (label, _)) in zip(axes.flat, HISTOGRAMS.items()):
        item = snapshot['histograms'][name]
        ax.stairs(item['counts'], item['edges'], fill=True, color=teal, alpha=.85)
        ax.set(xlabel=label, ylabel='Tracks',
               title=f"{item['valid_count']:,} valid measurements; {item['outside_range']:,} outside displayed range")
        ax.title.set_fontsize(10)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Acoustic measurements for history-matched tracks', fontsize=14)
    fig.supxlabel('One value per catalog row; each panel uses its own available subset.', fontsize=9)
    save(fig, 'acoustic_distributions.png')

    if snapshot['comparison']:
        results = snapshot['comparison']
        labels = [r['variant'].replace('_', '\n') for r in results]
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
        for ax, metric, title in zip(axes, ('hit_at_k', 'ndcg_at_k'), ('Hit', 'NDCG')):
            values = [r['validation'][metric] for r in results]
            k_values = {r['validation']['k'] for r in results}
            title += '@' + '/'.join(map(str, sorted(k_values)))
            bars = ax.bar(labels, values, color=[blue, '#6285a6', teal, '#7d9c83'][:len(results)])
            ax.bar_label(bars, fmt='%.3f', padding=5)
            ax.set(ylim=(0, 1), title=title, ylabel='Sampled validation score')
            ax.grid(axis='y', alpha=.15)
        protocols = {(r['validation']['examples'], r['validation']['mean_candidates']) for r in results}
        protocol = '; '.join(f'{n:,} examples, {c:g} mean candidates' for n, c in sorted(protocols))
        fig.suptitle('Saved feature-group comparison', fontsize=14)
        fig.supxlabel(f'{protocol}. Validation results; no uncertainty estimates.', fontsize=9)
        save(fig, 'model_comparison.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', type=Path, default=Path('data/master_track_table.csv'))
    parser.add_argument('--history', type=Path, default=Path('sensitive_info/spotify_listening_history.csv'))
    parser.add_argument('--comparison', type=Path, help='Explicit saved comparison_results.json; no training is run')
    parser.add_argument('--output', type=Path, default=Path('docs/images'))
    args = parser.parse_args()
    snapshot = collect_snapshot(args.master, args.history, comparison=args.comparison)
    render_figures(snapshot, args.output)
    (args.output / 'readme_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote aggregate snapshot and figures to {args.output}')


if __name__ == '__main__':
    main()
