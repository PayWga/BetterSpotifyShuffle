"""Feature-group comparisons with common validation targets and candidates."""
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from time import perf_counter, sleep

from .data import prepare_data
from .features import FeatureConfig, LOCAL
from .model import SpotifySASRec
from .training import build_evaluation_plan, fit, seed_everything


def default_variants():
    audio = FeatureConfig(artist=False, year=False, duration=False, genre=False,
                          artist_genre=False, local=False, provenance=False,
                          key_encoding='raw', time_signature_encoding='numeric')
    metadata = replace(audio, artist=True, year=True, duration=True, genre=True, artist_genre=True)
    compact = replace(metadata, key_encoding='circular', time_signature_encoding='categorical')
    return {'audio_baseline': audio, 'metadata': metadata, 'compact_encoding': compact,
            'extended_acoustics': replace(compact, local=True, provenance=True)}


def individual_local_variants(base=None):
    """Same metadata/context, adding exactly one local measurement per fit."""
    base = default_variants()['compact_encoding'] if base is None else base
    baseline = replace(base, local=True, **{name: False for name in LOCAL})
    return {'no_local': baseline, **{'add_' + name: replace(baseline, **{name: True}) for name in LOCAL}}


def estimated_duration_variants(base=None):
    """Exploratory comparison; does not recompute duration estimates as of train."""
    base = default_variants()['extended_acoustics'] if base is None else base
    return {'without_estimated_durations': replace(base, duration=True, include_estimated_durations=False),
            'with_estimated_durations': replace(base, duration=True, include_estimated_durations=True)}


def _key(example):
    return example.context, example.positive, example.timestamp.isoformat()


def run_comparison(catalog, history, *, variants=None, model_config=None, max_length=10,
                   epochs=5, preference_epochs=3, batch_size=32, num_negatives=99,
                   max_examples=500, seed=42, device='cpu', logger=None,
                   learning_rate=1e-3, preference_learning_rate=1e-4, patience=3,
                   cooldown_seconds=10):
    if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
        raise ValueError('cooldown_seconds must be finite and nonnegative')
    variants = default_variants() if variants is None else variants
    if not variants:
        raise ValueError('Provide at least one feature configuration')
    common_examples, common_candidates = None, None
    first_examples = None
    # Sequential passes avoid retaining several catalog-sized tensor stores.
    for config in variants.values():
        prepared = prepare_data(catalog, history, config, max_length)
        examples = prepared.examples['validation']['nip']
        keys = {_key(e) for e in examples}
        common_examples = keys if common_examples is None else common_examples & keys
        candidates = prepared.store.usable_ids
        common_candidates = candidates.copy() if common_candidates is None else common_candidates & candidates
        if first_examples is None:
            first_examples = list(examples)
        del prepared
        gc.collect()
    shared = [e for e in first_examples if _key(e) in common_examples]
    if not shared:
        raise ValueError('No validation examples are shared across the feature configurations')
    plan = build_evaluation_plan(shared, common_candidates, num_negatives=num_negatives,
                                 max_examples=max_examples, seed=seed, scope='common supplied-catalog snapshot')
    fingerprint = hashlib.sha256(json.dumps([asdict(c) for c in plan.cases], sort_keys=True).encode()).hexdigest()
    results = []
    training_options = dict(epochs=epochs, preference_epochs=preference_epochs, batch_size=batch_size,
                            learning_rate=learning_rate, preference_learning_rate=preference_learning_rate,
                            patience=patience)
    for index, (name, config) in enumerate(variants.items()):
        started = perf_counter()
        prepared = prepare_data(catalog, history, config, max_length)
        seed_everything(seed)
        model = SpotifySASRec(prepared.preprocessor, max_length, model_config).to(device)
        if logger:
            logger({'variant': name, 'parameters': model.parameter_report(), 'partitions': prepared.summary()})
        variant_logger = (lambda record, variant=name: logger({'variant': variant, **record})) if logger else None
        result = fit(model, prepared, plan, seed=seed, logger=variant_logger, **training_options)
        results.append({'variant': name, 'features': asdict(config), 'parameters': model.parameter_report(),
                        'validation': result['best_validation'], 'partitions': prepared.summary(),
                        'evaluation_fingerprint': fingerprint, 'test_evaluated': False,
                        'training_options': training_options.copy(), 'history': result['history'],
                        'elapsed_seconds': perf_counter() - started,
                        'active_local_features': list(config.enabled_local_features),
                        'uses_history_derived_duration_estimates': bool(config.duration and config.include_estimated_durations),
                        'duration_estimate_note': ('Estimates may incorporate held-out listening events; '
                            'this comparison is not a strictly chronological estimate experiment.'
                            if config.duration and config.include_estimated_durations else None),
                        'configured_cooldown_seconds': cooldown_seconds})
        del model, prepared
        gc.collect()
        if index < len(variants) - 1 and cooldown_seconds:
            if logger:
                logger({'event': 'model_cooldown', 'completed_variant': name, 'seconds': cooldown_seconds})
            sleep(cooldown_seconds)
    return results
