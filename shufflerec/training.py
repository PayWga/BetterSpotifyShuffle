"""Train-only optimization, fixed-candidate validation, and versioned artifacts."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import SequenceDataset, sample_negatives
from .features import FeaturePreprocessor
from .model import ModelConfig, SpotifySASRec


EVALUATION_CACHE_TRACKS = 8192
EVALUATION_BATCH_SIZE = 1024


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Seeds keep runs repeatable on the same backend.
    # Different GPU kernels can still produce different results.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def to_device(features, device):
    return {name: value.to(device) for name, value in features.items()}


@dataclass(frozen=True)
class EvaluationCase:
    context: tuple[str, ...]
    positive: str
    candidates: tuple[str, ...]
    timestamp: str | None = None


@dataclass(frozen=True)
class EvaluationPlan:
    cases: tuple[EvaluationCase, ...]
    seed: int
    num_negatives: int
    scope: str


def build_evaluation_plan(examples, candidate_ids, *, num_negatives=99, max_examples=500,
                          seed=42, scope='catalog snapshot'):
    if not examples or num_negatives < 1 or max_examples < 1:
        raise ValueError('Evaluation needs examples, positive limits, and at least one negative')
    pool = tuple(sorted(set(candidate_ids)))
    membership = frozenset(pool)
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(examples)), min(max_examples, len(examples))))
    cases = []
    for i in indexes:
        sample = examples[i]
        if sample.negative is not None:
            raise ValueError('Evaluation needs next-item examples')
        if sample.positive not in membership:
            raise ValueError('Evaluation target is outside the candidate pool')
        negatives = sample_negatives(pool, set(sample.context) | {sample.positive}, num_negatives,
                                     rng, membership=membership)
        if not negatives:
            raise ValueError('Evaluation has no eligible negative tracks')
        candidates = negatives + [sample.positive]
        rng.shuffle(candidates)  # Shuffle ties so the target has no fixed position advantage.
        cases.append(EvaluationCase(sample.context, sample.positive, tuple(candidates), sample.timestamp.isoformat()))
    return EvaluationPlan(tuple(cases), seed, num_negatives, scope)


def evaluate(model, store, plan, *, k=10):
    if not plan.cases or k < 1:
        raise ValueError('Evaluation needs nonempty cases and positive k')
    device = next(model.parameters()).device
    was_training = model.training
    hits, ndcg, counts = [], [], []
    model.eval()
    try:
        with torch.no_grad():
            for case in plan.cases:
                if len(set(case.candidates)) != len(case.candidates) or case.positive not in case.candidates:
                    raise ValueError('Candidates must be unique and contain the target')
            candidate_ids = sorted({track for case in plan.cases for track in case.candidates})
            vectors = None
            # Cache a bounded candidate snapshot for this evaluation pass only.
            # Custom rankers keep their scoring contract, and later epochs start fresh.
            if (type(model) is SpotifySASRec
                    and getattr(model.score_candidates, '__func__', None) is SpotifySASRec.score_candidates
                    and len(candidate_ids) <= EVALUATION_CACHE_TRACKS):
                chunks = [model.track_encoder(to_device(store.take(candidate_ids[start:start + EVALUATION_BATCH_SIZE]), device))
                          for start in range(0, len(candidate_ids), EVALUATION_BATCH_SIZE)]
                vectors = torch.cat(chunks)
                candidate_index = {track: i for i, track in enumerate(candidate_ids)}
            for case in plan.cases:
                context, padding = store.context(case.context, model.max_length)
                context = to_device({name: value.unsqueeze(0) for name, value in context.items()}, device)
                state = model(context, padding.unsqueeze(0).to(device))
                if vectors is None:
                    scores = model.score_candidates(state, to_device(store.take(case.candidates), device))[0]
                else:
                    indexes = torch.tensor([candidate_index[track] for track in case.candidates], device=device)
                    scores = (state @ vectors.index_select(0, indexes).T)[0]
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Non-finite evaluation scores')
                target_index = case.candidates.index(case.positive)
                target_score = scores[target_index]
                # Break score ties using the fixed shuffled order.
                # This keeps evaluation repeatable without favoring the target.
                earlier = torch.arange(len(scores), device=device) < target_index
                rank = 1 + int(((scores > target_score) | ((scores == target_score) & earlier)).sum())
                hit = rank <= k
                hits.append(float(hit))
                ndcg.append(1 / math.log2(rank + 1) if hit else 0.)
                counts.append(len(scores))
    finally:
        model.train(was_training)
    return {'hit_at_k': float(np.mean(hits)), 'ndcg_at_k': float(np.mean(ndcg)), 'k': k,
            'examples': len(hits), 'mean_candidates': float(np.mean(counts)),
            'min_candidates': min(counts), 'max_candidates': max(counts),
            'candidate_scope': plan.scope, 'seed': plan.seed}


def fit(model, prepared, validation_plan, *, epochs=5, preference_epochs=3, batch_size=32,
        learning_rate=1e-3, preference_learning_rate=1e-4, patience=3, seed=42, logger=None):
    if epochs < 1 or preference_epochs < 0 or batch_size < 1 or patience < 1:
        raise ValueError('Invalid epoch, batch size, or patience setting')
    if learning_rate <= 0 or preference_learning_rate <= 0:
        raise ValueError('Learning rates must be positive')
    if model.max_length != prepared.max_length:
        raise ValueError('Model and prepared context lengths differ')
    allowed = {(e.context, e.positive, e.timestamp.isoformat()) for e in prepared.examples['validation']['nip']}
    if not validation_plan.cases or any((c.context, c.positive, c.timestamp) not in allowed for c in validation_plan.cases):
        raise ValueError('Selection must use the validation partition')
    seed_everything(seed)
    device = next(model.parameters()).device
    best_score, best_weights, best_validation = -float('inf'), None, None
    history = []
    sizes = {}
    for phase, key, steps, lr in [('next_item', 'nip', epochs, learning_rate),
                                 ('preference', 'preference', preference_epochs, preference_learning_rate)]:
        samples = prepared.examples['train'][key]
        sizes[phase] = len(samples)
        if not steps or not samples:
            continue
        dataset = SequenceDataset(samples, prepared.store, prepared.max_length, seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        stale = 0
        for epoch in range(steps):
            model.train()
            total_loss, seen = 0., 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                context = model(to_device(batch['context'], device), batch['padding_mask'].to(device))
                pos = model.paired_scores(context, to_device(batch['positive'], device))
                neg = model.paired_scores(context, to_device(batch['negative'], device))
                loss = -F.logsigmoid(pos - neg).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                size = batch['padding_mask'].shape[0]
                total_loss += loss.item() * size
                seen += size
            metrics = evaluate(model, prepared.store, validation_plan)
            record = {'phase': phase, 'epoch': epoch + 1, 'loss': total_loss / seen, 'validation': metrics}
            history.append(record)
            if logger:
                logger(record)
            score = metrics['ndcg_at_k']
            if score > best_score:
                best_score, best_validation, stale = score, metrics, 0
                best_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= patience:
                break
        # Start preference training from the best next-item weights.
        # Keep those weights if preference training does not improve validation.
        if best_weights is not None:
            model.load_state_dict(best_weights)
    if best_weights is None:
        raise ValueError('No training examples were optimized')
    return {'history': history, 'best_validation': best_validation, 'training_examples': sizes,
            'seed': seed, 'test_evaluated': False}


def save_checkpoint(path, model, preprocessor, *, metadata=None):
    payload = {'version': 1, 'preprocessor': preprocessor.state_dict(),
               'model_config': asdict(model.config), 'max_length': model.max_length,
               'weights': {name: value.detach().cpu() for name, value in model.state_dict().items()},
               'metadata': metadata or {}}
    # Create a new file so an existing experiment cannot be overwritten.
    with Path(path).open('xb') as stream:
        torch.save(payload, stream)


def load_checkpoint(path, *, device='cpu'):
    # Load only data and tensors; checkpoint loading must not execute pickled code.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != 1:
        raise ValueError('Unsupported model checkpoint version')
    preprocessor = FeaturePreprocessor.from_state_dict(payload['preprocessor'])
    model = SpotifySASRec(preprocessor, payload['max_length'], ModelConfig(**payload['model_config']))
    model.load_state_dict(payload['weights'], strict=True)
    return model.to(device).eval(), preprocessor, payload['metadata']
