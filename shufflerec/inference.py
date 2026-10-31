"""Read-only local ranking with a frozen checkpoint and cached track vectors."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import TrackStore
from .features import clean_catalog
from .training import load_checkpoint, to_device


@dataclass(frozen=True)
class RankedTrack:
    track_id: str
    score: float


class InferenceEngine:
    """Own a model/catalog snapshot; construct a new engine to refresh either.

    Do not train or mutate ``model`` after construction: candidate vectors are
    cached for these exact weights. No preprocessing is fitted here.
    """

    def __init__(self, model, preprocessor, catalog, *, batch_size=1024):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.model = model.eval()
        self.preprocessor = preprocessor
        self.catalog = clean_catalog(catalog)
        self.store = TrackStore(self.catalog, preprocessor)
        self.device = next(model.parameters()).device
        self.ids = tuple(sorted(self.store.usable_ids))
        self.index = {track: i for i, track in enumerate(self.ids)}
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(self.ids), batch_size):
                features = to_device(self.store.take(self.ids[start:start + batch_size]), self.device)
                vectors = self.model.track_encoder(features)
                if not torch.isfinite(vectors).all():
                    raise FloatingPointError('Non-finite candidate embeddings')
                chunks.append(vectors)
            self._vectors = torch.cat(chunks) if chunks else None

    @classmethod
    def from_checkpoint(cls, path, catalog, *, device='cpu', batch_size=1024):
        model, preprocessor, _ = load_checkpoint(path, device=device)
        return cls(model, preprocessor, catalog, batch_size=batch_size)

    def usable_context(self, context_ids):
        """Unknown tracks reset context, just as they do in training examples."""
        context = []
        for track in context_ids:
            track = str(track).strip()
            if track in self.index:
                context.append(track)
            else:
                context.clear()
        return context[-self.model.max_length:]

    def rank(self, context_ids, candidate_ids, *, exclude=()):
        context = self.usable_context(context_ids)
        excluded = set(map(str, exclude))
        candidates = sorted({str(t).strip() for t in candidate_ids} & self.index.keys() - excluded)
        if not context or not candidates:
            return []
        self.model.eval()
        with torch.inference_mode():
            features, padding = self.store.context(context, self.model.max_length)
            features = to_device({k: v.unsqueeze(0) for k, v in features.items()}, self.device)
            state = self.model(features, padding.unsqueeze(0).to(self.device))
            indexes = torch.tensor([self.index[t] for t in candidates], device=self.device)
            scores = (state @ self._vectors.index_select(0, indexes).T)[0]
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite inference scores')
            ranked = [RankedTrack(track, score) for track, score in zip(candidates, scores.tolist())]
        return sorted(ranked, key=lambda item: (-item.score, item.track_id))
