"""A compact shared track encoder and one-layer causal sequence ranker."""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    nhead: int = 8
    feedforward: int = 2048
    dropout: float = 0.1
    artist_dim: int = 8
    genre_dim: int = 4
    album_dim: int = 4
    time_signature_dim: int = 2

    def __post_init__(self):
        if min(self.d_model, self.nhead, self.feedforward, self.artist_dim,
               self.genre_dim, self.album_dim, self.time_signature_dim) < 1:
            raise ValueError('Model dimensions must be positive')
        if self.d_model % self.nhead or not 0 <= self.dropout < 1:
            raise ValueError('Invalid head count or dropout')


class TrackEncoder(nn.Module):
    def __init__(self, preprocessor, config):
        super().__init__()
        self.numeric_size = len(preprocessor.numeric_names)
        dimensions = {'artist': config.artist_dim, 'genre': config.genre_dim,
                      'artist_genre': config.genre_dim, 'album': config.album_dim,
                      'time_signature': config.time_signature_dim}
        self.embeddings = nn.ModuleDict({name: nn.Embedding(len(vocab) + 2, dimensions[name], padding_idx=0)
                                         for name, vocab in preprocessor.vocabularies.items()})
        for embedding in self.embeddings.values():
            with torch.no_grad():
                # Start unknown identities at zero instead of a random embedding.
                embedding.weight[1].zero_()
        width = self.numeric_size * 2 + sum(e.embedding_dim + 1 for e in self.embeddings.values())
        self.projection = nn.Linear(width, config.d_model)

    def forward(self, features):
        pieces = [features['numeric'] * features['numeric_mask'], features['numeric_mask']]
        for name, embedding in self.embeddings.items():
            ids = features[name]
            present = ids.ne(0).unsqueeze(-1)
            pooled = (embedding(ids) * present).sum(dim=-2) / present.sum(dim=-2).clamp_min(1)
            pieces.extend([pooled, features[name + '_mask']])
        return self.projection(torch.cat(pieces, dim=-1))


class SpotifySASRec(nn.Module):
    def __init__(self, preprocessor, max_length=10, config=None):
        super().__init__()
        if max_length < 1:
            raise ValueError('max_length must be positive')
        self.config = config or ModelConfig()
        self.max_length = max_length
        self.track_encoder = TrackEncoder(preprocessor, self.config)
        self.pos_emb = nn.Embedding(max_length, self.config.d_model)
        self.attention = nn.TransformerEncoderLayer(d_model=self.config.d_model,
            nhead=self.config.nhead, dim_feedforward=self.config.feedforward,
            dropout=self.config.dropout, batch_first=True)

    def forward(self, context, padding_mask):
        if padding_mask.dtype != torch.bool or padding_mask.ndim != 2:
            raise ValueError('padding_mask must be a two-dimensional boolean tensor')
        if padding_mask.all(dim=1).any():
            raise ValueError('Cannot encode an empty context')
        if padding_mask[:, -1].any() or (padding_mask[:, 1:] & ~padding_mask[:, :-1]).any():
            raise ValueError('Contexts must be right-aligned')
        x = self.track_encoder(context)
        batch, length, _ = x.shape
        if length != self.max_length or padding_mask.shape != (batch, length):
            raise ValueError('Context length or padding shape does not match model')
        x = x + self.pos_emb(torch.arange(length, device=x.device))
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        allowed = causal.unsqueeze(0) & ~padding_mask.unsqueeze(1)
        # Let padded queries attend to themselves so no attention row is empty.
        # Real queries still cannot see padding or future positions.
        identity = torch.eye(length, dtype=torch.bool, device=x.device)
        allowed |= padding_mask.unsqueeze(-1) & identity.unsqueeze(0)
        mask = (~allowed).repeat_interleave(self.config.nhead, dim=0)
        output = self.attention(x, src_mask=mask)
        return output[:, -1]

    def score_candidates(self, context_state, candidates):
        return context_state @ self.track_encoder(candidates).T

    def paired_scores(self, context_state, tracks):
        return (context_state * self.track_encoder(tracks)).sum(dim=-1)

    def parameter_report(self):
        return {'total': sum(p.numel() for p in self.parameters()),
                'numeric_inputs': self.track_encoder.numeric_size * 2,
                'embeddings': {name: embedding.weight.numel()
                               for name, embedding in self.track_encoder.embeddings.items()},
                'model_config': asdict(self.config)}
