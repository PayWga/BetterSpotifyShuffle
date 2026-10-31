# Model feature and evaluation changes

Historical design record from 2026-10-19. Branch references and execution notes
below describe that implementation; current usage is documented in
[model-training.md](model-training.md).

Approved direction: conversation on 2026-10-19. Implement on model_tweaks,
targeting the enriched extracting_features catalog without modifying that
branch's acquisition, metadata, datasets, tests, README, or .gitignore.

## Contract

- Audio values and availability remain distinct from sequence padding.
- Add compact pooled artist, track-genre, and separate artist-genre embeddings,
  scaled release year and confirmed duration. Explicit is excluded.
- Local acoustic descriptors are enabled by default, tolerate missing columns,
  and retain their own meanings. Extractor-source flags and observed analysis
  durations distinguish local estimates and partial/longer analyses. Duration
  candidates never substitute for confirmed duration. Analysis seconds are
  actual analyzed seconds, not a guaranteed full-recording indicator.
- Artist IDs are preferred. Unambiguous training-name aliases support records
  with names only. Literal/JSON lists are parsed; plain comma-containing names
  are not guessed apart. Collaborations use pooled embeddings.
- Key supports raw and circular encodings; time signature supports numerical
  and a small categorical embedding. Default: circular and categorical.
- Album and popularity are opt-in experiments. No track-ID embedding, larger
  Transformer, or new behavioral-history signals in this iteration.
- Split complete chronological sessions approximately 80/10/10 by session count,
  with at least one session per partition. No context crosses partitions.
  Vocabulary and numerical statistics fit only training-session tracks; frozen
  transformations are reused for held-out candidates and inference.
- Train next-item and skip-pair phases only on training sessions. Preserve
  existing event/session semantics; metadata-only tracks may be represented.
- Evaluate deterministic candidates and targets shared across experiments, with
  Hit@10 and NDCG@10, reporting candidate scope/count. Test remains optional and
  is run only after validation-based selection. Current catalog metadata is a
  snapshot, not historically reconstructed availability.
- Save versioned feature config, fitted preprocessing, architecture and weights
  together. Historical 24-input weights are not interchangeable with these.

## Files and validation

New shufflerec modules own preprocessing, datasets/splits, model, training,
evaluation, audit and experiments. Notebook is an orchestration/inspection entry
point. New tests use synthetic data and temporary outputs, never private rows.
Audit prints aggregate coverage only. No network jobs or dataset writes.

Implementation stages: preprocessing; chronological datasets; shared track
encoder; training/evaluation/artifacts; notebook/experiments; final verification.
Run focused RED/GREEN tests at each stage and the existing suite at completion.

## Decisions

- User explicitly authorized implementation after the in-chat plan; proceed
  without another approval round or automatic commits.
- Work in the branch the user selected, without creating another worktree.
- Do not merge/copy the other branch's catalog. Audit the current snapshot with
  clear schema limits; enriched-schema behavior is covered by synthetic tests.
