# Model workflow

Open `notebooks/ShuffleRec.ipynb` from the project root or `notebooks/`.
Install `requirements-model.txt` in the same Python environment as the notebook
kernel. Acquisition dependencies remain in their existing requirements file.

## Data contract and branch integration

This implementation consumes the enriched master table produced by
`extracting_features`; it does not import that branch's acquisition modules.
No acquisition file, existing test, source CSV, private history, README, or
.gitignore is edited by the model workflow. Merge the data/acquisition branch
normally when ready; no model code or column list should need changing for its
documented fields. Unknown optional columns produce missing inputs.

The model uses original audio properties plus artist, represented release year,
confirmed duration, track genre, separate artist genres, and local acoustics.
Album IDs and popularity are opt-in `FeatureConfig` settings; explicit is never
included. Original metadata remains in the source table, untouched.

Names are normalized with Unicode normalization, whitespace normalization and
case folding. Stable artist IDs preserve case. Explicit list representations
support multiple artists. A plain comma-containing name is preserved as one
display name instead of guessing whether commas separate artists. Training-only,
unambiguous name-to-ID aliases support records lacking IDs. Pooling retains a
fixed vector width; `max_labels=16` bounds each category list (configurable).
Album display names are not used as IDs because they can collide across artists.

Track genres and artist genres use distinct vocabularies. Artist genre labels
are not substituted for a missing track genre. `history_duration_candidate_ms`
is deliberately ignored. Year can describe a reissue rather than an original
recording. Positive/negative acoustic values are validated according to their
meaning: -1 dB RMS/loudness can be real; -1 key/duration cannot.

Local RMS dB, onset strength, spectral centroid, spectral contrast and beat
regularity are distinct inputs, not replacements for Spotify-style features.
For locally extractable properties, source flags and log analysis durations have
their own availability indicators. The source flag means a recognized local
extractor marker; an absent marker does not certify a Spotify measurement.
Analysis seconds reflect actual analyzed length. Unknown duration is not called
60 seconds or full-song, and a known duration is not automatically a claim of
full-song coverage. Audio-source URLs are not learned categorical inputs.

## Chronological protocol

UTC events are grouped using the existing 20-minute session gap. Complete sessions
are split approximately 80/10/10 by **session count**, with at least one session
per partition. Event/example proportions can differ. No context crosses a split.
Neutral/unrecognized events and unrepresentable tracks retain the existing
context-reset behavior. A metadata-only track is representable.

All vocabulary, artist aliases and numerical ranges fit training-session tracks.
Held-out values use the same transformations with range clipping. A numerical
feature with no training observations remains unavailable; uncalibrated future
values are not injected at inference. Padding ID 0, unknown category ID 1,
feature availability and sequence padding are separate concepts.

The notebook defaults to a candidate universe of catalog tracks occurring in
the supplied history, including held-out identities. This is a retrospective
personal-library snapshot, not proof of historical availability. Set scope to
`catalog` for the full supplied catalog; budget additional RAM/preparation time.
The default samples up to 99 unique negatives per held-out target, excluding
the context. Actual candidate counts and scope are reported. Small candidate
pools make Hit@10 easy; inspect NDCG and the reported counts too.

Validation plans select seeded examples throughout the partition and fix both
candidate identity and order. The same plan is used at each epoch. Candidate
ties follow the fixed shuffled order rather than automatically favoring targets.
Reported Hit@k/NDCG@k are fractions in [0,1]. They are sampled metrics, not
full-catalog ranking metrics. Snapshot metadata (especially popularity) is not
a historically reconstructed view; enabling popularity is an explicit experiment.

Next-item and skip-preference phases use only training examples. Validation
NDCG selects the best model across both phases. A degrading preference phase
does not replace better next-item weights. Empty preference data skips that
phase. Training seeds are controlled; identical floating-point results across
different devices/backends are not promised.

## Run controls and artifacts

- `RUN_AUDIT=True` performs a read-only aggregate scan.
- `RUN_TRAINING=False` by default. Preparation displays partition counts,
  candidate coverage and actual parameter costs before optimization.
- `RUN_COMPARISON=False` by default. Enable separately to run audio, metadata,
  compact encoding, and local-acoustic variants. All share common validation
  contexts/targets/candidates, and each reports its training coverage. Models
  are built sequentially to avoid storing several catalog-sized tensors.
- `RUN_FINAL_TEST=False` until configuration selection is complete. Final test
  output is a separate file. Do not tune on those results and continue claiming
  an untouched test set.

Training writes only new files in ignored `sensitive_info/model_runs/<run>/`.
Checkpoints contain artist vocabularies and trained personal weights; keep them
private. They package schema/config, fitted preprocessing, architecture, weights,
and run metadata together. `load_checkpoint` uses restricted tensor/data loading
and refuses incompatible schema versions. Saving refuses to overwrite a file.
These are inference/selection artifacts, not optimizer-state training resumptions.
Old 24-input checkpoints are not compatible with the new input contract.

To score new inputs, load the checkpoint, transform the inference catalog with
the loaded preprocessor, construct `TrackStore`, and use its context/candidate
tensor blocks with the loaded model. Never fit new vocabularies or scalers for
an already-trained checkpoint.

Read-only command-line audit:

```powershell
python -m shufflerec.audit --master data/master_track_table.csv --history sensitive_info/spotify_listening_history.csv
```

Tests use synthetic catalogs, history, audio and temporary outputs:

```powershell
python -m pytest tests/test_model_features.py tests/test_model_data.py tests/test_model_network.py tests/test_model_training.py tests/test_model_workflow.py -q
```

The starting checkout has an unrelated full-suite collection error:
`tests/test_feature_extraction.py` imports `core.feature_extraction` but the
tracked package is `scripts`. This work deliberately leaves that conflicting
acquisition-side test unchanged.
