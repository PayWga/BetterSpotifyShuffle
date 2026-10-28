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
is deliberately ignored. New preprocessing also masks durations explicitly
marked `duration_ms_is_estimated=True`; other track features remain eligible.
This duration policy is serialized with preprocessing. Older artifacts without
the policy field retain their legacy duration transformation when loaded.
`include_estimated_durations=True` opts into using flagged estimates and adds a
separate binary `duration_ms_is_estimated` input with an availability mask.
Unknown estimate status remains unknown; invalid/missing duration stays missing.
This opt-in uses the estimates already in the snapshot. They may incorporate
held-out listening events: it is exploratory, not a claim of point-in-time
duration estimation. Strict chronological experiments would need estimates
recomputed from pre-cutoff evidence only.
Year can describe a reissue rather than an original
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

- `RUN_AUDIT=False` currently skips repeating the read-only aggregate scan.
- `RUN_TRAINING=False` skips the separate single-model experiment. Enable it to
  train using `FEATURE_CONFIG`; it does not select comparison variants.
- `RUN_COMPARISON=True` currently configures the notebook to run audio, metadata,
  compact encoding, and local-acoustic variants. All share common validation
  contexts/targets/candidates, and each reports its training coverage. Models
  are built sequentially to avoid storing several catalog-sized tensors.
  `COMPARISON_MODE="individual_local"` is now the default: running all cells
  starts six fits (one baseline plus one per local measurement), each with up
  to five next-item epochs. `feature_groups` selects the prior four cumulative
  variants; `estimated_duration` selects two duration-policy variants.
  Preference epochs are zero for this first feature comparison. No optimization
  starts merely by opening the notebook.
  `COMPARISON_VARIANTS` is the editable mapping of names to `FeatureConfig`
  objects. The settings cell includes examples for add-one and leave-one-out
  experiments. `FEATURE_CONFIG` controls only single-model runs.
  Learning rates and patience now reach comparison training as well.
  `MODEL_COOLDOWN_SECONDS=10` adds an idle pause after releasing each model
  except the last, before starting preparation for the next fit. Set 0 to
  disable. The setting is recorded in reports; model elapsed times exclude
  the idle interval. This does not monitor or guarantee hardware temperature.
- `RUN_FINAL_TEST=False` until configuration selection is complete. Final test
  output is a separate file. Do not tune on those results and continue claiming
  an untouched test set.

Training writes only new files in ignored `sensitive_info/model_runs/<run>/`.
Successful comparisons save `comparison_results.json` in a new comparison run
folder: configurations, epoch histories, metrics, parameter costs, seeds,
input file signatures and candidate protocol. The notebook prints a compact
results table too. Comparison model weights are not saved; train the chosen
configuration separately to obtain an inference checkpoint.
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

The prior `core.feature_extraction` test import mismatch was resolved by the
merged acquisition branch. The merged baseline suite passed 372 tests before
the notebook-control and duration-policy changes described above.

## Notebook experiment controls

The notebook's Control reference section lists every supported feature flag,
encoding option, sampling/data setting, optimizer setting and model dimension.
Original audio columns are always included. Each local measurement has its own
boolean switch: `local_rms_db`, `local_onset_strength`,
`local_spectral_centroid`, `local_spectral_contrast`, `local_beat_regularity`.
Disabling one removes that value, availability mask and its own provenance/scope
inputs. The older `local=False` master gate remains compatible and overrides all
five; leave it True to use individual switches. Selected descriptors remain
separate inputs, never an averaged or composite acoustic value.

Per-column original-audio ablations, date-window filters, source cohorts, custom
session gaps and notebook split-fraction controls are not implemented. Do not
alter the master CSV to simulate those experiments.

`individual_local_variants()` builds the baseline plus five add-one cases with
provenance disabled by default, isolating numerical measurements.
`estimated_duration_variants()` keeps the other configuration choices fixed and
varies estimate inclusion; results explicitly record the potential held-out
history dependence. `INCLUDE_ESTIMATED_DURATIONS` applies to single-model runs
and all variants in the other two comparison modes. Duration-comparison mode
deliberately uses both policies regardless of that global switch.

Keep the candidate universe, seed, context length and validation limits fixed
within feature comparisons. Default variants are cumulative; use an add-one
variant to isolate one feature or a leave-one-out variant to assess its value
alongside the remaining features. Different configurations can retain different
training examples, which is why the result table includes training coverage.
