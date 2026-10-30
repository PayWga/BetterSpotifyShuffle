# Local inference and sequencing

The offline sequencer uses a selected model checkpoint to produce a rolling
five-song queue. It makes no Spotify requests and never writes its inputs.
Install the existing `requirements-model.txt`; no new dependencies are required.

The playlist CSV is only an input adapter for offline testing. The
[live pipeline](live-pipeline.md) loads the current Spotify playlist automatically
and passes its IDs directly to inference and sequencing. It follows playlist
changes and checks playlist snapshot versions for edits; no playlist CSV is
needed for live playback.

For offline testing now, from the repository root start a persistent session:

```powershell
python -m shufflerec.sequence --checkpoint sensitive_info/model_runs/YOUR_RUN/model.pt --catalog data/master_track_table.csv --playlist sensitive_info/playlist.csv --context sensitive_info/accepted_context.csv
```

Use the actual checkpoint filename from the selected training run. `playlist.csv`
needs a `track_id` column. `accepted_context.csv` needs the same column, ordered
oldest to newest, containing accepted plays only. Do not pass unfiltered listening
history: its skip and interruption rows are not accepted context. IDs remain
strings, including numeric-looking and local IDs. The master catalog uses `id`.
Context tracks may be outside the playlist. Omitting context starts with a
reported playlist-random fallback until a model-usable context is available.

`--once` prints the initial queue and exits. Otherwise, each input line is a JSON
request and each output line is a JSON response. The model stays loaded until
EOF/Ctrl+C. For example, type these lines, using the active selection's actual ID:

```json
{"command":"start"}
{"command":"feedback","selection_id":1,"completion_rate":0.12,"reason_end":"track_change"}
{"command":"status"}
```

`start` marks the queue head as actually started; it does not initiate audio
playback. `feedback` resolves that active selection and replaces the unplayed
queue. At/above the threshold (default 0.25, or 25%) is acceptance; below it is
rejection for `track_change`/`skip`, otherwise neutral. Report a resolved playback
event once per selection, rather than each polling sample. Duplicate/stale
feedback is rejected. A pause does not resolve a selection. This sequencer rule
does not change the poller's separate 70% PLAY classification or historical
training events.

`status` reads state. `plan` replaces provisional suggestions when no selection is
active. It consumes new randomness, but does not change learned preferences or
committed variety penalties. Resolve an active item before starting another or
replanning. No queue item can appear twice within the same planned queue.

## Why inference is cached

The checkpoint restores both weights and frozen feature preprocessing. Catalog
rows are transformed once with the existing missing-feature and attention masks.
The track encoder produces reusable candidate vectors in bounded batches. Each
slot encodes context anew, then scores candidates using dot products with the
cached vectors. Scores rank candidates; they are not acceptance probabilities.

Later slots use prior planned songs as provisional context. Those songs are not
written into confirmed history and cannot earn feedback until started. Unknown
or model-unusable tracks reset model context; only the trailing usable suffix is
encoded, matching training's reset convention. Unknown playlist tracks can still
be selected from the random pool.

The engine owns a snapshot. Restart with a new engine/session after updating a
checkpoint or catalog; do not train its model or mutate its features in place.
The CLI reads only playlist/context rows from the master, and caches those rows.
Default inference is on CPU. Use `--device` and `--batch-size` to change the device
and candidate encoding batch size. Actual latency depends on hardware and playlist
size; no real-checkpoint performance benchmark has been run for this change.

## Sampling rules and defaults

| Setting | Default | Effect |
| --- | --- | --- |
| `queue_size` | 5 | Number of future selections |
| `base_weights` | [1, 1, 1] | Random, top, filtered initial pool weights |
| `randomize_initial_weights` | false | Optionally multiply base weights by bounded random factors |
| `initial_weight_low`, `initial_weight_high` | 0.8, 1.2 | Initial random multiplier limits |
| `min_probability`, `max_probability` | 0.05, 0.75 | Final bounds for each available pool |
| `top_k`, `top_weight_low` | 20, 0.5 | Top pool depth and lowest relative rank weight |
| `initial_candidates`, `max_candidates` | 50, 200 | Filtered pool starting and maximum ranking depth |
| `min_filtered` | 10 | Expansion target, not a minimum required to use the pool |
| `genre_weight_high`, `genre_weight_low` | 0.75, 0.375 | Genre-only rank weight limits |
| `fatigue_decay`, `fatigue_strength` | 0.65, 0.8 | Variety penalty recovery per selection and strength |
| `learning_rate`, `preference_limit` | 0.15, 2.0 | Signed feedback step and absolute log-preference bound |
| `early_skip_threshold` | 0.25 | Completion fraction separating early skip and acceptance |
| `recent_window` | 10 | Recent started/initial-context tracks excluded from selection |

Pass any overrides as a JSON file with `--config`. For example:

```json
{"queue_size":5,"early_skip_threshold":0.25,"randomize_initial_weights":true}
```

Playlist random is uniform. Top recommendations have gentle linear rank weights.
The filtered pool admits either an artist or genre match to the preceding song.
Artist matches always have weight 1, regardless of rank. This includes matches
on both artist and genre. Genre-only matches receive linear rank weights within
the inspected recommendation prefix. Stable artist IDs are case sensitive; names
are normalized and used when IDs are missing. Track genres match track genres;
artist genres separately match artist genres. Missing metadata cannot manufacture
a match. Explicit list representations support multiple artists and genre labels.

Expand the prefix by doubling until enough matches exist or the configured limit
is reached. Expansion does not repeat inference. No quotas guarantee that any
artist/category wins; relative candidate counts naturally affect the result.

Pool weights combine base weights, learned preferences and temporary fatigue.
Acceptance increases the source pool's log-preference; rejection decreases it;
neutral feedback does neither. Each started selection adds fatigue to its pool;
all fatigue decays with subsequent starts. Planning simulates this on a copy, so
rebuilding unheard suggestions cannot accumulate penalties. Final probabilities
are normalized subject to the configured floor/ceiling, not merely clipped and
renormalized. Pool overlap is allowed; feedback credits only the selected source.

Each queue item includes its source pool, process-local selection ID, probability
distribution, inspected candidate depth, and diagnostic codes. Empty pools are
removed without punishment. If only one pool is available, the user-approved
100% exception is reported as `single_available_pool`. Missing model context,
empty model candidates and runtime scoring failures have separate diagnostics.
If no eligible tracks remain, return a shorter queue with `playlist_exhausted`.
Small playlists may require reducing `recent_window` to permit further playback.

## Scope and verification

Adaptive weights, recent history and the queue are session-local. Restarting resets
them. Adaptive playback observations and submitted-selection feedback are described
in [the playback guide](adaptive-playback.md), including private SOCKS routing.
Spotify queue additions and automatic playlist loading are connected through the
[live entry point](live-pipeline.md). Captured output
contains personal track IDs and selection history, so keep it out of commits.

Tests use synthetic checkpoints/catalogs and temporary files only:

```powershell
python -m pytest tests/test_inference.py tests/test_sequencing.py tests/test_sequence_cli.py -q --basetemp .test-runs/sequencing-check
```

Choose a fresh `--basetemp` directory for each run; pytest clears a reused base.
