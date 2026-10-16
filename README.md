# ShuffleRec

Personal music recommendation and sequencing experiments using Spotify listening
history, track metadata and audio features, and a SASRec-inspired Transformer.

- `scripts/polling.py` collects plays and skips from Spotify playback.
- `scripts/import_etended_spotify_history.py` imports extended listening history.
- `scripts/data_merging.py` manages the master catalog and Spotify metadata sync.
- `scripts/spotify_metadata.py` provides shared missing-only Spotify enrichment.
- `scripts/metadata_fetcher.py` tracks play/skip counts and library age.
- `notebooks/ShuffleRec.ipynb` prepares sequences and trains next-item and preference models.
- `scripts/feature_extraction.py` finds audio and fills missing acoustic measurements.

## Audio feature extraction

Install Python dependencies with `python -m pip install -r requirements-audio.txt`.
Install FFmpeg and ffprobe on PATH. Node.js 22+ or Deno 2.3+ is recommended for
YouTube extraction. The script automatically enables either installed runtime.
The `yt-dlp[default]` dependency includes its JavaScript challenge solver; see the
[upstream setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

From the repository root:

```powershell
# Inspect up to 10 entries using existing audio only; no files or downloads change.
python scripts/feature_extraction.py --dry-run --limit 10

# Search/download missing audio and extract features for up to 10 tracks.
python scripts/feature_extraction.py --limit 10

# Process all eligible tracks without online search.
python scripts/feature_extraction.py --local-only

# Include entries that reached the attempt cap, after resolving their errors.
python scripts/feature_extraction.py --retry-failed --limit 10

# Keep acquisition ahead of analysis (these are the default pool sizes).
python scripts/feature_extraction.py --limit 100 --acquisition-workers 8 --analysis-workers 4

# Analyze complete recordings rather than the first minute.
python scripts/feature_extraction.py --limit 100 --analysis-seconds full
```

Default inputs are `data/dsp_backlog.csv` (`track_id`, optionally `track_name`,
`artist_name`, `duration_ms`) and `data/master_track_table.csv` (`id`). Missing
search metadata is filled from the master catalog. `artists` can be a single
artist or a string containing a Python list of artist names. Use `--backlog`,
`--master`, and `--audio-dir` to select separate files/directories. Relative CLI
paths are resolved from the working directory; default paths use the project root.

Local audio must be named `<track_id>.wav`, `.flac`, `.mp3`, `.ogg`, or `.m4a` in
`data/temp_audio`. If absent, the script searches five YouTube results, checks
artist and title, rejects conflicting version labels, and compares duration when
available (tolerance: 12 seconds or 12%, whichever is greater). It tries ranked
matches until a download succeeds. If none succeeds, a second, broader query
examines up to ten results without the "official audio" qualifier. Already tried
URLs are not downloaded again. The same matching checks apply to both searches.
These checks are heuristic and cannot prove
that a result is the exact Spotify recording. Use recordings with appropriate
permissions, and inspect cached `<track_id>.source.json` files when reviewing matches.
Downloaded audio is converted to WAV and retained for retries. By default the
first 60 seconds are analyzed. `--analysis-seconds full` analyzes the entire
recording; a positive number selects another limit, e.g. `--analysis-seconds 120`.
Full-song mode offers broader coverage, not a guarantee of more accurate key or
tempo, and takes more CPU time and memory. Allow disk space for full recordings
in either mode; download quality and source selection are unchanged.
Each candidate downloads into a separate staging directory. Failed staging files
are retained for inspection; only successful WAV conversions and their source
sidecars are published into the cache root.

Matching tolerates apostrophe variations, `feat`/`ft` credits, equivalent
`remaster`/`remastered` wording, comma/semicolon-separated artist credits, and
VEVO channel suffixes. Every named collaborator must still be present. Different
titles, cover/live/remix versions, different requested remaster years, and
duration conflicts are rejected. After download, resolved metadata is checked
again before the WAV and source sidecar are published; rejected downloads stay
in their isolated staging directories.

The extractor estimates `tempo`, `key`, and `mode`, and adds separate descriptors:
`local_rms_db`, `local_onset_strength`, `local_spectral_centroid` (Hz),
`local_spectral_contrast`, and `local_beat_regularity` (0–1). They are **not**
replacements for Spotify energy, loudness, danceability, or instrumentalness.
Unreliable measurements remain NaN. Every newly filled feature has a
`<feature>_source` extractor version, `<feature>_audio_source` path/URL, and
`<feature>_analysis_seconds` actual analyzed duration. Existing features retain
their original provenance and duration. Old measurements with no duration remain
unknown; their analysis scope is not guessed.
Existing nonmissing values and unrelated metadata are preserved.

Each run visits eligible backlog entries once. Acquisition and analysis have
separate thread pools: `--acquisition-workers` defaults to **8**, and
`--analysis-workers` to **4**. More downloads can proceed while the analysis pool
processes ready recordings. One coordinator still owns all catalog updates.
`--queue-size` limits ready audio paths plus reserved slots for acquisitions in
flight; it defaults to the greater of acquisition workers and twice analysis
workers. A smaller queue can intentionally limit acquisition concurrency.
The queue contains paths and metadata, not decoded waveforms. Only analysis
workers load signals, keeping analysis memory usage bounded.

`--workers N` remains a compatibility shorthand that sets both pools to N;
explicit stage-specific options override it. Thus `--workers 1` means one worker
in each stage, which can still overlap acquisition with analysis. Scientific
library work can run outside Python's interpreter lock; actual gains depend on
the computer, recordings, network responses, and memory pressure. Reduce
analysis workers if memory is tight, especially in full-song mode. Workers
receive individual track records, not copies of the master catalog. A slow
recording does not block the checkpointing of completed recordings.

The master is read once per run. Newly added rows are appended together once per
checkpoint, rather than rebuilding the master for every track. `--batch-size`
now defaults to **100 completed tracks per checkpoint**, instead of 10. This
reduces full CSV rewrites; it does not control concurrency. Use `--batch-size 10`
for more frequent saves at greater disk/serialization cost. `--max-attempts`
still defaults to 3. Failed entries retain an attempt count and error reason;
successful entries are removed only after their master updates are saved.

Ctrl+C attempts to checkpoint results already collected by the coordinator;
in-flight tracks retain their backlog entries and cached audio for a later run.
Active workers may take time to finish before shutdown. If Ctrl+C interrupts a
checkpoint itself, it is not repeated, avoiding double-counted failure attempts.
A forced termination can require reanalysis since the previous checkpoint.

CSV writes use atomic replacement, but concurrent catalog/backlog writers are
not supported. A timestamp/size check aborts saving if another process changed a
CSV since the previous checkpoint; this is a conflict check, not a multi-process
lock. Back up the catalog before large runs and finish other writer jobs first.
If Windows denies CSV replacement (for example, a reader holds it open without
delete sharing), the writer retries the same prepared file up to six times over
about eight seconds; it does not serialize the whole table again. It aborts if
the destination changes during preparation/retries. Persistent access denial
produces an actionable error and a nonzero exit. The complete prepared CSV is
retained beside the destination as a hidden `.master_track_table_*.csv` or
`.dsp_backlog_*.csv` snapshot; its path is printed. A retained snapshot is not
automatically applied on restart because a later run or another writer may have
changed the catalog. Do not replace newer data with it without reviewing the
current state. Incomplete serialization artifacts are not kept as snapshots.
If the master replacement fails, that checkpoint does not remove successful
tracks from the backlog. Downloaded audio remains cached, so a normal rerun can
reuse it, although unsaved feature analyses will run again.
Logs report load time, acquisition and analysis time per track, checkpoint time,
and total wall time. Acquisition also reports search, resolving/downloading, and
FFmpeg conversion time. Summed worker times can exceed wall time because stages overlap.

Failures now distinguish `no_search_results`, `no_matching_recording`,
`missing_search_metadata`, `invalid_audio`, and search/download errors with codes
such as `http_403`, `http_429`, `authentication_required`, and `network_timeout`.
Rejected candidate counts are logged by reason, and the run ends with a failure
histogram. Authentication challenges and HTTP 429 stop that track's candidate
and fallback-search retries immediately. Three authentication-challenged tracks
in one run, or a single HTTP 429, pause new acquisition for the rest of that run.
Already-started acquisitions and queued/active analyses finish and checkpoint
normally; there is no automatic timed resume. Challenged tracks record their
error without increasing `attempts`. Unstarted tracks remain unchanged. The
final JSON includes `deferred` when service challenges occurred; after a pause
it includes both challenged and unstarted eligible tracks. These errors also
appear in the error histogram, but are excluded from the `failed` track count.
The threshold counts challenges across the run, not only consecutive failures.
Check YouTube access before retrying; no cookie/browser access is automatic.
To use the authentication options suggested by yt-dlp, first open YouTube in
your browser on the same connection and complete any verification it presents,
then select that browser explicitly:

```powershell
python scripts/feature_extraction.py --cookies-from-browser chrome --acquisition-workers 1 --limit 5
```

Replace `chrome` with `edge` or `firefox` as appropriate. A specific profile is
accepted as `BROWSER:PROFILE`, for example `--cookies-from-browser "chrome:Default"`.
If browser cookie access/decryption fails, close the browser and retry, or use
an exported Netscape-format cookie file:

```powershell
python scripts/feature_extraction.py --cookies sensitive_info/youtube-cookies.txt --acquisition-workers 1 --limit 5
```

Choose one cookie source. Authentication is loaded once before processing,
then copied into each yt-dlp context in memory. The script does not export
browser cookies or overwrite the supplied cookie file. Loading errors stop the
run before track attempts/CSV updates. Dry-run and local-only modes never read
cookie files or browser cookies. Cookies are session credentials: keep exports
private in the ignored `sensitive_info` directory. Account cookies can still be
challenged or expire, and automated account access can lead to restrictions;
see the official [cookie FAQ](https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)
and [YouTube guidance](https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-youtube-cookies).

For a small subsequent test, use `--acquisition-workers 1 --limit 5`.
This reduces concurrent requests but does not enforce a request-rate limit.
Earlier runs may already have exhausted attempts; `--retry-failed` makes those
entries eligible again without resetting their stored attempt counts.
These improvements cannot guarantee recovery of unavailable or
unidentifiable recordings; older `audio_not_found` records do not identify the
original cause. Retrying with `--retry-failed` uses the new search/matching logic
but includes all attempt-exhausted entries, not just the last run's failures.

The calculations reuse magnitude and log-mel spectra. Rhythm retains Librosa's
median onset aggregation, while the onset descriptor retains mean aggregation;
the feature definitions and key-estimation algorithm are unchanged.

The extractor deliberately preserves existing `-1` values because they are
nonmissing CSV values. Use the placeholder cleanup below to normalize known
unknown values to NaN before extraction. The notebook still needs explicit model
availability/sequence masks and does not yet consume `local_*` features.

## Fill missing Spotify metadata

`--sync` still adds tracks from listening history that are absent from the master
catalog. `--backfill` checks **all existing master tracks**, independent of listening
history, and fills missing metadata only. Both use the same Spotify response mapping
and the credentials in `sensitive_info/spotify_credentials.json` (`client_id`,
`client_secret`, `redirect_uri`) and private OAuth cache. Install `pandas` and
`spotipy` if these are not already installed. Spotify may open its OAuth login flow.

**While audio extraction is running**, fetch into a separate checkpoint:

```powershell
# Preview candidate count; no authentication, API calls, or file writes.
python scripts/data_merging.py --backfill --dry-run

# Fetch and checkpoint metadata without writing the master CSV or DSP backlog.
python scripts/data_merging.py --backfill --fetch-only

# AFTER extraction and other catalog writers have stopped:
python scripts/data_merging.py --apply-backfill --dry-run
python scripts/data_merging.py --apply-backfill
```

For later runs, when no other catalog writer is active, `--backfill` without
`--fetch-only` fetches and then applies in one command. Applying always reloads
the latest master CSV and fills only its remaining gaps, preserving extraction
results, provenance, row order, IDs, existing values, and unrelated columns.
Atomic replacement protects against interrupted writes, but **does not support
concurrent master writers**. A running extractor cannot acquire a newly added
lock, so staging is required during your current extraction run. Run only one
metadata fetcher per checkpoint at a time.

Default fields are `name`/`track_name`, `artists`/`artist_name`, `artist_ids`,
`album`/`album_name`, `album_id`, `release_date`, `release_date_precision`, `year`,
`duration_ms`, `explicit`, `track_number`, `disc_number`, `popularity`, and `isrc`.
Equivalent existing names and a year derived from `release_date` are used before
requesting Spotify. `year` describes this album release, which may be a reissue,
rather than necessarily the original recording's year. Values Spotify omits
remain missing; false and zero are valid values. Artist IDs/names use comma-separated
strings, consistent with the existing sync. Existing list-form artist strings are
preserved. Blank/whitespace cells and numeric NaNs count as missing; existing `-1`
values are preserved (use the separate placeholder cleanup when appropriate).

Optional album and artist requests can fill `label` and a new `artist_genres`
JSON list. Album/artist responses are cached within a run. Artist genres are kept
separate from `genre` and `track_genre` because they describe the artist, not an
individual track. Stream counts, country, and audio features are outside this
metadata workflow; they are not inferred or fabricated. Optional fields may be
unavailable to your Spotify app.

```powershell
# Smaller resumable runs; repeat the identical command to visit the next entries.
python scripts/data_merging.py --backfill --fetch-only --limit 100

# Select fields and use a separate checkpoint for this selection.
python scripts/data_merging.py --backfill --fetch-only --fields artist_name year --checkpoint data/names-and-years.spotify-metadata.json

# Include album label and artist genres in a separate run.
python scripts/data_merging.py --backfill --fetch-only --extra-metadata --checkpoint data/extended.spotify-metadata.json
python scripts/data_merging.py --apply-backfill --checkpoint data/extended.spotify-metadata.json
```

Use `--master path/to/copy.csv` for a separate catalog. Relative paths resolve from
the current directory; defaults resolve from the project root. The default
checkpoint is `data/master_track_table.spotify-metadata.json`. It is bound to the
absolute master path, requested fields, and optional-request setting; mismatched
options fail rather than silently reuse another run's results. Checkpoints are
saved every 50 checked tracks and on orderly exit, including Ctrl+C. Completed
lookups (including omitted fields), unavailable tracks, and invalid/local IDs are
skipped on resume. Transient failures are retried on subsequent runs; HTTP 429
respects `Retry-After`, and transient request errors have bounded retries.
Authentication/permission failures stop the run with prior progress saved.
To revisit completed or unavailable lookups, choose a **new checkpoint filename**.
Custom checkpoint files can contain your track IDs: keep them private; default
`data/*.spotify-metadata.json` files are ignored by Git.

Requests use the individual-track endpoint for compatibility with current
[Spotify development-mode endpoint changes](https://developer.spotify.com/documentation/web-api/references/changes/february-2026).
This can take substantial time for a large catalog; progress and summary counts
show fetched, resumed, unavailable, invalid, and failed entries and filled cells
per column. Relinked Spotify responses retain the requested master ID. Duplicate
or empty master IDs are rejected before fetching so metadata is never applied to
ambiguous records. This command deliberately does not run the general feature
cleaner, modify DSP retry state, or refresh existing popularity values.

## Clear legacy placeholders

Wait for extraction and any other catalog writer to finish. From the repository
root, preview the number of cells to clear, then apply the cleanup:

```powershell
python scripts/data_merging.py --clear-placeholders --dry-run
python scripts/data_merging.py --clear-placeholders
```

`clear_feature_placeholders(df)` returns a cleaned copy and counts per column.
`clear_placeholders(master_path=None, dry_run=False)` reads the master CSV and
applies that function. Use `--master path/to/copy.csv` to select another catalog.

Only exact numeric `-1` values (including strings such as `-1.0`) are cleared
from known nonnegative fields: the 0–1 audio features, tempo, key, mode, time
signature, duration, track/disc numbers, year/release_year, popularity, stream
count, explicit, and the nonnegative `local_*` descriptors. Spotify's `key=-1`
means no key was detected; it becomes NaN under the project's convention.
`loudness=-1` and `local_rms_db=-1` are preserved because -1 dB can be a valid
measurement. Custom columns are left alone rather than guessing their ranges.

NaN is saved as an empty CSV cell. Other values, including zero, other negative
numbers, text like `NA` and `0007`, row order, duplicate IDs, and provenance are
preserved. This command does not perform broader repairs or add tracks to the
backlog. It reports counts per changed column, replaces the CSV atomically, and
does not rewrite files with no placeholders. `--dry-run` never writes the catalog.
The existing general cleaner now uses the same placeholder rule, so merge,
initialization, and repair no longer discard valid -1 dB loudness measurements.

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```

Tests use temporary catalogs and generated audio. Online acquisition is simulated
at the yt-dlp boundary; the default suite makes no network requests and never uses
private listening data or the real catalog/backlog. In a restricted environment,
choose a fresh writable `--basetemp` directory under `.test-runs`.
