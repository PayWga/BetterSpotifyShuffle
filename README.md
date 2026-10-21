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
URLs are not downloaded again. If both fail, a third query examines up to fifteen
results using the primary artist and core title, retaining requested version
checks. A final primary-artist/core-title query examines up to fifteen lyric
video results without requiring a remaster label in the query. Queries can
overlap; a candidate is attempted only once per track/run.
The same final matching checks apply to all searches.
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
VEVO channel suffixes, and presentation labels such as "Audio Only", "Full
Audio", "Unofficial Lyrics Video", "Fan Made Lyric Video", and "Reupload".
An official uploader is not required. Controlled presentation brackets (including
lyric/Letra/Paroles labels and lyric-video author credits) are ignored when
checking the song title; a different song merely containing the requested title
is still rejected. When a search result
matches the title/version/duration and primary artist but lacks collaborator
credits, full metadata is fetched without downloading audio. Structured artist
and creator fields can then establish the missing credits. If guest credits
remain absent, a candidate can qualify only when the primary artist/title match
and known duration differs by no more than **3 seconds or 2%, whichever is
greater**. The same tighter duration rule permits a requested remaster to use an
upload that omits its remaster label/year. This does not verify the exact master;
an explicitly conflicting remaster year remains rejected. Missing duration is
insufficient for either relaxation. Complete credits and explicit remaster
matches rank above relaxed alternatives with otherwise equal evidence.
Matching separates the base title from featured credits and recording labels.
`(with Future)` and `(feat. Future)` are equivalent; Japanese `Version`/`Ver.`
and explicit `(Japanese)` labels agree. Explicit language labels are required:
an unmarked/Korean upload cannot substitute a Japanese version. Live/studio and
stripped/regular recordings remain distinct. A non-Latin title with an explicit
Latin-script alias can match that alias, e.g. `붐바야 (BOOMBAYAH)`. Artist/title
order and recognized pipe-delimited lyric/uploader branding are handled without
accepting unrelated additional song names. Fallback queries retain known
language/live/stripped constraints.
Source sidecars record `match_evidence` when guest/remaster identity is relaxed,
using matching policy `structured_recording_identity_v6`.

**A remix is eligible only when the backlog track itself requests a remix.** Remix/RMX,
re-mix/remixed, bootleg, mashup, rework and alternate club/dance/DJ/extended/dub/
festival mix labels are checked before presentation text is stripped. Candidate
title, track, alternate-title, version and tag fields are checked, together with
remixer credit fields. A requested named remix must match its title/version;
an original or a different named remix is rejected. RMX, re-mix and remixed
spellings normalize to remix. Plain "Original Mix" is allowed for originals;
a generic unexplained "Mix" label is not. These checks depend on supplied metadata and cannot detect
an unlabelled remix by listening to the waveform.
The previous unconditional filter's `excluded_remix` status no longer suppresses
tracks: these entries are eligible again even above the normal attempt limit,
without resetting stored counters or requiring a manual CSV edit. No rows or
existing features are deleted. Cached audio with incompatible remix sidecar
metadata is bypassed and retained on disk while a matching recording is sought;
cached audio for the requested remix remains usable.
Different titles, conflicting cover/live versions, and duration conflicts remain
rejected. After download, resolved metadata is checked
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
With `--retry-failed`, tracks are selected by lowest attempt count first, keeping
backlog order for ties, before applying `--limit`. Thus genuine failures move
behind less-attempted tracks on later limited runs. Service-deferred tracks keep
their attempt counts and may be selected again until service access is restored.

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
such as `http_403`, `http_429`, `authentication_required`, `network_timeout`,
`geo_restricted`, `session_reload_required`, and `js_challenge_failed`. Geographic
classification requires an explicit country/region restriction in the error;
ordinary unavailable/deleted recordings remain separate. Not every upstream
error exposes a specific cause.
Rejected candidate counts are deduplicated by URL/reason. Unsuccessful searches
also log up to five candidate examples with rejection reason, title, uploader and
URL. URL credentials, fragments and arbitrary query tokens are omitted. Redirect
console output to a private log if you want to review these examples later.
The run ends with a failure histogram. Authentication, session-reload, JavaScript
challenge and HTTP 429 errors stop that track's candidate and fallback-search
retries immediately. Three challenged tracks (any mix of authentication,
session-reload or JavaScript errors) in one run, or a single HTTP 429, pause new
acquisition for the rest of that run.
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

### Enrich tracks in the audio backlog

Backlog-only tracks may have no row in the master, so master backfill cannot
supply their missing durations. With extraction and other backlog writers
stopped, use the dedicated action:

```powershell
python scripts/data_merging.py --backfill-backlog --dry-run
python scripts/data_merging.py --backfill-backlog --limit 50
```

After checking that small enrichment run, omit `--limit` to enrich the remaining
backlog before retrying extraction. The enrichment limit and extraction limit
can select different rows, so enriching only 50 rows does not guarantee that the
next 50 extraction candidates have metadata.

This reuses the existing private Spotify credentials/OAuth client and track API
retry handling. It fills missing `duration_ms`, `isrc`, `track_name` and
`artist_name`, and stores separate `spotify_track_name`, JSON
`spotify_artist_names`, `spotify_resolved_id`, `spotify_metadata_source` and UTC
`spotify_metadata_checked_at` fields. Existing values, attempts, last extraction
errors, IDs and unrelated columns are preserved. Full Spotify artist credits
are used for matching; the original displayed backlog identity is retained.
Successful extraction carries enriched metadata into remaining gaps in the
master, including tracks already present there.

The action writes only the selected backlog (`--backlog PATH` can override it),
not the master. It checkpoints approximately every 25 requests and on ordinary
error/interrupt exit. Reruns skip rows with duration and both Spotify identity
fields already present. ISRC is optional; unavailable ISRC alone does not cause
repeated requests. API permission/authentication errors stop the run after saving
completed enrichment; unavailable/invalid IDs get a separate
`spotify_metadata_error`, leaving extraction attempt counts unchanged. Unknown
response IDs are rejected unless Spotify explicitly links them to the requested
ID. A timestamp/size conflict check aborts rather than overwriting a detected
external update; concurrent backlog writers remain unsupported. Dry-run reads
only: no authentication, API requests or writes.

Metadata HTTP attempts start at least **10 seconds apart**, with an additional
**five-minute rest after every 50 attempts**. Failed lookups, retries, and optional
album/artist requests all count. Slow responses count toward the interval, so
no extra interval delay is added when a request already took 10 seconds or longer.
A monotonic gate protects calls within one process; persistent transport pacing
also coordinates this project's metadata processes and survives restarts. Rest
waits release the state-file lock and remain interruptible with Ctrl+C.

There is **no default daily request cap**. The previous 200-attempts-per-day cap
was a local guess, not an established Spotify limit, and is now disabled.
The retained 24-hour attempt history is for diagnostics, not a quota restriction.
The interval/rest defaults are conservative settings for sustained metadata
fetching, not a promise of a safe Spotify threshold. Adjust `REQUEST_INTERVAL`,
`REQUEST_REST_EVERY`, and `REQUEST_REST_SECONDS` in `scripts/spotify_metadata.py`
when necessary. Playback polling and apps outside this metadata client are not
coordinated; run one metadata job at a time and account for other API activity.

Every Spotify 429 now stops immediately, even with a short `Retry-After`; it is
never automatically retried. The transport recognizes `QUOTA_EXCEEDED` and
persists the cooldown until `Retry-After` plus a one-minute safety margin.
Missing or invalid retry times cause a conservative 24-hour pause. Completed backlog
updates or master metadata checkpoints are saved first; rate limits do not spend
audio-extraction attempts. The command exits with code 2 and displays the next
UTC retry time when available. The HTTP session preserves original 429 headers
so Spotipy cannot discard them in a `Max Retries` exception. This shared handling
also applies to master metadata backfill. Rerun the same command only after the
reported cooldown; restarting does not clear recorded cooldowns or batch rests.
The command does not schedule a restart or shorten Spotify's restriction.
`--sync` also stops on cooldown and saves successfully fetched records into
missing master cells before exiting.

Request timestamps, batch-rest state, and Spotify cooldowns live in
`sensitive_info/spotify_request_budget.json`, protected by a sidecar file lock and
atomic replacement. It contains no credentials or listening history. Keep the
file: replacing a checkpoint does not reset it, unreadable budget state blocks
requests, and manually deleting the file would discard this protection. Calls
made before this safeguard was installed cannot be counted retroactively.
Existing running metadata processes must be stopped and restarted to load it.

```powershell
# Inspect pacing/cooldown status without authenticating or making API requests.
python scripts/data_merging.py --spotify-budget-status

# Record a cooldown you already received. Seconds are measured from now;
# this never shortens an existing pause or resets counted attempts.
python scripts/data_merging.py --set-spotify-cooldown 80000
```

Dry-run, checkpoint application, and local-only fills consume no HTTP attempts.
On a future 429, diagnostics include the available error reason and local counts
for the preceding 30 seconds and 15 minutes. An absent reason is reported as
unspecified; neither a long cooldown nor successful playback polling proves
whether a metadata failure was a rate limit or a quota restriction. Spotify
documents both [rolling-window rate limits](https://developer.spotify.com/documentation/web-api/concepts/rate-limits)
and [separate quota buckets](https://developer.spotify.com/documentation/web-api/concepts/quota-modes).

After enrichment, test extraction with your usual cookie option plus
`--retry-failed --limit 20 --acquisition-workers 1`. Filling metadata does not
reset the extraction attempt limit. Inspect rejected-candidate examples before
scaling up; offline regression fixtures do not establish the recovery rate on
your full catalog.

### Fill audio-search metadata directly in the backlog

Tracks absent from master also need metadata to acquire the right audio. This
offline command fills missing cells in `data/dsp_backlog.csv` directly:

```powershell
python scripts/data_merging.py --backfill-backlog-local --dry-run
python scripts/data_merging.py --backfill-backlog-local
```

It matches exact Spotify track IDs against the supported history CSV/JSON
exports and master catalog, preserving backlog row order, existing values,
attempt counts and error state. It supplies missing names/artists/albums,
confirmed recording durations and available metadata. Existing structured
`spotify_artist_names` lists from master are copied as lists, without guessing
artist boundaries from a comma-separated credit. Conflicting local candidates
are skipped and newly filled cells receive per-field source provenance.

Confirmed durations include history rows labeled `polling` or `live_polling`,
and explicit supported duration provenance. Unqualified durations are retained
only as `history_duration_candidate_ms`; playback time never becomes recording
duration. Original track/version labels are preserved. Audio extraction uses
the recovered `duration_ms`, `track_name`, `artist_name` and available Spotify
artist lists on its next run. No acoustic features are inferred from history.

By default this action never opens Spotify checkpoint files and never makes
API calls. It writes only the backlog, so it can run alongside the existing
`--backfill --fetch-only` job, which writes a separate checkpoint. Do not run it
alongside extraction or backlog API enrichment, because those also write the
backlog. Changing metadata does not alter an already-running job's in-memory
catalog. The master and all history/source files remain unchanged.

After fetch jobs finish, saved checkpoints can optionally contribute too:

```powershell
python scripts/data_merging.py --backfill-backlog-local --include-local-checkpoints --dry-run
python scripts/data_merging.py --backfill-backlog-local --include-local-checkpoints
```

Use `--backlog PATH` to select a different target, and `--local-source-root PATH`
for a source project root containing `data/` and `sensitive_info/`. A detected
change to the target during processing aborts the write. The action neither
inserts new backlog tracks nor resets extraction retries.

### Fill the master from all supported local sources

Run this before spending Spotify requests on missing metadata. Stop extraction
and other master-table writers before applying changes:

```powershell
# Preview all fills and conflicts; no files change and no Spotify calls occur.
python scripts/data_merging.py --backfill-local --dry-run

# Apply the same missing-only recovery to the master CSV.
python scripts/data_merging.py --backfill-local
```

This single offline command combines:

- Existing master aliases (`name`/`track_name`, `artists`/`artist_name`,
  `album`/`album_name`), year from a valid stored release date, and that date's
  represented precision. Derived precision describes the stored date's detail;
  its provenance does not claim Spotify supplied the precision.
- History CSV files whose names contain `history` or `listening`, under
  `sensitive_info/`, including the old `artist` column and history copies.
- Retained audio history JSON under `sensitive_info/history/`. These exports
  preserve album names that the normalized CSV dropped. Only explicit Spotify
  track URIs are matched; playback timestamps never become release years and
  `ms_played`/`listened_ms` never become recording durations.
- Saved version-1 `data/*.spotify-metadata.json` results, including the basic
  metadata checkpoint. Only supported metadata fields listed in the checkpoint
  are imported; catalog IDs must already exist in the selected master.
- Successful Spotify metadata saved in `data/dsp_backlog.csv`. Acquisition retry
  counters, error state, source files, and cached results are left unchanged.

Master aliases take priority, followed by agreement among external sources.
Conflicting nonempty candidates are skipped and reported per column. Existing
nonmissing master values, acoustic features/provenance, IDs, and row order are
preserved. No new tracks are inserted and no generic feature cleanup runs.
Only newly filled metadata cells receive `<field>_metadata_source` provenance.
The command uses column operations for master aliases and scans each supported
source once; it does not checkpoint a million local-only rows.

History `duration_ms` is treated as a confirmed recording duration only when
the row explicitly identifies polling or a recognized Spotify duration source.
Other positive history durations may have been estimated by the importer, so
unambiguous values go into **`history_duration_candidate_ms`**, never silently
into `duration_ms`. Saved Spotify responses and backlog durations with an explicit
`duration_ms_source` marker can fill `duration_ms` directly. A row-level Spotify
source marker is insufficient: API enrichment preserves an older nonmissing
duration. New backlog API fills now record field-specific duration provenance;
old unqualified backlog durations remain candidates. A consistent CSV value alone
does not establish that it was measured by Spotify.

This command works during a Spotify cooldown and never authenticates. It writes
the master once, atomically, only if cells changed; repeated unchanged runs do not
rewrite it. A detected concurrent master change aborts the save, but concurrent
catalog writers remain unsupported. Missing sources are skipped; malformed
supported source files abort before any master write.

Use `--master path/to/copy.csv` to select another output catalog. Optional
`--local-source-root path/to/project` selects a root containing `data/` and
`sensitive_info/` for source discovery; it defaults to this project. This is
also how offline tests isolate synthetic sources. Arbitrary datasets, audio
extraction, and network metadata requests are outside this command.

### Enrich tracks already in the master

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

`--fetch-only` queues only tracks that need Spotify lookups after checking existing
name aliases and release dates. It scans catalog columns using pandas operations,
then skips local-only fills, invalid Spotify IDs, completed checkpoint entries,
and known unavailable tracks before entering the fetch loop. The startup report
separates remaining API lookups from local-only fills. With `--fetch-only`,
`--limit` counts queued track lookups (individual retries and optional album/artist
requests can add HTTP calls). Add `--fetch-only` to a dry-run command to preview
that same API-only queue. Reading the whole catalog once is still necessary.
Local-only rows are not added to the fetch-only checkpoint; use normal backfill
when you also want equivalent metadata copied between existing columns.

To prioritize basic metadata while deferring year, popularity, ISRC, and date
precision, use:

```powershell
python scripts/data_merging.py --backfill --fetch-only --fields track_name artist_name album_name duration_ms explicit --checkpoint data/basic.spotify-metadata.json
```

Existing checkpoints remain compatible. Local-only records already saved by
older runs are retained; choose a new checkpoint filename if you prefer a fresh,
smaller checkpoint (prior fetched results will not be reused by the new file).

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
stops immediately and persists its cooldown, while other transient request
errors have bounded retries that each consume HTTP budget.
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
