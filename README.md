# ShuffleRec

Personal music recommendation and sequencing experiments using Spotify listening
history, track metadata and audio features, and a SASRec-inspired Transformer.

- `scripts/polling.py` collects plays and skips from Spotify playback.
- `scripts/import_etended_spotify_history.py` imports extended listening history.
- `scripts/data_merging.py` manages the master catalog and Spotify metadata sync.
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
matches until a download succeeds. These checks are heuristic and cannot prove
that a result is the exact Spotify recording. Use recordings with appropriate
permissions, and inspect cached `<track_id>.source.json` files when reviewing matches.
Downloaded audio is converted to WAV and retained for retries; allow disk space
for full recordings even though only the first 60 seconds are analyzed.
Each candidate downloads into a separate staging directory. Failed staging files
are retained for inspection; only successful WAV conversions and their source
sidecars are published into the cache root.

The extractor estimates `tempo`, `key`, and `mode`, and adds separate descriptors:
`local_rms_db`, `local_onset_strength`, `local_spectral_centroid` (Hz),
`local_spectral_contrast`, and `local_beat_regularity` (0–1). They are **not**
replacements for Spotify energy, loudness, danceability, or instrumentalness.
Unreliable measurements remain NaN. Every newly filled feature has a
`<feature>_source` extractor version and `<feature>_audio_source` path/URL.
Existing nonmissing values and unrelated metadata are preserved.

Each run visits eligible backlog entries once. Failed entries retain an attempt
count and error reason; successful entries are removed only after their master
updates are saved. `--batch-size` defaults to 10 and `--max-attempts` to 3.
CSV writes use atomic replacement, but concurrent catalog/backlog writers are
not supported. Back up the catalog before large runs.

The extractor deliberately preserves existing `-1` values because they are
nonmissing CSV values. Use the placeholder cleanup below to normalize known
unknown values to NaN before extraction. The notebook still needs explicit model
availability/sequence masks and does not yet consume `local_*` features.

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
