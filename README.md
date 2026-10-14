# ShuffleRec

Personal music recommendation and sequencing experiments using Spotify listening
history, track metadata and audio features, and a SASRec-inspired Transformer.

- `core/polling.py` collects plays and skips from Spotify playback.
- `core/import_etended_spotify_history.py` imports extended listening history.
- `core/data_merging.py` manages the master catalog and Spotify metadata sync.
- `core/metadata_fetcher.py` tracks play/skip counts and library age.
- `notebooks/ShuffleRec.ipynb` prepares sequences and trains next-item and preference models.
- `core/feature_extraction.py` finds audio and fills missing acoustic measurements.

## Audio feature extraction

Install Python dependencies with `python -m pip install -r requirements-audio.txt`.
Install FFmpeg and ffprobe on PATH. Node.js 22+ or Deno 2.3+ is recommended for
YouTube extraction. The script automatically enables either installed runtime.
The `yt-dlp[default]` dependency includes its JavaScript challenge solver; see the
[upstream setup guide](https://github.com/yt-dlp/yt-dlp/wiki/EJS).

From the repository root:

```powershell
# Inspect up to 10 entries using existing audio only; no files or downloads change.
python core/feature_extraction.py --dry-run --limit 10

# Search/download missing audio and extract features for up to 10 tracks.
python core/feature_extraction.py --limit 10

# Process all eligible tracks without online search.
python core/feature_extraction.py --local-only

# Include entries that reached the attempt cap, after resolving their errors.
python core/feature_extraction.py --retry-failed --limit 10
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

The older data manager and notebook still use legacy `-1` missing-value
sentinels. This extractor deliberately preserves existing `-1` values because
they are nonmissing CSV values. A separate migration to NaN and explicit model
availability/sequence masks is needed before those older components follow the
current project conventions. The notebook does not yet consume `local_*` features.

## Tests

```powershell
python -m pip install pytest
python -m pytest -q
```

Tests use temporary catalogs and generated audio. Online acquisition is simulated
at the yt-dlp boundary; the default suite makes no network requests and never uses
private listening data or the real catalog/backlog. In a restricted environment,
choose a fresh writable `--basetemp` directory under `.test-runs`.
