# Live Spotify pipeline

The live entry point connects the current Spotify playlist to local model
inference, adaptive sequencing, queue additions and playback feedback. It uses
the existing private credentials, proxy settings, token cache and request budget.
No playlist CSV is needed.

## Start

Install the existing model dependencies and Spotify transport dependencies:

```powershell
python -m pip install -r requirements-model.txt -r requirements-spotify.txt
```

Start a playlist in Spotify on your PC or phone, then from the repository root:

```powershell
# Read the current playlist, load the model, and preview without queue changes.
python -m shufflerec.live --dry-run --once

# Run the complete pipeline continuously.
python -m shufflerec.live
```

The private `sensitive_info/spotify_pipeline.json` selects the checkpoint and
runtime defaults. The selected checkpoint has been configured there. Its relative
paths resolve from the settings file's directory. CLI `--checkpoint`, `--catalog`
and `--buffer-size` override the saved values. Example settings using placeholders:

```json
{
  "checkpoint": "model_runs/YOUR_RUN/model.pt",
  "catalog": "../data/master_track_table.csv",
  "buffer_size": 5,
  "follow_playlist": true,
  "playlist_refresh_seconds": 300,
  "request_interval_seconds": 1.0,
  "seed": 42,
  "sequencer": {"queue_size": 5, "early_skip_threshold": 0.25},
  "polling": {"early_poll_seconds": 15, "max_gap_seconds": 75, "idle_poll_seconds": 30}
}
```

Queue modification needs `user-modify-playback-state` and playlist reading needs
the private/collaborative playlist scopes. Spotify may ask you to authorize those
additional permissions. Your account and active device must permit queue changes.
The current playlist-item endpoint documents owner/collaborator restrictions;
inaccessible playlists produce `source_unavailable`, not a fallback to a different
catalog. See [Spotify's playlist-item reference](https://developer.spotify.com/documentation/web-api/reference/get-playlists-items).

`--proxy` and `--no-proxy` override `sensitive_info/spotify_proxy.json` for the run.
OAuth token and Web API requests share that routing. `--help` makes no API calls.
`--dry-run` still performs authenticated reads, which consume request budget and
can refresh the token cache; it does not submit songs or write listening/evidence
logs. Stop with Ctrl+C. Already queued songs remain in Spotify.

## Behavior

The pipeline automatically follows changes to the playback playlist and targets
five of its acknowledged upcoming songs. Existing manually queued songs remain
in place. It does not issue playback-start, transfer, skip, clear-queue or playlist
editing commands. When playback is paused, inactive, non-playlist, or on a
restricted device, it withholds additions and keeps observing as appropriate.

Songs already submitted to Spotify are committed. Acceptance or rejection changes
the unsubmitted plan and future replenishment; it cannot freely replace previously
submitted songs. Existing entries, including old-source submissions after a
playlist switch, can therefore play before newly generated songs. Short playlists
may yield a shorter buffer. The live planner can relax recent-play exclusions
when exhausted, while preserving current-track and pending-duplicate exclusions.

One worker handles model preparation and planning. The checkpoint and fitted
preprocessing stay frozen. Playlist response metadata only fills missing values
in an in-memory catalog; the master CSV is never updated by this pipeline.
Model scoring is recomputed for each planned slot. Committed future selections
and identifiable manual queue entries form provisional future context; those
unheard entries are not confirmed listening history and earn no feedback.

The controller owns mutable state. Source generation and feedback revision tags
discard obsolete background results, including switching away from and back to
the same playlist during a load. Obsolete scans and shutdown stop fetching at page
boundaries; an already running HTTP request may wait for its timeout.

Playlist contents are loaded in pages of 50 and cached by snapshot version.
Unchanged playlists are checked every five minutes by default. API calls are
serialized and spaced by the configured minimum; all still honor the shared
request budget and cooldowns. Queue snapshots are read around transitions,
submission batches, device/source changes and uncertain results, rather than on
every unchanged playback poll. Background work waking up does not itself poll.

Playback uses the [adaptive tracker](adaptive-playback.md): up to 15 seconds before
25% acceptance, then up to 75 seconds capped by remaining duration, with a
two-second pre-end guard. Full inferred early skips receive full negative weight,
weak inferred skips receive half weight, and unresolved transitions receive none.
Once accepted, a song cannot be rewarded twice or later penalized. Device changes,
outages and queue mismatches invalidate the relevant inference assumptions.

Queue evidence cannot reveal every unobserved manual edit. Reconciliation verifies
the visible order and suppresses unsupported attribution; it does not claim an
exact playback log. Unavailable features and unknown durations retain their masks
and uncertainty.

## Command recovery and status

Before an add request, the controller records an intent in private
`sensitive_info/queue_commands.jsonl`, then records acknowledgement. Requests are
sent serially; no failed POST is blindly retried. If the response is lost, the
controller checks queue evidence. A newly visible candidate can resolve the
command without a duplicate. If the outcome cannot be established, additions
pause with `submission_uncertain` while observations continue.

To accept that unknown outcome after checking Spotify, stop the process and run:

```powershell
python -m shufflerec.live --acknowledge-uncertain
```

Acknowledgement abandons the uncertain attribution rather than replaying the
request. A visible song is retained as an unowned buffer entry; an absent song is
not rewarded or punished. The controller avoids recommending that uncertain track
again during this run. This operation changes the private command journal, so it
cannot be combined with dry-run.

On restart, journaled queued entries still visible in Spotify count against the
buffer, but do not receive invented feedback from the previous process. Model
preferences, fatigue and confirmed context remain session-local. Explicit 400/
401/403/404 rejections are recorded as rejected commands, not ambiguous successes.
Permission failures stop additions. Reported cooldowns stop the pipeline until a
later restart; a proxy mode change does not reset the shared budget.

Console status includes state, playlist ID and upcoming count. Useful states:
`loading_source`, `running`, `preview`, `waiting_for_playlist`,
`awaiting_queue_confirmation`, `submission_uncertain`, `source_unavailable`,
`playback_permission_required`, `authorization_required`, and `spotify_cooldown`.
Queue confirmations are retried at bounded polling intervals if Spotify's first
response has not yet reflected a successful addition.

Historical CSV and evidence JSONL behavior is preserved. Only observed measured
listens go to `spotify_listening_history.csv`; inferred bounds stay in
`playback_evidence.jsonl`. Uncertain elapsed time is never manufactured as a
historical GHOST_SKIP. Both files and the command journal contain private IDs and
listening information and remain under ignored `sensitive_info`.

A shared `playback_owner.lock` lease prevents starting this pipeline and the old
standalone poller at the same time. Do not run a separate `scripts/polling.py`
instance alongside it. Windows local media-session observation remains future
work; the live pipeline currently observes both PC and phone through Web API.

## Validation

Tests use a real synthetic checkpoint with a simulated Spotify client to exercise
playlist loading, inference, sequencing, queue writes, rapid skips, feedback,
replenishment, dry-run, source changes, restarts and ambiguous responses. They do
not contact Spotify or change the real queue. The selected personal checkpoint
also passed a read-only inference smoke test on 100 catalog rows without printing
track IDs or rewriting source files.
