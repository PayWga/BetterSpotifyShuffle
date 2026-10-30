# Adaptive playback tracking

The shared tracker accepts normalized observations from a playback backend. The
existing polling daemon now uses it with one Spotify playback read per iteration.
No model scoring, artist metadata lookup, or queue lookup adds API reads to that
iteration. Windows media-session observation and SOCKS setup remain separate
backend/connection work; the current implementation uses the Web API for PC and
phone observation.

## Polling and credentials

Inspect options without authentication or network calls:

```powershell
python scripts/polling.py --help
```

Once the connection is configured, the existing entry point runs the observer:

```powershell
python scripts/polling.py --early-poll 15 --max-gap 75 --skip-threshold 0.25
```

The gateway reads `sensitive_info/spotify_credentials.json` and reuses
`sensitive_info/.spotify_cache`. Credentials are neither embedded in source nor
printed. API and OAuth HTTP requests pass through the same budgeted session;
SDK/transport automatic retries are disabled. The existing
`sensitive_info/spotify_request_budget.json` provides shared pacing/cooldown state
with metadata jobs. A blocked budget stops the observer instead of repeatedly
probing Spotify. Background enrichment is not run by this daemon. Independent
metadata jobs still share any persisted rests/cooldowns; there is no central
priority service coordinating separately launched processes yet.

Before acceptance, the default scheduled delay is at most 15 seconds. Once
measured listening reaches 25%, it becomes at most 75 seconds and never exceeds
predicted remaining song time. A two-second guard schedules a check just before
the predicted ending, preserving measured listening coverage for the historical
70% PLAY classifier. At the guard position the next check targets the remaining
time; an already-ended position uses a one-second retry rather than a zero-delay
loop. Paused/inactive playback uses 30 seconds; missing playback for 60 seconds
ends the observation neutrally. Network latency/cooldowns can extend actual gaps.

Acceptance is emitted once while the song remains active. Seeking does not add
listening time, pause/resume intervals contribute uncertainty rather than measured
listening, and network outages invalidate queue/timing continuity. A device or
playlist change requires reconciliation before inferring any intervening songs.

## Confidence and history

For a known submitted order, the entire gap between observations is a conservative
upper bound for each intervening track. It is deliberately not recorded as its
actual listening duration. Durations come from the catalog's supplied metadata;
unknown duration cannot establish early rejection.

| Evidence | Outcome | Weight |
| --- | --- | --- |
| Upper bound below the song's 25% threshold | Rejected | 1.0 |
| Upper bound reaches 25% but is below full duration | Weak rejection | 0.5 |
| Full listen possible, unknown duration, or broken order | Unresolved | 0.0 |

Long gaps can still prove early rejection for sufficiently long songs. Short gaps
can be ambiguous for short songs. Once accepted, an occurrence cannot receive a
later rejection. Pending duplicate track IDs do not establish occurrence identity;
consumed occurrences are retired so later legitimate repeats remain identifiable.

The legacy `spotify_listening_history.csv` schema and historical 70% PLAY / below
15% GHOST_SKIP rules are preserved. Only observed occurrences with measured
listening are written there. A transition whose upper bound cannot establish a
historical early skip uses `reason_end=uncertain_transition`, so unknown elapsed
listening cannot become a GHOST_SKIP. Fully inferred tracks are not fabricated as
zero-duration history rows. New evidence is written separately to private
`sensitive_info/playback_evidence.jsonl`, with UTC timestamps and fields including
`selection_id`, `inferred`, `weak`, `weight`, `gap_ms`, and `upper_bound_ms`.
Occurrence/selection IDs are local to a process, not globally persistent IDs.

## Sequencer coordination

`PlaybackSession` joins a sequencer to a tracker. It freezes a submitted queue
prefix separately from replannable suggestions. Register a submission only after
the playback transport acknowledges it; no playback submission is performed by
this stage. Starts inferred from verified progression commit variety fatigue just
like observed starts. Feedback changes only the supplying pool, once per selection.
External songs can enter confirmed context on acceptance without pool attribution.

The offline sequencing CLI supports these additional JSON-lines commands for
testing an adapter without Spotify:

```json
{"command":"submit","count":3}
{"command":"observe","continuity":true,"snapshot":{"track_id":"ACTUAL_FIRST_SUBMITTED_ID","duration_ms":180000,"progress_ms":0,"is_playing":true,"observed_at":0,"device_id":"pc","context_uri":"playlist:x"}}
{"command":"observe","continuity":true,"snapshot":{"track_id":"ACTUAL_FIRST_SUBMITTED_ID","duration_ms":180000,"progress_ms":45000,"is_playing":true,"observed_at":45,"device_id":"pc","context_uri":"playlist:x"}}
```

Use IDs from the emitted `submitted` list. Snapshot `observed_at` is strictly
increasing monotonic seconds, not a UTC event timestamp. `continuity=true` is an
explicit assertion from the adapter that the submitted order is still valid; it
is not inferred merely because we once planned those songs. The `interrupt`
command invalidates that assumption after an outage. Reconfirm the submitted
order through the adapter before resuming recovery. The standalone polling
daemon has no submitted-order source, so it cannot infer unheard queue entries.
Do not run a second standalone poller alongside an integrated session: feed one
observation stream to the tracker, logger, and sequencer.

An integrated adapter can pass its `PlaybackSession` as `tracker` to
`run_polling`, with a `continuity_check(snapshot)` callback returning a boolean
from its acknowledged queue state. That callback should not add another Spotify
request per iteration. The same loop then updates pool feedback, writes evidence,
and forwards observed historical events. The adapter must reconcile external
queue edits rather than permanently assuming continuity.

Manual offline `start`/`feedback` commands remain available, but must not be mixed
with an active submitted-playback session. `status` exposes current playback and
next polling delay in addition to the speculative queue.

## Validation

Tests use synthetic playback snapshots, fake Spotify/OAuth boundaries and
temporary outputs. They make no real Spotify requests and do not modify private
history, credentials, budget state or token caches:

```powershell
python -m pytest tests/test_playback.py tests/test_playback_session.py tests/test_playback_gateway.py -q --basetemp .test-runs/adaptive-check-UNIQUE
```

Use a fresh temporary base per run. This tests the integration logic, not actual
Spotify quota capacity or SOCKS connectivity.
