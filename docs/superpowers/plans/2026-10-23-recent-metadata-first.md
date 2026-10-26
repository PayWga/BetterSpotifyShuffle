# Recent Listening Metadata Priority

The user selected most recently listened to (not newest release). Add opt-in
`--recent-first` to master metadata backfill and `--history PATH` to select a
listening CSV, defaulting to the existing HISTORY_DB. API/library parameters are
`recent_first=False, history_path=None`; the data_merging wrapper supplies the
default history path only when enabled. Without the flag retain catalog order.

Read only track_id and timestamp in chunks, filter to remaining eligible IDs,
normalize mixed timestamp formats to UTC, and aggregate latest timestamp per
track. Sort descending, stably: ties/unknown dates retain catalog order. Require
both columns; invalid timestamps provide no priority. Fail on missing/malformed
selected history before authentication or writes. Do not open history when
priority is off or the queue is empty. Explicit naive history timestamps are
interpreted as UTC, following the project convention.

Apply completed/unavailable/API-gap filtering before ranking, then apply --limit
in the existing fetch loop. Do not change checkpoint format/options validation,
catalog row order, pacing, source files, or active processes. Resume can enable
priority on an existing compatible checkpoint. Print aggregate ordering coverage
in dry-run and actual runs, without exposing events or IDs.

- [x] Add synthetic tests for latest event vs first event, UTC/mixed formats,
  stable ties/unknown dates, limit+resume, missing history/schema, CLI wiring,
  default opt-out and no writes in dry-run.
- [x] Run RED, implement minimal queue-ranking helper and CLI wiring, run GREEN.
- [x] Document the flag and duration-only command, run full suite and diff check.

No commits or production-data writes are part of this change.

Verification: RED 6 failed/20 passed before implementation; GREEN 26 targeted
tests passed. Full suite `python -m pytest -q --basetemp
.test-runs/recent-metadata-suite`: 343 passed in 64.00s. Scoped diff whitespace
check passed. Independent review found no correctness issues. A real CLI dry-run
found 2,517 remaining duration lookups, all with usable listening timestamps;
no authentication, requests, or writes occurred. Active jobs were untouched.
