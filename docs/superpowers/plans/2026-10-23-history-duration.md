# Completed-play Duration Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Spotify duration lookups with labelled, consistent local estimates.
**Architecture:** Aggregate deduplicated completed plays while reading raw exports
in local_backfill; apply estimates after existing confirmed candidates. Keep API
fetching independent of history scans.
**Tech Stack:** Python, pandas, pytest.
**Spec:** ../specs/2026-10-23-history-duration-design.md

## Global Constraints

- Preserve nonmissing values, private inputs, and running processes.
- Tests use temporary synthetic catalogs only; no network calls.
- Two agreeing events minimum; 80% support within 0.1% of median.

## Review Focus

- Repeated exports must not inflate evidence.
- Conflicting copies, malformed values, and skips must not become durations.
- Confirmed local values must win over estimates.
- An estimated duration must not be reclassified as confirmed on reuse.
- Other requested missing fields must remain eligible for Spotify lookups.

### Task 1: Integrate estimates into existing offline workflow

**Files:** scripts/local_backfill.py, tests/test_local_backfill.py, README.md.
**Interface:** Existing run/backfill CLI signatures remain unchanged.

- [x] Add integration tests for accepted median/provenance, duplicate and invalid
  evidence rejection, conflicting timestamps, outlier support, confirmed priority,
  dry-run/idempotence, both targets, and API queue reduction.
- [x] Run new tests and confirm failures reflect missing estimation behavior.
- [x] Add bounded-to-relevant-IDs event collection and missing-only estimate fill.
- [x] Document thresholds, provenance, and offline-then-fetch commands.
- [x] Run targeted tests, then `python -m pytest -q`, and inspect the diff.
- [x] Review changes; leave them uncommitted for the user.

## Execution notes

- Initial integration tests: 4 failed / 13 passed before implementation, then
  17 passed after implementation. Initial full suite: 333 passed.
- Independent review found orphaned evidence could survive a newly filled
  duration. Added regression test (failed as expected), refreshed estimate
  evidence only for newly filled durations, then all 18 local-backfill tests
  passed. This deliberately updates stale evidence, never existing durations.
- Final full-suite verification: `python -m pytest -q --basetemp
  .test-runs/duration-estimate-final` -> 334 passed in 57.44s. Diff whitespace
  check passed. Real catalogs, history, and running processes were untouched.
- Follow-up authorized by user: lower support minimum to two. Updated tests for
  acceptance of exact/near matching pairs, single-play rejection, duplicate and
  conflict rejection at the new boundary, and backlog behavior. RED: 4 failed /
  17 passed with the old threshold. GREEN: 21 local-backfill tests passed. Final
  suite: `python -m pytest -q --basetemp .test-runs/two-play-duration-final` ->
  337 passed in 58.61s. Diff whitespace check passed; production data untouched.
