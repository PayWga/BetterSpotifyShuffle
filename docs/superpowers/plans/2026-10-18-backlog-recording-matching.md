# Backlog recording matching implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Recover real metadata/title-format failures without confusing artists, languages or recording versions.

**Architecture:** Reuse spotify_metadata.track_metadata and request handling and the existing data_merging CLI to enrich backlog rows directly. Keep missing-only updates separate from extraction. Introduce structured title parsing within the existing matcher, preserving remix/version guards and source provenance.

**Tech stack:** Existing Python, pandas, Spotipy, yt-dlp; offline pytest fixtures.

**Spec:** User-approved scope in this chat: backlog metadata enrichment plus structured matching, grounded in supplied failed examples. Requested remixes remain eligible; originals must not borrow remix audio.

## Constraints

- Preserve existing nonmissing metadata, measurements, IDs, attempts and unrelated Git changes.
- No real credentials, Spotify requests, audio downloads or private CSV writes during testing.
- UTC provenance; backlog track_id and master id stay distinct.
- Refuse conflicting catalog writers. No deletion; enriched metadata fills only missing cells when extraction saves a master row.

## Tasks

1. Backlog enrichment
   - [x] Write failing tests for absent-master tracks, missing-only fields, preserved text/counters, dry run, auth failure and concurrent updates.
   - [x] Implement a directly callable backlog enrichment function and --backfill-backlog CLI, reusing track mapping/auth/retries.
   - [x] Save incremental checkpoints to the backlog using the existing writer and conflict guard; allow reruns to skip completed metadata.
2. Structured matching
   - [x] Add the real Japanese Version/Ver., with/feat., bilingual-live and lyric-branding examples as failing tests.
   - [x] Separate recording labels and featured credits from title identity, canonicalize language/version aliases, and preserve explicit language/live/remix conflicts.
   - [x] Use additional Spotify artist credits without replacing preserved backlog identity; retain metadata/provenance through extraction.
3. Verification and documentation
   - [x] Run full offline regression suite and targeted review.
   - [x] Document dry-run, enrichment and small extraction commands; clearly distinguish offline checks from real recovery validation.

## Review focus

- Japanese uploads cannot substitute an unmarked/Korean recording.
- Live requests cannot accept studio audio, or vice versa.
- Featured credits cannot conceal a conflicting named remix.
- Spotify relinking cannot attach a different requested recording's metadata.
- API failures/conflicting writers must preserve completed checkpoints and user data.

## Validation

- Full offline suite: 270 passed (41.32s).
- Review found literal numeric NaN duration handling; failing regressions reproduced
  the issue before correction, and now pass.
- Additional fixtures verify valid/invalid Spotify relinks and the 25-request
  checkpoint before the next API request.
- ISRC is optional for resume eligibility so unavailable ISRC does not trigger
  endless refetches; duration and both Spotify identity fields determine eligibility.
- No real API requests, downloads, credentials or private dataset writes were used.
- Real backlog recovery rate remains to be measured after user-run enrichment.
