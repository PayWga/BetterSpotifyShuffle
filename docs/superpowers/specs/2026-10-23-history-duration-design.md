# Completed-play duration estimates

The user authorized implementing the completed-play method evaluated against
saved Spotify durations. Extend the existing offline local backfill rather than
adding a second command or introducing history scans into API fetching.

Use raw Spotify audio exports only: `reason_end=trackdone`, positive integral
`ms_played`, a track URI, and a valid UTC event timestamp. Deduplicate by track
and timestamp across exports; conflicting copies of an event provide no evidence.
Take the median of distinct completed plays. Accept only with at least two
supporting events and at least 80% of events within 0.1% of the median. These are
consistency filters, not an accuracy guarantee. Never reverse imported completion
percentages, whose denominator can already be an estimate.

Apply confirmed local metadata first, then fill only remaining missing durations.
Record `duration_ms_metadata_source=history_completed_play_median`,
`duration_ms_is_estimated=True`, `duration_ms_estimate_observations`, and
`duration_ms_estimate_support`. Preserve existing values and all source files.
Evidence fields for a newly filled duration are refreshed together, replacing
orphaned flags/counts that cannot describe the newly calculated estimate.
The existing API gap selection then skips filled durations; other explicitly
requested missing fields can still trigger API requests. The same offline
mechanism serves master and backlog targets, with dry-run and concurrent-write
protection retained.

Alternatives: percentage inversion is circular for imported history; promoting
every single completed play has avoidable outliers. The selected repeated-play
median trades coverage for evidence. The user chose two supporting plays after
validation: all 40 additional confirmed tracks met the 0.1% tolerance; across
all 780 accepted validation tracks, 776 met it. No production data is applied during
implementation or testing.
