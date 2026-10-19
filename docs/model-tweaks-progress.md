# Model tweaks progress — docs/model-tweaks-design.md

- Approved in-chat plan and user instruction are the implementation brief.
- Existing local .gitignore edit must remain untouched.
- No commits, merges, network jobs, or real dataset writes are planned.
- Ruling: use the enriched schema with optional-column handling, not a copy of
  the old catalog schema; this avoids a second model-input redesign after merge.
- Ruling: current metadata cannot establish historical catalog availability;
  evaluation will disclose this rather than promise point-in-time metadata.

Stages:
1. Preprocessing — six RED failures, then six passing tests.
2. Chronological datasets — six RED failures, then twelve cumulative passes.
3. Model encoder — six RED failures, then eighteen cumulative passes.
4. Training, evaluation and artifacts — five RED failures, then twenty-three passes.
5. Audit, experiments and notebook — three RED failures, then three workflow passes.
6. Final verification — 29 model tests passed; compileall and git diff --check
   passed. Full pytest still fails collection on the pre-existing core import.

Baseline: full pytest collection fails in existing test_feature_extraction.py,
which imports core although this checkout contains scripts. Preserve that file.

Ruling: notebook candidate scope defaults to history-matched catalog tracks for
a practical personal-library evaluation; full catalog remains configurable.
Report the retrospective snapshot limitation and actual candidate counts.

Notebook fixture initially exhausted its three-track negative pool at the last
context. Added a fourth synthetic neutral history track; source CSV byte checks
and checkpoint reload tests pass. No real training has run.

Independent review found two P2 issues, both reproduced with RED tests and fixed:
- Validation identity now includes timestamps, preventing a test-period repeated
  sequence from passing as a validation example.
- Year-precision numeric dates are parsed as calendar years, not epoch values;
  invalid year sentinels can fall back to valid release_year aliases.
Reviewer rechecked both changes: ten focused tests passed, no remaining findings.

Read-only preparation on the current older catalog:
- Training: 4,993 sessions, 36,980 next-item examples, 1,197 skip pairs.
- Validation: 624 sessions, 3,058 next-item examples.
- Test: 625 sessions, 3,592 next-item examples (counts only; not scored).
- History-matched usable candidates: 2,107.
- Default model: 614,860 parameters versus previous 597,504 (about +2.9%).
- Extended local/artist-genre columns absent in this checkout; enriched-schema
  tests cover them. Do not interpret these counts as the enriched branch's data.

File-path comparison: all 17 implementation/document/test paths have zero
overlap with extracting_features changes. Existing .gitignore edits are preserved.
No real training, final-test scoring, network requests, source data writes,
commits, or merges were performed. Synthetic reviewer outputs were preserved
under ignored .test-runs, not deleted.
