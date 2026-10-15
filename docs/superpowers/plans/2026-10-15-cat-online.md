# CAT online implementation plan

Spec: `docs/superpowers/specs/2026-10-15-cat-online-design.md`.
User explicitly requested implementation; execute inline without further gates.

1. Extract pinned deployment oracle functions, save compact fixtures with hashes.
   Add failing comparisons, then implement GPU-capable obstacle preprocessing
   and source-equivalent deployment field construction.
2. Add tests for online lifecycle (updates, missing/stale data, goals, partial
   reset), then implement independent online field provider and mapper facade.
3. Add ray acquisition tests, implement depth/LiDAR adapters and CAT native hooks.
   Preserve training behavior and isolate deployment defaults/diagnostics.
4. Implement explicit online playback and inference compatibility validation;
   add CLI and native smoke tests, document source differences and commands.
5. Run focused CPU and available CUDA regression, lint changed files, request
   independent final review, resolve important findings and record evidence.

## Progress and rulings

- Source files downloaded read-only to `/tmp/cat-online-source` from pinned SHA.
- Ruling: work in existing `feat/g1-cat-source` checkout to preserve IDE context
  and user edits; do not write Git metadata or commit unrelated changes.
- Ruling: source CPU Fast Marching remains the numerical oracle/backend; GPU
  acceleration applies to ray fusion, occupancy preprocessing and querying.
- Ruling: preserve occupancy-only source semantics explicitly; measured known
  masks remain diagnostics and do not silently change CAT's navigation field.
- Initial CUDA detection in sandbox returned unavailable; native validation
  requires a separate check of actual device access.

## Execution evidence

- Task 1 complete: missing-module RED, pinned author morphology/field fixtures
  GREEN; CPU 3 passed/1 CUDA skip; subsequent GPU run includes all 4 passes.
- Task 2 complete: lifecycle RED→GREEN; out-of-grid scan falsely renewing fields
  found and reproduced. Mapper now returns which slots actually measured map
  cells. Generic mapping/lifecycle suite: 22 passed, 2 CUDA skips in sandbox.
- Task 3 complete: missing adapter RED, range/filter tests GREEN (3); native
  depth and LiDAR mapping and partial reset GREEN on actual CUDA (2).
- Task 4 complete: checkpoint/CLI/training-boundary RED→GREEN. Actual sensorless
  model checkpoint loads through both online sensor modes. Focused GPU suite
  (native, policy, deployment, lifecycle, occupancy): 35 passed in 24.63 s.
- Task 5: full GPU regression passed 204 tests in 43.70 s, no skips
  (`pytest --ignore=tests/test_scan.py -q`, actual CUDA access). Final independent
  review in progress. CLI --help passes; source fixtures regenerate exactly.
- CPU sandbox regression: 179 passed, 19 skipped, 1 failure in the existing
  test_register_gpu_status assertion (GPU inaccessible inside sandbox). Native
  checks run outside sandbox confirm an RTX 4060 Laptop GPU. Full regression
  uses GPU access. Exclude existing tests/test_scan.py: it is a top-level CUDA
  demo that writes terrain_scan.npy during collection, not a unit test.
- Ruling: retain source sampler's X/Z weight/corner mismatch, including grid
  boundary discontinuities — required source fidelity — costs smoothness.
- Ruling: native multi-scene batches require identical grid origin/shape and
  0.04 m resolution — source morphology is grid-index-based — costs flexibility;
  generic perception still accepts independent mapper origins.
- Ruling: no automatic Git metadata changes/commits or skill workspace cleanup;
  this plan itself is the execution ledger in the existing dirty checkout.
  User changes in examples/depth_mapping_demo.py, robots/g1.py, tasks/field_assets.py
  and untracked src/embodied_learning/README.md were preserved.
- Measured one LiDAR smoke rebuild at 0.8169 s for one 126×126×56 map;
  document synchronous CPU FMM cost and do not claim wall-clock 10 Hz.
- Depth CLI smoke also passed: 3072 valid rays, 144537 observed cells, 840
  occupied cells, available field, 0.8585 s last rebuild.
- Final independent review: 1 Important finding, no Critical/Minor findings.
  Current field invalidity could be overridden by valid delayed odometry fields.
  Reproduced with a real online provider in 3 RED tests (current outside,
  delayed outside, stale); fixed by combined deployment validity gating of
  current/delayed commands and field features after gait handling. All 3 GREEN,
  including recovery/other-slot isolation; source task tests remain GREEN.
- Final review rulings: synchronous timing remains a documented limitation;
  rolling maps/localization/deskew/hardware transport remain outside this task
  and are needed before real-device integration. Unknown traversability and
  interpolation discontinuities remain explicit source-fidelity choices.
  No trained navigation success claim is made; smoke verifies wiring only.
  Existing user edits were excluded from review and preserved.
- Finish: preserve the current feature checkout and uncommitted changes for
  IDE review; no merge, push, integration menu or cleanup is needed for the
  user's implementation request. Final full-suite result follows below.
- Task 5 complete: final GPU regression after review fix: **207 passed in
  42.16 s**, no skips (`pytest -p no:cacheprovider --ignore=tests/test_scan.py -q`).
  Ruff on all changed Python files and `git diff --check` both pass. No deferred
  review minors. Source reference regeneration was numerically exact.
