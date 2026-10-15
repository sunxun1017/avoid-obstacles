# CAT asynchronous and GPU implementation plan

> Execution: use superpowers:executing-plans inline, then one fresh final reviewer.

Goal: decouple online work from CAT control and measure optimized fresh builds
against 100 ms. Spec: `docs/superpowers/specs/2026-10-15-cat-realtime-design.md`.
Stack: Python multiprocessing/shared_memory, Torch, Warp 1.16, source scikit-fmm.

- [x] Task 1: bounded process transport and asynchronous field lifecycle.
  Create `perception/field_worker.py`, `perception/async_field.py`; tests
  `test_async_field.py`. Extend OnlineFieldMap with reuse accounting. Demonstrate
  busy/nonblocking, latest-only, original timestamps, reset/goal rejection,
  worker failure and cleanup before integrating into control.
- [x] Task 2: source-compatible Warp DDA fusion.
  Create `perception/occupancy_warp.py`, select via explicit mapper backend;
  test randomized/multiframe CPU-reference agreement, ties, invalids, clipping,
  duplicate hits and isolated resets. Benchmark fusion independently.
- [x] Task 3: GPU Eikonal fields.
  Create `perception/eikonal_warp.py` and `perception/cat_gpu.py`. Test source
  fixtures/analytic fields/detours/disconnected regions and convergence failure.
  Quantify source numerical error and optimize measured kernels.
- [x] Task 4: native/CLI integration and benchmark.
  Add execution/backend options to CAT provider/factory/CLI; explicit process
  cleanup, wall-clock pacing/freshness and source checkpoint compatibility.
  Add native async tests and reproducible profiling tool with source/GPU timings.
- [x] Task 5: full GPU regression, lint, independent review, fixes and docs.

## Review focus

An old episode/goal must not be resurrected by a late result. Failed/absent scans
must not renew age. Cached work must not masquerade as new computations. Process
start/close/errors must not leak children/shared memory. GPU solving must preserve
obstacle masks and reachability and expose approximation/convergence limits.

## Ledger and rulings

- Baseline measured last turn: 863.5 ms pipeline, 68.4 ms fusion, 11.4 ms
  morphology, 598.5 ms FMM and ~175 ms other CPU math; 5 warmed LiDAR repeats.
- Ruling: existing user authorization covers this implementation; preserve
  checkout/IDE context and use this file as ledger, without additional review
  approval or Git metadata mutation. Cost: changes remain uncommitted for review.
- Preflight: async publication consumes the same VoxelPotentialField/FieldSample
  as sync; epochs and timestamps belong to transport, not field mathematics.
  Warp/GPU imports stay in optional backends; no simulator imports in perception.
- Task 1: complete — cache/async lifecycle RED→GREEN; 16 focused tests pass,
  including original timestamps, future rejection, reset/goal generations,
  busy mailboxes and terminated worker cleanup.
- Task 2: complete — Warp traversal RED→GREEN; randomized 3-slot/3-frame CPU
  parity and boundary/hit precedence pass. Full LiDAR fusion ~0.33 ms vs ~68 ms.
- Task 3: complete — 3 CUDA numerical tests pass (planar, disconnected, source
  fixture, obstacle detour/removal, convergence failure). 126×126×56 fresh fields
  ~28–40 ms after warmup. CPU oracle is unchanged; GPU remains explicitly approximate.
- Task 4: in progress — actual native depth/LiDAR async tests pass (2/2), CLI
  exposes execution/backends and paces wall-clock control. Profiling separates
  initialization, fresh builds, cache hits and foreground transfer cost.
- Ruling: library defaults stay sync/source for compatibility; the online CLI
  defaults to async/CUDA/Warp with explicit source fallback. Cost: CLI numerical
  results have documented Eikonal discretization differences from CPU FMM.
- Task 4: complete — both sensors measured on configured grid, 20/20 fresh async
  publications per 2-second window, no dropped scans after separate initialization.
  LiDAR GPU pipeline P50/P95 55.9/59.2 ms; depth 44.7/47.9 ms. Worker P95
  70.2/63.9 ms; foreground P95 12.64/10.40 ms. JSON evidence in docs/benchmarks.
  Source distance/direction errors, startup and synthetic changed-map workload
  are documented; these do not establish navigation success or hard realtime.
- Task 5: in progress — full CUDA regression and default CLI smoke, then one
  fresh whole-change reviewer as required by the executing-plans skill.
- Verification: 220/220 CUDA regression tests passed in 51.70 s, excluding the
  pre-existing test_scan.py demonstration script. Changed files pass Ruff.
- Default CLI smoke: 150 steps, finite observations, final field available and no
  worker errors. Full step mean 35.35 ms/max 85.36 ms; 123 missed 20 ms deadlines.
  Documented separately: mapping replay achieves 10 Hz, full simulation does not
  establish 50 Hz. Zero-action smoke is not navigation validation.
- Final review: one fresh gpt-6-astra reviewer identified three Important findings,
  no Critical/Minor findings: duplicate worker cadence can starve phase-offset
  slots, shared-memory/process-constructor failure cleanup, and native acceptance
  allowing stale finite defaults. All accepted into the single final fix pass.
- Final: fixed duplicate cadence — dropped/overwritten two-slot offset regression
  RED ([1,6] instead of [5,6]) → GREEN. Parent is sole admission authority;
  worker processes preselected ray masks without a second independent cadence.
- Final: fixed startup cleanup — injected semaphore and process-construction
  failures RED→GREEN; allocated shared memory is no longer attachable afterward.
  CPU async/cache suite 16/16 passed after these fixes.
- Final: Ruling: trained-policy closed-loop navigation success stays outside this
  implementation's acceptance, as in the approved scope — no trained checkpoint
  was supplied; smoke/numerical/latency tests cannot establish navigation success.
  Cost: GPU approximation still needs closed-loop policy evaluation before claiming
  source-equivalent navigation performance.
- Final: fixed native acceptance — explicitly pause after publication to force a
  stale first field (RED: guidance_valid false); advance native observation clocks
  and wait for fresh current/delayed queries, asserting valid age and nonzero actor
  navigation command (GREEN: both depth/LiDAR, 2/2 in 10.65 s).
- Final reviewer reported no deferred Minor findings. No second review dispatch;
  fixes are verified with reproducing tests and the final full regression.
- Task 5: complete — final full CUDA suite 223/223 passed in 53.14 s;
  all three Important review findings fixed with reproducing tests. Final Ruff
  and git diff --check pass. No deferred Minor findings. Existing user edits
  remain in the current checkout; no commit, merge, push or workspace deletion.
- Final: fixed findings verified against full suite 223/223: phase-offset scan
  admission, constructor-failure cleanup, native stale-to-valid observation recovery.
