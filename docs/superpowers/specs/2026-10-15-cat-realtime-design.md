# CAT asynchronous mapping and GPU fields

The user approved the proposed two steps: remove fusion/field work from control,
then optimize it toward a 100 ms fresh-field budget. This extends the existing
online implementation. Work inline in the current dirty checkout, preserving
previous changes; no commit or extra approval gates are needed.

## Execution

Use one spawned worker process, never fork a CUDA context. The simulation thread
captures sensor rays/poses, submits a bounded latest-scan mailbox, polls completed
fields, and samples the latest acceptable field. Occupancy fusion, morphology,
FMM/GPU solve and field construction belong to the worker. Submission/publication
locks are nonblocking on the consumer; no per-step join or queue wait. Shared
memory carries bounded input/output arrays, avoiding large pickled field queues.
Explicit close stops/joins the worker and releases shared memory, including on
initialization or worker failure.

Use monotonic wall time for async capture/freshness, with original measurement
timestamps retained through build and publication. A new computation is not a
new measurement. Per-environment episode and goal generations reject obsolete
in-flight results. Reset invalidates only its slot; goal changes invalidate its
old field immediately. Latest-only input may drop superseded scans explicitly.
Worker errors are reported and fields fall back to the existing stop defaults.
Do not enlarge max_age to disguise slow fields. Startup/JIT is measured separately.

Keep the synchronous source path for numerical comparison and existing tests.
Expose execution/backend options explicitly in the CLI. Async headless playback
uses a 50 Hz wall-clock target and reports missed deadlines, not a realtime claim.
Viewers retain their native pacing. Training continues to reject online envs.

## Computation

Add an optional Warp ray-traversal kernel that preserves the existing DDA's
per-scan deduplication, hit precedence, boundary/invalid semantics. Keep Torch
reference support for CPU/float64; test CUDA parity before selecting Warp.

Cache fields when processed occupancy and goal are unchanged after a new valid
scan. Reuse must preserve measurement freshness and coverage separately from
build count, and never cross episode/goal generations.

Add an explicitly named GPU Eikonal solver/backend; do not rename graph distance
as FMM. Use source interface initialization, obstacle masks, signed distances,
gradients and progressive CAT guidance. CPU skfmm remains the oracle. Document
discretization/convergence differences and measured distance/direction errors;
reject nonconverged results. Validate obstacle detours, disconnected regions,
empty grids, occupied/outside goals and changed maps (including obstacle removal).
The source backend remains available independently of CUDA field dependencies.

## Acceptance

Tests must demonstrate nonblocking polling/submission while the worker is busy,
latest-only buffering, cleanup, stale/future/obsolete result rejection, per-slot
reset, goal switching and worker failure. Native depth/LiDAR async loops must
consume actual published fields. Existing source contracts remain intact.

Benchmark warmed single-env configured 126×126×56 grids with measured sensor
scans, changed occupancy/goal as well as cache hits. Record stages, P50/P95,
wall-clock field age, new-build count and foreground latency. The 100 ms target
is an acceptance target to measure; do not report success from cached repeats or
from unchanged 0.1-second scheduling alone. Final source comparisons, regression,
lint and an independent review are required.
