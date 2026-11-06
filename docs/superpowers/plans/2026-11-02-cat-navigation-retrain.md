# Root navigation correction and 800-update trial

> **For agentic workers:** Use superpowers:executing-plans inline. Test behavior before implementation.

**Goal:** Fix demonstrated root/pelvis reference mismatch and amplification of weak goal fields, preserve risk blending and original smoothness, train randomized stage4 for800 additional updates.

**Architecture:** Opt-in arrival_control=root_planar_blend_v2. Query the base field at the root for projected navigation. In a clear goal neighborhood smoothly mix with an analytic horizontal root-to-goal command: clearance blend smoothstep .3-.5m, distance blend full inside .25m/off outside .35m. Analytic command has speed=min(1.05*distance,.315), zero inside existing8cm stop radius. Do not normalize or amplify weak field vectors in this path. Existing root_linear_v1 unchanged; no phase or stop hysteresis changes this trial.

**Tech Stack:** Existing Torch/mjlab, generic train_cat_curriculum entry, 9x9 offline route pools.

- [x] RED tests: weak negative field near goal must not produce opposite command; finite/continuous mixing near bounds and obstacle gate; zero-distance; no change legacy profile. Native initial/current root query and strict transfer guards.
- [x] GREEN: pure command helper, source task opt-in profile, root query via base bank (not foot override). Explicit SHA-bound Actor-only adapter allows declared arrival/smoothness/route-cache changes, robot fingerprint/risk/PD/obstacle geometry verified. Strict full resume unchanged.
- [x] Verify tests/lint, exact logged negative-input regressions; GPU empty-scene frozen-policy experiment comparing old/corrected with source observation noise, random PD and RFI retained and pushes suppressed for isolation; formal training retains pushes. Record startup weights/risk/route diversity.
- [x] Snapshot immutable source/new run directory; parent latest risk600, Actor only, fresh matched Critic/Adam. Candidate and old-navigation control each800 updates, both original smoothness and9x9 routes. GPUs5/6, no0.40update chunks,256evaluation plus fixed-route and empty navigation diagnostics; final independent seed. Preserve all older runs; no promotion.
- [x] Verify actual first updates, parent Actor hashes, losses and workers before reporting launch.
