# 8cm/10cm stop hysteresis and coordinated transition

User approved implementation and a NEW800-update trial. Retain original smoothness, risk blend, root navigation fixes and9x9 routes. Freeze/archive old trial checkpoints; the old trial completed naturally; no old workers needed termination.

Opt-in task.stop_transition version stop8_resume10_v1, stop8cm/release10cm, normal fade0.2s. Strong dynamics (horizontal speed>.35m/s or gyro norm>.8rad/s) release the position latch when a nonzero source navigation command exists, and use immediate source-style recovery phase/GF to avoid starving necessary balance steps. These are explicitly experimental guards, not a certified capture-step controller.

New gait state keeps a latent anti-phase clock across normal restarts, fades phase features toward standing and move flag over0.2s, and slews the walking GF vector toward its new direction/zero without normalization. Emergency starts retain immediate source phase reset; initial walking observations retain source-compatible initialization. Both nominal gait/reward masks and observations use declared transition state; old profile untouched.

Tasks:
- [x] Pure-state RED/GREEN: exact8/10cm latch, emergency release, phase/GF continuity, partial reset isolation, no RNG, decay to standing.
- [x] Integrate optional state and Actor/Critic phase features; snapshot/restore every new per-slot state for reset refreshes; strict SHA-only transfer for optional task.stop_transition and no other contract changes.
- [x] Full tests/review; frozen corrected800 randomized-route256-case checks with full perturbations; direct state tests cover the8/10cm quiet latch. Treat new observation semantics as experimental; no claim all jitter solved.
- [x] Create immutable new800 trial from validated original-model checkpoint, fresh Critic/Adam and matching controls; original smoothness,9x9 routes, source PD/noise/RFI/push. Preserve old outputs and stop only replaced own workers. GPU availability check, nonzero only.
- [x] Verify fresh formal initialization, actual losses and source hashes; document launch.
