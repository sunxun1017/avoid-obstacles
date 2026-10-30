# CAT Contact Force Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans, test-driven-development and verification-before-completion.

Goal: preserve baseline and all bug fixes while evaluating a measured contact-force excess reward.
Spec: docs/superpowers/specs/2026-10-30-cat-contact-force-design.md.

## Review focus
- Independent baseline model validation permits sensor-field changes only, rejects dynamics changes.
- Every2ms sample exactly once; peak survives last substep, repeated refresh and local reset.
- Sensor caches must not reuse old force after sense; no extra forward/RNG.
- Baseline remains old reward, descent penalty disabled; actor-only transfer cannot mask unrelated schema changes.

## Task
- [ ] Write failing tests for threshold/normalization/force shape, control-peak and reset, native sensor call order, physical-model validation, default reward identity, exact opt-in reward, strict transfer/resume.
- [ ] Implement cat_contact_force.py, native measurement setup, controller lifecycle, reward and metadata hooks; no other reward changes.
- [ ] Run targeted and full tests, fresh code review, fix important findings with RED→GREEN evidence.
- [ ] GPU verify physical identity, parent Actor, identical fresh Critic/Adam and nonzero neutral-phase force penalty; then launch matched128update jobs with per32 evaluation and retained raw traces.
- [ ] Integrate and commit only task files preserving user dirty config, attempt existing-authorized push; save concrete status.
