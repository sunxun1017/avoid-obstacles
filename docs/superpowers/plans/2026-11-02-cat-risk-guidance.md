# Risk-guidance implementation plan

1. Pure risk/blend schema and exact Actor adapter. Write RED tests for geometry/risk/prediction/amplitude/guards, implement, verify GREEN.
2. Add opt-in risk reward and task integration before reset, pass through native/common training entry. RED tests for unchanged non-GF rewards, progress gate, no x threshold, Actor162/Critic250 and untouched command. Defaultgoldens/fullsuite.
3. Independent code review, fix substantive findings with regression tests, commit branch.
4. Snapshot hashed source to new remote trial, reuse immutable stage4 assets. GPU smoke must verify alpha range, effective observation/reward changes, exact parentActor, unchanged PD/physics and unrelated contract; matched Critic seeds.
5. Launch candidate/control600updates via40updatechunks, separate status/history/evaluations; verify real first updates/finite losses and initialization. Existing25x400trial stays untouched. Record paths, branch commit, initialization limitations and budget.

Verification ledger:
- Pure/integration RED observed before implementation;18 new tests GREEN.
- Independent read-only review: clean,43 targeted passed2 skipped.
- Additional partial-reset snapshot regression: RED demonstrated continuing-slot alpha/observation mismatch; retain cached risk for continuing worlds on same-step partial resets; GREEN24passed2skipped.
- Final full suite592passed34skipped, Ruff clean, git diff --check clean. No old source bugfix reverted. Defaultgoldens unchanged. Native GPU verification remains before launch.
