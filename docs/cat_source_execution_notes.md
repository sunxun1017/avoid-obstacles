# SDD ledger — plan: docs/superpowers/plans/2026-10-09-g1-public-source-training.md
Spec: docs/superpowers/specs/2026-10-09-g1-public-source-training-design.md
Base: caaf4ce; worktree /tmp/embodied-learning-cat-source; branch feat/g1-cat-source.
Pre-flight: Task1->4/5 CatRewardInputs; Task2->5 gait and command; Task3->5 source FieldSample; Task4->5 native adapter; Task5->6/7 protocol; all names consistent.
Ruling: use separate source task and native factory instead of altering old FieldTaskState — preserve old checkpoint playback — shared fixes must be backward compatible.
Ruling: source oracle runs in isolated /tmp/cat-oracle environment — JAX is absent in mjlab environment and must not become runtime dependency — fixtures must record source hashes.
Tasks 1–8 pending.
Task 1: Ruling: add hands_pos and required body_velocities to input contract — crossed masks and field rewards need real point data omitted from plan — no fabricated zero/root velocity substitute.
Baseline: 125 passed, one known MULTICCD warning. Reward tests: RED absent module -> GREEN 3/3 against author JAX fixtures.
Task 1: complete (commits caaf4ce..2406e3b, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_reward.py -q → 3 passed in 0.77s)
Task 2: complete (commits 2406e3b..2a57d10, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_command_gait.py tests/test_cat_reward.py -q → 7 passed in 0.80s)
Task 3: Ruling: preserve source interpolation corner/weight permutation — author sample_field swaps X/Z weights relative to standard interpolation — downstream behavior differs from corrected interpolation; oracle locks source behavior.
Task 3: Ruling: source caches use version 1 plus mandatory source_profile identity — old manifests remain loadable by old task, source mix rejected — explicit profile must travel in task/checkpoint.
Task 3: Ruling: empty maps use infinite SDF and zero BF — skfmm cannot derive obstacle surface without obstacles — documented extension only, no invented wall.
Task 3: complete (commits 2a57d10..02b9386, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_fields.py tests/test_field_scenes.py tests/test_potential_field.py -q → 30 passed, 1 skipped in 1.86s)
Task 4: Ruling: defer read_cat_reward_inputs and contact extraction to Task5 — they require CatTaskState from Task5 — controller remains independently verified; source inputs will be exercised by Task5 native integration.
Task 4: native CUDA 4 G1 x20 steps passed, finite joint state, all targets [4,29]; explicit BuiltinMotor controls only.
Task 4: complete (commits 02b9386..e18124d, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_control.py -q → 4 passed in 3.46s)
Task 5: complete (commits e18124d..ed94f3f, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_task.py tests/test_cat_reward.py -q → 9 passed in 6.58s)
Task 6: complete (commits ed94f3f..c61e79b, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_training.py tests/test_field_ppo.py tests/test_dagger.py tests/test_field_cli.py -q → 28 passed in 13.49s)
Task 7: complete (commits c61e79b..70fd967, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_cat_play.py tests/test_field_cli.py tests/test_dagger.py -q → 21 passed in 7.97s)
Task 5: Ruling: use refresh(payload, controller, steps, common_step) instead of coupling task state to env — simulator adapter supplies real kinematics, task stays independent — adapter/state interface must remain tested.
Task 5/6: Ruling: consolidate terminal/reset assertions into native reads test and PPO/resume/DAgger assertions into one three-scene integration test — same behaviors verified together — test name granularity differs from plan.
Task 7: Ruling: explicit native contact capacity 128 and constraints 512 — single-env playback exceeded autodetected buffer (49 needed), repaired with allocation only — more GPU memory per environment; not a source dynamics change.
Task 7: native source config prepare/PPO2 updates passed; Viser 500-step play HTTP200, clean stop after capacity repair; first attempt hit transient PyTorch non-monotonic ApproximateClock initialization, successful retries did not require code changes.
Final review: fresh read-only gpt-6-astra reviewer, base caaf4ce..f7c57b6. Six Important findings accepted; one Minor deferred. No re-review.
Final: fixed current actor move flag — test_source_observation_fixture_162_250 with differing current/delayed flags RED→GREEN.
Final: fixed reset field normalization/command — test_reset_normalization_and_command_match_executed_author with extracted upstream reset assignments RED→GREEN.
Final: fixed counter-driven gait advancement — test_resume_counter_does_not_advance_physical_state RED→GREEN.
Final: fixed pelvis IMU velocity location — test_source_site_velocity_and_asset_contracts RED→GREEN.
Final: fixed legacy PPO directory config compatibility — test_legacy_checkpoint_and_config_resume_in_original_directory RED→GREEN.
Final: fixed cache/model identities in source contract — test_verified_cache_identity_changes_for_a_different_valid_cache and test_model_identity_changes_with_native_numeric_asset RED→GREEN.
Final: fixed evaluation first cached noisy frame — test_no_perturbations_cleans_the_first_cached_observation (52/52 noisy prefix mismatch) RED→GREEN.
Final: minor (deferred): source clamped queries have no separate out-of-bounds diagnostic; source reward/query behavior is preserved and docs disclose this.
Final: Ruling: exact MJX/Brax trajectory and optimizer/native dynamics equivalence excluded — chosen mjlab/RSL backends intentionally differ — cannot claim exact original training replication.
Final: Ruling: long-run obstacle performance not judged by short smoke — authorized acceptance only tests pipeline — success rates remain unproven until server training/evaluation.
Final: Ruling: retain enumerated scene reset starts/goals, omit upstream continuous XY/yaw/joint/root-velocity reset distributions — approved design specifies these starts and PD/RFI/push — reduced reset diversity may affect robustness; docs disclose omission.
Task 8: complete (commits 70fd967..d026fb3, tests: /home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests -q → 166 passed, 1 warning in 24.72s)

Final verification: 166/166 passed on CUDA; 25 changed Python files Ruff check/format clean. Four-env source checkpoint played in one-env Viser for 500 steps with normal shutdown.
