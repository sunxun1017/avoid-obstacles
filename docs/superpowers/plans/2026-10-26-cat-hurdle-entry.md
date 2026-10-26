# CAT hurdle training entry integration

User authorized integration into the general project entry, commit and push. Related run directories must be retained.

Goal: reproduce the validated4cmhurdle experiment from tracked project code instead of per-run Python copies.

1. Add portable failing tests for strict feetgf Actor migration, recipe initialization, persistent RFI attenuation across resets, and restartable training/evaluation entry.
2. Integrate the existing feetgf declaration alongside tracking-weight migration. Add explicit PD/RFI/smoothness metadata in generic environment construction,keeping legacy defaults unchanged.
3. Extend generic init with feetgf_progress_stage1_v1 and supplied parent path;reuse normal prepare/train/evaluate pipeline. Add detached repository launcher and no-push evaluation,with no machine-specific paths.
4. Test native Actor-only warmstart and strict reload in a tiny run;run full suite and Ruff;independent review.
5. Merge only inspected files into the active project,preserve existing work,commit relevant CAT source/dependencies/tests/docs,normal push to existing branch. Retain all independent run directories per latest instruction.

Do not change active training snapshots or defaults of the separate initialization experiment. Historical benchmark images and diagnostic JSON stay unstaged.
