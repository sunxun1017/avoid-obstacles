# Hurdle training through the project entry

The `feetgf_progress_stage1_v1` recipe brings the validated single-reward
experiment into the ordinary curriculum trainer. Run these commands from the
repository root with the environment described in [field training](field_training.md).
Model weights, fields and logs remain in ignored `runs/` directories. No per-run
copy of project source or machine-specific Python script is required.

## Initialize and prepare

Supply an **ungated stage 1 hurdle checkpoint** that declares its PD gains,
RFI amplitude and reward contract:

```bash
python tools/experiments/train_cat_curriculum.py init \
  --root runs/hurdle-progress \
  --recipe feetgf_progress_stage1_v1 \
  --parent-checkpoint /path/to/stage1/latest.pt

python tools/experiments/train_cat_curriculum.py prepare \
  --root runs/hurdle-progress --stage 1
```

The generated protocol records the parent SHA-256 and experiment identity.
It inherits actual PD gains, RFI attenuation, smoothness weight, task options,
field interpolation and PPO settings from that checkpoint. It adds only
`feetgf_progress_gate: root_displacement_0p5s_v1` to the task. In particular,
an old checkpoint with fixed initialization is not silently switched to a
different initialization profile from the default YAML.

The task remains a **4 cm high, 8 cm deep single hurdle**. The default new
budget is 611 updates with 4096 environments, approximately 400 million new
environment transitions. This is a local experimental curriculum, not a claim
to reproduce the paper's complete difficulty schedule.

## Launch, observe and resume

```bash
python tools/experiments/run_cat_hurdle.py \
  --root runs/hurdle-progress --gpu 0
```

The command starts a detached process. A GPU UUID may be supplied instead of
its physical index. Without `--gpu`, existing CUDA visibility is inherited.
Use a GPU supported by the installed PyTorch build. Add `--foreground` to run
attached to the current terminal.

The driver uses the existing `run_cat_curriculum_pipeline.py` implementation:

- Actor weights are inherited exactly; Critic, optimizer and counters start fresh.
- Each subsequent chunk resumes its own full PPO checkpoint.
- Every 32 updates, 256 complete episodes are evaluated with `source_train`.
- After training, another evaluation suppresses only pushes, keeping noise,
  RFI, random PD, gait initialization and field delay unchanged.
- Difficulty never advances automatically.

Inspect `driver-status.json`, `launch.json`, `logs/`, and
`stage1-session/status.json` under the run root. Logs and models are not committed.
The driver records hashes of the project Python files and checks them before
every child process. Keep that project checkout unchanged until the job ends;
editing it causes the next chunk to stop explicitly instead of mixing versions.

To resume an interrupted, unfinished budget, use an **unused session directory**:

```bash
python tools/experiments/run_cat_hurdle.py \
  --root runs/hurdle-progress --gpu 0 \
  --output runs/hurdle-progress/resumed-session \
  --resume /path/to/this-experiment/checkpoint.pt
```

An active driver prevents a second detached launch. Previous launch records are
retained. `--smoke` uses 32 environments but does not silently change the update
budget. For a short native check, use the ordinary `train` command with
`--updates 2 --smoke` and the declared `--warmstart-actor` checkpoint.

## Evaluate an existing checkpoint

```bash
python tools/experiments/train_cat_curriculum.py evaluate \
  --root runs/hurdle-progress --stage 1 \
  --checkpoint /path/to/checkpoint.pt \
  --output runs/hurdle-progress/no-push-check \
  --execution-profile source_train --disable-push --seed 20261026
```

This override is recorded separately from the checkpoint's training contract.
The original push random draws are consumed without applying root-velocity
impulses. `author_playback` changes several conditions and is not a substitute
for this single-factor test.

The reward multiplies only positive `feetgf` during commanded motion by the
net body progress over the last 0.5 seconds, projected along the command and
normalized by requested speed. Negative directional rewards, stopping rewards,
other terms and their weights are preserved. Small stationary foot lifting loses
its high score; larger body oscillations can still receive partial progress
credit. Higher learned success must be checked by training and evaluation.

## Existing run directories

No existing independent training directory is removed by this integration.
Running jobs, unique checkpoints, cached fields and their referenced parents
must be retained. The new entry removes the need to create more per-run source
copies; it does not make old provenance or model files disposable.
