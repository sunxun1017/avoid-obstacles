# Experimental occlusion probe

`occlusion_probe.py` is an isolated, disposable finite-grid experiment. It does
not modify the production perception/CAT chain, train a policy, implement the
paper's ADMM solver, or control G1. Its purpose is to test representation and
planning contracts before attempting integration.

Run from the repository root:

```bash
/home/sx/miniconda3/envs/mjlab/bin/python -m pytest tests/test_occlusion_probe.py -q
/home/sx/miniconda3/envs/mjlab/bin/python tools/experiments/occlusion_probe.py --seed 17 --output docs/benchmarks/occlusion-probe-2026-10-19.json
```

The JSON records the script SHA-256, Python/NumPy/SciPy versions, parameters,
raw-ray hashes, per-stage wall times, all individual outcomes and counterexamples.
Timing is ordinary Python CPU timing, not a real-time deployment claim. The raw
and near sensor-fixture generation times are separate from compression, mapping,
risk construction and candidate evaluation.

## What is implemented

1. **Near 3-D geometry:** a fixed-resolution voxel observation memory whose
   radius is `max(2 m, body_radius + margin + v*latency + v²/(2*braking))`.
   Observation age is added to effective latency in the sweep. The virtual depth
   sensor ray-casts to predetermined voxel centers; occluded cells remain unknown.
   Radius grows without coarsening voxels, so ray count and compute grow too.
2. **Far compression:** a full-circle angular grid with three sampled heights.
   Uniform and adaptive versions receive exactly the same 540 raw far rays,
   missing-beam mask, near rays and angular-bin budget. Adaptive density favors
   heading, intended turning direction and measured range/occlusion boundaries.
   It is compression after acquisition, not an active hardware scan scheduler.
3. **Explicit evidence:** one retained measured ray per height/bin, nearest hit
   range, hit-only occlusion start, unknown mask and observation timestamp.
   Free-space updates follow the retained ray only. An observed bin does not
   certify its angular wedge or the gaps between height layers. No-return rays
   clear their measured segments but never invent an occluder. Invalid rays
   provide no evidence. Voxel ages are independent; a new local scan cannot make
   old cells fresh. Stale free cells become unknown.
4. **Risk and common-prefix planning:** unknown/free boundaries seed a multi-source
   Dijkstra expansion at three assumed hidden-agent speeds. The graph blocks
   observed walls and permits unknown space, including unknown walls. It cannot
   read world-truth obstacles. Nine possible shared control prefixes each have
   81 suffix candidates per branch, using acceleration-limited planar unicycle
   rollouts. Equal-weight suffix costs choose the prefix. Every branch checks
   a stopping tail, including actual zero terminal speed, footprint clearance,
   subcell position and between-sample travel margin. Failure returns
   `feasible=False, action=None`; it is not relabeled a safe stop.

The center reachability model is **four-neighbor grid motion**. It is not a
conservative continuous-plane model: diagonal motion can arrive sooner.
Euclidean inflation subsequently makes a collision-volume mask and can cover
the far side of a thin wall. That extra conservatism does not mean a hidden
center crossed the wall. All unknown/free frontiers are retained; there is no
learned source probability or selective route pruning.

The near voxel size is **0.25 m**. “Near detail” means retaining 3-D voxels relative
to the compressed far representation; it does not establish centimeter-scale
foot placement or body clearance. Navigation checks an upright column, excludes
the ground layer, and models no articulated feet, legs, stepping or crouching.
Integration requires substantially finer geometry and a body collision envelope.
This probe cannot resolve the production CAT foot-collision issue.

## Baselines and reference scope

The representation baseline is uniform angular compression with matched raw
observations and bin budget. The planning baselines use identical geometry,
candidate controls, horizon and cost: visible-only risk disabled; one worst-speed
risk configuration; and three configurations with shared prefix.

These speed-bound risk sets are nested. A feasible worst-case suffix can serve
every weaker branch, so the feasible prefix set must match the single-worst
baseline. The multi-branch experiment can change cost-selected controls; it
cannot demonstrate an enlarged safety-feasible region in this setup.

The script also evaluates the tangent-aligned risk-disk formulas from equations
7/8 of [the CMPC paper](https://arxiv.org/html/2503.04563v5) on one canonical
visible-circle fixture. It does not substitute a forbidden full cone starting
at the sensor. That fixture is a formula reference, not a full paper-performance
baseline. The near-zero-speed denominator is preserved and its large radii are
reported. The adaptive binning heuristic is not the trained actor or motor
policy in [ADAPT](https://arxiv.org/html/2603.16328v1).

## Recorded result, seed 17

There are four geometric scenes (corner, doorway, rear poles, overhang), two bin
budgets (12/24), two initial speeds (0.6/1.6 m/s), and two observation ages
(0/0.4 s): **32 paired parameter rows, not 32 independent scenarios**.
Compression has 16 distinct scene/speed/bin settings because age repeats the raw
compression evidence. These are snapshot decisions, not navigation episodes.

| Diagnostic | Adaptive versus uniform |
| --- | --- |
| Full-circle display range MAE | 18 rows lower, 6 higher, 8 equal |
| Occlusion-edge bin error | 22 rows lower, 2 higher, 8 without edges |
| False-free voxels in ideal binary-voxel worlds | 0 for both |
| Visible-only candidate feasibility | 28/32 for both |
| Single-worst candidate feasibility | 18/32 for both |
| Shared-prefix candidate feasibility | 18/32 for both |
| Shared-prefix first action differs from single-worst | 4 uniform rows, 6 adaptive rows |

There is **no demonstrated navigation-success or collision-rate benefit**.
The feasibility equality is expected from nesting, not independent validation
of multi-branch benefits. An action difference alone is not evidence of better
control. No closed-loop speed variance, success or safety claim follows.

A concrete negative case is rear poles at 1.6 m/s with 12 bins: display range MAE
increases from 0.872 m to 1.831 m when angular allocation favors the front/turn.
The overhang lies between far sampled heights; near 3-D sensing detects it while
the far representation alone cannot. Conservative unknown handling also rejects
otherwise geometrically plausible candidates, particularly with age and larger
hidden-speed bounds. Infeasible cases remain in the report.

## Behavioral verification

Tests cover invalid/no-return distinctions, no clearing behind first hits,
unobserved beam gaps, local freshness, speed/latency/braking radius dependence,
matched angular budgets, wall-constrained center paths, speed/age/horizon risk
monotonicity, preserved overhead geometry, identical common controls, checked
terminal stopping, subcell footprint overlap, infeasibility, and hidden-world
noninterference. In the last test two worlds differ only behind an occluding
wall: delivered observations and subsequent planning must be identical.

The 16 focused tests pass. A full repository test attempt fails during collection
in the existing `tests/test_scan.py`, which instantiates `cuda:0` despite no CUDA
GPU being available in this environment. This probe needs only NumPy and SciPy.

## Existing CAT expert diagnosis

The separate checkpoint investigation completed **3,448 new inference episodes**
on `zjut`: seven 64-environment counterfactual cohorts and three 1,000-environment
single-perturbation cohorts. Each slot contributes exactly its first episode;
later resets are excluded. These are evaluation episodes, not PPO iterations.
Server code was synchronized through Git bundles and isolated worktrees. No
production task behavior or checkpoint weights were changed by these probes.

- [Combined diagnosis and verified counts](../../docs/benchmarks/cat-diagnosis-2026-10-19.json)
- [Stopping traces, denominators and source hashes](../../docs/benchmarks/cat-stopping-diagnosis-2026-10-19.json)
- [Native collision versus point-SDF audit](../../docs/benchmarks/cat-geometry-audit-2026-10-19.json)

`tools/diagnose_cat_checkpoint.py --help` lists the inference interventions.
It records first-contact/first-goal/terminal states before auto-reset, plus the
first eight slots every 0.1 s. Single-factor noise/controller overrides are
applied after strict checkpoint validation and do not rebuild the initial cached
observation. The separate CLI `--no-perturbations` override rebuilds initial
observations without noise. Individual RFI and PD overrides apply to first
episodes only. Raw logs and traces are under `runs/cat-diagnosis-20261019-26f42f9` and
`runs/cat-diagnosis-isolation-20261019-6e6a153` on both machines.

The hurdle probe reproduces foot contact while all eleven query SDF values are
positive. The wall probe separates high-speed overshoot during the stopping
transition from perturbation-dependent angular motion afterward. Neither
actor-command coherence nor naive gait rearming is an accepted fix. Disabling
RFI, observation noise or PD randomization individually does not restore the
defined continuous settling condition in the 1,000-episode cohorts.

The local 41-test targeted run passes. A subsequent diagnostic guard keeps
NaN/Inf body positions from aborting field queries and marks their placeholder
fields invalid; all archived episodes had `invalid=false`. Full-suite success is not claimed:
besides CUDA initialization in `test_scan.py`, the existing
`test_adapter.py::test_register_gpu_status` explicitly requires a CUDA GPU and
fails on this CPU-only machine.
