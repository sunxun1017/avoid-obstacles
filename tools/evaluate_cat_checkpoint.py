"""Evaluate a fixed cohort of complete CAT episodes using strict CLI loading.

This benchmark keeps the source 20-second horizon and training perturbations.
Slots receive quotas before simulation, avoiding selection of fastest episodes.
The original checkpoint and its field cache are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch


def wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    return [
        max(0.0, center - half / denominator),
        min(1.0, center + half / denominator),
    ]


def distribution(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return dict(count=0, mean=None, median=None, p95=None)
    data = torch.tensor(values, dtype=torch.float64)
    return dict(
        count=len(values),
        mean=float(data.mean()),
        median=float(data.quantile(0.5)),
        p95=float(data.quantile(0.95)),
    )


@torch.inference_mode()
def evaluate_cohort(env, policy, *, episodes, episode_path=None):
    """Consume terminal snapshots, ignoring later episodes in completed slots."""
    if episodes < env.num_envs or episodes < 1:
        raise ValueError("episodes must be at least num_envs and positive")
    obs = env.get_observations()
    device = env.device
    quota = torch.full((env.num_envs,), episodes // env.num_envs, device=device)
    quota[: episodes % env.num_envs] += 1
    completed = torch.zeros_like(quota)
    age = torch.zeros_like(quota)
    first_goal = torch.zeros_like(quota)
    first_goal_speed = torch.full(
        (env.num_envs,), math.nan, device=device, dtype=torch.float64
    )
    ever_goal_hold = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    sums = {
        name: torch.zeros(env.num_envs, device=device, dtype=torch.float64)
        for name in (
            "reward",
            "stop_command",
            "settled_stop",
            "root_speed",
            "goal_hold",
            "pre_goal_steps",
            "pre_goal_stop_command",
            "pre_goal_settled_stop",
        )
    }
    records, definitions = [], {}
    started = time.perf_counter()
    step = 0
    stream = episode_path.open("w") if episode_path else None
    try:
        while bool((completed < quota).any()):
            active = completed < quota
            pre_goal = active & (first_goal == 0)
            obs, rewards, dones, extras = env.step(policy.predict(obs))
            age[active] += 1
            outcomes = extras["field_outcomes"]
            metrics = extras["cat_metrics"]
            definitions.update(extras["cat_metric_definitions"])
            reached = active & (first_goal == 0) & outcomes["goal"]
            first_goal[reached] = age[reached]
            first_goal_speed[reached] = metrics["root_speed"][reached].double()
            goal_hold = (
                (metrics["goal_distance"] < 0.1)
                & metrics["settled_stop"].bool()
                & ~outcomes["fall"].bool()
                & ~outcomes["invalid"].bool()
            )
            ever_goal_hold |= active & goal_hold
            sums["reward"][active] += rewards[active]
            sums["goal_hold"][active] += goal_hold[active].double()
            for name in ("stop_command", "settled_stop", "root_speed"):
                sums[name][active] += metrics[name][active].double()
            sums["pre_goal_steps"][pre_goal] += 1
            for name in ("stop_command", "settled_stop"):
                sums["pre_goal_" + name][pre_goal] += metrics[name][
                    pre_goal
                ].double()
            done = dones.bool() & active
            for slot in done.nonzero().flatten().tolist():
                length = int(age[slot])
                flags = {name: bool(value[slot]) for name, value in outcomes.items()}
                record = dict(
                    slot=slot,
                    episode_in_slot=int(completed[slot]),
                    steps=length,
                    duration_seconds=length * 0.02,
                    first_goal_seconds=float(first_goal[slot]) * 0.02
                    if first_goal[slot]
                    else None,
                    first_goal_root_speed=float(first_goal_speed[slot])
                    if first_goal[slot]
                    else None,
                    ever_goal_hold=bool(ever_goal_hold[slot]),
                    final_goal_hold=bool(goal_hold[slot]),
                    final_root_speed=float(metrics["root_speed"][slot]),
                    final_root_angular_speed=float(
                        metrics["root_angular_speed"][slot]
                    ),
                    outcomes=flags,
                    timeout_without_goal=flags["timeout"] and not flags["goal"],
                    final_goal_distance=float(metrics["goal_distance"][slot]),
                    min_goal_distance=float(metrics["min_goal_distance"][slot]),
                    sums={name: float(value[slot]) for name, value in sums.items()},
                )
                for name in (
                    "final_goal_distance",
                    "min_goal_distance",
                    "first_goal_root_speed",
                    "final_root_speed",
                    "final_root_angular_speed",
                ):
                    if record[name] is not None and not math.isfinite(record[name]):
                        record[name] = None
                if not all(math.isfinite(x) for x in record["sums"].values()):
                    raise ValueError("Nonfinite episode metric; investigate physics")
                records.append(record)
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            completed[done] += 1
            age[done] = first_goal[done] = 0
            first_goal_speed[done] = math.nan
            ever_goal_hold[done] = False
            for value in sums.values():
                value[done] = 0
            step += 1
            if step % 100 == 0 or len(records) == episodes:
                if stream:
                    stream.flush()
                print(
                    json.dumps(
                        dict(
                            progress_steps=step,
                            completed=len(records),
                            target=episodes,
                            elapsed_seconds=time.perf_counter() - started,
                        )
                    ),
                    flush=True,
                )
            if step > int(quota.max()) * 1001:
                raise RuntimeError(
                    "A designated episode did not terminate in 20 seconds"
                )
    finally:
        if stream:
            stream.close()
    assert len(records) == episodes
    counts = {
        name: sum(r["outcomes"][name] for r in records)
        for name in records[0]["outcomes"]
    }
    counts["timeout_without_goal"] = sum(r["timeout_without_goal"] for r in records)
    total_steps = sum(r["steps"] for r in records)
    pre_steps = sum(r["sums"]["pre_goal_steps"] for r in records)
    totals = {name: sum(r["sums"][name] for r in records) for name in sums}
    ever_hold_count = sum(r["ever_goal_hold"] for r in records)
    final_hold_count = sum(r["final_goal_hold"] for r in records)
    definitions["goal_hold"] = dict(
        goal_distance_below_m=0.1,
        settled_stop_required=True,
        excludes_current_fall_or_invalid=True,
        step_fraction_denominator="all transitions in designated complete episodes",
        ever="At least one qualifying step during the designated episode.",
        final="Qualifies on the terminal step before automatic reset.",
    )
    return dict(
        episodes=episodes,
        envs=env.num_envs,
        simulation_steps=step,
        included_transitions=total_steps,
        elapsed_seconds=time.perf_counter() - started,
        sampling=(
            "Fixed per-slot quotas; all designated episodes complete; "
            "later resets excluded"
        ),
        outcomes=counts,
        rates={k: v / episodes for k, v in counts.items()},
        wilson_95={k: wilson(v, episodes) for k, v in counts.items()},
        mean_step_reward=totals["reward"] / total_steps,
        mean_root_speed=totals["root_speed"] / total_steps,
        stop_command_fraction=totals["stop_command"] / total_steps,
        settled_stop_fraction=totals["settled_stop"] / total_steps,
        goal_hold_fraction=totals["goal_hold"] / total_steps,
        ever_goal_hold_count=ever_hold_count,
        ever_goal_hold_rate=ever_hold_count / episodes,
        final_goal_hold_count=final_hold_count,
        final_goal_hold_rate=final_hold_count / episodes,
        pre_goal_stop_command_fraction=totals["pre_goal_stop_command"] / pre_steps
        if pre_steps
        else None,
        pre_goal_settled_stop_fraction=totals["pre_goal_settled_stop"] / pre_steps
        if pre_steps
        else None,
        duration_seconds=distribution([r["duration_seconds"] for r in records]),
        first_goal_seconds=distribution([r["first_goal_seconds"] for r in records]),
        first_goal_root_speed=distribution(
            [r["first_goal_root_speed"] for r in records]
        ),
        final_root_speed=distribution([r["final_root_speed"] for r in records]),
        final_root_angular_speed=distribution(
            [r["final_root_angular_speed"] for r in records]
        ),
        successful_first_goal_seconds=distribution(
            [r["first_goal_seconds"] for r in records if r["outcomes"]["success"]]
        ),
        final_goal_distance=distribution(
            [r["final_goal_distance"] for r in records]
        ),
        min_goal_distance=distribution([r["min_goal_distance"] for r in records]),
        metric_definitions=definitions,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261019)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= args.episodes:
        parser.error("Require 1 <= num-envs <= episodes")
    fields = (args.run_dir / "fields").resolve()
    if not (fields / "manifest.json").is_file():
        parser.error("Original field cache is required")
    args.result_dir.mkdir(parents=True, exist_ok=False)
    (args.result_dir / "fields").symlink_to(fields, target_is_directory=True)
    torch.set_num_threads(1)
    metadata = dict(
        started_at=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        config=str(args.config.resolve()),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        fields=str(fields),
        scene=args.scene,
        seed=args.seed,
        episodes=args.episodes,
        num_envs=args.num_envs,
        perturbations="training settings retained",
        python=platform.python_version(),
        torch=torch.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device=torch.cuda.get_device_name(0),
    )
    (args.result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    from embodied_learning.cli.field_training import main as field_main

    def evaluate(env, policy, *, steps):
        return evaluate_cohort(
            env,
            policy,
            episodes=args.episodes,
            episode_path=args.result_dir / "episodes.jsonl",
        )

    with patch("embodied_learning.algorithms.dagger.evaluate_student", evaluate):
        return field_main(
            [
                "evaluate",
                "--config",
                str(args.config),
                "--output-dir",
                str(args.result_dir),
                "--checkpoint",
                str(args.checkpoint),
                "--scene",
                args.scene,
                "--num-envs",
                str(args.num_envs),
                "--seed",
                str(args.seed),
            ]
        )


if __name__ == "__main__":
    raise SystemExit(main())
