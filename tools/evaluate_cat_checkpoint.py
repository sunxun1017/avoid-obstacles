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

from embodied_learning.evaluation.cat_terminal_gate import (
    HOLD_STEPS,
    VERSION as TERMINAL_PROTOCOL,
    classify_terminal,
    update_hold,
)


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
def evaluate_cohort(env, policy, *, episodes, episode_path=None, terminal_gate=False):
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
    if terminal_gate:
        ordinary_hold = torch.zeros_like(quota)
        strict_hold = torch.zeros_like(quota)
        first_ordinary_hold = torch.zeros_like(quota)
        first_strict_hold = torch.zeros_like(quota)
        contact = torch.zeros_like(quota, dtype=torch.bool)
        reached_v1 = torch.zeros_like(contact)
    first_goal_speed = torch.full(
        (env.num_envs,), math.nan, device=device, dtype=torch.float64
    )
    ever_goal_hold = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
    paper_success = torch.zeros_like(ever_goal_hold)
    unsafe_prefix = torch.zeros_like(ever_goal_hold)
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
            # Prefix safety is latched before testing arrival, so a collision
            # on the arrival step fails. Collisions after clean arrival do not
            # retroactively change this independent short-horizon result.
            unsafe_prefix |= active & (
                outcomes["physical_collision"].bool()
                | outcomes["fall"].bool()
                | outcomes["invalid"].bool()
            )
            paper_success |= (
                active
                & ~unsafe_prefix
                & (age <= 250)
                & (metrics["goal_distance"] < 0.1)
            )
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
            if terminal_gate:
                ordinary_hold, strict_hold = update_hold(
                    ordinary_hold, strict_hold, metrics["goal_distance"],
                    metrics["root_speed"], metrics["root_angular_speed"],
                )
                reached_v1 |= active & (metrics["goal_distance"] <= 0.1)
                contact |= active & outcomes["prohibited_contact"].bool()
                safe_now = active & ~contact & ~outcomes["fall"].bool() & ~outcomes["invalid"].bool()
                ordinary_new = safe_now & (ordinary_hold >= HOLD_STEPS) & (first_ordinary_hold == 0)
                strict_new = safe_now & (strict_hold >= HOLD_STEPS) & (first_strict_hold == 0)
                first_ordinary_hold[ordinary_new] = age[ordinary_new]
                first_strict_hold[strict_new] = age[strict_new]
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
                    paper_success=bool(paper_success[slot]),
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
                if terminal_gate:
                    success, strict, reason = classify_terminal(
                        length, int(ordinary_hold[slot]), int(strict_hold[slot]),
                        invalid=flags["invalid"], fall=flags["fall"],
                        contact=bool(contact[slot]), reached=bool(reached_v1[slot]),
                    )
                    record["terminal_v1"] = dict(
                        success=success, strict_success=strict, failure_reason=reason,
                        first_ordinary_hold_seconds=int(first_ordinary_hold[slot]) * 0.02
                        if first_ordinary_hold[slot] else None,
                        first_strict_hold_seconds=int(first_strict_hold[slot]) * 0.02
                        if first_strict_hold[slot] else None,
                    )
                records.append(record)
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
            if terminal_gate:
                ordinary_hold[done] = strict_hold[done] = 0
                first_ordinary_hold[done] = first_strict_hold[done] = 0
                contact[done] = reached_v1[done] = False
            completed[done] += 1
            age[done] = first_goal[done] = 0
            first_goal_speed[done] = math.nan
            ever_goal_hold[done] = False
            paper_success[done] = unsafe_prefix[done] = False
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
    paper_count = sum(r["paper_success"] for r in records)
    definitions["paper_success"] = dict(
        horizon_seconds=5.0,
        goal_xy_distance_below_m=0.1,
        collision_window="Episode start through first qualifying arrival, inclusive",
        excludes="Physical obstacle contact, fall or invalid before arrival",
        independent_self_contact_veto=False,
        episode_termination=(
            "Source task termination retained; "
            "5 seconds is the maximum arrival window"
        ),
        later_failures="Reported independently in full-episode outcomes",
        scope=(
            "Paper success definition on this local task; "
            "not a paper benchmark replication"
        ),
    )
    definitions["goal_hold"] = dict(
        goal_distance_below_m=0.1,
        settled_stop_required=True,
        excludes_current_fall_or_invalid=True,
        step_fraction_denominator="all transitions in designated complete episodes",
        ever="At least one qualifying step during the designated episode.",
        final="Qualifies on the terminal step before automatic reset.",
    )
    terminal_summary = {}
    if terminal_gate:
        scores = [r["terminal_v1"] for r in records]
        count = sum(r["success"] for r in scores)
        strict_count = sum(r["strict_success"] for r in scores)
        terminal_summary["terminal_v1"] = dict(
            protocol=TERMINAL_PROTOCOL, success_count=count, episodes=episodes,
            success_wilson_95=wilson(count, episodes),
            strict_success_count=strict_count,
            strict_success_wilson_95=wilson(strict_count, episodes),
            first_ordinary_hold_seconds=distribution([r["first_ordinary_hold_seconds"] for r in scores]),
            first_strict_hold_seconds=distribution([r["first_strict_hold_seconds"] for r in scores]),
            failures={
                reason: sum(r["failure_reason"] == reason for r in scores)
                for reason in ("invalid", "fall", "prohibited_contact", "not_reached", "not_settled")
            },
        )
    return dict(
        **terminal_summary,
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
        paper_success_count=paper_count,
        paper_success_rate=paper_count / episodes,
        paper_success_wilson_95=wilson(paper_count, episodes),
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
    parser.add_argument("--terminal-gate", action="store_true")
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
            terminal_gate=args.terminal_gate,
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
