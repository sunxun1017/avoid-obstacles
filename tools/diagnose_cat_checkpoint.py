"""Instrument existing CAT checkpoints before auto-reset; research diagnostics only.

Variants alter inference observations/state only and never update checkpoint weights.
They are counterfactual probes, not changes to the source reproduction contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from evaluate_cat_checkpoint import evaluate_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--no-perturbations", action="store_true")
    parser.add_argument("--disable-push", action="store_true")
    parser.add_argument("--disable-rfi", action="store_true")
    parser.add_argument("--disable-observation-noise", action="store_true")
    parser.add_argument("--disable-random-pd", action="store_true")
    parser.add_argument(
        "--variant",
        choices=("baseline", "coherent_stop", "rearm"),
        default="baseline",
    )
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=False)
    (args.result_dir / "fields").symlink_to((args.run_dir / "fields").resolve())
    torch.set_num_threads(1)
    metadata = dict(
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        scene=args.scene,
        num_envs=args.num_envs,
        seed=args.seed,
        variant=args.variant,
        no_perturbations=args.no_perturbations,
        disable_push=args.disable_push,
        disable_rfi=args.disable_rfi,
        disable_observation_noise=args.disable_observation_noise,
        disable_random_pd=args.disable_random_pd,
        override_timing=(
            "After strict checkpoint load, before first rollout action; initial "
            "cached observation retains the original noise. Controller overrides "
            "apply to the designated first episode only; later resets are excluded."
        ),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    (args.result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    from embodied_learning.cli.field_training import main as field_main

    def evaluate(env, policy, *, steps):
        state, native = env.cat_state, env.unwrapped
        if args.disable_push:
            state.options = {**state.options, "push": False}
        if args.disable_observation_noise:
            state.options = {**state.options, "noise_level": 0.0}
        if args.disable_rfi:
            native.cat_controller.rfi_scale.zero_()
        if args.disable_random_pd:
            native.cat_controller.kp_scale.fill_(1.0)
            native.cat_controller.kd_scale.fill_(1.0)
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        contact_seen = torch.zeros_like(active)
        goal_seen = torch.zeros_like(active)
        trace = []
        events = []
        pairs = Counter()
        contact_by_obstacle = {
            n: torch.zeros_like(active) for n in native.cat_obstacle_contacts
        }
        original = state.refresh

        def refresh(payload, controller, lengths, common):
            fresh = state._last_common != common
            continuing = ~state.initial.clone()
            selected = active & continuing
            if fresh and args.variant == "rearm":
                distance = (
                    payload["root_position"][:, :2] - state.goals[:, :2]
                ).norm(dim=-1)
                resume = (
                    selected & (state.gait.stop_timestep <= 0) & (distance > 0.3)
                )
                state.gait.stop_timestep[resume] = 100
            original(payload, controller, lengths, common)
            if not fresh or not bool(selected.any()):
                return
            if args.variant == "coherent_stop":
                stopped = state.gait.last_command[:, 1:].norm(dim=-1) < 0.2
                state._obs["actor"][selected & stopped, 77:80] = 0.0
            flags = state.saved_outcomes
            terminal = selected & (
                flags["fall"]
                | flags["collision"]
                | flags["invalid"]
                | flags["timeout"]
            )
            first_contact = selected & payload["physical_collision"] & ~contact_seen
            first_goal = selected & flags["goal"] & ~goal_seen
            contact_seen.logical_or_(selected & payload["physical_collision"])
            goal_seen.logical_or_(selected & flags["goal"])
            new_obstacle_contact = torch.zeros_like(active)
            for name in contact_by_obstacle:
                touching = selected & native.scene[name].data.found.flatten(1).any(1)
                new_obstacle_contact |= touching & ~contact_by_obstacle[name]
                contact_by_obstacle[name] |= touching
            sampled = state.bank.sample_source(
                payload["body_positions"], state.scene_ids, state.goal_ids
            )
            values = dict(
                root=payload["root_position"],
                goal=state.goals,
                velocity=payload["reward_inputs"].root_vel,
                gyro=payload["gyro"],
                body_positions=payload["body_positions"],
                sdf=sampled.sdf,
                raw_guidance=sampled.guidance,
                raw_boundary=sampled.sdf_gradient,
                command=state.gait.last_command,
                actor_command=state._obs["actor"][:, 76:80],
                stop_counter=state.gait.stop_timestep,
                phase=state.gait.phase,
                actions=controller.actions,
                targets=controller.targets,
                joint_pos=payload["reward_inputs"].joint_pos,
                joint_vel=payload["reward_inputs"].joint_vel,
                torques=payload["reward_inputs"].torques,
                head_z=payload["body_positions"][:, 0, 2],
                up_z=payload["up_z"],
                goal_distance=state.last_metrics["goal_distance"],
                physical_collision=payload["physical_collision"],
                move_flag=payload["reward_inputs"].move_flag,
                feet_ground_contact=payload["reward_inputs"].feet_contact,
                settled_steps=state.settled_steps,
                push_interval=state.push_interval,
            )
            # Save eight slots densely and all first-contact/terminal events.
            count = min(8, env.num_envs)
            if common % 5 == 0:
                trace.append(
                    dict(
                        step=common,
                        active=selected[:count].cpu().numpy(),
                        **{
                            k: v[:count].detach().cpu().numpy()
                            for k, v in values.items()
                        },
                    )
                )
            event_slots = (
                (first_contact | new_obstacle_contact | first_goal | terminal)
                .nonzero()
                .flatten()
                .tolist()
            )
            step_events = {}
            for slot in event_slots:
                event = dict(
                    slot=slot,
                    step=int(lengths[slot]),
                    first_contact=bool(first_contact[slot]),
                    first_goal=bool(first_goal[slot]),
                    terminal=bool(terminal[slot]),
                    outcomes={k: bool(v[slot]) for k, v in flags.items()},
                    **{
                        k: v[slot].detach().cpu().tolist() for k, v in values.items()
                    },
                )
                events.append(event)
                step_events[slot] = event
                event["contacts"] = []
            contact_events = new_obstacle_contact | first_contact | terminal
            if bool(contact_events.any()):
                data = native.sim.data
                n = min(int(data.nacon[0]), data.contact.dist.shape[0])
                c = data.contact
                world = c.worldid[:n]
                geom = c.geom[:n]
                valid_world = world.clamp(0, env.num_envs - 1).long()
                mask = (
                    contact_events[valid_world]
                    & (world >= 0)
                    & (world < env.num_envs)
                )
                # Raw pairs supplement per-obstacle sensors with body attribution.
                for slot, ids, dist, normal in zip(
                    world[mask].cpu().tolist(),
                    geom[mask].cpu().tolist(),
                    c.dist[:n][mask].cpu().tolist(),
                    c.frame[:n][mask, 0].cpu().tolist(),
                    strict=True,
                ):
                    names = tuple(native.sim.mj_model.geom(int(i)).name for i in ids)
                    if any("obstacle_" in name for name in names):
                        step_events[slot]["contacts"].append(
                            dict(geoms=names, distance=dist, normal=normal)
                        )
                        if bool(first_contact[slot]) and dist <= 0:
                            pairs[" | ".join(names)] += 1
            active[terminal] = False

        with patch.object(state, "refresh", refresh):
            result = evaluate_cohort(
                env,
                policy,
                episodes=env.num_envs,
                episode_path=args.result_dir / "episodes.jsonl",
            )
        if trace:
            arrays = {k: np.stack([row[k] for row in trace]) for k in trace[0]}
            np.savez_compressed(args.result_dir / "trace.npz", **arrays)
        (args.result_dir / "events.json").write_text(json.dumps(events) + "\n")
        result["first_contact_pairs"] = dict(pairs)
        result["episodes_with_contact_by_obstacle"] = {
            k: int(v.sum()) for k, v in contact_by_obstacle.items()
        }
        result["diagnostic_variant"] = args.variant
        result["diagnostic_disable_push"] = args.disable_push
        result["diagnostic_disable_rfi"] = args.disable_rfi
        result["diagnostic_disable_observation_noise"] = (
            args.disable_observation_noise
        )
        result["diagnostic_disable_random_pd"] = args.disable_random_pd
        return result

    command = [
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
    if args.no_perturbations:
        command.append("--no-perturbations")
    with patch("embodied_learning.algorithms.dagger.evaluate_student", evaluate):
        return field_main(command)


if __name__ == "__main__":
    raise SystemExit(main())
