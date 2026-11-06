"""Trace a fixed cohort of frozen CAT hurdle and arrival trajectories."""

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from evaluate_cat_checkpoint import evaluate_cohort
import train_cat_curriculum as curriculum
from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.evaluation.cat_route_probe import route_goal_ids
from embodied_learning.policies.field_policy import load_teacher
from embodied_learning.simulators.mjlab import cat_traversal
from embodied_learning.tasks import cat_traversal as cat_task
from embodied_learning.tasks.cat_foot_guidance import load_foot_bank
from embodied_learning.tasks.cat_hurdle import evaluation_push_mode
from embodied_learning.tasks.field_assets import load_field_cache


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_arrays(model):
    return {
        name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for name in dir(model)
        if not name.startswith(("_", "sensor", "name"))
        and isinstance((value := getattr(model, name)), np.ndarray)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Candidate experiment arm")
    parser.add_argument("fixed_source", type=Path, help="Arm with fixed hurdle assets")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--route", default="original",
        choices=("original", "hurdle-near", "hurdle-far", "empty-near", "empty-stand"),
    )
    parser.add_argument("--empty-cache", type=Path)
    parser.add_argument("--disturbance", choices=("full", "no-push", "quiet"), default="full")
    args = parser.parse_args()
    empty = args.route.startswith("empty-")
    if args.disturbance != "full" and args.route != "empty-stand":
        parser.error("Disturbance ablation is restricted to the empty standing probe")
    if empty and args.empty_cache is None:
        parser.error("Empty routes require --empty-cache")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)

    protocol_path = args.source / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    settings = dict(num_envs=args.episodes, seed=args.seed)
    base = curriculum.make_env(args.source, 4, **settings)
    try:
        contract = curriculum.contract(base, curriculum.config(protocol, 1))
        arrays = model_arrays(base.unwrapped.sim.mj_model)
    finally:
        base.close()
    del base

    sensor_config = cat_traversal.ContactSensorCfg

    def read_force(**kwargs):
        return sensor_config(**{**kwargs, "fields": ("found", "force")})

    with patch.object(cat_traversal, "ContactSensorCfg", read_force):
        env = curriculum.make_env(args.source, 4, **settings)
    reset_patch = None
    obs_patch = None
    try:
        native = env.unwrapped
        state = env.cat_state
        sim, controller = native.sim, native.cat_controller
        model = sim.mj_model
        if arrays != model_arrays(model):
            raise RuntimeError("Force instrumentation changed non-sensor model arrays")
        stage = args.fixed_source / "stages" / "4"
        config = stage / "config.json"
        assets = json.loads((stage / "assets.json").read_text())
        manifest = load_scene_manifest(config)
        if empty:
            manifest = replace(
                manifest, scenes=tuple(replace(scene, boxes=()) for scene in manifest.scenes),
            )
        field_path = args.empty_cache if empty else Path(assets["fields"]) / "manifest.json"
        cache = load_field_cache(
            field_path, manifest,
            device=env.device, source_profile={"version": "cat_source_v1"},
        )
        cache = replace(cache, source_interpolation=protocol["source_interpolation"])
        state.manifest = manifest
        scene = next(scene for scene in manifest.scenes if scene.id == "hurdle_crouch")
        robot = native.scene["robot"]
        if empty:
            # With no SDF voxel near the obstacles, both foot radii yield
            # exactly the same goal guidance.
            influence = max(protocol["foot_radius_m"], 5 * scene.voxel_size)
            scene_idx = cache.scene_names.index(scene.id)
            if any(float(field.sdf.min()) < influence for field in cache.fields[scene_idx]):
                raise ValueError("Empty field is inside the foot guidance influence radius")
            state.bank = cache
        else:
            state.bank = load_foot_bank(
                cache, assets["foot_fields"], radius=protocol["foot_radius_m"],
            )
        if args.route != "original":
            routes = torch.tensor(
                route_goal_ids(scene.goals, far=args.route == "hurdle-far"),
                device=env.device,
            )
            goal_points = state.goals.new_tensor(scene.goals)
            original_reset = cat_traversal._reset_field

            def reset_route(world, slots):
                original_reset(world, slots)
                state.goal_ids[slots] = routes[state.goal_ids[slots]]
                state.goals[slots] = goal_points[state.goal_ids[slots]]
                if empty:
                    poses = state.goals.new_zeros((len(slots), 7))
                    poses[:, 3] = 1
                    poses[:, 2] = -100
                    poses[:, :3] += world.scene.env_origins[slots]
                    for name in world.cat_obstacle_contacts:
                        world.scene[name.removeprefix("cat_")].write_mocap_pose_to_sim(
                            poses, env_ids=slots,
                        )
                if args.route == "empty-stand":
                    roots = robot.data.default_root_state[slots].clone()
                    roots[:, :3] = state.goals[slots] + world.scene.env_origins[slots]
                    roots[:, 3:7] = roots.new_tensor([1, 0, 0, 0])
                    roots[:, 7:] = 0
                    robot.write_root_state_to_sim(roots, env_ids=slots)

            reset_patch = patch.object(cat_traversal, "_reset_field", reset_route)
            reset_patch.start()
        if args.disturbance == "quiet":
            original_obs = cat_task.build_cat_observations

            def clean_observations(**kwargs):
                original_level = kwargs.pop("noise_level")
                return original_obs(
                    **kwargs, noise_level=0.0, consume_noise_draws=bool(original_level),
                )

            obs_patch = patch.object(cat_task, "build_cat_observations", clean_observations)
            obs_patch.start()
        torch.manual_seed(args.seed)
        env.reset_slots(torch.arange(env.num_envs, device=env.device))
        policy = load_teacher(args.checkpoint, expected_contract=contract, device=env.device)
        sites = [robot.site_names.index(name) for name in ("left_foot", "right_foot")]
        ground = [native.scene[f"cat_{side}_ground"] for side in ("left", "right")]
        prohibited = [native.scene[name] for name in native.cat_obstacle_contacts]
        prohibited += [native.scene[name] for name in (
            "cat_feet_self", "cat_left_foot_shin", "cat_right_foot_shin"
        )]
        body_mass = sum(
            model.body_mass[i] for i in range(model.nbody)
            if (model.body(i).name or "").startswith("robot/")
        )
        if not 10 < body_mass < 100:
            raise ValueError(f"Unexpected robot mass {body_mass}")
        active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        substep_contact = torch.zeros_like(active)
        physics = []
        control = []
        motion = []
        update = native.scene.update
        refresh = state.refresh

        def trace_update(*args, **kwargs):
            update(*args, **kwargs)
            sim.sense()
            for sensor in prohibited:
                substep_contact.logical_or_(
                    active & sensor.data.found.flatten(1).any(1)
                )
            fz = torch.stack([s.data.force.sum(1)[:, 2] for s in ground], dim=1)
            foot = robot.data.site_pos_w[:, sites] - native.scene.env_origins[:, None]
            vz = robot.data.site_vel_w[:, sites, 2]
            physics.append(torch.cat([fz, foot.flatten(1), vz, active[:, None]], dim=1))

        def trace_refresh(payload, action, lengths, common):
            selected = active & ~state.initial
            fresh = state._last_common != common
            result = refresh(payload, action, lengths, common)
            if fresh and bool(selected.any()):
                x = payload["reward_inputs"]
                flags = state.saved_outcomes
                flags["prohibited_contact"] |= substep_contact
                channels = (
                    state.goal_distance, state.last_metrics["root_speed"],
                    state.last_metrics["root_angular_speed"], x.torso_rpy[:, 1],
                    x.feet_contact[:, 0], x.feet_contact[:, 1],
                    selected, state.gait.last_command[:, 0],
                )
                control.append(torch.cat([
                    torch.stack(channels, dim=1), x.joint_pos, x.joint_vel,
                    controller.targets,
                    (x.foot_vel[:, :, :2].norm(dim=-1)),
                ], dim=1))
                motion.append(torch.cat([
                    payload["gyro"], x.pelvis_rpy, x.torso_ang_vel, x.torso_rpy,
                    state.gait.last_command, state.gait.phase,
                    state.gait.stop_timestep[:, None], payload["root_position"],
                ], dim=1))
                active.logical_and_(~(
                    selected & (flags["fall"] | flags["collision"] | flags["invalid"] | flags["timeout"])
                ))
            return result

        native.scene.update = trace_update
        state.refresh = trace_refresh
        root_path = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(["git", "-C", str(root_path), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(root_path), "status", "--porcelain"], text=True)
        if status.strip():
            raise RuntimeError(f"Probe checkout must be clean: {status}")
        initial_hashes = {
            key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
            for key, value in dict(
                root=robot.data.root_link_pose_w, joints=robot.data.joint_pos,
                kp=controller.kp_scale, kd=controller.kd_scale, rfi=controller.rfi_scale,
            ).items()
        }
        metadata = dict(
            label=args.label, route=args.route, code_commit=commit,
            disturbance=args.disturbance, pd_randomization_unchanged=True,
            suppress_push=args.disturbance != "full",
            suppress_observation_noise=args.disturbance == "quiet",
            suppress_rfi=args.disturbance == "quiet", checkpoint=str(args.checkpoint),
            checkpoint_sha256=sha256(args.checkpoint), evaluation_seed=args.seed,
            evaluation_episodes=args.episodes, protocol_sha256=sha256(protocol_path),
            config_sha256=sha256(config),
            environment_config_sha256=sha256(args.source / "stages/4/config.json"),
            scene_manifest=manifest.identity, initial_hashes=initial_hashes,
            field_cache_identity=cache.cache_identity,
            initial_goals=state.goals.detach().cpu().tolist(),
            initial_roots=(robot.data.root_link_pos_w - native.scene.env_origins).detach().cpu().tolist(),
            motion_columns=[
                "gyro_roll", "gyro_pitch", "gyro_yaw",
                "pelvis_roll", "pelvis_pitch", "pelvis_yaw",
                "torso_rate_roll", "torso_rate_pitch", "torso_rate_yaw",
                "torso_roll", "torso_pitch", "torso_yaw",
                "move", "command_x", "command_y", "command_z", "phase_left", "phase_right",
                "stop_timestep", "root_x", "root_y", "root_z",
            ],
            robot_mass_kg=float(body_mass),
            boxes=[dict(center=b.center, half_size=b.half_size) for b in scene.boxes],
            joint_names=list(env.schema["joint_names"]), control_hz=50, physics_hz=500,
            python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            probe="fixed_hurdle_contact_v1", physics_arrays_unchanged=True,
        )
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        with ExitStack() as intervention:
            intervention.enter_context(evaluation_push_mode(env, args.disturbance != "full"))
            if args.disturbance == "quiet":
                original_torque = controller.compute_torques

                def torque_without_rfi(pos, vel, *, rfi):
                    return original_torque(pos, vel, rfi=torch.zeros_like(rfi))

                intervention.enter_context(
                    patch.object(controller, "compute_torques", torque_without_rfi)
                )
            report = evaluate_cohort(
                env, policy, episodes=args.episodes,
                episode_path=args.output / "episodes.jsonl", terminal_gate=True,
            )
        np.savez_compressed(
            args.output / "traces.npz",
            physics=torch.stack(physics).cpu().numpy(),
            control=torch.stack(control).cpu().numpy(),
            motion=torch.stack(motion).cpu().numpy(),
        )
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if obs_patch is not None:
            obs_patch.stop()
        if reset_patch is not None:
            reset_patch.stop()
        env.close()


if __name__ == "__main__":
    main()
