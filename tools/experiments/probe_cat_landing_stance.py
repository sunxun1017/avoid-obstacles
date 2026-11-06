"""Trace a fixed cohort of frozen CAT hurdle and arrival trajectories."""

import argparse
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
from embodied_learning.policies.field_policy import load_teacher
from embodied_learning.simulators.mjlab import cat_traversal
from embodied_learning.tasks.cat_foot_guidance import load_foot_bank
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
    args = parser.parse_args()
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
        cache = load_field_cache(
            Path(assets["fields"]) / "manifest.json", manifest,
            device=env.device, source_profile={"version": "cat_source_v1"},
        )
        cache = replace(cache, source_interpolation=protocol["source_interpolation"])
        state.manifest = manifest
        state.bank = load_foot_bank(
            cache, assets["foot_fields"], radius=protocol["foot_radius_m"],
        )
        torch.manual_seed(args.seed)
        env.reset_slots(torch.arange(env.num_envs, device=env.device))
        policy = load_teacher(args.checkpoint, expected_contract=contract, device=env.device)
        robot = native.scene["robot"]
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
        scene = next(scene for scene in manifest.scenes if scene.id == "hurdle_crouch")
        metadata = dict(
            label=args.label, code_commit=commit, checkpoint=str(args.checkpoint),
            checkpoint_sha256=sha256(args.checkpoint), evaluation_seed=args.seed,
            evaluation_episodes=args.episodes, protocol_sha256=sha256(protocol_path),
            config_sha256=sha256(config),
            environment_config_sha256=sha256(args.source / "stages/4/config.json"),
            scene_manifest=manifest.identity, initial_hashes=initial_hashes,
            robot_mass_kg=float(body_mass),
            boxes=[dict(center=b.center, half_size=b.half_size) for b in scene.boxes],
            joint_names=list(env.schema["joint_names"]), control_hz=50, physics_hz=500,
            python=platform.python_version(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
            probe="fixed_hurdle_contact_v1", physics_arrays_unchanged=True,
        )
        (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        report = evaluate_cohort(
            env, policy, episodes=args.episodes,
            episode_path=args.output / "episodes.jsonl", terminal_gate=True,
        )
        np.savez_compressed(
            args.output / "traces.npz",
            physics=torch.stack(physics).cpu().numpy(),
            control=torch.stack(control).cpu().numpy(),
        )
        (args.output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
