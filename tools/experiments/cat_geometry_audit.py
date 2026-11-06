"""Reproduce static CAT collision/field counterexamples with native CPU MuJoCo.

This audit does not load a policy, step dynamics, or infer an episode's fall cause.
Run from the repository with its mjlab environment and PYTHONPATH=src.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.entity import EntityCfg
from mjlab.scene import Scene

from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.robots.g1 import (
    CAT_DEFAULT_JOINTS,
    CAT_JOINT_NAMES,
    G1_BODY_SCHEMA,
)
from embodied_learning.simulators.mjlab.cat_traversal import (
    build_cat_cfg,
    native_model_identity,
)
from embodied_learning.simulators.mjlab.field_traversal import _box_spec
from embodied_learning.tasks.field_assets import load_field_cache


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_geometry_model(manifest):
    """Use production robot, terrain, box specs and MuJoCo settings on CPU.

    Sensor attachments are omitted: this changes the compiled model identity,
    but not the collision geometry. No task options or dynamics are evaluated.
    """
    cfg = build_cat_cfg(num_envs=1, seed=42)
    cfg.scene.sensors = ()
    for scene_id, scene in enumerate(manifest.scenes):
        for box_id, box in enumerate(scene.boxes):
            cfg.scene.entities[f"obstacle_{scene_id}_{box_id}"] = EntityCfg(
                spec_fn=lambda half=box.half_size: _box_spec(half),
                init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, -100.0)),
            )
    native_scene = Scene(cfg.scene, device="cpu")
    model = native_scene.compile()
    cfg.sim.mujoco.apply(model)
    origin = native_scene.env_origins[0].numpy().copy()
    return model, origin


def inspect_pose(model, origin, manifest, bank, scene_id, root, label):
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    root_joint = model.joint("robot/floating_base_joint").id
    root_adr = int(model.jnt_qposadr[root_joint])
    data.qpos[root_adr : root_adr + 3] = np.asarray(root) + origin
    data.qpos[root_adr + 3 : root_adr + 7] = (1.0, 0.0, 0.0, 0.0)
    for name, angle in zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS, strict=True):
        joint_id = model.joint(f"robot/{name}").id
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    for sid, scene in enumerate(manifest.scenes):
        for bid, box in enumerate(scene.boxes):
            prefix = f"obstacle_{sid}_{bid}/"
            mocap_ids = [
                int(model.body_mocapid[i])
                for i in range(model.nbody)
                if model.body(i).name.startswith(prefix)
                and model.body_mocapid[i] >= 0
            ]
            if len(mocap_ids) != 1:
                raise ValueError(f"Expected one native mocap root for {prefix}")
            position = np.asarray(box.center).copy()
            if sid != scene_id:
                position[2] = -100.0
            data.mocap_pos[mocap_ids[0]] = position + origin
            data.mocap_quat[mocap_ids[0]] = (1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    obstacle_contacts = []
    other_contacts = []
    for contact in data.contact:
        geom_ids = [int(contact.geom1), int(contact.geom2)]
        names = [model.geom(i).name for i in geom_ids]
        record = dict(
            geom_ids=geom_ids,
            geom_names=names,
            body_names=[
                model.body(int(model.geom_bodyid[i])).name for i in geom_ids
            ],
            signed_distance_m=float(contact.dist),
            penetration_m=max(0.0, -float(contact.dist)),
            position_scene_m=(contact.pos - origin).tolist(),
            normal_geom0_to_geom1=contact.frame.reshape(3, 3)[0].tolist(),
        )
        if any(n.startswith("obstacle_") for n in names):
            obstacle_contacts.append(record)
        else:
            other_contacts.append(record)
    positions = np.array(
        [data.site(f"robot/{name}").xpos - origin for name in G1_BODY_SCHEMA]
    )
    scene = manifest.scenes[scene_id]
    source = bank.sample_source(
        torch.tensor(positions[None], dtype=torch.float32),
        torch.tensor([scene_id]),
        torch.tensor([0]),
    )
    sdf = source.sdf[0].numpy()
    lower = np.asarray(scene.origin)
    upper = lower + (np.asarray(scene.shape) - 1) * scene.voxel_size
    inside_grid = bool(((positions >= lower) & (positions <= upper)).all())
    return dict(
        label=label,
        scene=scene.id,
        root_position_scene_m=list(root),
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        obstacle_contacts=obstacle_contacts,
        other_contacts=other_contacts,
        obstacle_contact_count=len(obstacle_contacts),
        maximum_obstacle_penetration_m=max(
            (r["penetration_m"] for r in obstacle_contacts), default=0.0
        ),
        all_sites_inside_field_bounds=inside_grid,
        source_field_goal_index=0,
        sites=[
            dict(name=name, position_scene_m=pos.tolist(), source_sdf_m=float(value))
            for name, pos, value in zip(G1_BODY_SCHEMA, positions, sdf, strict=True)
        ],
        minimum_site_source_sdf_m=float(sdf.min()),
        all_site_source_sdf_positive=bool((sdf > 0).all()),
        source_collision_from_sdf_at_threshold_004=bool((sdf < -0.04).any()),
    )


def run_audit(config, cache):
    manifest = load_scene_manifest(config)
    bank = load_field_cache(
        cache, manifest, device="cpu", source_profile={"version": "cat_source_v1"}
    )
    metadata = json.loads(cache.read_text())
    model, origin = build_geometry_model(manifest)
    endpoint_checks = []
    for scene_id, scene in enumerate(manifest.scenes):
        for kind, points in (("start", scene.starts), ("goal", scene.goals)):
            for index, root in enumerate(points):
                endpoint_checks.append(
                    inspect_pose(
                        model,
                        origin,
                        manifest,
                        bank,
                        scene_id,
                        root,
                        f"{kind}_{index}",
                    )
                )
    hurdle_id = bank.scene_names.index("hurdle_crouch")
    counterexamples = [
        inspect_pose(model, origin, manifest, bank, hurdle_id, root, label)
        for root, label in (
            ((-0.60, 0.0, 0.8), "torso_and_head_against_overhead_bar"),
            ((0.30, 0.0, 0.8), "toes_against_low_obstacle"),
        )
    ]
    endpoints_clear = all(
        r["obstacle_contact_count"] == 0 and r["all_sites_inside_field_bounds"]
        for r in endpoint_checks
    )
    blindspots = all(
        r["maximum_obstacle_penetration_m"] > 0 and r["all_site_source_sdf_positive"]
        for r in counterexamples
    )
    if not endpoints_clear or not blindspots:
        raise AssertionError(
            "Static geometry audit did not reproduce expected evidence"
        )
    return dict(
        audit_date="2026-10-19",
        scope="Static collision/point-field counterexamples; no policy or dynamics",
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        script=dict(path=str(Path(__file__).resolve()), sha256=sha256(__file__)),
        packages={
            name: importlib.metadata.version(name)
            for name in ("mjlab", "mujoco", "torch")
        },
        config=dict(path=str(config.resolve()), sha256=sha256(config)),
        cache=dict(
            manifest_path=str(cache.resolve()),
            manifest_sha256=sha256(cache),
            scene_identity=manifest.identity,
            verified_bank_cache_identity=bank.cache_identity,
            source_profile=bank.source_profile,
            verified_files=[
                {**r, "path": str((cache.parent / r["file"]).resolve())}
                for r in metadata["fields"]
            ],
            provenance=(
                "Available local verification cache, not the server evaluation path"
            ),
        ),
        geometry_model=dict(
            construction=(
                "build_cat_cfg + native Scene + production _box_spec; CPU mj_forward"
            ),
            sensors_omitted=True,
            identity_is_checkpoint_contract=False,
            compiled_sensorless_model_sha256=native_model_identity(model),
            environment_origin_m=origin.tolist(),
            source_scenes=[asdict(scene) for scene in manifest.scenes],
            default_joints=dict(
                zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS, strict=True)
            ),
        ),
        result=dict(
            endpoint_pose_count=len(endpoint_checks),
            all_declared_endpoints_obstacle_clear=endpoints_clear,
            positive_point_sdf_despite_native_penetration_reproduced=blindspots,
            actual_episode_fall_cause_established=False,
        ),
        limitations=[
            "Frozen default poses are illustrative, not measured policy states.",
            "Only forward collision/kinematics; no integration or settling.",
            "Endpoint obstacle clearance does not prove traversal feasibility.",
            "Source SDF uses 4 cm voxels and the preserved source interpolation.",
            "Sensors omitted; robot/terrain/obstacle collision geometry is native.",
            "Contact alone does not distinguish support from a destabilizing hit.",
            "Fall diagnosis requires pre-reset episode contact and motion traces.",
        ],
        endpoint_checks=endpoint_checks,
        static_counterexamples=counterexamples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/g1_cat_source.yaml")
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("runs/cat-training-verified-20261015/fields/manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = run_audit(args.config, args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(args.output), result=report["result"])))


if __name__ == "__main__":
    main()
