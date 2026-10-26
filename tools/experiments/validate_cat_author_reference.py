"""Validate a released author policy in its declared scene and original player.

This is a single-trajectory compatibility reference, not a success-rate estimate.
The player writes a temporary XML, so all source/assets are copied first.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort


def namespace(value):
    if isinstance(value, dict):
        return types.SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("Use 1 to 1000 steps for this bounded reference.")
    source, model_dir, output = (
        args.source.resolve(),
        args.model_directory.resolve(),
        args.output.resolve(),
    )
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((model_dir / "config.json").read_text())
    scene = Path(config["env_config"]["pf_config"]["path"])
    if (
        scene.is_absolute()
        or ".." in scene.parts
        or scene.parts[:2] != ("data", "assets")
    ):
        raise ValueError("Reference scene must be a declared relative author asset.")
    copied = output / "reference-source"
    for relative in (Path("cat_ppo"), Path("data/assets/unitree_g1"), scene):
        shutil.copytree(
            source / relative,
            copied / relative,
            ignore=shutil.ignore_patterns("__pycache__", "*_tmp.xml"),
        )
    hashes = {
        str(p.relative_to(copied)): digest(p)
        for p in copied.rglob("*")
        if p.is_file()
    }
    for package in ("cat_ppo", "cat_ppo.utils", "cat_ppo.envs", "cat_ppo.envs.g1"):
        module = types.ModuleType(package)
        module.__path__ = [str(copied.joinpath(*package.split(".")))]
        sys.modules[package] = module
    os.environ["GLI_PATH"] = str(copied / "cat_ppo")
    os.environ["WANDB_PROJECT"] = "cat-author-reference"
    sys.modules["cat_ppo"].registry = importlib.import_module(
        "cat_ppo.utils.registry"
    )
    player = importlib.import_module("cat_ppo.envs.g1.play_cat")
    config["env_config"]["pf_config"]["path"] = str(copied / scene)
    env = player.PlayG1CatEnv(config=namespace(config["env_config"]), headless=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    policy = ort.InferenceSession(
        str(model_dir / "policy.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    np.random.seed(args.seed)
    state = env.reset()
    initial_qpos = env.mj_data.qpos.copy().tolist()
    first_goal, min_distance, fell, invalid = None, float("inf"), False, False
    for step in range(1, args.steps + 1):
        obs = state.obs["state"].reshape(1, -1).astype(np.float32)
        action = policy.run(["continuous_actions"], {"obs": obs})[0][0]
        if (
            obs.shape != (1, 162)
            or action.shape != (12,)
            or not np.isfinite(action).all()
        ):
            raise ValueError("Author policy interface or output is invalid.")
        state = env.step(state, action)
        distance = float(
            np.linalg.norm(env.mj_data.qpos[:2] - env.current_goal_global[:2])
        )
        min_distance = min(min_distance, distance)
        invalid = not (
            np.isfinite(env.mj_data.qpos).all()
            and np.isfinite(env.mj_data.qvel).all()
        )
        fell = bool(
            state.info["head_pos"][2] < 0.7 or env.get_gravity("pelvis")[2] < 0
        )
        if first_goal is None and distance < 0.1 and not (fell or invalid):
            first_goal = step * 0.02
        if fell or invalid:
            break
    result = dict(
        scene=str(scene),
        config_sha256=digest(model_dir / "config.json"),
        policy_sha256=digest(model_dir / "policy.onnx"),
        source_asset_sha256=hashes,
        initial_qpos=initial_qpos,
        goal=env.current_goal_global.tolist(),
        control_dt_seconds=0.02,
        simulation_dt_seconds=float(env.mj_model.opt.timestep),
        steps=step,
        first_goal_seconds=first_goal,
        min_goal_xy_distance_m=min_distance,
        final_goal_xy_distance_m=distance,
        fall=fell,
        invalid=invalid,
        mujoco_version=mujoco.__version__,
        seed=args.seed,
        limits=[
            "One original-player trajectory; not a success-rate estimate.",
            "Original XML collision masks preserved; "
            "contact-free physical success is not certified.",
            "Original fields/interpolation preserved for this reference; "
            "local training retains corrected interpolation and 30 cm foot GF.",
        ],
    )
    (output / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k != "source_asset_sha256"})
    )


if __name__ == "__main__":
    main()
