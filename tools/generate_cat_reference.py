"""Execute pinned upstream methods, without importing its training stack.

Development only: requires JAX and a checkout supplied as --source.
No upstream formula is reimplemented in this oracle.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jp
import numpy as np

COMMIT = "866ba392f1c1e84b92ad75fa66550f26e8af8e48"


def load_methods(source):
    namespace = {"jax": jax, "jp": jp, "np": np, "EPS": 1e-6}
    methods = {}
    hashes = {}
    for name, cls in [("env_loco.py", "G1LocoEnv"), ("env_cat.py", "G1CatEnv")]:
        path = source / "cat_ppo/envs/g1" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = ast.parse(path.read_text())
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and (
                fn.name.startswith(("_reward_", "_cost_", "_re_"))
                or fn.name
                in ("_get_reward", "compute_cmd_from_rtf", "_update_phase")
            ):
                module = ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        fn,
                    ],
                    type_ignores=[],
                )
                exec(
                    compile(ast.fix_missing_locations(module), str(path), "exec"),
                    namespace,
                )
                methods[fn.name] = namespace[fn.name]
    return type("AuthorMethods", (), methods), hashes


def generate(source, output):
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != COMMIT:
        raise ValueError("Reference checkout must match pinned commit.")
    cls, hashes = load_methods(source)
    rng = np.random.default_rng(72)
    b = 16

    def rand(shape, scale=1):
        return (rng.normal(size=shape) * scale).astype(np.float32)

    values = dict(
        pelvis_rpy=rand((b, 3), 0.2),
        torso_rpy=rand((b, 3), 0.2),
        head_pos=rand((b, 3)),
        cmd_vel=rand((b, 3)),
        root_vel=rand((b, 3)),
        torso_ang_vel=rand((b, 3)),
        leg_rotation=rand((b, 6, 3, 3)),
        foot_pos=rand((b, 2, 3)),
        foot_vel=rand((b, 2, 3)),
        hands_pos=rand((b, 2, 3)),
        pelvis_subtree_com=rand((b, 3)),
        feet_contact=rng.integers(0, 2, (b, 2)).astype(np.float32),
        gait_mask=rng.integers(-1, 2, (b, 2)).astype(np.float32),
        move_flag=rng.choice([0.0, 0.75, 1.0], b).astype(np.float32),
        foot_height=np.full(b, 0.07, np.float32),
        joint_pos=rand((b, 29)),
        joint_vel=rand((b, 29)),
        previous_vel=rand((b, 29)),
        torques=rand((b, 29)),
        actions=rand((b, 12)),
        last_actions=rand((b, 12)),
        last_last_actions=rand((b, 12)),
        gf=rand((b, 11, 3)),
        bf=rand((b, 11, 3)),
        sdf=rand((b, 11), 0.2),
        navigation_rotation=np.broadcast_to(
            np.eye(3, dtype=np.float32), (b, 3, 3)
        ).copy(),
    )
    values["root_vel"][0] = 0
    values["gf"][1] = 0
    values["sdf"][2] = 5
    values["sdf"][3] = -1
    values["body_velocities"] = rand((b, 11, 3))
    lower = np.full(29, -0.5, np.float32)
    upper = np.full(29, 0.5, np.float32)
    author = cls()
    author.dt = 0.02
    author._soft_lowers = jp.array(lower)
    author._soft_uppers = jp.array(upper)
    author.obs_joint_ids = jp.array(list(range(19)) + list(range(22, 26)))
    author._knee_indices = [3, 9]
    author._feet_site_id = jp.array([0, 1])
    author.body_id_pelvis = 0
    author.body_ids_left_leg = jp.array([0, 1, 2])
    author.body_ids_right_leg = jp.array([3, 4, 5])
    author._foot_linvel_sensor_adr = jp.arange(6).reshape(2, 3)
    author._config = NS(
        torso_height=[0.5, 1.0],
        ang_vel_yaw=[-0.5, 0.5],
        reward_config=NS(foot_height_stance=0),
    )
    output_values = {f"input_{k}": v for k, v in values.items()}
    rewards = {}
    for i in range(b):
        x = {k: jp.array(v[i]) for k, v in values.items()}
        data = NS(
            qpos=jp.concatenate([jp.array([0, 0, 0.8, 1, 0, 0, 0]), x["joint_pos"]]),
            qvel=jp.concatenate([jp.zeros(6), x["joint_vel"]]),
            site_xpos=x["foot_pos"],
            subtree_com=x["pelvis_subtree_com"][None],
            xmat=x["leg_rotation"],
            actuator_force=x["torques"],
            sensordata=x["foot_vel"].reshape(-1),
        )
        info = dict(
            command=jp.concatenate([x["move_flag"][None], x["cmd_vel"]]),
            navi_pelvis_rpy=x["pelvis_rpy"],
            navi_torso_rpy=x["torso_rpy"],
            head_pos=x["head_pos"],
            feet_pos=x["foot_pos"],
            hands_pos=x["hands_pos"],
            global_lin_vel=x["root_vel"],
            navi_torso_ang_vel=x["torso_ang_vel"],
            navi2world_rot=x["navigation_rotation"],
            navi2world_pose=jp.eye(4).at[:3, :3].set(x["navigation_rotation"]),
            gait_mask=x["gait_mask"],
            foot_height=x["foot_height"],
            last_joint_vel=x["previous_vel"],
            last_act=x["last_actions"],
            last_last_act=x["last_last_actions"],
        )
        for label, ids in [
            ("head", [0]),
            ("feet", [3, 4]),
            ("hands", [5, 6]),
            ("knees", [7, 8]),
            ("shlds", [9, 10]),
        ]:
            info[label + "gf"] = x["gf"][jp.array(ids)]
            info[label + "df"] = x["sdf"][jp.array(ids)]
            if label in ("head", "feet", "hands"):
                info[label + "_vel"] = x["body_velocities"][jp.array(ids)]
        # Independent point velocities are included in the production input.
        info["head_vel"] = x["body_velocities"][jp.array([0])]
        result = author._get_reward(
            data, x["actions"], info, jp.array(False), x["feet_contact"]
        )
        for k, v in result.items():
            rewards.setdefault(k, []).append(np.asarray(v))
    output_values.update({f"reward_{k}": np.asarray(v) for k, v in rewards.items()})
    author._stop_cmd = jp.zeros(4)
    output_values["command"] = np.stack(
        [
            np.asarray(
                author.compute_cmd_from_rtf(
                    jp.array(values["gf"][i, 1]),
                    jp.array(values["gf"][i, [0, 3, 4, 5, 6]]),
                    jp.array(values["bf"][i, [0, 3, 4, 5, 6]]),
                )
            )
            for i in range(b)
        ]
    )
    author._gait_bound = 0.6
    author._stop_cmd = jp.zeros(4)
    author._stance_phase = jp.zeros(2)
    state = NS(
        info=dict(
            last_command=jp.zeros(4),
            phase=jp.array([0.0, np.pi]),
            phase_dt=jp.float32(2 * np.pi * 0.02 * 1.4),
            stop_timestep=jp.array(100),
        )
    )
    commands = []
    results = []
    phases = []
    masks = []
    for i in range(64):
        command = jp.array([0.75, 0.4, 0.0, 0.0]) if i < 3 else jp.zeros(4)
        commands.append(np.asarray(command))
        state.info["command"] = command
        author._update_phase(state)
        results.append(np.asarray(state.info["command"]))
        phases.append(np.asarray(state.info["phase"]))
        masks.append(np.asarray(state.info["gait_mask"]))
        state.info["last_command"] = state.info["command"]
    output_values.update(
        gait_commands=np.stack(commands),
        gait_result=np.stack(results),
        gait_phase=np.stack(phases),
        gait_mask=np.stack(masks),
    )
    output_values.update(soft_lower=lower, soft_upper=upper)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "cat_source_reference.npz", **output_values)
    (output / "cat_source_reference.json").write_text(
        json.dumps(
            dict(commit=COMMIT, files=hashes, seed=72, batch=b, jax=jax.__version__),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures"))
    args = parser.parse_args()
    generate(args.source, args.output)
