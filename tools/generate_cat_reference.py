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
        if name=='env_cat.py':
            for function in tree.body:
                if isinstance(function,ast.FunctionDef) and function.name in ('world_to_navi_vel','world_to_navi_pos'):
                    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and (
                fn.name.startswith(("_reward_", "_cost_", "_re_"))
                or fn.name
                in (
                    "_get_reward",
                    "compute_cmd_from_rtf",
                    "_update_phase",
                    "_get_obs", "sample_field",
                    "world_to_grid",
                )
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
    obs_values=dict(gyro=rand((b,3)),gravity=rand((b,3)),joint_pos=values['joint_pos'],joint_vel=values['joint_vel'],default_joints=np.zeros((b,29),np.float32),last_actions=values['last_actions'],targets=rand((b,29)),command=np.concatenate([values['move_flag'][:,None],values['cmd_vel']],axis=1),foot_height=values['foot_height'],phase=rand((b,2)),root_local_vel=values['root_vel'],gf=values['gf'],bf=values['bf'],sdf=values['sdf'],body_positions=rand((b,11,3)),body_velocities=values['body_velocities'],torso_rpy=values['torso_rpy'],gait_mask=values['gait_mask'],feet_contact=values['feet_contact'],kp_scale=np.ones((b,1),np.float32),kd_scale=np.ones((b,1),np.float32),rfi_scale=rand((b,29)),navigation_rotation=values['navigation_rotation'],gf_delayed=values['gf'],bf_delayed=values['bf'],sdf_delayed=values['sdf'],command_delayed=np.concatenate([values['move_flag'][:,None],values['cmd_vel']],axis=1))
    obs_values['body_positions'][:,0]=values['head_pos'];obs_values['body_positions'][:,3:5]=values['foot_pos'];obs_values['body_positions'][:,5:7]=values['hands_pos']
    actor_outputs=[];critic_outputs=[]
    author.action_joint_ids=jp.arange(12);author._default_qpos=jp.zeros(29)
    author._config.noise_config=NS(level=0.,scales=NS(gyro=.2,gravity=.05,joint_pos=.03,joint_vel=1.5))
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
        ox={k:jp.array(v[i]) for k,v in obs_values.items()}
        author.get_gyro=lambda data,frame:ox['gyro']
        author.get_local_linvel=lambda data,frame:ox['root_local_vel']
        # Use an exact identity orientation with an independently supplied gravity vector below.
        data.site_xmat=jp.stack([jp.eye(3),jp.eye(3)])
        author._pelvis_imu_site_id=0
        ox['gravity']=jp.array([0.,0.,-1.]);obs_values['gravity'][i]=np.array([0.,0.,-1.])
        info.update(rng=jax.random.PRNGKey(i),phase=ox['phase'],motor_targets=ox['targets'],kp_scale=jp.array(1.),kd_scale=jp.array(1.),rfi_lim_scale=ox['rfi_scale'],command_delay=ox['command_delayed'],pelv_pos=ox['body_positions'][1],tors_pos=ox['body_positions'][2],knees_pos=ox['body_positions'][7:9],shlds_pos=ox['body_positions'][9:11])
        for label,ids in [('head',[0]),('pelv',[1]),('tors',[2]),('feet',[3,4]),('hands',[5,6]),('knees',[7,8]),('shlds',[9,10])]:
            for suffix,key in [('gf','gf'),('bf','bf'),('df','sdf')]:
                val=ox[key][jp.array(ids)]
                if suffix=='df':val=val[:,None]
                info[label+suffix]=val
                info[label+suffix+'_delay']=val
        actor_obs=author._get_obs(data,info,ox['feet_contact'])
        actor_outputs.append(np.asarray(actor_obs['state']));critic_outputs.append(np.asarray(actor_obs['privileged_state']))
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
    import skfmm

    path = source / "procedural_obstacle_generation/pf_modular.py"
    hashes["pf_modular.py"] = hashlib.sha256(path.read_bytes()).hexdigest()
    ns = {"np": np, "skfmm": skfmm}
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("make_sdf", "grad3", "make_guidance_field_progressive")
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
    occupied = np.zeros((9, 9, 9), bool)
    occupied[3:5, 3:5, 2:6] = True
    grid = np.meshgrid(*(np.arange(9) * 0.1 for _ in range(3)), indexing="ij")
    goal = np.array([0.6, 0.6, 0.6])
    cfg = NS(voxel=0.1, v_max=0.6, goal_seed_r=0.12)
    sdf = ns["make_sdf"](occupied, 0.1)
    bf = ns["grad3"](sdf, 0.1)
    t, gf = ns["make_guidance_field_progressive"](cfg, grid, occupied, goal, bf, sdf)
    output_values.update(
        grid_occupied=occupied,
        grid_goal=goal,
        grid_sdf=sdf,
        grid_geodesic=t,
        grid_guidance=gf,
    )
    author.pf_origin = jp.zeros(3)
    author.dx = 0.1
    author.Nx = author.Ny = author.Nz = 9
    queries = jp.array([[-4.0, 0.25, 0.33], [0.21, 0.35, 0.19]])
    output_values["query_guidance"] = np.asarray(
        author.sample_field(jp.array(gf), queries)
    )[:, None, :]
    output_values.update({f'obs_{k}':v for k,v in obs_values.items()})
    output_values.update(obs_actor=np.stack(actor_outputs),obs_critic=np.stack(critic_outputs))
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
