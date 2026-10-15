"""Run CAT with measured depth/LiDAR fields; no prepare/cache stage required."""

import argparse
import json
import time
from pathlib import Path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/g1_cat_source.yaml")
    )
    parser.add_argument("--sensor", choices=("depth", "lidar"), default="lidar")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint", type=Path)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Use zero latent actions to test wiring, not navigation.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene", action="append")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--update-period", type=float, default=0.1)
    parser.add_argument("--max-age", type=float, default=0.3)
    parser.add_argument("--execution", choices=("async", "sync"), default="async")
    parser.add_argument(
        "--field-backend",
        choices=("cuda", "source"),
        default="cuda",
        help="CUDA Eikonal approximation or source CPU FMM.",
    )
    parser.add_argument(
        "--mapping-backend", choices=("warp", "torch"), default="warp"
    )
    parser.add_argument(
        "--viewer", choices=("none", "native", "viser"), default="none"
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.num_envs < 1 or args.steps < 1:
        raise ValueError("num-envs and steps must be positive.")
    from embodied_learning.data.config import load_experiment_config
    from embodied_learning.data.field_scenes import load_scene_manifest

    settings = load_experiment_config(args.config)
    if (
        settings.get("task_profile") != "cat_source_v1"
        or settings.get("robot") != "unitree_g1"
        or settings.get("source_profile") != {"version": "cat_source_v1"}
    ):
        raise ValueError("Online playback requires a G1 CAT source configuration.")
    manifest = load_scene_manifest(args.config)
    scenes = tuple(args.scene or (manifest.scenes[0].id,))
    if len(set(scenes)) != len(scenes):
        raise ValueError("Repeated scene selection.")
    import torch

    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.simulators.mjlab.cat_online import make_cat_online_env
    from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig

    saved = (
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.checkpoint
        else None
    )
    env = make_cat_online_env(
        manifest,
        scene_ids=scenes,
        num_envs=args.num_envs,
        device=args.device,
        seed=settings.get("seed", 42),
        options=settings.get("task", {}),
        sensor=OnlineSensorConfig(source=args.sensor),
        update_period=args.update_period,
        max_age=args.max_age,
        execution=args.execution,
        field_backend=args.field_backend,
        mapping_backend=args.mapping_backend,
    )
    viewer_owns_env = False
    missed_deadlines = 0
    step_seconds = []
    try:
        if saved is not None:
            from embodied_learning.policies.cat_online import load_online_policy

            contract = PolicyContract.from_dict(saved["field_contract"])
            expected = PolicyContract.from_env(
                env,
                contract.hidden_dims,
                critic_hidden_dims=contract.critic_hidden_dims,
            )
            policy = load_online_policy(
                args.checkpoint, online_contract=expected, device=args.device
            )
        else:

            class SmokePolicy:
                def predict(self, observations):
                    return torch.zeros(env.num_envs, 12, device=env.device)

            policy = SmokePolicy()
        # Validate saved task options first; deployment disables training noise.
        env.cat_state.options = {
            **env.cat_state.options,
            "noise_level": 0.0,
            "random_pd": False,
            "random_rfi": False,
            "push": False,
        }
        action = env.unwrapped.action_manager.get_term("joint_pos")
        action.cfg.random_pd = action.cfg.random_rfi = False
        controller = env.unwrapped.cat_controller
        controller.kp_scale.fill_(1.0)
        controller.kd_scale.fill_(1.0)
        controller.rfi_scale.zero_()
        env.unwrapped.reset()
        if args.viewer != "none":
            from embodied_learning.cli.field_play import play_policy

            viewer_owns_env = True
            play_policy(
                env, policy, viewer=args.viewer, port=args.port, steps=args.steps
            )
        else:
            obs = env.get_observations()
            deadline = time.monotonic()
            with torch.inference_mode():
                for _ in range(args.steps):
                    started = time.monotonic()
                    obs, _, _, _ = env.step(policy.predict(obs))
                    if not torch.isfinite(obs["actor"]).all():
                        raise RuntimeError("Nonfinite online actor observations.")
                    step_seconds.append(time.monotonic() - started)
                    if args.execution == "async":
                        deadline += env.unwrapped.step_dt
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            time.sleep(remaining)
                        else:
                            missed_deadlines += 1
                            deadline = time.monotonic()
        print(
            json.dumps(
                dict(
                    sensor=args.sensor,
                    smoke=args.smoke,
                    headless_timing=dict(
                        target_control_hz=50 if args.execution == "async" else None,
                        missed_deadlines=missed_deadlines,
                        steps=len(step_seconds),
                        mean_step_seconds=(
                            sum(step_seconds) / len(step_seconds)
                            if step_seconds
                            else None
                        ),
                        max_step_seconds=max(step_seconds, default=None),
                    ),
                    **env.cat_state.bank.diagnostics(),
                )
            )
        )
    finally:
        if not viewer_owns_env:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
