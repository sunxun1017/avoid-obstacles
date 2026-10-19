"""Portable G1 field PPO/DAgger application composition.

Run from the uploaded repository root. Help does not import Torch or mjlab.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from embodied_learning.data.config import load_experiment_config


def read_teachers(path: Path, scenes: tuple[str, ...]) -> dict[str, Path]:
    """Resolve portable teacher paths relative to the teacher manifest."""
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or any(s not in values for s in scenes):
        raise ValueError("Missing teacher for a requested scene.")
    result = {}
    for scene in scenes:
        if not isinstance(values[scene], str):
            raise ValueError("Teacher paths must be strings.")
        checkpoint = (path.parent / values[scene]).resolve()
        if not checkpoint.is_file():
            raise ValueError(f"Missing teacher checkpoint: {scene}.")
        result[scene] = checkpoint
    return result


def preflight_teachers(
    paths: dict[str, Path],
    *,
    scene_hash: str,
    scene_names: tuple[str, ...],
    task_profile: str | None = None,
) -> dict[str, dict]:
    """Check checkpoint schema and scene routing before constructing physics."""
    import torch

    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.robots.g1 import G1_BODY_SCHEMA

    records = {}
    first = None
    dimensions = {
        None: (173, 173, 29),
        "cat_source_v1": (162, 250, 12),
        "cat_recovery_v1": (171, 256, 12),
    }
    if task_profile not in dimensions:
        raise ValueError("Unknown teacher task profile.")
    actor_dim, critic_dim, action_dim = dimensions[task_profile]
    for name, path in paths.items():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        try:
            contract = PolicyContract.from_dict(saved["field_contract"])
            schema = contract.schema
            if (
                schema["scene_hash"] != scene_hash
                or tuple(schema["body_sites"]) != G1_BODY_SCHEMA
                or schema["scene_names"] != list(scene_names)
                or contract.actor_dim != actor_dim
                or contract.critic_dim != critic_dim
                or contract.action_dim != action_dim
                or schema.get("task_profile") != task_profile
                or scene_names.index(name) not in saved["training_scenes"]
            ):
                raise ValueError("Teacher contract mismatch.")
            if first:
                first.require(contract, network=False)
            else:
                first = contract
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Teacher contract is missing or invalid.") from error
        records[name] = saved
    return records


def check_rsl_rl() -> None:
    """Detect an incompatible editable RSL package before physics startup."""
    import importlib
    import sys

    try:
        module = importlib.import_module("rsl_rl.models")
        if not hasattr(module, "MLPModel"):
            raise ImportError("MLPModel is missing")
    except ImportError as error:
        package = sys.modules.get("rsl_rl")
        path = getattr(package, "__file__", "not installed")
        raise RuntimeError(
            f"Incompatible RSL-RL import from {path}, Python={sys.executable}. "
            "Use the project .venv-field and official rsl-rl-lib==5.4.2; "
            "see docs/cat_source_training.md."
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1 paper-field PPO and DAgger")
    commands = parser.add_subparsers(dest="stage", required=True)
    for stage in ("prepare", "ppo", "dagger", "evaluate", "play"):
        command = commands.add_parser(stage)
        command.add_argument(
            "--config", type=Path, default=Path("configs/experiments/g1_field.yaml")
        )
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--seed", type=int, default=None)
        if stage == "prepare":
            continue
        command.add_argument("--scene", action="append", help="Repeat for subsets.")
        command.add_argument(
            "--num-envs", type=int, default=1 if stage == "play" else 1024
        )
        command.add_argument("--hidden-dims", nargs="+", type=int)
        if stage in ("ppo", "dagger"):
            command.add_argument(
                "--iterations", type=int, default=30000 if stage == "ppo" else 1000
            )
            command.add_argument(
                "--rollout-steps", type=int, default=None if stage == "ppo" else 64
            )
            initialization = command.add_mutually_exclusive_group()
            initialization.add_argument("--resume", type=Path)
            if stage == "ppo":
                initialization.add_argument(
                    "--warmstart",
                    type=Path,
                    help="Initialize research weights from a source checkpoint.",
                )
        if stage == "ppo":
            command.add_argument("--epochs", type=int)
            command.add_argument("--mini-batches", type=int, default=None)
        if stage == "dagger":
            command.add_argument("--teachers", type=Path)
            command.add_argument("--updates", type=int, default=100)
            command.add_argument("--batch-size", type=int, default=512)
            command.add_argument("--capacity", type=int, default=200000)
            command.add_argument("--beta-start", type=float, default=1.0)
            command.add_argument("--beta-rounds", type=int, default=20)
        if stage in ("evaluate", "play"):
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument(
                "--steps", type=int, default=None if stage == "play" else 1000
            )
            command.add_argument("--no-perturbations", action="store_true")
        if stage == "play":
            command.add_argument(
                "--viewer", choices=("native", "viser"), default="native"
            )
            command.add_argument("--port", type=int, default=8765)
    return parser


def resolve_training_defaults(args, settings):
    """CLI overrides experiment defaults; reject ambiguous booleans as integers."""
    if args.seed is None:
        args.seed = settings.get("seed", 42)
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if args.stage == "ppo":
        if args.mini_batches is None:
            args.mini_batches = settings.get("ppo", {}).get("mini_batches", 4)
        if type(args.mini_batches) is not int or args.mini_batches < 1:
            raise ValueError("mini_batches must be a positive integer.")


def main(argv: list[str] | None = None) -> int:
    """Validate inputs, compose the native backend, execute one selected stage."""
    args = _parser().parse_args(argv)
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields

    settings = load_experiment_config(args.config)
    manifest = load_scene_manifest(args.config)
    resolve_training_defaults(args, settings)
    if settings.get("robot") != "unitree_g1":
        raise ValueError("The training configuration must select unitree_g1.")
    from embodied_learning.tasks.field_traversal import parse_task_options

    profile = settings.get("task_profile")
    if profile not in (None, "cat_source_v1", "cat_recovery_v1"):
        raise ValueError("Unknown task profile.")
    source_profile = settings.get("source_profile") if profile else None
    if profile and source_profile != {"version": "cat_source_v1"}:
        raise ValueError("Invalid source field profile.")
    if not profile:
        parse_task_options(settings.get("task", {}))
    scene_names = tuple(s.id for s in manifest.scenes)
    output = args.output_dir or Path(
        "runs/g1-cat-recovery"
        if profile == "cat_recovery_v1"
        else "runs/g1-cat-source"
        if profile
        else "runs/g1-field"
    )
    cache_path = output / "fields" / "manifest.json"
    if args.stage == "prepare":  # 也就是在准备离线地图和障碍是吗
        from embodied_learning.simulators.mjlab.field_traversal import (
            validate_g1_clearance,
        )

        source_joints = None
        if profile:
            from embodied_learning.robots.g1 import (
                CAT_DEFAULT_JOINTS,
                CAT_JOINT_NAMES,
            )

            source_joints = dict(
                zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS, strict=True)
            )
        validate_g1_clearance(manifest, joint_positions=source_joints)
        prepared = prepare_fields(  # 为配置里的场景和目标离线计算并缓存体素势场
            manifest, output / "fields", source_profile=source_profile
        )
        print(
            json.dumps(
                dict(field_cache=str(prepared.path), scene_hash=prepared.identity)
            )
        )
        return 0
    scenes = tuple(args.scene or scene_names)
    if (
        not scenes
        or len(set(scenes)) != len(scenes)
        or any(s not in scene_names for s in scenes)
    ):
        raise ValueError("Requested scenes must be unique known scene names.")
    if args.num_envs < 1:
        raise ValueError("num-envs must be positive.")
    if args.stage == "ppo" and len(scenes) != 1:
        raise ValueError("A PPO specialist requires exactly one --scene.")
    import torch

    check_rsl_rl()
    from embodied_learning.algorithms.dagger import (
        DaggerConfig,
        evaluate_student,
        train_dagger,
    )
    from embodied_learning.algorithms.ppo import PpoConfig, train_specialist
    from embodied_learning.policies.field_policy import (
        FieldStudent,
        load_student,
        load_teacher,
    )
    from embodied_learning.protocols.imitation import PolicyContract

    teachers, records = {}, {}
    if args.stage == "ppo":
        ppo = settings.get("ppo", {})
        config = PpoConfig(
            iterations=args.iterations,
            rollout_steps=args.rollout_steps or ppo.get("rollout_steps", 24),
            hidden_dims=tuple(
                args.hidden_dims or ppo.get("hidden_dims", [512, 256, 128])
            ),
            critic_hidden_dims=tuple(ppo["critic_hidden_dims"])
            if ppo.get("critic_hidden_dims")
            else None,
            normalization=ppo.get("normalization", True),
            gamma=ppo.get("gamma", 0.99),
            learning_rate=ppo.get("learning_rate", 0.001),
            epochs=args.epochs or ppo.get("epochs", 5),
            mini_batches=args.mini_batches,
            save_interval=ppo.get("save_interval", 100),
            seed=args.seed,
        )
    elif args.stage == "dagger":
        teachers = read_teachers(args.teachers or output / "teachers.json", scenes)
        records = preflight_teachers(
            teachers,
            scene_hash=manifest.identity,
            scene_names=scene_names,
            task_profile=profile,
        )
        config = DaggerConfig(
            iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            updates=args.updates,
            batch_size=args.batch_size,
            capacity=args.capacity,
            beta_start=args.beta_start,
            beta_rounds=args.beta_rounds,
            seed=args.seed,
        )
    else:
        if args.steps is not None and args.steps < 1:
            raise ValueError("steps must be positive.")
        records["evaluation"] = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )
    if getattr(args, "resume", None) and not args.resume.is_file():
        raise ValueError("Resume checkpoint does not exist.")
    if getattr(args, "warmstart", None) and not args.warmstart.is_file():
        raise ValueError("Warmstart checkpoint does not exist.")
    bank = load_field_cache(
        cache_path, manifest, device=args.device, source_profile=source_profile
    )
    from embodied_learning.simulators.mjlab.field_traversal import make_field_env

    if profile:
        from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env

        factory = make_cat_env
    else:
        factory = make_field_env
    research = (
        dict(task_profile=profile, recovery_options=settings.get("recovery", {}))
        if profile == "cat_recovery_v1"
        else {}
    )
    env = factory(
        manifest,
        bank,
        scene_ids=scenes,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        options=settings.get("task", {}),
        **research,
    )
    playback_owns_env = False
    try:
        if args.stage == "ppo":
            checkpoint = train_specialist(
                env,
                config,
                output / "specialists" / scenes[0],
                resume=args.resume,
                warmstart=args.warmstart,
            )
            teacher_manifest = output / "teachers.json"
            values = (
                json.loads(teacher_manifest.read_text())
                if teacher_manifest.exists()
                else {}
            )
            values[scenes[0]] = checkpoint.relative_to(output).as_posix()
            temporary = teacher_manifest.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, indent=2) + "\n")
            temporary.replace(teacher_manifest)
            print(json.dumps(dict(checkpoint=str(checkpoint))))
        elif args.stage == "dagger":
            experts = {}
            for name, path in teachers.items():
                saved_contract = PolicyContract.from_dict(
                    records[name]["field_contract"]
                )
                expected = PolicyContract.from_env(
                    env,
                    saved_contract.hidden_dims,
                    critic_hidden_dims=saved_contract.critic_hidden_dims,
                    normalization=saved_contract.normalization,
                )
                experts[scene_names.index(name)] = load_teacher(
                    path, expected_contract=expected, device=args.device
                )
            contract = PolicyContract.from_env(
                env,
                tuple(
                    args.hidden_dims
                    or settings.get("ppo", {}).get("hidden_dims", [512, 256, 128])
                ),
                normalization=settings.get("ppo", {}).get("normalization"),
            )
            student = FieldStudent(contract, args.device)
            checkpoint = train_dagger(
                env,
                student,
                experts,
                config,
                output / "generalist",
                resume=args.resume,
            )
            print(json.dumps(dict(checkpoint=str(checkpoint))))
        else:
            saved = records["evaluation"]
            contract = PolicyContract.from_dict(saved["field_contract"])
            expected = PolicyContract.from_env(
                env,
                contract.hidden_dims,
                critic_hidden_dims=contract.critic_hidden_dims,
                normalization=contract.normalization,
            )
            loader = load_student if "student" in saved else load_teacher
            policy = loader(
                args.checkpoint, expected_contract=expected, device=args.device
            )
            overrides = {}
            if args.no_perturbations:
                if not profile:
                    raise ValueError(
                        "--no-perturbations is available for source task only."
                    )
                overrides = dict(
                    noise_level=0.0, push=False, random_pd=False, random_rfi=False
                )
                env.cat_state.options = {**env.cat_state.options, **overrides}
                c = env.unwrapped.cat_controller
                action = env.unwrapped.action_manager.get_term("joint_pos")
                action.cfg.random_pd = False
                action.cfg.random_rfi = False
                c.kp_scale.fill_(1.0)
                c.kd_scale.fill_(1.0)
                c.rfi_scale.zero_()
                env.cat_state._obs = None
                env.cat_state._key = None
                env.unwrapped.observation_manager.reset()
            if args.stage == "play":
                from embodied_learning.cli.field_play import play_policy

                playback_owns_env = True
                play_policy(
                    env, policy, viewer=args.viewer, port=args.port, steps=args.steps
                )
                return 0
            metrics = evaluate_student(env, policy, steps=args.steps)
            metrics["evaluation_overrides"] = overrides
            metrics_path = output / "evaluation.json"
            metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics))
    finally:
        if not playback_owns_env:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
