"""Prepare, train and evaluate one explicit geometry-curriculum stage."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(CODE / "tools"))
from embodied_learning.data.config import load_experiment_config
from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.tasks.cat_curriculum import (
    require_original_task_contract,
    robot_fingerprint,
    stage_settings,
    validate_transfer,
)
from embodied_learning.tasks.cat_foot_guidance import (
    load_foot_bank,
    prepare_foot_cache,
    sha256,
)
from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def config(protocol, updates):
    from embodied_learning.algorithms.ppo import PpoConfig

    values = dict(protocol["ppo"])
    values["hidden_dims"] = tuple(values["hidden_dims"])
    if values["critic_hidden_dims"] is not None:
        values["critic_hidden_dims"] = tuple(values["critic_hidden_dims"])
    return replace(PpoConfig(**values), iterations=updates)


def initialize(
    root, *, recipe="foundation_v1", parent_checkpoint=None,
    training_seed=42, terminal_gate_seeds=None,
):
    """Create a new, reviewable protocol; never rewrite an existing run."""
    from embodied_learning.tasks.cat_hurdle import parent_settings

    if recipe not in ("foundation_v1", "feetgf_progress_stage1_v1"):
        raise ValueError("Unknown curriculum recipe.")
    if type(training_seed) is not int or training_seed < 0:
        raise ValueError("Training seed must be a nonnegative integer")
    gated = terminal_gate_seeds is not None
    if gated:
        if recipe != "foundation_v1":
            raise ValueError("Automatic gate initialization requires the from-scratch recipe")
        gate_seeds = tuple(terminal_gate_seeds)
        if (
            len(gate_seeds) != 2
            or any(type(seed) is not int or seed < 0 for seed in gate_seeds)
            or len({training_seed, *gate_seeds}) != 3
        ):
            raise ValueError("Training and both evaluation seeds must be distinct nonnegative integers")
    saved = None
    if recipe == "feetgf_progress_stage1_v1":
        if parent_checkpoint is None:
            raise ValueError("The hurdle recipe requires a parent checkpoint.")
        parent_checkpoint = Path(parent_checkpoint).resolve()
        saved = parent_settings(parent_checkpoint)
    elif parent_checkpoint is not None:
        raise ValueError("Parent checkpoint requires the hurdle recipe.")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Initialization requires an empty experiment directory.")
    original = load_experiment_config(
        CODE / "configs/experiments/g1_cat_source.yaml"
    )
    write_json(root / "original.yaml", original)
    write_json(
        root / "protocol.json",
        dict(
            experiment=root.name,
            curriculum_profile="foundation_v1",
            progression="terminal_20s_v1" if gated else "manual",
            stage_initialization="actor_only",
            foot_radius_m=0.30,
            source_interpolation="xyz_weights_v1",
            optimization_profile="brax_0_12_3_v1",
            training_envs=4096,
            total_updates=600 if gated else 611,
            chunk_updates=50 if gated else 32,
            evaluation_episodes=128 if gated else 256,
            evaluation_profile="source_train" if gated else "author_playback",
            **(
                {"terminal_gate_seeds": list(gate_seeds)}
                if gated else {"gate_seed": 20261022, "final_seed": 20261022}
            ),
            ppo=dict(
                iterations=600 if gated else 611,
                rollout_steps=160,
                hidden_dims=[256, 128, 64],
                critic_hidden_dims=[512, 256, 128],
                normalization=False,
                gamma=0.97,
                learning_rate=3e-4,
                epochs=4,
                mini_batches=32,
                save_interval=50 if gated else 16,
                seed=training_seed,
                optimization_profile="brax_0_12_3_v1",
                source_unroll_steps=20,
            ),
        ),
    )
    if saved is not None:
        from embodied_learning.tasks.cat_hurdle import configure_recipe

        protocol = json.loads((root / "protocol.json").read_text())
        configure_recipe(original, protocol, parent_checkpoint, saved)
        write_json(root / "original.yaml", original)
        write_json(root / "protocol.json", protocol)


def stages_for(root, protocol):
    return stage_settings(
        load_experiment_config(root / "original.yaml"),
        profile=protocol.get("curriculum_profile", "legacy_combination_v1"),
    )


def prepare(root, level=None):
    from embodied_learning.robots.g1 import CAT_DEFAULT_JOINTS, CAT_JOINT_NAMES
    from embodied_learning.simulators.mjlab.field_traversal import (
        validate_g1_clearance,
    )

    protocol = json.loads((root / "protocol.json").read_text())
    original = load_experiment_config(root / "original.yaml")
    prepared = []
    stages = stages_for(root, protocol)
    if level is not None and not 0 <= level < len(stages):
        raise ValueError("Invalid curriculum stage.")
    for i, settings in enumerate(stages):
        if level is not None and i != level:
            continue
        level_index = i
        directory = root / "stages" / str(level_index)
        path = directory / "config.json"
        if path.exists() and json.loads(path.read_text()) != settings:
            raise ValueError("Existing stage configuration mismatch.")
        write_json(path, settings)
        manifest = load_scene_manifest(path)
        validate_g1_clearance(
            manifest,
            joint_positions=dict(
                zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS, strict=True)
            ),
        )
        if level_index == len(stages) - 1 and protocol.get("reference_fields"):
            # Original geometry and base fields; generate the requested 30 cm GF.
            fields = Path(protocol["reference_fields"])
            foot_fields = directory / "foot_fields"
        else:
            fields, foot_fields = directory / "fields", directory / "foot_fields"
            prepare_fields(
                manifest, fields, source_profile=settings["source_profile"]
            )
        base = load_field_cache(
            fields / "manifest.json",
            manifest,
            device="cpu",
            source_profile=settings["source_profile"],
        )
        bank = prepare_foot_cache(
            base, manifest, foot_fields, radius=protocol["foot_radius_m"]
        )
        record = dict(
            stage=level_index,
            curriculum_profile=protocol.get(
                "curriculum_profile", "legacy_combination_v1"
            ),
            boxes=next(
                s["boxes"] for s in settings["scenes"] if s["id"] == "hurdle_crouch"
            ),
            scene_identity=manifest.identity,
            fields=str(fields),
            foot_fields=str(foot_fields),
            composite_cache_identity=bank.cache_identity,
            config_sha256=sha256(path),
        )
        write_json(directory / "assets.json", record)
        prepared.append(record)
        print(json.dumps({"prepared": record}), flush=True)
        del base, bank
    assert stages[-1] == original
    write_json(
        root / "prepared.json", dict(stages=prepared, final_geometry_exact=True)
    )


def make_env(root, level, *, num_envs, seed, execution_profile="source_train"):
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env

    protocol = json.loads((root / "protocol.json").read_text())
    directory = root / "stages" / str(level)
    assets = json.loads((directory / "assets.json").read_text())
    settings = load_experiment_config(directory / "config.json")
    manifest = load_scene_manifest(directory / "config.json")
    assert assets["config_sha256"] == sha256(directory / "config.json")
    base = load_field_cache(
        Path(assets["fields"]) / "manifest.json",
        manifest,
        device="cuda:0",
        source_profile=settings["source_profile"],
    )
    base = replace(
        base, source_interpolation=protocol.get("source_interpolation", "legacy_xz")
    )
    bank = load_foot_bank(
        base, assets["foot_fields"], radius=protocol["foot_radius_m"]
    )
    assert bank.cache_identity == assets["composite_cache_identity"]
    env = make_cat_env(
        manifest,
        bank,
        foot_contact_force=protocol.get("foot_contact_force"),
        risk_guidance=protocol.get("risk_guidance"),
        scene_ids=["hurdle_crouch"],
        num_envs=num_envs,
        device="cuda:0",
        seed=seed,
        options=settings["task"],
        execution_profile=execution_profile,
        rfi_amplitude_multiplier=protocol.get("rfi_amplitude_multiplier", 1.0),
    )
    env.schema["foot_guidance_experiment"] = bank.experiment_schema
    slip_mode = protocol.get("standing_slip_mode", "phase_mask_v1")
    if slip_mode not in ("phase_mask_v1", "ground_contact_v1"):
        env.close()
        raise ValueError("Unknown standing slip mode.")
    if slip_mode == "ground_contact_v1":
        env.cat_state.standing_contact_slip = True
        env.schema["standing_slip_mode"] = slip_mode
    balance_mode = protocol.get("standing_balance_mode", "original_v1")
    if balance_mode not in ("original_v1", "stand_free_v1"):
        env.close()
        raise ValueError("Unknown standing balance mode.")
    if balance_mode == "stand_free_v1":
        env.cat_state.standing_balance_free = True
        env.schema["standing_balance_mode"] = balance_mode
    from embodied_learning.tasks.cat_hurdle import apply_runtime_settings

    apply_runtime_settings(env, protocol)
    if "source_interpolation" in protocol:
        env.schema["source_interpolation"] = protocol["source_interpolation"]
    if protocol.get("optimization_profile") == "brax_0_12_3_v1":
        env.schema["optimization_profile"] = "brax_0_12_3_v1"
        env.schema["policy_activation"] = "swish"
    env.curriculum_metadata = dict(
        stage=level,
        experiment=root.name,
        protocol_sha256=sha256(root / "protocol.json"),
        robot_fingerprint=robot_fingerprint(env.unwrapped.sim.mj_model),
    )
    if "reward_weight_warmstart" in protocol:
        env.curriculum_metadata["reward_weight_warmstart"] = protocol[
            "reward_weight_warmstart"
        ]
    if "feetgf_progress_refinement" in protocol:
        env.curriculum_metadata["feetgf_progress_refinement"] = protocol[
            "feetgf_progress_refinement"
        ]
    if "field_reward_refinement" in protocol:
        env.curriculum_metadata["field_reward_refinement"] = protocol[
            "field_reward_refinement"
        ]
    if "whole_body_velocity_refinement" in protocol:
        env.curriculum_metadata["whole_body_velocity_refinement"] = protocol[
            "whole_body_velocity_refinement"
        ]
    if "landing_speed_refinement" in protocol:
        env.curriculum_metadata["landing_speed_refinement"] = protocol[
            "landing_speed_refinement"
        ]
    if "contact_force_refinement" in protocol:
        env.curriculum_metadata["contact_force_refinement"] = protocol[
            "contact_force_refinement"
        ]
    if "smoothness_refinement" in protocol:
        env.curriculum_metadata["smoothness_refinement"] = protocol[
            "smoothness_refinement"
        ]
    if "standing_slip_refinement" in protocol:
        env.curriculum_metadata["standing_slip_refinement"] = protocol[
            "standing_slip_refinement"
        ]
    if "standing_balance_refinement" in protocol:
        env.curriculum_metadata["standing_balance_refinement"] = protocol[
            "standing_balance_refinement"
        ]
    if "stop_transition_refinement" in protocol:
        env.curriculum_metadata["stop_transition_refinement"] = protocol[
            "stop_transition_refinement"
        ]
    if "navigation_refinement" in protocol:
        env.curriculum_metadata["navigation_refinement"] = protocol[
            "navigation_refinement"
        ]
    if "risk_guidance_refinement" in protocol:
        env.curriculum_metadata["risk_guidance_refinement"] = protocol[
            "risk_guidance_refinement"
        ]
    return env


def contract(env, cfg):
    from embodied_learning.protocols.imitation import PolicyContract

    result = PolicyContract.from_env(
        env,
        cfg.hidden_dims,
        critic_hidden_dims=cfg.critic_hidden_dims,
        normalization=cfg.normalization,
    )
    assert (result.actor_dim, result.critic_dim, result.action_dim) == (162, 250, 12)
    return result


def train(
    root, level, updates, output, resume=None, *, warmstart_actor=None, smoke=False
):
    from embodied_learning.algorithms.ppo import train_specialist
    from embodied_learning.policies.field_policy import load_teacher

    protocol = json.loads((root / "protocol.json").read_text())
    if (output / "config.json").exists():
        raise ValueError("Each training chunk requires a new directory.")
    if resume is not None and warmstart_actor is not None:
        raise ValueError("Resume and Actor warmstart are mutually exclusive.")
    if resume is None and warmstart_actor is None and level != 0:
        raise ValueError("Fresh initialization must begin at the easiest stage.")
    if resume is not None:
        parent = torch.load(resume, map_location="cpu", weights_only=False)
        if parent["curriculum"]["stage"] != level:
            raise ValueError(
                "Resume must stay in the same stage; "
                "use --warmstart-actor to change stage."
            )
    env = make_env(
        root,
        level,
        num_envs=32 if smoke else protocol["training_envs"],
        seed=protocol["ppo"]["seed"],
    )
    cfg = config(protocol, updates)
    expected = contract(env, cfg)
    phase_pushes = None
    try:
        if (push_version := protocol.get("training_push_curriculum")) is not None:
            from embodied_learning.tasks.cat_phase_push_curriculum import S0PhasePushCurriculum

            if (
                push_version != S0PhasePushCurriculum.version
                or level != 0
                or not env.cat_state.options["push"]
                or env.cat_state.arrival_control != "root_planar_blend_v2"
            ):
                raise ValueError(
                    "Phase push curriculum requires S0 arrival control and original pushes."
                )
            phase_pushes = S0PhasePushCurriculum(env.num_envs, env.device)
            env.unwrapped.cat_phase_pushes = phase_pushes
        before = (
            torch.load(resume, map_location="cpu", weights_only=False)
            if resume
            else None
        )
        write_json(
            output / "initialization.json",
            dict(
                stage=level,
                resume=str(resume) if resume else None,
                warmstart_actor=str(warmstart_actor) if warmstart_actor else None,
                initialization="full_ppo_resume"
                if resume
                else "actor_only_warmstart"
                if warmstart_actor
                else "fresh_random",
                additional_updates=updates,
                num_envs=env.num_envs,
                contract=expected.to_dict(),
                metadata=env.curriculum_metadata,
            ),
        )
        checkpoint = train_specialist(
            env, cfg, output, resume=resume, warmstart_actor=warmstart_actor
        )
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        prior_updates = before["completed_iterations"] if before else 0
        assert saved["completed_iterations"] == prior_updates + updates
        assert (
            saved["environment_steps"]
            == saved["completed_iterations"] * cfg.rollout_steps
        )
        assert saved["parameter_delta"] > 0
        if warmstart_actor:
            assert saved["warmstart"]["migration"] == "cat_curriculum_actor_only_v1"
        optimizer_steps = {
            float(s["step"]) for s in saved["optimizer_state_dict"]["state"].values()
        }
        assert optimizer_steps == {
            float(saved["completed_iterations"] * cfg.epochs * cfg.mini_batches)
        }
        source_batch = saved.get("source_batch")
        if cfg.source_unroll_steps is not None:
            width = (
                env.num_envs
                * cfg.rollout_steps
                // cfg.source_unroll_steps
                // cfg.mini_batches
            )
            assert source_batch["actual_minibatch_shapes"] == [
                (cfg.source_unroll_steps, width)
            ] * (cfg.epochs * cfg.mini_batches)
            assert (
                saved["collected_transitions"]
                == saved["completed_iterations"] * cfg.rollout_steps * env.num_envs
            )
        policy = load_teacher(
            checkpoint, expected_contract=expected, device="cuda:0"
        )
        assert torch.isfinite(policy.predict(env.get_observations())).all()
        # Same 11 keypoints, original SDF/BF, and unchanged 9 non-foot GFs.
        bank = env.cat_state.bank
        pos = env.cat_state.last_positions
        original = bank.base.sample_source(
            pos, env.cat_state.scene_ids, env.cat_state.goal_ids
        )
        modified = bank.sample_source(
            pos, env.cat_state.scene_ids, env.cat_state.goal_ids
        )
        assert torch.equal(original.sdf, modified.sdf)
        assert torch.equal(original.sdf_gradient, modified.sdf_gradient)
        nonfoot = [0, 1, 2, 5, 6, 7, 8, 9, 10]
        assert torch.equal(
            original.guidance[:, nonfoot], modified.guidance[:, nonfoot]
        )
        write_json(
            output / "complete.json",
            dict(
                checkpoint=str(checkpoint),
                checkpoint_sha256=sha256(checkpoint),
                completed_iterations=saved["completed_iterations"],
                stage=level,
                parameter_delta=saved["parameter_delta"],
                optimizer_steps=sorted(optimizer_steps),
                source_batch=source_batch,
                collected_transitions=saved["collected_transitions"],
                gpu_memory=dict(
                    free_bytes=torch.cuda.mem_get_info()[0],
                    total_bytes=torch.cuda.mem_get_info()[1],
                    peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
                ),
                strict_reload_passed=True,
                field_routing_passed=True,
                **(
                    {
                        "phase_pushes": dict(zip(
                            phase_pushes.phase_names,
                            phase_pushes.counts.tolist(),
                            strict=True,
                        ))
                    }
                    if phase_pushes is not None else {}
                ),
                completed_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        print("TRAIN_CHUNK_PASSED", flush=True)
    finally:
        env.close()


def evaluate(
    root,
    level,
    checkpoint,
    output,
    seed,
    *,
    execution_profile="author_playback",
    smoke=False,
    disable_push=False,
    terminal_gate=False,
):
    from contextlib import nullcontext

    from evaluate_cat_checkpoint import evaluate_cohort
    from embodied_learning.evaluation.cat_substep_contacts import capture_prohibited_contacts

    from embodied_learning.policies.field_policy import load_teacher
    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.tasks.cat_hurdle import evaluation_push_mode

    if disable_push and execution_profile != "source_train":
        raise ValueError("Disabling only pushes requires source_train.")

    if terminal_gate and (disable_push or execution_profile != "source_train"):
        raise ValueError("Terminal gate evaluation requires the original training disturbances")

    protocol = json.loads((root / "protocol.json").read_text())
    if terminal_gate:
        git = ["git", "-C", str(CODE)]
        if subprocess.check_output([*git, "status", "--porcelain"], text=True).strip():
            raise ValueError("Terminal gate evaluation requires a clean committed checkout")
        evaluation_code_commit = subprocess.check_output(
            [*git, "rev-parse", "HEAD"], text=True,
        ).strip()
    count = 32 if smoke else 128 if terminal_gate else protocol["evaluation_episodes"]
    env = make_env(
        root, level, num_envs=count, seed=seed, execution_profile=execution_profile
    )
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = contract(env, config(protocol, 1))
        changes = validate_transfer(
            saved["field_contract"],
            expected.to_dict(),
            saved["curriculum"],
            env.curriculum_metadata,
            evaluation=True,
        )
        # Frozen cross-layout evaluation is explicit; never rewrite the checkpoint.
        policy = load_teacher(
            checkpoint,
            expected_contract=PolicyContract.from_dict(saved["field_contract"]),
            device="cuda:0",
        )
        if level == len(stages_for(root, protocol)) - 1 and protocol.get(
            "reference_25cm_checkpoint"
        ):
            reference = torch.load(
                protocol["reference_25cm_checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
            old = reference["field_contract"]
            training_contract = expected.to_dict()
            # Compare training semantics without the declared playback adaptation.
            if execution_profile == "author_playback":
                training_contract["schema"].pop("execution_profile")
                training_contract["schema"]["task"] = stages_for(root, protocol)[-1][
                    "task"
                ]
            require_original_task_contract(old, training_contract)
            assert (
                env.schema["foot_guidance_experiment"]["foot_projection_radius_m"]
                == 0.30
            )
        output.mkdir(parents=True, exist_ok=False)
        from cat_motion_probe import record_motion

        contacts = capture_prohibited_contacts(env) if terminal_gate else nullcontext()
        with evaluation_push_mode(env, disable_push), record_motion(env, output), contacts:
            result = evaluate_cohort(
                env, policy, episodes=count, episode_path=output / "episodes.jsonl",
                terminal_gate=terminal_gate,
            )
        if terminal_gate:
            result["contact_sampling"] = "physics_substep"
            result["protocol_sha256"] = sha256(root / "protocol.json")
            result["evaluation_code_commit"] = evaluation_code_commit
            result["training_seed"] = protocol["ppo"]["seed"]
        result.update(
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            seed=seed,
            evaluation_stage=level,
            training_stage=saved["curriculum"]["stage"],
            completed_updates=saved["completed_iterations"],
            contract_changes=changes,
            contract=expected.to_dict(),
            execution_profile=execution_profile,
        )
        if disable_push:
            result["diagnostic_override"] = {
                "push": False,
                "original_push_draws_consumed": True,
                "other_conditions": "source_train unchanged",
            }
        write_json(output / "evaluation.json", result)
        print(
            json.dumps({"evaluation": result["rates"], "stage": level}), flush=True
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "prepare", "train", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=int)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warmstart-actor", type=Path)
    parser.add_argument(
        "--recipe",
        default="foundation_v1",
        choices=("foundation_v1", "feetgf_progress_stage1_v1"),
    )
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument(
        "--execution-profile",
        choices=("source_train", "author_playback"),
        default="author_playback",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--terminal-gate-seeds", nargs=2, type=int)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--disable-push", action="store_true")
    parser.add_argument("--terminal-gate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command != "init" and (args.training_seed is not None or args.terminal_gate_seeds is not None):
        parser.error("Training and terminal gate seeds apply only to initialization")
    if args.terminal_gate and args.command != "evaluate":
        parser.error("--terminal-gate applies only to evaluation")
    root = args.root.resolve()
    if args.command in ("train", "evaluate"):
        protocol = json.loads((root / "protocol.json").read_text())
        if args.stage is None or not 0 <= args.stage < len(
            stages_for(root, protocol)
        ):
            parser.error("Select an existing --stage explicitly.")
        if (
            args.output is None
            or args.command == "evaluate"
            and args.checkpoint is None
        ):
            parser.error(
                "--output is required; evaluation also requires --checkpoint."
            )
        if args.updates < 1:
            parser.error("--updates must be positive.")
    if args.command == "init":
        initialize(
            root, recipe=args.recipe, parent_checkpoint=args.parent_checkpoint,
            training_seed=42 if args.training_seed is None else args.training_seed,
            terminal_gate_seeds=args.terminal_gate_seeds,
        )
    elif args.command == "prepare":
        prepare(root, args.stage)
    elif args.command == "train":
        train(
            root,
            args.stage,
            args.updates,
            args.output,
            args.resume,
            warmstart_actor=args.warmstart_actor,
            smoke=args.smoke,
        )
    else:
        evaluate(
            root,
            args.stage,
            args.checkpoint,
            args.output,
            args.seed,
            execution_profile=args.execution_profile,
            smoke=args.smoke,
            disable_push=args.disable_push,
            terminal_gate=args.terminal_gate,
        )


if __name__ == "__main__":
    main()
