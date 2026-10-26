"""Portable protocol and runtime settings for the feetgf hurdle experiment."""

import copy
from contextlib import contextmanager
from pathlib import Path

import torch

from embodied_learning.robots.g1 import CAT_JOINT_NAMES
from embodied_learning.tasks.cat_foot_guidance import sha256

PROGRESS_PROFILE = "root_displacement_0p5s_v1"


def parent_settings(checkpoint: Path) -> dict:
    """Validate a stage1 parent before creating a new experiment directory."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = saved["field_contract"]["schema"]
    if saved["curriculum"]["stage"] != 1:
        raise ValueError("Hurdle reward refinement requires a stage1 parent.")
    if schema["task"].get("feetgf_progress_gate") is not None:
        raise ValueError("Parent already uses the progress gate; resume its run.")
    required = {
        "controller_pd",
        "rfi_amplitude_multiplier",
        "reward_smoothness_action_weight",
        "foot_guidance_experiment",
    }
    if required - set(schema):
        raise ValueError("Parent must declare PD, RFI and field/reward settings.")
    pd = schema["controller_pd"]
    if pd.get("joint_names") != list(CAT_JOINT_NAMES):
        raise ValueError("Parent PD joint order mismatch.")
    for name in ("kp", "kd"):
        gains = torch.tensor(pd[name])
        if (
            gains.shape != (29,)
            or not torch.isfinite(gains).all()
            or (gains <= 0).any()
        ):
            raise ValueError("Parent PD gains must be positive finite vectors.")
    from embodied_learning.simulators.mjlab.cat_control import CatActionCfg

    CatActionCfg(
        entity_name="robot",
        rfi_amplitude_multiplier=schema["rfi_amplitude_multiplier"],
    )
    selected = saved["training_scenes"]
    names = schema.get("scene_names", [])
    if selected and all(type(i) is int and 0 <= i < len(names) for i in selected):
        selected = [names[i] for i in selected]
    if selected != ["hurdle_crouch"]:
        raise ValueError("Parent must use the hurdle_crouch training scene.")
    return saved


def configure_recipe(
    original: dict, protocol: dict, parent: Path, saved: dict
) -> None:
    """Fill a new protocol from the actual parent, not host-specific constants."""
    schema = saved["field_contract"]["schema"]
    original["method"] = "cat_root_arrival_control_experiment"
    original["task"] = copy.deepcopy(schema["task"])
    original["task"]["feetgf_progress_gate"] = PROGRESS_PROFILE
    protocol.update(
        recipe="feetgf_progress_stage1_v1",
        active_training_stage=1,
        parent_checkpoint=str(parent.resolve()),
        evaluation_profile="source_train",
        gate_seed=20261026,
        final_seed=20261026,
        controller_pd=copy.deepcopy(schema["controller_pd"]),
        rfi_amplitude_multiplier=schema["rfi_amplitude_multiplier"],
        reward_smoothness_action_weight=schema["reward_smoothness_action_weight"],
        source_interpolation=schema.get("source_interpolation", "legacy_xz"),
        foot_radius_m=schema["foot_guidance_experiment"]["foot_projection_radius_m"],
        feetgf_progress_refinement=dict(
            version="cat_feetgf_progress_actor_transfer_v1",
            parent_sha256=sha256(parent),
            parent_experiment=saved["curriculum"]["experiment"],
            parent_protocol_sha256=saved["curriculum"]["protocol_sha256"],
            before=None,
            after=PROGRESS_PROFILE,
        ),
    )
    # Checkpoints record the complete PPO configuration; the new run has its
    # own update budget while retaining the parent's optimization settings.
    if "ppo_config" in saved:
        protocol["ppo"] = copy.deepcopy(saved["ppo_config"])
        protocol["ppo"]["iterations"] = protocol["total_updates"]
        protocol["optimization_profile"] = protocol["ppo"]["optimization_profile"]


def apply_runtime_settings(env, protocol: dict) -> None:
    """Apply explicit gains/weights and report them in the policy contract."""
    controller = env.unwrapped.cat_controller
    pd = protocol.get("controller_pd")
    # Existing verified runs recorded ankle gains in control_intervention.
    legacy = protocol.get("control_intervention", {})
    if pd is None and legacy.get("version") == "cat_ankle_pd30_2_history_v1":
        ids = [i for i, n in enumerate(CAT_JOINT_NAMES) if "_ankle_" in n]
        controller.kp[ids] = legacy["kp"]
        controller.kd[ids] = legacy["kd"]
        pd = dict(
            version="cat_ankle_pd30_2_v1",
            joint_names=list(CAT_JOINT_NAMES),
            kp=controller.kp.tolist(),
            kd=controller.kd.tolist(),
        )
    if pd is not None:
        if pd["joint_names"] != list(CAT_JOINT_NAMES):
            raise ValueError("Controller PD joint order mismatch.")
        for name in ("kp", "kd"):
            values = torch.tensor(pd[name], device=env.device)
            if (
                values.shape != (29,)
                or not torch.isfinite(values).all()
                or (values <= 0).any()
            ):
                raise ValueError(
                    "Controller PD gains must be 29 positive finite values."
                )
            getattr(controller, name).copy_(values)
        env.schema["controller_pd"] = copy.deepcopy(pd)
    weight = protocol.get("reward_smoothness_action_weight")
    if weight is None and "new_weight" in protocol.get("reward_intervention", {}):
        weight = protocol["reward_intervention"]["new_weight"]
    if weight is not None:
        if type(weight) not in (float, int) or not torch.isfinite(
            torch.tensor(weight)
        ):
            raise ValueError("Smoothness reward weight must be finite.")
        env.cat_state.scales["smoothness_action"] = weight
        env.schema["reward_smoothness_action_weight"] = weight
    if "rfi_amplitude_multiplier" in protocol:
        env.schema["rfi_amplitude_multiplier"] = protocol["rfi_amplitude_multiplier"]


@contextmanager
def evaluation_push_mode(env, disable: bool):
    """Suppress only push impulses, retaining source random draw timing."""
    if not disable:
        yield
        return
    state = env.cat_state
    options, original = state.options, state.advance_before_physics
    state.options = {**options, "push": False}

    def advance(actions):
        if options["push"]:
            count = int(
                (
                    ((env.unwrapped.episode_length_buf + 1) % state.push_interval)
                    == 0
                ).sum()
            )
            if count:
                torch.rand(count, device=env.device)
                torch.rand(count, device=env.device)
        original(actions)

    state.advance_before_physics = advance
    try:
        yield
    finally:
        state.advance_before_physics = original
        state.options = options
