"""Research stopping must use physical motion and cannot restart while moving."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch


def navigator(count=1, **options):
    from embodied_learning.tasks.cat_recovery import (
        RecoveryConfig,
        RecoveryNavigator,
    )

    return RecoveryNavigator(count, "cpu", RecoveryConfig(**options))


def advance(nav, *, distance=1.0, speed=0.0, angular=0.0, contact=True, active=True):
    count = len(nav.mode)
    return nav.advance(
        goal_delta=torch.tensor([[distance, 0.0]]).repeat(count, 1),
        velocity=torch.tensor([[speed, 0.0, 0.0]]).repeat(count, 1),
        angular_velocity=torch.tensor([[0.0, 0.0, angular]]).repeat(count, 1),
        guidance_command=torch.tensor([[1.0, 0.5, 0.0, 0.0]]).repeat(count, 1),
        feet_contact=torch.full((count, 2), float(contact)),
        active=torch.full((count,), active),
    )


def test_high_speed_near_goal_brakes_without_claiming_settle():
    nav = navigator()
    nav.last_velocity[:, 0] = 0.5
    command, restart = advance(nav, distance=0.15, speed=2.2)
    assert nav.mode.item() == nav.BRAKE
    assert nav.settle_steps.item() == 0
    assert command[0, 0] == 1
    assert 0 <= command[0, 1] < 0.5
    assert not restart.any()


def test_braking_distance_uses_actual_speed_not_command_scale():
    fast, slow = navigator(), navigator()
    advance(fast, distance=0.8, speed=2.2)
    advance(slow, distance=0.8, speed=0.3)
    assert fast.mode.item() == fast.BRAKE
    assert slow.mode.item() == slow.DRIVE


def test_hold_requires_goal_and_both_velocity_norms_for_continuous_time():
    nav = navigator()
    for _ in range(24):
        advance(nav, distance=0.04)
    assert nav.mode.item() == nav.SETTLE
    advance(nav, distance=0.04, angular=0.2)
    assert nav.settle_steps.item() == 0
    for _ in range(25):
        command, _ = advance(nav, distance=0.04)
    assert nav.mode.item() == nav.HOLD
    assert not command.any()
    assert nav.settle_steps.item() == 25


def test_three_dimensional_velocity_blocks_hold():
    nav = navigator()
    args = dict(
        goal_delta=torch.tensor([[0.04, 0.0]]),
        velocity=torch.tensor([[0.0, 0.0, 0.2]]),
        angular_velocity=torch.zeros(1, 3),
        guidance_command=torch.zeros(1, 4),
        feet_contact=torch.ones(1, 2),
        active=torch.ones(1, dtype=torch.bool),
    )
    for _ in range(30):
        nav.advance(**args)
    assert nav.mode.item() != nav.HOLD
    assert nav.settle_steps.item() == 0


def test_restart_requires_stationary_supported_dwell_and_hysteresis():
    nav = navigator()
    for _ in range(25):
        advance(nav, distance=0.04)
    # Small errors wait longer than the far-goal restart dwell.
    for _ in range(30):
        _, restart = advance(nav, distance=0.2)
        assert not restart.any()
    for _ in range(30):
        _, restart = advance(nav, distance=0.4, speed=0.3)
        assert not restart.any()
    for _ in range(30):
        _, restart = advance(nav, distance=0.4, contact=False)
        assert not restart.any()
    for _ in range(19):
        _, restart = advance(nav, distance=0.4)
        assert not restart.any()
    command, restart = advance(nav, distance=0.4)
    assert restart.item()
    assert nav.mode.item() == nav.DRIVE
    assert 0 < command[0, 1] <= 0.5 * 0.02 + 1e-7
    assert nav.resume_steps.item() == 0


def test_small_goal_drift_recovers_after_longer_continuous_safe_dwell():
    nav = navigator()
    for _ in range(25):
        advance(nav, distance=0.04)
    for _ in range(49):
        _, restart = advance(nav, distance=0.2)
        assert not restart.any()
    assert nav.mode.item() == nav.SETTLE
    # Returning inside the goal resets the recovery dwell, preventing chatter.
    advance(nav, distance=0.09)
    for _ in range(49):
        _, restart = advance(nav, distance=0.2)
        assert not restart.any()
    command, restart = advance(nav, distance=0.2)
    assert restart.item()
    assert nav.mode.item() == nav.DRIVE
    assert 0 < command[0, 1] <= 0.5 * 0.02 + 1e-7


def test_settle_waits_for_reference_to_decelerate_before_stance():
    nav = navigator()
    nav.mode.fill_(nav.BRAKE)
    nav.last_velocity[:, 0] = 0.5
    previous = nav.last_velocity.clone()
    for _ in range(25):
        command, restart = advance(nav, distance=0.08, speed=0.14)
        assert not restart.any()
        assert (command[:, 1:] - previous).norm().item() <= 0.02 + 1e-7
        if command[0, 1] > 1e-6:
            assert nav.mode.item() == nav.BRAKE
            assert command[0, 0] == 1
        previous = command[:, 1:].clone()
    assert nav.mode.item() == nav.SETTLE
    assert not command.any()


def test_goal_overshoot_cannot_reverse_command_before_safe_restart():
    nav = navigator()
    nav.last_velocity[:, 0] = 0.5
    advance(nav, distance=0.15, speed=2.2)
    for _ in range(20):
        command, restart = advance(nav, distance=-0.5, speed=0.8)
        assert command[0, 1] >= 0
        assert not restart.any()
    for _ in range(19):
        _, restart = advance(nav, distance=-0.5, speed=0.0)
        assert not restart.any()


def test_inactive_slots_preserve_every_state_and_reset_is_per_slot():
    nav = navigator(2)
    for _ in range(4):
        advance(nav, distance=0.04)
    before = {
        k: v.clone() for k, v in vars(nav).items() if isinstance(v, torch.Tensor)
    }
    advance(nav, distance=3.0, speed=1.0, active=False)
    for key, value in before.items():
        torch.testing.assert_close(getattr(nav, key), value)
    nav.reset(torch.tensor([0]))
    assert nav.mode[0] == nav.DRIVE and nav.settle_steps[0] == 0
    assert not nav.last_velocity[0].any()
    for key, value in before.items():
        torch.testing.assert_close(getattr(nav, key)[1], value[1])


def test_first_order_plant_reaches_goal_instead_of_stopping_at_deadband():
    nav = navigator()
    position, speed = 0.0, 0.0
    for _ in range(1000):
        delta = 1.0 - position
        # The source command has a 0.2 m/s deadband near its target.
        guidance = 0.0 if abs(delta) < 0.3 else 0.5
        command, _ = nav.advance(
            goal_delta=torch.tensor([[delta, 0.0]]),
            velocity=torch.tensor([[speed, 0.0, 0.0]]),
            angular_velocity=torch.zeros(1, 3),
            guidance_command=torch.tensor(
                [[float(guidance > 0), guidance, 0.0, 0.0]]
            ),
            feet_contact=torch.ones(1, 2),
            active=torch.ones(1, dtype=torch.bool),
        )
        speed += 0.2 * (command[0, 1].item() - speed)
        position += 0.02 * speed
    assert abs(1.0 - position) < 0.1
    assert nav.mode.item() == nav.HOLD


@pytest.mark.parametrize(
    "options",
    [
        {"dt": 0},
        {"max_speed": -1},
        {"resume_distance": 0.05},
        {"latency": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        navigator(**options)


def reward_case():
    from embodied_learning.tasks.cat_reward import (
        CatRewardInputs,
        CatRewardParameters,
        cat_reward_terms,
    )

    fixture = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    inputs = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(fixture[k]).clone()
            for k in fixture.files
            if k.startswith("input_")
        }
    )
    inputs.root_vel.zero_()
    inputs.cmd_vel.zero_()
    inputs.move_flag.zero_()
    inputs.body_velocities.zero_()
    parameters = CatRewardParameters(
        torch.from_numpy(fixture["soft_lower"]),
        torch.from_numpy(fixture["soft_upper"]),
    )
    return inputs, cat_reward_terms(inputs, parameters=parameters)


def test_research_costs_penalize_overspeed_vertical_spin_and_wrong_goal():
    from embodied_learning.tasks.cat_recovery import research_reward_terms

    inputs, source = reward_case()
    kwargs = dict(
        inputs=inputs,
        goal_delta=torch.zeros(16, 2),
        angular_velocity=torch.zeros(16, 3),
        mode=torch.full((16,), 3),
    )
    quiet, scales = research_reward_terms(source, **kwargs)
    inputs.root_vel[:, 0] = 2.2
    inputs.root_vel[:, 2] = 0.4
    kwargs["angular_velocity"][:, 2] = 0.5
    kwargs["goal_delta"][:, 0] = 0.7
    moving, actual_scales = research_reward_terms(source, **kwargs)
    assert scales == actual_scales
    assert torch.all(
        moving["tracking_root_field"] < quiet["tracking_root_field"] - 4
    )
    for key in (
        "recovery_linear_speed",
        "recovery_angular_speed",
        "recovery_goal_error",
    ):
        assert torch.all(moving[key] > quiet[key]) and scales[key] < 0
    assert torch.all(quiet["recovery_goal_hold"] > moving["recovery_goal_hold"])


def test_field_reward_has_no_stop_or_world_x_bonus_or_overspeed_advantage():
    from embodied_learning.tasks.cat_recovery import research_reward_terms

    inputs, source = reward_case()
    inputs.gf.zero_()
    inputs.gf[:, :, 0] = 1
    inputs.sdf.fill_(0.1)
    kwargs = dict(
        inputs=inputs,
        goal_delta=torch.zeros(16, 2),
        angular_velocity=torch.zeros(16, 3),
        mode=torch.full((16,), 3),
    )
    held, _ = research_reward_terms(source, **kwargs)
    for name in ("headgf", "feetgf", "handsgf"):
        assert not held[name].any()
    kwargs["mode"].zero_()
    inputs.cmd_vel[:, 0] = 0.5
    inputs.body_velocities[:, :, 0] = 0.5
    matched, _ = research_reward_terms(source, **kwargs)
    shifted = replace(
        inputs,
        head_pos=inputs.head_pos + 100,
        foot_pos=inputs.foot_pos + 100,
        hands_pos=inputs.hands_pos + 100,
    )
    kwargs["inputs"] = shifted
    translated, _ = research_reward_terms(source, **kwargs)
    inputs.body_velocities[:, :, 0] = 2.2
    fast, _ = research_reward_terms(source, **kwargs)
    for name in ("headgf", "feetgf", "handsgf"):
        torch.testing.assert_close(matched[name], translated[name])
        assert torch.all(fast[name] <= matched[name])


def test_recovery_aggregation_preserves_negative_cost_instead_of_clipping():
    from embodied_learning.tasks.cat_recovery import aggregate_recovery_reward

    result = aggregate_recovery_reward(
        {"cost": torch.tensor([2.0, 10.0])}, {"cost": -2.0}, dt=0.02
    )
    torch.testing.assert_close(result, torch.tensor([-0.08, -0.4]))


def test_envelope_clearance_cost_is_bounded_and_continuous():
    from embodied_learning.tasks.cat_recovery import research_reward_terms

    inputs, source = reward_case()
    inputs.sdf[:4] = torch.tensor([[0.1], [0.03], [0.0], [-1.0]])
    terms, _ = research_reward_terms(
        source,
        inputs=inputs,
        goal_delta=torch.zeros(16, 2),
        angular_velocity=torch.zeros(16, 3),
        mode=torch.zeros(16, dtype=torch.long),
    )
    for key in ("headdf", "handsdf", "feetdf", "kneesdf", "shldsdf"):
        torch.testing.assert_close(
            terms[key][:4], torch.tensor([0.0, 0.0, -0.018, -0.8])
        )


def test_goal_hold_cost_does_not_penalize_distance_during_approach():
    from embodied_learning.tasks.cat_recovery import research_reward_terms

    inputs, source = reward_case()
    mode = torch.arange(4).repeat(4)
    terms, _ = research_reward_terms(
        source,
        inputs=inputs,
        goal_delta=torch.full((16, 2), 3.0),
        angular_velocity=torch.zeros(16, 3),
        mode=mode,
    )
    torch.testing.assert_close(
        terms["recovery_goal_error"][:4], torch.tensor([0.0, 0.0, 18.0, 18.0])
    )


@pytest.mark.parametrize("core_body", [1, 2])
def test_core_only_penetration_reduces_total_reward(core_body):
    from embodied_learning.tasks.cat_recovery import (
        aggregate_recovery_reward,
        research_reward_terms,
    )

    inputs, source = reward_case()
    inputs.sdf.fill_(0.1)
    kwargs = dict(
        inputs=inputs,
        goal_delta=torch.zeros(16, 2),
        angular_velocity=torch.zeros(16, 3),
        mode=torch.zeros(16, dtype=torch.long),
    )
    clear, scales = research_reward_terms(source, **kwargs)
    inputs.sdf[:, core_body] = -0.02
    contact, contact_scales = research_reward_terms(source, **kwargs)
    assert scales == contact_scales
    assert torch.all(
        aggregate_recovery_reward(contact, scales, dt=0.02)
        < aggregate_recovery_reward(clear, scales, dt=0.02)
    )
    torch.testing.assert_close(
        contact["recovery_core_clearance"], torch.full((16,), -0.025)
    )
