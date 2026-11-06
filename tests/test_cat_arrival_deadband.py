"""Navigation and playback gait must agree on declared slow commands."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodied_learning.tasks.cat_command import compute_cat_command
from embodied_learning.tasks.cat_gait import CatPlaybackGaitState
from embodied_learning.tasks.cat_traversal import CatTaskState


def test_lower_deadband_retains_the_projected_velocity_without_inflating_it():
    root = torch.tensor([[0.03, 0.0, 0.0]])
    guidance = root[:, None].expand(1, 5, 3)
    boundary = torch.zeros_like(guidance)
    assert compute_cat_command(root, guidance, boundary).eq(0).all()
    command = compute_cat_command(root, guidance, boundary, deadband=0.001)
    torch.testing.assert_close(command, torch.tensor([[0.75, 0.01575, 0.0, 0.0]]))


def test_slow_command_keeps_playback_gait_active_and_can_restart_after_stop():
    gait = CatPlaybackGaitState(1, "cpu", command_deadband=0.001)
    gait.reset(
        torch.tensor([0]), torch.tensor([1.5]), torch.tensor([[0.0, math.pi]])
    )
    slow = torch.tensor([[0.75, 0.01575, 0.0, 0.0]])
    for _ in range(120):
        result = gait.advance(slow)
    assert result[0, 0] == 1
    for _ in range(52):
        result = gait.advance(torch.zeros_like(slow))
    assert result[0, 0] == 0
    assert gait.advance(slow)[0, 0] == 1


@pytest.mark.parametrize("profile", ["source_train", "author_playback"])
def test_task_routes_declared_deadband_to_current_navigation(profile):
    from embodied_learning.robots.g1 import CAT_JOINT_NAMES
    from embodied_learning.simulators.mjlab.cat_control import CatJointController
    from embodied_learning.tasks.cat_reward import CatRewardInputs

    class Bank:
        source_profile = {"version": "cat_source_v1"}

        def sample_source(self, positions, scene, goal):
            guidance = torch.zeros_like(positions)
            guidance[..., 0] = 0.03
            return SimpleNamespace(
                guidance=guidance,
                sdf_gradient=torch.zeros_like(positions),
                sdf=torch.ones(positions.shape[:2]),
            )

    state = CatTaskState(
        SimpleNamespace(),
        Bank(),
        (0,),
        1,
        "cpu",
        {"noise_level": 0.0, "command_deadband_m_s": 0.001},
        execution_profile=profile,
    )
    state.reset(torch.tensor([0]))
    controller = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 1, "cpu"
    )
    with np.load(
        Path(__file__).parent / "fixtures/cat_source_reference.npz"
    ) as data:
        inputs = CatRewardInputs(
            **{
                k[6:]: torch.from_numpy(data[k][:1]).clone()
                for k in data.files
                if k.startswith("input_")
            }
        )
    payload = dict(
        reward_inputs=inputs,
        body_positions=torch.ones(1, 11, 3),
        root_position=torch.zeros(1, 3),
        root_rotation=torch.eye(3)[None],
        gyro=torch.zeros(1, 3),
        gravity=torch.zeros(1, 3),
        up_z=torch.ones(1),
        self_contact=torch.zeros(1, dtype=torch.bool),
        physical_collision=torch.zeros(1, dtype=torch.bool),
        root_local_vel=torch.zeros(1, 3),
        invalid=torch.zeros(1, dtype=torch.bool),
    )
    state.refresh(payload, controller, torch.tensor([0]), 0)
    for i in range(1, 120):
        state.refresh(payload, controller, torch.tensor([i]), i)
    torch.testing.assert_close(inputs.cmd_vel[0, 0], torch.tensor(0.01575))
    assert state.gait.last_command[0, 0] == 1
    assert state.options["command_deadband_m_s"] == 0.001
    assert state.observations()["actor"].shape == (1, 162)


@pytest.mark.parametrize("value", [0.0, -0.1, math.inf, math.nan, True])
def test_invalid_deadband_is_rejected_by_command_and_gait(value):
    with pytest.raises(ValueError, match="deadband"):
        compute_cat_command(
            torch.zeros(1, 3),
            torch.zeros(1, 5, 3),
            torch.zeros(1, 5, 3),
            deadband=value,
        )
    with pytest.raises(ValueError, match="deadband"):
        CatPlaybackGaitState(1, "cpu", command_deadband=value)


def test_legacy_task_contract_does_not_gain_an_implicit_option():
    state = CatTaskState(
        SimpleNamespace(),
        SimpleNamespace(source_profile={"version": "cat_source_v1"}),
        (0,),
        1,
        "cpu",
        {},
    )
    assert "command_deadband_m_s" not in state.options


def test_root_arrival_retimes_only_clear_goal_neighborhood_and_stops_inside():
    from embodied_learning.tasks.cat_command import retime_cat_arrival

    command = torch.tensor([[0.75, -0.03, 0.0, 0.0]] * 4)
    root = torch.tensor(
        [[0.18, 0.0, 0.8], [0.8, 0.0, 0.8], [0.18, 0.0, 0.8], [0.07, 0.0, 0.8]]
    )
    goals = torch.tensor([[0.0, 0.0, 0.8]] * 4)
    sdf = torch.full((4, 11), torch.inf)
    sdf[2, 3] = 0.1
    result = retime_cat_arrival(command, root, goals, sdf)
    torch.testing.assert_close(result[0, 1], torch.tensor(-0.189))
    torch.testing.assert_close(result[1:3], command[1:3])
    assert result[3].eq(0).all()
    torch.testing.assert_close(command[0, 1], torch.tensor(-0.03))


@pytest.mark.parametrize("profile", ["source_train", "author_playback"])
def test_arrival_profile_aligns_actor_reward_and_restartable_gait(profile):
    from embodied_learning.robots.g1 import CAT_JOINT_NAMES
    from embodied_learning.simulators.mjlab.cat_control import CatJointController
    from embodied_learning.tasks.cat_reward import CatRewardInputs

    class Bank:
        source_profile = {"version": "cat_source_v1"}

        def sample_source(self, positions, scene, goal):
            guidance = torch.zeros_like(positions)
            guidance[..., 0] = -0.03
            return SimpleNamespace(
                guidance=guidance,
                sdf_gradient=torch.zeros_like(positions),
                sdf=torch.full(positions.shape[:2], torch.inf),
            )

    state = CatTaskState(
        SimpleNamespace(),
        Bank(),
        (0,),
        1,
        "cpu",
        {
            "noise_level": 0.0,
            "command_deadband_m_s": 0.01,
            "arrival_control": "root_linear_v1",
        },
        execution_profile=profile,
    )
    state.reset(torch.tensor([0]))
    controller = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 1, "cpu"
    )
    with np.load(
        Path(__file__).parent / "fixtures/cat_source_reference.npz"
    ) as data:
        inputs = CatRewardInputs(
            **{
                k[6:]: torch.from_numpy(data[k][:1]).clone()
                for k in data.files
                if k.startswith("input_")
            }
        )
    payload = dict(
        reward_inputs=inputs,
        body_positions=torch.ones(1, 11, 3),
        root_position=torch.tensor([[0.18, 0.0, 0.8]]),
        root_rotation=torch.eye(3)[None],
        gyro=torch.zeros(1, 3),
        gravity=torch.zeros(1, 3),
        up_z=torch.ones(1),
        self_contact=torch.zeros(1, dtype=torch.bool),
        physical_collision=torch.zeros(1, dtype=torch.bool),
        root_local_vel=torch.zeros(1, 3),
        invalid=torch.zeros(1, dtype=torch.bool),
    )
    state.refresh(payload, controller, torch.tensor([0]), 0)
    assert state.gait.step.item() == 0  # Reset observations do not advance time.
    state.refresh(payload, controller, torch.tensor([1]), 1)
    torch.testing.assert_close(
        state.observations()["actor"][0, 77], torch.tensor(-0.189)
    )
    torch.testing.assert_close(inputs.cmd_vel[0, 0], torch.tensor(-0.189))
    assert not state.last_metrics["stop_command"][0]
    payload["root_position"][:, 0] = 0.07
    for i in range(2, 55):
        state.refresh(payload, controller, torch.tensor([i]), i)
    assert state.gait.last_command[0].eq(0).all()
    assert state.last_metrics["stop_command"][0]
    payload["root_position"][:, 0] = 0.2
    state.refresh(payload, controller, torch.tensor([55]), 55)
    assert state.gait.last_command[0, 0] == 1
    torch.testing.assert_close(
        state.observations()["actor"][0, 77], inputs.cmd_vel[0, 0]
    )
    previous_step = state.gait.step.clone()
    state.refresh(payload, controller, torch.tensor([55]), 55)
    torch.testing.assert_close(state.gait.step, previous_step)
