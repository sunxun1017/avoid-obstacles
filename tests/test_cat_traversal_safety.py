from types import SimpleNamespace

import pytest
import torch

from embodied_learning.data.field_scenes import SceneBox
from embodied_learning.tasks.cat_traversal_safety import TraversalSafety


def safety(boxes):
    return TraversalSafety(
        SimpleNamespace(scenes=[SimpleNamespace(boxes=boxes)]), "cpu"
    )


def test_trailing_heel_clearance_remains_after_front_foot_passes():
    s = safety([SceneBox((0.52, 0.0, 0.06), (0.12, 0.5, 0.06))])
    # Front foot fully past; back foot toe and heel still straddle the bar.
    p = torch.tensor(
        [[[0.9, 0.0, 0.2], [1.0, 0.0, 0.2], [0.45, 0.0, 0.14], [0.65, 0.0, 0.14]]]
    )
    radii = torch.full((4,), 0.02)
    groups = torch.tensor([0, 0, 1, 1])
    args = (radii, groups, torch.tensor([[-1.0, -1.0]]), torch.tensor([0]))
    assert s.foot_clearance_cost(p, *args).item() > 0
    p[:, 2:, 2] = 0.18  # whole sole clears .12 + .03 margin
    assert s.foot_clearance_cost(p, *args).item() == 0
    p[:, 2:, 2] = 0.02
    p[:, 2:, 0] += 0.6
    assert s.foot_clearance_cost(p, *args).item() == 0


def test_clearance_is_local_swing_only_and_ignores_overhead():
    p = torch.tensor([[[0.52, 0.0, 0.05], [0.52, 0.1, 0.05]]])
    args = (
        torch.full((2,), 0.02),
        torch.tensor([0, 1]),
        torch.full((1, 2), -1.0),
        torch.tensor([0]),
    )
    for boxes in ([], [SceneBox((0.52, 0.0, 1.32), (0.25, 0.9, 0.12))]):
        assert safety(boxes).foot_clearance_cost(p, *args).item() == 0
    s = safety([SceneBox((0.52, 0.0, 0.06), (0.12, 0.5, 0.06))])
    assert s.foot_clearance_cost(p, *args).item() > 0
    assert (
        s.foot_clearance_cost(p, args[0], args[1], torch.ones(1, 2), args[3]).item()
        == 0
    )
    p[:, :, 1] += 1.0
    assert s.foot_clearance_cost(p, *args).item() == 0


def test_corridor_correction_is_observable_local_and_sign_symmetric():
    s = safety([SceneBox((0.52, 0.0, 0.06), (0.12, 0.5, 0.06))])
    root = torch.tensor(
        [[0.5, 0.1, 0.8], [0.5, 0.3, 0.8], [0.5, -0.3, 0.8], [1.5, 0.3, 0.8]]
    )
    command = torch.tensor([[1.0, 0.5, 0.0, 0.0]]).repeat(4, 1)
    result, cost = s.corridor(command, root, torch.zeros(4, dtype=torch.long))
    assert result[:, 2].tolist() == pytest.approx([0.0, -0.2, 0.2, 0.0])
    assert cost.tolist() == pytest.approx([0.0, 0.64, 0.64, 0.0])
    torch.testing.assert_close(command[:, 2], torch.zeros(4))
    empty = safety([])
    out, cost = empty.corridor(command, root, torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(out, command)
    assert not cost.any()


def test_actual_contact_terminates_first_step_zeroes_reward_and_survives_reset():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_traversal import CatTaskState

    old, controller, payload, _ = task_case()
    manifest = SimpleNamespace(scenes=[SimpleNamespace(boxes=[])])
    state = CatTaskState(
        manifest,
        old.bank,
        (0,),
        16,
        "cpu",
        {"noise_level": 0.0, "traversal_robustness": "obstacle_clearance_v1"},
    )
    payload["foot_cover"] = (
        torch.ones(16, 2, 3),
        torch.full((2,), 0.02),
        torch.tensor([0, 1]),
    )
    state.reset(torch.arange(16))
    state.refresh(payload, controller, torch.zeros(16, dtype=torch.long), 0)
    payload["physical_collision"][0] = True
    state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 1)
    assert state.outcomes()["collision"].tolist() == [True] + [False] * 15
    assert state.reward()[0].item() == 0
    assert state.saved_outcomes["prohibited_contact"][0]
    state.reset(torch.tensor([0]))
    payload["physical_collision"].zero_()
    state.refresh(payload, controller, torch.tensor([0] + [1] * 15), 1)
    assert not state.outcomes()["collision"][0]
    assert state.saved_outcomes["prohibited_contact"][0]


def test_corridor_command_reaches_actor_and_critic_including_reset_observation():
    from test_cat_review_regressions import task_case

    from embodied_learning.tasks.cat_traversal import CatTaskState

    old, controller, payload, _ = task_case()
    manifest = SimpleNamespace(
        scenes=[
            SimpleNamespace(boxes=[SceneBox((0.52, 0.0, 0.06), (0.12, 0.5, 0.06))])
        ]
    )
    state = CatTaskState(
        manifest,
        old.bank,
        (0,),
        16,
        "cpu",
        {"noise_level": 0.0, "traversal_robustness": "obstacle_clearance_v1"},
    )
    payload["root_position"][:] = torch.tensor([0.5, 0.32, 0.8])
    payload["reward_inputs"].navigation_rotation[:] = torch.eye(3)
    payload["foot_cover"] = (
        torch.ones(16, 2, 3),
        torch.full((2,), 0.02),
        torch.tensor([0, 1]),
    )
    state.reset(torch.arange(16))
    for step in (0, 1):
        state.refresh(payload, controller, torch.full((16,), step), step)
        for group in ("actor", "critic"):
            torch.testing.assert_close(
                state.observations()[group][:, 78], torch.full((16,), -0.25)
            )
