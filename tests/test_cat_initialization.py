"""Author reset ranges, frame mapping, determinism and partial-reset isolation."""

import math
from types import SimpleNamespace

import pytest
import torch

from embodied_learning.tasks.cat_initialization import sample_author_initial_state


def inputs(n=4096):
    root = torch.zeros(n, 13)
    root[:, :3] = torch.tensor([-0.8, 0.0, 0.8])
    root[:, 3] = 1
    default = torch.linspace(-1, 1, 29)
    default[14] = 0
    return root, default, torch.full((29,), -0.8), torch.full((29,), 0.8)


def test_author_draw_order_ranges_and_local_angular_mapping():
    root, default, lower, upper = inputs()
    g = torch.Generator().manual_seed(42)
    actual, joints, qvel = sample_author_initial_state(
        root, default, lower, upper, generator=g
    )
    ref = torch.Generator().manual_seed(42)
    xy = 2 * torch.rand(4096, 2, generator=ref) - 1
    yaw = (torch.rand(4096, generator=ref) - 0.5) * math.pi
    expected_joints = (default * (0.5 + torch.rand(4096, 29, generator=ref))).clamp(
        lower, upper
    )
    expected_vel = torch.rand(4096, 6, generator=ref) - 0.5
    torch.testing.assert_close(actual[:, :2], root[:, :2] + xy)
    torch.testing.assert_close(actual[:, 2], root[:, 2])
    torch.testing.assert_close(actual[:, 3], torch.cos(yaw / 2))
    torch.testing.assert_close(actual[:, 6], torch.sin(yaw / 2))
    torch.testing.assert_close(joints, expected_joints)
    torch.testing.assert_close(qvel, expected_vel)
    torch.testing.assert_close(actual[:, 7:10], qvel[:, :3])
    local_x = yaw.cos() * actual[:, 10] + yaw.sin() * actual[:, 11]
    local_y = -yaw.sin() * actual[:, 10] + yaw.cos() * actual[:, 11]
    torch.testing.assert_close(local_x, qvel[:, 3], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(local_y, qvel[:, 4], atol=1e-7, rtol=1e-5)
    assert actual[:, 4:6].count_nonzero() == 0
    assert joints[:, 14].count_nonzero() == 0
    assert (qvel.abs() <= 0.5).all()
    assert torch.all(root[:, 0] == -0.8)


def test_seed_reproducibility_and_empty_batch():
    args = inputs(7)
    a = sample_author_initial_state(
        *args, generator=torch.Generator().manual_seed(6)
    )
    b = sample_author_initial_state(
        *args, generator=torch.Generator().manual_seed(6)
    )
    for x, y in zip(a, b, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    r, q, v = sample_author_initial_state(*inputs(0))
    assert (r.shape, q.shape, v.shape) == ((0, 13), (0, 29), (0, 6))


def test_native_hook_only_writes_selected_slots_and_keeps_controller_targets():
    from embodied_learning.simulators.mjlab.cat_traversal import (
        _apply_author_initialization,
    )

    n = 5
    root, default, lower, upper = inputs(n)
    qpos = torch.cat([root[:, :7], default.repeat(n, 1)], 1)
    targets = default.repeat(n, 1).clone()
    expected_targets = targets.clone()
    writes = []
    robot = SimpleNamespace(
        indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)),
        data=SimpleNamespace(default_root_state=root),
        write_root_state_to_sim=lambda value, env_ids: writes.append(
            ("root", value, env_ids)
        ),
        write_joint_state_to_sim=lambda pos, vel, env_ids: writes.append(
            ("joint", pos, vel, env_ids)
        ),
    )
    controller = SimpleNamespace(
        default=default,
        lower=lower,
        upper=upper,
        indices=torch.arange(28, -1, -1),
        targets=targets,
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = torch.zeros(n, 3)
    env = SimpleNamespace(
        scene=scene,
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
        cat_controller=controller,
    )
    ids = torch.tensor([1, 4])
    _apply_author_initialization(env, ids)
    assert len(writes) == 2
    assert writes[0][1].shape == (2, 13)
    assert writes[1][1].shape == (2, 29)
    assert torch.equal(writes[0][-1], ids)
    assert torch.equal(writes[1][-1], ids)
    assert writes[1][2].count_nonzero() == 0
    torch.testing.assert_close(controller.targets, expected_targets, atol=0, rtol=0)
    # Source order must be scattered back to the robot's native joint order.
    assert (writes[1][1][:, 14] == 0).all()

    assert (writes[1][1][:, 0] >= 0.5).all()
    assert (writes[1][1][:, 28] <= -0.5).all()


@pytest.mark.parametrize(
    ("profile", "execution", "randomized"),
    [
        ("fixed_v1", "source_train", False),
        ("author_random_v1", "source_train", True),
        ("author_random_v1", "author_playback", False),
    ],
)
def test_profile_gates_training_only(monkeypatch, profile, execution, randomized):
    from embodied_learning.simulators.mjlab import cat_traversal as native

    calls = []
    state = SimpleNamespace(
        training_initialization=profile,
        execution_profile=execution,
        reset=lambda ids: calls.append(("state", ids)),
    )
    env = SimpleNamespace(
        cat_state=state, cat_sensor_config=None, num_envs=3, device="cpu"
    )
    monkeypatch.setattr(
        native, "_reset_field", lambda env, ids: calls.append(("field", ids))
    )
    monkeypatch.setattr(
        native,
        "_apply_author_initialization",
        lambda env, ids: calls.append(("random", ids)),
    )
    native._source_reset(env, None)
    assert [x[0] for x in calls] == (
        ["field", "random", "state"] if randomized else ["field", "state"]
    )
    assert calls[-1][1].tolist() == [0, 1, 2]


def test_omitted_profile_preserves_legacy_contract_and_unknown_profile_rejected():
    from embodied_learning.tasks.cat_traversal import CatTaskState

    bank = SimpleNamespace(source_profile={"version": "cat_source_v1"})
    args = (SimpleNamespace(), bank, (0,), 1, "cpu")
    state = CatTaskState(*args, {"noise_level": 0.0})
    assert state.training_initialization == "fixed_v1"
    assert "training_initialization" not in state.options
    with pytest.raises(ValueError, match="initialization"):
        CatTaskState(*args, {"training_initialization": "typo"})
