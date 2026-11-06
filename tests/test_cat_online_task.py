"""Online validity must cover both current and delayed policy field queries."""

from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.data.field_scenes import FieldScene, SceneManifest
from embodied_learning.perception.rays import RayMeasurements
from embodied_learning.robots.g1 import CAT_JOINT_NAMES
from embodied_learning.simulators.mjlab.cat_control import CatJointController
from embodied_learning.tasks.cat_observation import field_features
from embodied_learning.tasks.cat_online import CatOnlineFields
from embodied_learning.tasks.cat_reward import CatRewardInputs
from embodied_learning.tasks.cat_traversal import CatTaskState


@pytest.mark.parametrize("failure", ["current_outside", "delayed_outside", "stale"])
def test_unavailable_current_or_delayed_field_stops_actor_and_critic(failure):
    scene = FieldScene(
        "test",
        (0.0, 0.0, 0.0),
        (35, 35, 40),
        0.04,
        (),
        ((0.2, 0.2, 0.8),),
        ((0.8, 0.4, 0.8),),
    )
    manifest = SceneManifest((scene,))
    bank = CatOnlineFields(manifest, num_envs=2, device="cpu")
    state = CatTaskState(manifest, bank, (0,), 2, "cpu", {"noise_level": 0.0})
    c = CatJointController(
        CAT_JOINT_NAMES, torch.full((29,), -2.0), torch.full((29,), 2.0), 2, "cpu"
    )
    state.goals[:] = torch.tensor(scene.goals)
    origins = torch.tensor([[[0.03, 0.217, 0.813]]]).repeat(2, 1, 1)
    ends = origins.clone()
    ends[..., 0] = 0.6
    mask = torch.ones(2, 1, dtype=torch.bool)
    rays = RayMeasurements(origins, ends, mask, mask)
    bank.update(rays, state.goals, timestamp=0.0, env_origins=torch.zeros(2, 3))
    reference = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    reward = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(reference[k][:2]).clone()
            for k in reference.files
            if k.startswith("input_")
        }
    )
    positions = origins.repeat(1, 11, 1)
    payload = dict(
        reward_inputs=reward,
        body_positions=positions,
        root_position=positions[:, 1].clone(),
        root_rotation=torch.eye(3).repeat(2, 1, 1),
        gyro=torch.zeros(2, 3),
        gravity=torch.zeros(2, 3),
        up_z=torch.ones(2),
        self_contact=torch.zeros(2, dtype=torch.bool),
        physical_collision=torch.zeros(2, dtype=torch.bool),
        root_local_vel=torch.zeros(2, 3),
        invalid=torch.zeros(2, dtype=torch.bool),
    )
    state.refresh(payload, c, torch.zeros(2, dtype=torch.long), 0)
    assert state.observations()["actor"][:, 77:79].norm(dim=-1).min() > 0.2
    if failure == "current_outside":
        positions[0, :, 0] = -0.001
        payload["root_position"][0, 0] = -0.001
    elif failure == "delayed_outside":
        state.odom_position[0, 0] = -0.001
    else:
        # Renew only the second slot; the first keeps its expired field.
        bank.online.publish_snapshot(
            bank.online.mapper.snapshot(),
            state.goals,
            timestamp=0.5,
            env_ids=torch.tensor([1]),
        )
    bank.timestamp = 0.5 if failure == "stale" else 0.04
    state.refresh(payload, c, torch.tensor([2, 2]), 2)
    obs = state.observations()
    assert not obs["actor"][0, 76:80].any()
    assert not obs["critic"][0, 76:80].any()
    assert not state.gait.last_command[0].any()
    default = field_features(
        torch.zeros(1, 11, 3), torch.zeros(1, 11, 3), torch.full((1, 11), 0.5)
    )[0]
    torch.testing.assert_close(obs["actor"][0, 85:], default)
    torch.testing.assert_close(obs["critic"][0, 88:165], default)
    assert obs["actor"][1, 77:79].norm() > 0.2
    assert bank.diagnostics()["field_available"] == [False, True]
    # A fresh scan and an in-bounds odometry update restore this slot only.
    positions[0, :, 0] = 0.03
    payload["root_position"][0, 0] = 0.03
    bank.update(rays, state.goals, timestamp=0.6, env_origins=torch.zeros(2, 3))
    state.refresh(payload, c, torch.tensor([6, 6]), 6)
    assert state.observations()["actor"][0, 77:79].norm() > 0.2
    assert bank.diagnostics()["field_available"] == [True, True]
