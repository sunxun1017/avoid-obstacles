import math

import torch

from embodied_learning.tasks.cat_gait import CatPlaybackGaitState


def test_playback_gait_matches_unmodified_pinned_author_function_fixture():
    from pathlib import Path

    import numpy as np

    data = np.load(Path(__file__).parent / "fixtures/cat_author_playback_gait.npz")
    gait = CatPlaybackGaitState(1, "cpu")
    gait.reset(
        torch.tensor([0]), torch.tensor([1.5]), torch.tensor([[0.0, math.pi]])
    )
    for i, cmd in enumerate(data["commands"]):
        command = torch.tensor([[0.75, *cmd]], dtype=torch.float32)
        result = gait.advance(command)
        assert result[0, 0] == data["moving"][i], i
        assert gait.stop_timestep[0] == data["timestamp"][i], i
        expected = torch.from_numpy(data["phase"][i]).float()
        # +pi and -pi encode the same gait; compare the actual policy inputs.
        torch.testing.assert_close(
            gait.phase[0].cos(), expected.cos(), atol=5e-5, rtol=0
        )
        torch.testing.assert_close(
            gait.phase[0].sin(), expected.sin(), atol=5e-5, rtol=0
        )
        torch.testing.assert_close(
            gait.gait_mask[0], torch.from_numpy(data["mask"][i])
        )


def test_playback_restarts_during_countdown_and_after_stopping():
    gait = CatPlaybackGaitState(2, "cpu")
    gait.reset(
        torch.arange(2), torch.full((2,), 1.5), torch.tensor([[0.0, math.pi]] * 2)
    )
    drive = torch.tensor([[1.0, 0.315, 0.0, 0.0]] * 2)
    stop = torch.zeros(2, 4)
    assert gait.advance(drive)[:, 0].tolist() == [1.0, 1.0]
    for _ in range(20):
        gait.advance(stop)
    result = gait.advance(torch.stack((drive[0], stop[1])))
    assert result[0, 0] == 1 and result[0, 1] == 0.315
    torch.testing.assert_close(gait.phase[0], torch.tensor([0.0, math.pi]))
    for _ in range(55):
        result = gait.advance(torch.stack((drive[0], stop[1])))
    assert result[1, 0] == 0
    assert gait.advance(drive)[1, 0] == 1
    torch.testing.assert_close(gait.phase[1], torch.tensor([0.0, math.pi]))


def test_playback_uses_current_fields_raw_command_and_raw_reset_fields():
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from embodied_learning.robots.g1 import CAT_JOINT_NAMES
    from embodied_learning.simulators.mjlab.cat_control import CatJointController
    from embodied_learning.tasks.cat_reward import CatRewardInputs
    from embodied_learning.tasks.cat_traversal import CatTaskState

    class Bank:
        source_profile = {"version": "cat_source_v1"}

        def sample_source(self, p, scene, goal):
            guidance = torch.zeros_like(p)
            guidance[..., 0] = 0.6
            return SimpleNamespace(
                guidance=guidance,
                sdf_gradient=torch.zeros_like(p),
                sdf=0.2 + p[..., 0] * 0.01,
            )

    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    for profile in ("source_train", "author_playback"):
        state = CatTaskState(
            SimpleNamespace(),
            Bank(),
            (0,),
            1,
            "cpu",
            {"noise_level": 0.0},
            execution_profile=profile,
        )
        state.reset(torch.tensor([0]))
        controller = CatJointController(
            CAT_JOINT_NAMES,
            torch.full((29,), -2.0),
            torch.full((29,), 2.0),
            1,
            "cpu",
        )
        inputs = CatRewardInputs(
            **{
                k[6:]: torch.from_numpy(x[k][:1]).clone()
                for k in x.files
                if k.startswith("input_")
            }
        )
        payload = dict(
            reward_inputs=inputs,
            body_positions=torch.tensor([[[0.0, 0.0, 1.0]]] * 11).reshape(1, 11, 3),
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
        if profile == "author_playback":
            assert (
                state.options["push"] is False
                and state.options["random_rfi"] is False
            )
            torch.testing.assert_close(
                state.observations()["actor"][0, 85], torch.tensor(0.6)
            )
            assert state.observations()["actor"][0, 76] == 0
        state.refresh(payload, controller, torch.tensor([1]), 1)
        payload["body_positions"][..., 0] = 0.5
        payload["root_position"][:, 0] = 0.5
        state.refresh(payload, controller, torch.tensor([2]), 2)
        obs = state.observations()["actor"][0]
        torch.testing.assert_close(
            obs[77],
            torch.tensor(0.315 if profile == "author_playback" else 0.525),
            atol=2e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            obs[91], torch.tensor(0.205 if profile == "author_playback" else 0.2)
        )


def test_motion_probe_supports_empty_and_single_bar_scenes(tmp_path):
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    from cat_motion_probe import record_motion

    for name, boxes in (
        ("empty", []),
        (
            "bar",
            [SimpleNamespace(center=(0.0, 0.0, 0.02), half_size=(0.04, 0.5, 0.02))],
        ),
    ):
        directory = tmp_path / name
        directory.mkdir()
        state = SimpleNamespace(
            manifest=SimpleNamespace(
                scenes=[SimpleNamespace(id="hurdle_crouch", boxes=boxes)]
            ),
            refresh=lambda *args: None,
        )
        env = SimpleNamespace(cat_state=state, num_envs=1, device="cpu")
        with record_motion(env, directory):
            pass
        assert (directory / "motion-summary.json").exists()


def test_empty_field_infinity_has_bounded_observations_without_changing_finite_sdf():
    from pathlib import Path

    import numpy as np

    from embodied_learning.tasks.cat_observation import build_cat_observations

    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    payload = {
        k[4:]: torch.from_numpy(x[k]).clone()
        for k in x.files
        if k.startswith("obs_") and k not in ("obs_actor", "obs_critic")
    }
    payload["sdf"][:] = torch.inf
    payload["sdf_delayed"][:] = torch.inf
    obs = build_cat_observations(**payload, noise_level=0.0)
    assert obs["critic"].abs().max() < 1000
    assert obs["actor"].abs().max() < 1000
