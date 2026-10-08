from types import SimpleNamespace

import pytest
import torch

from embodied_learning.simulators.mjlab.sensors import (
    DepthCameraReader,
    add_depth_camera,
)


def test_attach_preserves_existing_sensors():
    original = SimpleNamespace(name="terrain_scan")
    cfg = SimpleNamespace(scene=SimpleNamespace(sensors=(original,)))
    camera = SimpleNamespace(name="depth", data_types=("depth",))
    add_depth_camera(cfg, camera)
    assert cfg.scene.sensors == (original, camera)
    with pytest.raises(ValueError, match="already"):
        add_depth_camera(cfg, camera)


def test_attach_requires_depth():
    cfg = SimpleNamespace(scene=SimpleNamespace(sensors=()))
    camera = SimpleNamespace(name="rgb", data_types=("rgb",))
    with pytest.raises(ValueError, match="depth"):
        add_depth_camera(cfg, camera)
    assert cfg.scene.sensors == ()


def test_read_is_independent_and_preserves_batch():
    buffer = torch.ones((2, 3, 4, 1))
    sensor = SimpleNamespace(data=SimpleNamespace(depth=buffer))
    env = SimpleNamespace(scene={"depth": sensor})
    reader = DepthCameraReader(env, "depth")
    result = reader.read()
    assert result.shape == (2, 3, 4)
    buffer.zero_()
    assert torch.all(result == 1)
    assert torch.all(reader.read() == 0)


def test_read_missing_depth():
    sensor = SimpleNamespace(data=SimpleNamespace(depth=None))
    reader = DepthCameraReader(SimpleNamespace(scene={"depth": sensor}), "depth")
    with pytest.raises(RuntimeError, match="depth"):
        reader.read()


def test_read_rejects_wrong_shape():
    sensor = SimpleNamespace(data=SimpleNamespace(depth=torch.zeros(2, 3, 4)))
    reader = DepthCameraReader(SimpleNamespace(scene={"depth": sensor}), "depth")
    with pytest.raises(ValueError, match="shape"):
        reader.read()


def test_frame_fov_and_optical_pose():
    model = SimpleNamespace(
        cam_fovy=torch.tensor([[90.0]]),
        cam_intrinsic=torch.zeros(1, 1, 4),
    )
    data = SimpleNamespace(
        cam_xpos=torch.tensor([[[1.0, 2.0, 3.0]]]),
        cam_xmat=torch.eye(3).reshape(1, 1, 3, 3),
    )
    sim = SimpleNamespace(
        model=model,
        data=data,
        mj_model=SimpleNamespace(cam_projection=[0], cam_sensorsize=[(0.0, 0.0)]),
    )
    sensor = SimpleNamespace(
        camera_idx=0, data=SimpleNamespace(depth=torch.ones(1, 2, 4, 1))
    )
    reader = DepthCameraReader(
        SimpleNamespace(scene={"depth": sensor}, sim=sim), "depth"
    )
    frame = reader.read_frame()
    expected_k = torch.tensor([[[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])
    torch.testing.assert_close(frame.intrinsics, expected_k)
    torch.testing.assert_close(
        frame.rotation[0], torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    )
    data.cam_xpos.zero_()
    assert frame.position.tolist() == [[1.0, 2.0, 3.0]]


def test_frame_intrinsics_match_aspect_adjustment():
    model = SimpleNamespace(
        cam_fovy=torch.tensor([[45.0]]),
        cam_intrinsic=torch.tensor([[[2.0, 3.0, 0.2, -0.1]]]),
    )
    sim = SimpleNamespace(
        model=model,
        data=SimpleNamespace(
            cam_xpos=torch.zeros(1, 1, 3),
            cam_xmat=torch.eye(3).reshape(1, 1, 3, 3),
        ),
        mj_model=SimpleNamespace(cam_projection=[0], cam_sensorsize=[(2.0, 2.0)]),
    )
    sensor = SimpleNamespace(
        camera_idx=0, data=SimpleNamespace(depth=torch.ones(1, 2, 4, 1))
    )
    reader = DepthCameraReader(
        SimpleNamespace(scene={"depth": sensor}, sim=sim), "depth"
    )
    k = reader.read_frame().intrinsics[0]
    torch.testing.assert_close(
        k, torch.tensor([[4.0, 0.0, 1.6], [0.0, 6.0, 1.2], [0.0, 0.0, 1.0]])
    )
    sim.mj_model.cam_projection = [1]
    with pytest.raises(ValueError, match="Orthographic"):
        reader.read_frame()
