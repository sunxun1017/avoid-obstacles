"""Independent paper-formula checks for field-conditioned training."""

import math

import pytest
import torch

from embodied_learning.perception.field_observation import (
    FieldParameters,
    evaluate_body_field,
    yaw_rotate_vectors,
)
from embodied_learning.perception.field_reward import vmf_direction_reward
from embodied_learning.perception.voxel_field import FieldSample


def tensors(vectors, *, dtype=torch.float64):
    return torch.tensor([vectors], dtype=dtype)


def sample(*, invalid=False, empty=False):
    scalar = torch.ones((1, 2), dtype=torch.float64)
    mask = torch.ones((1, 2), dtype=torch.bool)
    gradient = tensors([[1, 0, 0], [1, 0, 0]])
    sdf = scalar * (math.inf if empty else 2.0)
    if invalid:
        mask[0, 1] = False
        gradient[0, 1] = math.nan
    return FieldSample(
        sdf,
        scalar,
        torch.zeros_like(gradient),
        gradient,
        torch.zeros_like(gradient),
        mask,
        mask,
        mask,
    )


def test_vmf_matches_explicit_3d_density():
    force = tensors([[2, 0, 0]])
    velocity = tensors([[1, 1, 0]])
    got = vmf_direction_reward(
        force,
        velocity,
        concentration_scale=1.0,
        valid_mask=torch.tensor([[True]]),
    )
    want = math.log(2 / (4 * math.pi * math.sinh(2))) + 2 / math.sqrt(2)
    assert got.item() == pytest.approx(want, abs=1e-12)


def test_vmf_zero_concentration():
    force = tensors([[0, 0, 0], [1, 0, 0]])
    got = vmf_direction_reward(
        force,
        force,
        concentration_scale=0.0,
        valid_mask=torch.tensor([[True, True]]),
    )
    assert got.item() == pytest.approx(-2 * math.log(4 * math.pi))


def test_zero_velocity_not_aligned():
    force = tensors([[1, 0, 0]])
    valid = torch.tensor([[True]])

    def reward(velocity):
        return vmf_direction_reward(
            force,
            velocity,
            concentration_scale=3.0,
            valid_mask=valid,
        ).item()

    assert reward(force) > reward(tensors([[0, 0, 0]]))
    assert reward(tensors([[0, 1, 0]])) == pytest.approx(
        reward(tensors([[0, 0, 0]]))
    )
    assert reward(tensors([[0, 0, 0]])) > reward(-force)


@pytest.mark.parametrize("magnitude", [0.0, 1e-10, 1.0, 100.0, 1e6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vmf_large_concentration_finite(magnitude, dtype):
    guidance = tensors([[magnitude, 0, 0]], dtype=dtype)
    guidance.requires_grad_()
    velocity = tensors([[1, 0, 0]], dtype=dtype)
    got = vmf_direction_reward(
        guidance,
        velocity,
        concentration_scale=1.0,
        valid_mask=torch.tensor([[True]]),
    )
    assert torch.isfinite(got).all()
    got.sum().backward()
    assert torch.isfinite(guidance.grad).all()


def test_vmf_ignores_invalid_nonfinite_points():
    got = vmf_direction_reward(
        tensors([[math.nan, 0, 0]]),
        tensors([[math.inf, 0, 0]]),
        concentration_scale=1,
        valid_mask=torch.tensor([[False]]),
    )
    assert got.item() == 0.0


def test_invalid_field_mask():
    result = evaluate_body_field(
        sample(invalid=True),
        tensors([[0, 0, 0], [0, 0, 0]]),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        parameters=FieldParameters(),
    )
    assert result.valid_mask.tolist() == [[True, False]]
    assert torch.isfinite(result.guidance).all()
    assert result.guidance[0, 1].tolist() == [0.0, 0.0, 0.0]


def test_root_priority():
    result = evaluate_body_field(
        sample(),
        tensors([[0, 0, 0], [0, 0, 0]]),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        parameters=FieldParameters(),
    )
    want_root = -0.5 * math.exp(-2)
    assert result.guidance[0, 0].tolist() == pytest.approx([want_root, 0, 0])
    assert result.guidance[0, 1].tolist() == pytest.approx([want_root / 2, 0, 0])


def test_empty_scene_retains_goal_guidance():
    result = evaluate_body_field(
        sample(empty=True),
        tensors([[0, 0, 0], [0, 0, 0]]),
        torch.tensor([1.0, 0.5], dtype=torch.float64),
        parameters=FieldParameters(),
    )
    assert result.valid_mask.all()
    assert torch.isfinite(result.guidance).all()
    assert (result.guidance[..., 0] < 0).all()
    assert torch.isposinf(result.sdf).all()


def test_yaw_rotation():
    vector = tensors([[1, 0, 0], [0, 0, 1]])
    q = torch.tensor([[math.sqrt(0.5), 0, 0, math.sqrt(0.5)]], dtype=torch.float64)
    got = yaw_rotate_vectors(vector, q)
    torch.testing.assert_close(
        got, tensors([[0, -1, 0], [0, 0, 1]]), atol=1e-12, rtol=0
    )


def test_yaw_rotation_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        yaw_rotate_vectors(
            tensors([[1, 0, 0]]), torch.zeros((1, 4), dtype=torch.float64)
        )


@pytest.mark.parametrize("scale", [-1, math.nan, math.inf])
def test_vmf_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        vmf_direction_reward(
            tensors([[1, 0, 0]]),
            tensors([[1, 0, 0]]),
            concentration_scale=scale,
            valid_mask=torch.tensor([[True]]),
        )
