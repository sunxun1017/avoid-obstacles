"""Author G1Cat reset distributions, adapted from upstream reset (Apache-2.0)."""

import math

import torch
from torch import Tensor


def sample_author_initial_state(
    root_state: Tensor,
    default_joints: Tensor,
    soft_lower: Tensor,
    soft_upper: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample source-order joints and source qvel, returning world-frame roots.

    XY offsets are relative to the selected scene start. The caller supplies
    ground-relative height 0.8 m. MuJoCo free-joint qvel uses world linear and
    body angular velocity; the native root-state writer expects both in world
    coordinates. Targets are deliberately not changed to randomized joints.
    """
    if root_state.ndim != 2 or root_state.shape[1] != 13:
        raise ValueError("Root state must have shape [N,13].")
    if any(x.shape != (29,) for x in (default_joints, soft_lower, soft_upper)):
        raise ValueError("Source joint vectors must contain 29 entries.")
    n = len(root_state)

    def rand(*shape):
        return torch.rand(
            *shape,
            device=root_state.device,
            dtype=root_state.dtype,
            generator=generator,
        )

    roots = root_state.clone()
    roots[:, :2] += 2 * rand(n, 2) - 1
    yaw = (rand(n) - 0.5) * math.pi
    roots[:, 3:7] = 0
    roots[:, 3] = torch.cos(yaw / 2)
    roots[:, 6] = torch.sin(yaw / 2)
    joints = (default_joints * (0.5 + rand(n, 29))).clamp(soft_lower, soft_upper)
    source_qvel = rand(n, 6) - 0.5
    roots[:, 7:10] = source_qvel[:, :3]
    c, s = yaw.cos(), yaw.sin()
    roots[:, 10] = c * source_qvel[:, 3] - s * source_qvel[:, 4]
    roots[:, 11] = s * source_qvel[:, 3] + c * source_qvel[:, 4]
    roots[:, 12] = source_qvel[:, 5]
    return roots, joints, source_qvel
