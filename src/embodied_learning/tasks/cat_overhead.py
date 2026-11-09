"""Geometry-based ducking permission for the flat-ground G1 CAT scenes."""

import torch
from torch import Tensor

from embodied_learning.data.field_scenes import SceneManifest

POSTURE_VERSION = "overhead_clearance_v1"


class OverheadClearance:
    """Query low suspended boxes independently of the current head height.

    These G1 defaults assume the scene ground is z=0. The 0.70 m lower bound
    is the task's head-height fall threshold. A wall extending to the ground
    is not a traversable ceiling. This is reward shaping, not collision proof.
    """

    upright_head_m = 1.35
    vertical_margin_m = 0.05
    footprint_margin_m = 0.15
    fade_distance_m = 0.15
    lookahead_s = 0.20
    max_lookahead_m = 0.30

    def __init__(self, manifest: SceneManifest, device: str) -> None:
        count = max(1, max(len(s.boxes) for s in manifest.scenes))
        shape = (len(manifest.scenes), count)
        self.lower = torch.zeros(*shape, 3, device=device)
        self.upper = torch.zeros_like(self.lower)
        self.vertical = torch.zeros(*shape, device=device)
        for i, scene in enumerate(manifest.scenes):
            for j, box in enumerate(scene.boxes):
                center = self.lower.new_tensor(box.center)
                half = self.lower.new_tensor(box.half_size)
                self.lower[i, j] = center - half
                self.upper[i, j] = center + half
                underside = box.center[2] - box.half_size[2]
                if underside > 0.70:
                    self.vertical[i, j] = min(
                        1.0,
                        max(
                            0.0,
                            (
                                self.upright_head_m
                                + self.vertical_margin_m
                                - underside
                            )
                            / self.vertical_margin_m,
                        ),
                    )

    def allowance(
        self, root: Tensor, head: Tensor, command_xy: Tensor, scene_ids: Tensor
    ) -> Tensor:
        """Return [B] permission, smoothly approaching zero outside low roofs.

        Query a conservative horizontal bounding rectangle spanning the root,
        head and their commanded lookahead. No current vertical body position
        enters the permission, so crouching cannot manufacture an exemption.
        """
        delta = self.lookahead_s * command_xy
        delta = delta * (
            self.max_lookahead_m / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ).clamp_max(1.0)
        points = torch.stack((root[:, :2], head[:, :2]), dim=1)
        points = torch.cat((points, points + delta[:, None]), dim=1)
        sweep_lo, sweep_hi = points.amin(1), points.amax(1)
        lo = self.lower[scene_ids, :, :2]
        hi = self.upper[scene_ids, :, :2]
        gap = torch.maximum(lo - sweep_hi[:, None], sweep_lo[:, None] - hi)
        distance = gap.clamp_min(0).amax(-1)
        z = ((distance - self.footprint_margin_m) / self.fade_distance_m).clamp(0, 1)
        horizontal = 1 - z.square() * (3 - 2 * z)
        return (horizontal * self.vertical[scene_ids]).amax(-1)
