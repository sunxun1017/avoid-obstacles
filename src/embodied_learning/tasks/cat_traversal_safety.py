"""Versioned offline obstacle clearance and corridor guidance for CAT experts.

Geometry costs are shaping, never physical collision certificates. Commands
carry the corridor correction through the existing actor/critic interface.
"""

import torch

TRAVERSAL_VERSION = "obstacle_clearance_v1"


class TraversalSafety:
    def __init__(self, manifest, device):
        count = max(1, max(len(s.boxes) for s in manifest.scenes))
        self.low = torch.zeros(len(manifest.scenes), count, 3, device=device)
        self.high = torch.zeros_like(self.low)
        self.ground = torch.zeros(
            self.low.shape[:2], dtype=torch.bool, device=device
        )
        self.front = torch.full((len(manifest.scenes),), torch.inf, device=device)
        self.back = torch.full_like(self.front, -torch.inf)
        self.bar_front = self.front.clone()
        self.bar_back = self.back.clone()
        for i, scene in enumerate(manifest.scenes):
            for j, box in enumerate(scene.boxes):
                c, h = (
                    self.low.new_tensor(box.center),
                    self.low.new_tensor(box.half_size),
                )
                self.low[i, j], self.high[i, j] = c - h, c + h
                self.front[i] = min(self.front[i], c[0] - h[0])
                self.back[i] = max(self.back[i], c[0] + h[0])
                if c[2] - h[2] <= 0.01 and 0 < c[2] + h[2] <= 0.35:
                    self.ground[i, j] = True
                    self.bar_front[i] = min(self.bar_front[i], c[0] - h[0])
                    self.bar_back[i] = max(self.bar_back[i], c[0] + h[0])
        # Roof-only stages still cover approach/crossing/exit disturbances.
        self.push_front = torch.where(
            torch.isfinite(self.bar_front), self.bar_front, self.front
        )
        self.push_back = torch.where(
            torch.isfinite(self.bar_back), self.bar_back, self.back
        )

    def corridor(self, command, root, scene_ids):
        active = root[:, 0] >= self.front[scene_ids] - 0.30
        active &= root[:, 0] <= self.back[scene_ids] + 0.30
        error = ((root[:, 1].abs() - 0.22) / 0.10).clamp(0, 1) * active
        corrected = command.clone()
        # Blend toward a restoring command, rather than adding an unbounded bias.
        corrected[:, 2] = (1 - error) * command[:, 2] - error * 0.25 * root[
            :, 1
        ].sign()
        return corrected, error.square()

    def foot_clearance_cost(self, centers, radii, groups, gait_mask, scene_ids):
        """Penalize deficient whole-foot sole height near low ground obstacles.

        Uses full native foot covers, independently for each foot. The root or
        leading foot crossing never disables the trailing foot's requirement.
        """
        low, high = self.low[scene_ids], self.high[scene_ids]
        ground = self.ground[scene_ids]
        costs = []
        for foot in (0, 1):
            select = groups == foot
            p, r = centers[:, select], radii[select]
            lo = (p - r[None, :, None]).amin(1)
            hi = (p + r[None, :, None]).amax(1)
            distance = torch.maximum(
                low[:, :, :2] - hi[:, None, :2], lo[:, None, :2] - high[:, :, :2]
            ).clamp_min(0)
            proximity = (1 - distance.amax(-1) / 0.12).clamp(0, 1) * ground
            deficit = ((high[:, :, 2] + 0.03 - lo[:, None, 2]) / 0.10).clamp(0, 2)
            cost = (proximity * deficit.square()).amax(1)
            costs.append(cost * (gait_mask[:, foot] == -1))
        return torch.stack(costs, 1).sum(1)
