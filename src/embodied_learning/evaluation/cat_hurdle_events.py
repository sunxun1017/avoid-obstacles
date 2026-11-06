"""Observable hurdle events for timing frozen-policy diagnostic impulses."""


def crossing_events(foot, grounded, seen_over_bar, front, back, lane, half_width, top):
    x, y, z = (foot[..., i] for i in range(3))
    in_lane = abs(y - lane) <= half_width
    over_bar = in_lane & (x >= front) & (x <= back) & (z > top) & ~grounded
    swing = over_bar & ~seen_over_bar
    seen = seen_over_bar | over_bar
    landed = seen & grounded & (x > back)
    return seen, swing, landed


class HurdleEventImpulse:
    stages = ("swing", "touchdown", "after")

    def __init__(self, robot, slots, device, box, mass, stage, *, dv=0.35):
        import torch

        self.robot = robot
        self.stage = stage
        self.dv = dv
        cx, cy, cz = box.center
        dx, dy, dz = box.half_size
        self.bounds = cx - dx, cx + dx, cy, dy, cz + dz
        self.ground_force = 0.05 * mass * 9.81
        self.seen = torch.zeros((slots, 2), device=device, dtype=torch.bool)
        self.events = torch.full((slots, 3), -1, device=device, dtype=torch.long)
        self.applied = torch.full((slots,), -1, device=device, dtype=torch.long)

    def observe(self, tick, foot, fz, active):
        self.seen, swing, landed = crossing_events(
            foot, fz.abs() > self.ground_force, self.seen, *self.bounds,
        )
        swing = swing.any(dim=1)
        landed = landed.any(dim=1)
        # 300 ms after touchdown, with both feet beyond the rear edge.
        after = (
            (foot[..., 0] > self.bounds[1]).all(dim=1)
            & (self.events[:, 1] >= 0)
            & (tick - self.events[:, 1] >= 150)
        )
        for index, occurred in enumerate((swing, landed, after)):
            stamp = self.events[:, index]
            stamp[active & occurred & (stamp < 0)] = tick

    def apply(self, tick, active):
        if self.stage == "baseline":
            return
        import torch

        occurred = self.events[:, self.stages.index(self.stage)] >= 0
        ids = torch.nonzero(active & occurred & (self.applied < 0), as_tuple=True)[0]
        if not len(ids):
            return
        root = torch.cat(
            (self.robot.data.root_link_pose_w[ids], self.robot.data.root_link_vel_w[ids]),
            dim=1,
        ).clone()
        root[:, 8] += self.dv
        self.robot.write_root_state_to_sim(root, env_ids=ids)
        self.applied[ids] = tick

    def report(self):
        events, impulses = self.events.cpu().tolist(), self.applied.cpu().tolist()
        return [
            {
                "slot": slot, "events_substeps": dict(zip(self.stages, stamps)),
                "impulse_substep": applied,
            }
            for slot, (stamps, applied) in enumerate(zip(events, impulses))
        ]
