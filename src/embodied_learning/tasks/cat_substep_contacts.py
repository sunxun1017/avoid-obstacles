"""Native prohibited-contact accumulation before reward and reset."""

import torch

SELF_SENSORS = ("cat_feet_self", "cat_left_foot_shin", "cat_right_foot_shin")


class SubstepContacts:
    def __init__(self, native):
        self.native = native
        self.names = (*native.cat_obstacle_contacts, *SELF_SENSORS)
        self.sensors = tuple(native.scene[name] for name in self.names)
        self.hits = torch.zeros(
            native.num_envs, len(self.names), device=native.device, dtype=torch.bool
        )
        self.original_update = native.scene.update
        self.active = False
        self.record_details = False
        self.pairs = {}
        native.scene.update = self.update

    def begin(self):
        self.hits.zero_()
        self.pairs = {}
        self.active = True

    def update(self, *args, **kwargs):
        result = self.original_update(*args, **kwargs)
        if self.active:
            self.native.sim.sense()
            for index, sensor in enumerate(self.sensors):
                self.hits[:, index] |= sensor.data.found.flatten(1).any(1)
            if self.record_details and self.obstacle.any():
                for slot, pairs in contact_pairs(
                    self.native.sim.mj_model, self.native.sim.data
                ).items():
                    self.pairs.setdefault(slot, set()).update(pairs)
        return result

    @property
    def obstacle(self):
        return self.hits[:, : len(self.native.cat_obstacle_contacts)].any(1)

    @property
    def prohibited(self):
        return self.hits.any(1)

    def close(self):
        self.native.scene.update = self.original_update
        self.active = False


def contact_pairs(model, data):
    """Decode active robot/obstacle geom pairs for evaluation diagnostics only."""
    count = int(data.nacon[0])
    contacts = data.contact
    geom = contacts.geom[:count].cpu().tolist()
    worlds = contacts.worldid[:count].cpu().tolist()
    active = (contacts.dist[:count] < contacts.includemargin[:count]).cpu().tolist()
    result = {}
    for world, pair, touching in zip(worlds, geom, active, strict=True):
        if not touching or min(pair) < 0:
            continue
        first, second = (model.geom(i).name for i in pair)
        if second.startswith("robot/"):
            first, second = second, first
        if first.startswith("robot/") and second.startswith("obstacle_"):
            result.setdefault(world, []).append((first, second))
    return result
