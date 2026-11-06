"""Evaluation-only physical contact accumulation between policy decisions."""

from contextlib import contextmanager
from unittest.mock import patch

import torch


@contextmanager
def capture_prohibited_contacts(env):
    """Add substep contacts to the existing pre-reset policy transition flags."""
    native = env.unwrapped
    sensors = tuple(native.scene[name] for name in (
        *native.cat_obstacle_contacts, "cat_feet_self", "cat_left_foot_shin", "cat_right_foot_shin",
    ))
    found = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    update, step = native.scene.update, env.step

    def update_sensors(*args, **kwargs):
        result = update(*args, **kwargs)
        native.sim.sense()
        for sensor in sensors:
            found.logical_or_(sensor.data.found.flatten(1).any(1))
        return result

    def step_with_contacts(actions):
        found.zero_()
        result = step(actions)
        outcomes = result[3]["field_outcomes"]
        result[3]["field_outcomes"] = {
            **outcomes, "prohibited_contact": outcomes["prohibited_contact"] | found,
        }
        return result

    with patch.object(native.scene, "update", update_sensors), patch.object(env, "step", step_with_contacts):
        yield
