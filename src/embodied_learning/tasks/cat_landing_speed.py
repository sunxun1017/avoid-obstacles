"""Opt-in, physics-substep landing speed cost for native flat-ground G1."""

import copy
import math
import re
from dataclasses import dataclass

import mujoco
import torch

from embodied_learning.tasks.cat_foot_guidance import sha256

KEY = "foot_landing_speed"


def validate_schema(value):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "height_m", "speed_m_s", "weight"}
        or value["version"] != "sole_descent_substep_v1"
    ):
        raise ValueError("Invalid landing speed schema.")
    result = dict(value)
    for key in ("height_m", "speed_m_s", "weight"):
        v = value[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError("Landing speed values must be finite and nonnegative.")
        if key != "speed_m_s" and v == 0:
            raise ValueError("Landing height and weight must be positive.")
        result[key] = float(v)
    return result


def landing_speed_cost(sole_height, vertical_velocity, *, height_m, speed_m_s):
    if (
        sole_height.ndim != 2
        or sole_height.shape[1] != 2
        or vertical_velocity.shape != sole_height.shape
        or vertical_velocity.device != sole_height.device
        or vertical_velocity.dtype != sole_height.dtype
        or not sole_height.is_floating_point()
        or height_m <= 0
        or speed_m_s < 0
    ):
        raise ValueError("Landing speed inputs must be paired floating [B,2].")
    gate = (1 - sole_height / height_m).clamp(0, 1)
    cost = (gate * (-vertical_velocity - speed_m_s).clamp_min(0).square()).sum(-1)
    # Existing task invalid-state handling still terminates invalid physics.
    return torch.where(torch.isfinite(cost), cost, 0.0)


@dataclass
class SoleGeometry:
    ids: tuple
    radii: tuple
    half_lengths: tuple

    @classmethod
    def from_model(cls, model, device):
        ids, radii, halves = [], [], []
        for side in ("left", "right"):
            found = []
            for i in range(model.ngeom):
                name = model.geom(i).name.rsplit("/", 1)[-1]
                if not re.fullmatch(rf"{side}_foot\d+_collision", name):
                    continue
                if not (model.geom_contype[i] or model.geom_conaffinity[i]):
                    continue
                kind = int(model.geom_type[i])
                if kind not in (
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                ):
                    raise ValueError(
                        "Landing geometry requires foot spheres/capsules."
                    )
                radius = float(model.geom_size[i, 0])
                half = (
                    float(model.geom_size[i, 1])
                    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE
                    else 0.0
                )
                if (
                    not math.isfinite(radius)
                    or radius <= 0
                    or not math.isfinite(half)
                    or half < 0
                ):
                    raise ValueError("Invalid landing collision dimensions.")
                found.append((i, radius, half))
            if not found:
                raise ValueError(f"Missing {side} enabled foot collision geometry.")
            ids.append(torch.tensor([x[0] for x in found], device=device))
            radii.append(torch.tensor([x[1] for x in found], device=device))
            halves.append(torch.tensor([x[2] for x in found], device=device))
        return cls(tuple(ids), tuple(radii), tuple(halves))

    def heights(self, geom_xpos, geom_xmat, env_origins):
        # Capsule support along world z is radius + half_length*abs(axis_z).
        return torch.stack(
            [
                (
                    geom_xpos[:, ids, 2]
                    - self.radii[foot]
                    - self.half_lengths[foot] * geom_xmat[:, ids, 2, 2].abs()
                ).amin(-1)
                - env_origins[:, 2]
                for foot, ids in enumerate(self.ids)
            ],
            -1,
        )


class FootLandingState:
    def __init__(self, num_envs, device, *, height_m, speed_m_s, control_dt):
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("Landing control dt must be positive.")
        validate_schema(
            dict(
                version="sole_descent_substep_v1",
                height_m=height_m,
                speed_m_s=speed_m_s,
                weight=1.0,
            )
        )
        self.integral = torch.zeros(num_envs, device=device)
        self.height_m = height_m
        self.speed_m_s = speed_m_s
        self.control_dt = control_dt
        self.geometry = None
        self.site_ids = None

    def begin_control(self):
        self.integral.zero_()

    def reset(self, env_ids):
        self.integral[env_ids] = 0

    def accumulate(self, heights, velocities, *, dt):
        self.integral.add_(
            landing_speed_cost(
                heights, velocities, height_m=self.height_m, speed_m_s=self.speed_m_s
            )
            * dt
        )

    @property
    def cost_rate(self):
        return self.integral / self.control_dt

    def sample(self, native):
        robot = native.scene["robot"]
        heights = self.geometry.heights(
            native.sim.data.geom_xpos,
            native.sim.data.geom_xmat,
            native.scene.env_origins,
        )
        velocity = robot.data.site_vel_w[:, self.site_ids, 2]
        self.accumulate(heights, velocity, dt=native.physics_dt)

    def install(self, native):
        original_step = native.sim.step

        def step_with_landing_sample():
            result = original_step()
            # mj_step exposes derived state at this integration's start.
            # Sampling afterward covers every substep exactly once, including
            # the final one before the task's control-boundary mj_forward.
            self.sample(native)
            return result

        native.sim.step = step_with_landing_sample


def apply_landing_speed(env, protocol):
    if KEY not in protocol:
        return
    schema = validate_schema(protocol[KEY])
    if schema is None:
        raise ValueError("Explicit landing configuration cannot be null.")
    if getattr(env.cat_state, "recovery", None) is not None:
        raise ValueError("Landing speed pilot requires source reward mode.")
    native = env.unwrapped
    if getattr(native.cat_controller, "landing", None) is not None:
        raise ValueError("Landing speed sampling is already installed.")
    model = native.sim.mj_model
    planes = [
        i
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
    ]
    if len(planes) != 1 or abs(float(model.geom_pos[planes[0], 2])) > 1e-6:
        raise ValueError(
            "Landing speed pilot requires the flat zero-height ground plane."
        )
    landing = FootLandingState(
        env.num_envs,
        env.device,
        height_m=schema["height_m"],
        speed_m_s=schema["speed_m_s"],
        control_dt=native.step_dt,
    )
    landing.geometry = SoleGeometry.from_model(model, env.device)
    landing.site_ids = [
        native.scene["robot"].site_names.index(n)
        for n in ("left_foot", "right_foot")
    ]
    native.cat_controller.landing = landing
    landing.install(native)
    env.cat_state.scales[KEY] = -schema["weight"]
    env.schema[KEY] = schema


def adapt_landing_transfer(checkpoint, old, target, source, curriculum):
    declaration = curriculum.get("landing_speed_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = validate_schema(old["schema"].get(KEY))
    after = validate_schema(target["schema"].get(KEY))
    expected = dict(
        version="cat_landing_speed_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if declaration != expected or source["stage"] != curriculum["stage"]:
        raise ValueError("Landing speed transfer parent/declaration/stage mismatch.")
    old = copy.deepcopy(old)
    if after is None:
        old["schema"].pop(KEY, None)
    else:
        old["schema"][KEY] = after
    source = {
        **source,
        "experiment": curriculum["experiment"],
        "protocol_sha256": curriculum["protocol_sha256"],
    }
    return (
        old,
        source,
        ({KEY: dict(before=before, after=after)} if before != after else {}),
    )
