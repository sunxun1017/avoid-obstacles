"""Measured foot-ground peak-force excess; opt-in single-term experiment."""

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodied_learning.tasks.cat_foot_guidance import sha256

KEY = "foot_contact_force"
SENSORS = ("cat_left_ground", "cat_right_ground")


def validate_schema(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "threshold_bw", "weight"}
        or value["version"] != "ground_force_peak_bw_v1"
    ):
        raise ValueError("Invalid foot contact force schema.")
    result = dict(value)
    for key in ("threshold_bw", "weight"):
        v = value[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(
                "Contact threshold and weight must be finite and positive."
            )
        result[key] = float(v)
    return result


def contact_force_cost(
    peak_force: torch.Tensor, *, weight_n: float, threshold_bw: float
) -> torch.Tensor:
    if (
        peak_force.ndim != 2
        or peak_force.shape[1] != 2
        or not peak_force.is_floating_point()
    ):
        raise ValueError("Contact force peaks must be floating [B,2].")
    if (
        not math.isfinite(weight_n)
        or weight_n <= 0
        or not math.isfinite(threshold_bw)
        or threshold_bw <= 0
    ):
        raise ValueError("Body weight and contact threshold must be positive.")
    value = (peak_force / weight_n - threshold_bw).clamp_min(0).sum(-1)
    return torch.where(torch.isfinite(value), value, 0.0)


def verify_measurement_model(actual: Any, reference: Any) -> None:
    """Allow sensor metadata changes only; verify all physical arrays/options."""
    for name in dir(reference):
        if name.startswith(("_", "sensor", "name")):
            continue
        a = getattr(reference, name)
        if isinstance(a, np.ndarray):
            b = getattr(actual, name)
            if a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                raise ValueError(
                    f"Force sensor changed physical model array {name}."
                )
    for name in dir(reference.opt):
        a = getattr(reference.opt, name)
        b = getattr(actual.opt, name)
        same = (
            np.array_equal(a, b, equal_nan=True)
            if isinstance(a, np.ndarray)
            else a == b
            if isinstance(a, (int, float))
            else True
        )
        if not same:
            raise ValueError(f"Force sensor changed physical option {name}.")


class FootContactForceState:
    def __init__(
        self,
        num_envs: int,
        device: str | torch.device,
        *,
        weight_n: float,
        threshold_bw: float,
    ) -> None:
        self.peak = torch.zeros(num_envs, 2, device=device)
        contact_force_cost(self.peak, weight_n=weight_n, threshold_bw=threshold_bw)
        self.weight_n = weight_n
        self.threshold_bw = threshold_bw

    def begin_control(self) -> None:
        self.peak.zero_()

    def reset(self, env_ids: torch.Tensor) -> None:
        self.peak[env_ids] = 0

    def accumulate(self, force: torch.Tensor) -> None:
        if force.shape != (*self.peak.shape, 3):
            raise ValueError("Foot forces must be [B,2,3].")
        self.peak.copy_(torch.maximum(self.peak, force.norm(dim=-1)))

    @property
    def cost_rate(self) -> torch.Tensor:
        return contact_force_cost(
            self.peak, weight_n=self.weight_n, threshold_bw=self.threshold_bw
        )

    def sample(self, native: Any) -> None:
        native.sim.sense()
        forces = []
        for name in SENSORS:
            sensor = native.scene[name]
            # The post-step hook runs before scene.update; clear cached cat()
            # output without advancing contact history/airtime (not enabled here).
            sensor.update(0.0)
            forces.append(sensor.data.force.sum(1))
        self.accumulate(torch.stack(forces, 1))

    def install(self, native: Any) -> None:
        original = native.sim.step

        def step_with_force() -> Any:
            result = original()
            self.sample(native)
            return result

        native.sim.step = step_with_force


def apply_contact_force(env: Any, protocol: dict[str, Any]) -> None:
    if KEY not in protocol:
        return
    schema = validate_schema(protocol[KEY])
    if schema is None:
        raise ValueError("Explicit contact force configuration cannot be null.")
    native = env.unwrapped
    c = native.cat_controller
    if (
        getattr(env.cat_state, "recovery", None) is not None
        or getattr(c, "landing", None) is not None
    ):
        raise ValueError(
            "Force pilot requires source baseline without landing-speed cost."
        )
    if getattr(c, "contact_force", None) is not None:
        raise ValueError("Contact force sampler already installed.")
    if getattr(native, "cat_force_baseline_identity", None) is None:
        raise ValueError(
            "Contact force sensors require verified native construction."
        )
    model = native.sim.mj_model
    robot_ids = [
        i for i in range(model.nbody) if model.body(i).name.startswith("robot/")
    ]
    if not robot_ids:
        raise ValueError("Missing named robot bodies for force normalization.")
    weight_n = float(model.body_mass[robot_ids].sum()) * 9.81
    state = FootContactForceState(
        env.num_envs,
        env.device,
        weight_n=weight_n,
        threshold_bw=schema["threshold_bw"],
    )
    c.contact_force = state
    state.install(native)
    env.cat_state.scales[KEY] = -schema["weight"]
    env.schema[KEY] = schema


def adapt_contact_force_transfer(
    checkpoint: str | Path,
    old: dict[str, Any],
    target: dict[str, Any],
    source: dict[str, Any],
    curriculum: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    declaration = curriculum.get("contact_force_refinement")
    if declaration is None or (
        source["experiment"] == curriculum["experiment"]
        and source["protocol_sha256"] == curriculum["protocol_sha256"]
    ):
        return old, source, {}
    before = validate_schema(old["schema"].get(KEY))
    after = validate_schema(target["schema"].get(KEY))
    expected = dict(
        version="cat_contact_force_actor_transfer_v1",
        parent_sha256=sha256(checkpoint),
        parent_experiment=source["experiment"],
        parent_protocol_sha256=source["protocol_sha256"],
        before=before,
        after=after,
    )
    if declaration != expected or source["stage"] != curriculum["stage"]:
        raise ValueError("Contact force transfer parent/declaration/stage mismatch.")
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
