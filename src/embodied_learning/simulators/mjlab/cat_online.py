"""Inference-only CAT sensor/field composition; no offline field loading."""

import time

import torch

from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig


class CatSensorSchedule:
    """Gate this environment's native sensing calls, including step/reset calls.

    mjlab 1.6 senses unconditionally at the end of step/reset. CAT also needs
    pre-reset observations for termination/rewards. Keep its Simulation intact,
    but route that instance's sense entry point through one acquisition schedule.
    A changed episode may require a second, post-reset capture in the same step.
    """

    def __init__(self, env):
        self.env = env
        self._sense = env.sim.sense
        self._key = None
        self._last_capture = torch.full(
            (env.num_envs,), -torch.inf, dtype=torch.float64, device=env.device
        )
        self._goals = torch.full_like(env.cat_state.goals, torch.nan)
        self.pending = False
        self.timestamp = None
        self.slots = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.sim.sense = self.sense

    def sense(self):
        env, bank = self.env, self.env.cat_state.bank
        key = (env.common_step_counter, env.cat_state.generation)
        if key == self._key:
            return
        self._key = key
        self.pending = False
        timestamp = (
            time.monotonic()
            if bank.execution == "async"
            else env.common_step_counter * env.step_dt
        )
        due = (
            timestamp - self._last_capture >= bank.online.update_period - 1e-9
        ) | (env.cat_state.goals != self._goals).any(-1)
        if not due.any():
            return
        self._sense()
        # The timestamp belongs to acquisition, never to subsequent submission.
        self.timestamp = timestamp
        self.slots = due
        self._last_capture[due] = timestamp
        self._goals[due] = env.cat_state.goals[due]
        self.pending = True

    def reset(self, env_ids):
        self._last_capture[env_ids] = -torch.inf
        self._goals[env_ids] = torch.nan
        self._key = None
        self.pending = False

    def close(self):
        self.env.sim.sense = self._sense


def install_sensor_schedule(env):
    if env.cat_sensor_schedule is None:
        env.cat_sensor_schedule = CatSensorSchedule(env)
    return env.cat_sensor_schedule


def make_cat_online_env(
    manifest,
    *,
    scene_ids,
    num_envs,
    device,
    seed,
    options,
    sensor: OnlineSensorConfig | None = None,
    update_period=0.1,
    max_age=0.3,
    execution="sync",
    field_backend="source",
    mapping_backend="torch",
):
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env
    from embodied_learning.tasks.cat_online import CatOnlineFields

    sensor = sensor or OnlineSensorConfig()
    bank = CatOnlineFields(
        manifest,
        num_envs=num_envs,
        device=device,
        update_period=update_period,
        max_age=max_age,
        execution=execution,
        field_backend=field_backend,
        mapping_backend=mapping_backend,
        max_rays=(
            sensor.width * sensor.height
            if sensor.source == "depth"
            else sensor.azimuth_count * sensor.elevation_count
        ),
    )
    try:
        bank.start()
        return make_cat_env(
            manifest,
            bank,
            scene_ids=scene_ids,
            num_envs=num_envs,
            device=device,
            seed=seed,
            options=options,
            online_sensor=sensor,
        )
    except BaseException:
        bank.close()
        raise


def refresh_online_fields(env):
    from embodied_learning.simulators.mjlab.cat_sensors import CatSensorReader

    bank = env.cat_state.bank
    if bank.execution == "async":
        bank.online.poll()
    schedule = install_sensor_schedule(env)
    schedule.sense()
    if schedule.pending:
        if env.cat_sensor_reader is None:
            env.cat_sensor_reader = CatSensorReader(env, env.cat_sensor_config)
        bank.update(
            env.cat_sensor_reader.read(),
            env.cat_state.goals,
            timestamp=schedule.timestamp,
            env_origins=env.scene.env_origins,
            capture_mask=schedule.slots,
        )
        schedule.pending = False
    # Account for publication/copy latency when checking field freshness.
    bank.timestamp = (
        time.monotonic()
        if bank.execution == "async"
        else env.common_step_counter * env.step_dt
    )
