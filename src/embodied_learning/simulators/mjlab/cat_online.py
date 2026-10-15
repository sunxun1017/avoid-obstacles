"""Inference-only CAT sensor/field composition; no offline field loading."""

import time

from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig


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


def refresh_online_fields(env, *, capture_timestamp=None):
    from embodied_learning.simulators.mjlab.cat_sensors import CatSensorReader

    bank = env.cat_state.bank
    timestamp = (
        (time.monotonic() if capture_timestamp is None else capture_timestamp)
        if bank.execution == "async"
        else env.common_step_counter * env.step_dt
    )
    if bank.execution == "async":
        bank.online.poll()
    if bank.online.due(timestamp, env.cat_state.goals).any():
        if env.cat_sensor_reader is None:
            env.cat_sensor_reader = CatSensorReader(env, env.cat_sensor_config)
        bank.update(
            env.cat_sensor_reader.read(),
            env.cat_state.goals,
            timestamp=timestamp,
            env_origins=env.scene.env_origins,
        )
    # Account for publication/copy latency when checking field freshness.
    bank.timestamp = time.monotonic() if bank.execution == "async" else timestamp
