"""Actual mjlab stepping must acquire CAT scans only at their sensor cadence."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from embodied_learning.data.field_scenes import FieldScene, SceneManifest
from embodied_learning.perception.rays import RayMeasurements
from embodied_learning.simulators.mjlab import cat_online, cat_traversal
from embodied_learning.tasks.cat_online import CatOnlineFields


class _Manager:
    def reset(self, *args, **kwargs):
        return {}

    def compute(self, *args, **kwargs):
        return None

    def compute_substep(self):
        pass


def _native_loop(
    monkeypatch,
    *,
    execution="sync",
    terminate_at=None,
    sensor_seconds=0.0,
):
    """Keep native step/reset, CAT refresh, ray masking and field lifecycle real.

    CPU stand-ins replace only unavailable GPU physics/rendering and unrelated
    managers. Sensing records the physical pose and produces a measured ray;
    changing acquisition cadence therefore changes actual map publications.
    """
    scene = FieldScene(
        "test",
        (0.0, 0.0, 0.0),
        (30, 30, 30),
        0.04,
        (),
        ((0.2, 0.2, 0.8),),
        ((0.9, 0.4, 0.8),),
    )
    bank = CatOnlineFields(SceneManifest((scene,)), num_envs=2, device="cpu")
    # The worker transport is tested separately. Keep real map admission/builds
    # here, while selecting the production scheduler's monotonic-clock branch.
    bank.execution = execution
    bank.online.poll = lambda: False
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(cat_online.time, "monotonic", lambda: clock.now)
    state = SimpleNamespace(
        _key=None,
        generation=0,
        goals=torch.tensor(scene.goals).repeat(2, 1),
        bank=bank,
        scene_ids=torch.zeros(2, dtype=torch.long),
        history=[],
    )
    sim = SimpleNamespace(
        position=torch.tensor([[0.2, 0.2, 0.8]]).repeat(2, 1),
        captures=[],
        data=SimpleNamespace(overflow=torch.zeros(2), nefc=torch.zeros(2)),
        wp_data=SimpleNamespace(njmax=10),
    )

    def forward():
        sim.kinematics = sim.position.clone()
        sim.contact = sim.position[:, 0] > 0.23

    def physics():
        sim.position[:, 0] += 0.001

    def sense():
        sim.measured_pose = sim.kinematics.clone()
        sim.captures.append((env.common_step_counter, clock.now, sim.measured_pose))
        clock.now += sensor_seconds

    def read():
        origins = sim.measured_pose[:, None].clone()
        ends = origins.clone()
        ends[..., 0] += 0.2
        mask = torch.ones(2, 1, dtype=torch.bool)
        return RayMeasurements(origins, ends, mask, mask)

    def refresh(payload, controller, steps, common):
        state._key = (common, state.generation)
        state.history.append((common, payload["position"], payload["contact"]))

    def reset_state(ids):
        state.generation += 1
        state._key = None

    def reset_robot(env, ids):
        sim.position[ids] = torch.tensor([0.2, 0.2, 0.8])

    def payload(env, state):
        return dict(position=sim.kinematics.clone(), contact=sim.contact.clone())

    state.refresh, state.reset = refresh, reset_state
    sim.forward, sim.step, sim.sense = forward, physics, sense
    sim.reset = lambda ids: None
    monkeypatch.setattr(cat_traversal, "read_cat_reward_inputs", payload)
    monkeypatch.setattr(cat_traversal, "_reset_field", reset_robot)

    def noop(*args, **kwargs):
        return None

    env = SimpleNamespace(
        cat_state=state,
        cat_sensor_config=object(),
        cat_sensor_schedule=None,
        cat_sensor_reader=SimpleNamespace(read=read),
        cat_controller=None,
        cfg=SimpleNamespace(auto_reset=True, decimation=10),
        num_envs=2,
        device="cpu",
        physics_dt=0.002,
        step_dt=0.02,
        sim=sim,
        common_step_counter=0,
        _sim_step_counter=0,
        extras={},
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        _manual_reset_pending=torch.zeros(2, dtype=torch.bool),
        _command_dt=torch.zeros(2),
        scene=SimpleNamespace(
            write_data_to_sim=noop,
            update=noop,
            reset=noop,
            env_origins=torch.zeros(2, 3),
        ),
        action_manager=_Manager(),
        reward_manager=_Manager(),
        metrics_manager=_Manager(),
        command_manager=_Manager(),
        curriculum_manager=_Manager(),
        event_manager=_Manager(),
        recorder_manager=SimpleNamespace(
            record_post_step=noop,
            record_pre_reset=noop,
            record_post_reset=noop,
        ),
    )
    env.action_manager.process_action = env.action_manager.apply_action = noop
    env.reward_manager.compute = lambda **kwargs: torch.zeros(2)
    env.event_manager.available_modes = ["reset"]
    env.event_manager.apply = lambda *, env_ids, **kwargs: (
        cat_traversal._source_reset(env, env_ids)
    )

    def terminate():
        cat_traversal._refresh(env)
        return torch.tensor([env.common_step_counter == terminate_at, False])

    def observations(**kwargs):
        cat_traversal._refresh(env)
        return {"actor": sim.kinematics.clone()}

    env.termination_manager = _Manager()
    env.termination_manager.compute = terminate
    env.termination_manager.terminated = torch.zeros(2, dtype=torch.bool)
    env.termination_manager.time_outs = torch.zeros(2, dtype=torch.bool)
    env.observation_manager = _Manager()
    env.observation_manager.compute = observations
    env._reset_idx = lambda ids: ManagerBasedRlEnv._reset_idx(env, ids)
    # Observation shape discovery calls this during real native construction.
    cat_traversal._refresh(env)
    sim.captures.clear()
    state.history.clear()
    return env, clock


@pytest.mark.parametrize("execution", ["sync", "async"])
def test_native_step_acquires_at_sensor_cadence_not_twice_per_control_step(
    monkeypatch,
    execution,
):
    # Restoring unconditional sense in _refresh or at native step end must fail.
    env, clock = _native_loop(monkeypatch, execution=execution)
    for step in range(1, 51):
        clock.now = 10.0 + step * (0.004 if execution == "async" else 0.02)
        ManagerBasedRlEnv.step(env, torch.zeros(2, 12))
    expected_steps = [25, 50] if execution == "async" else list(range(5, 51, 5))
    assert [item[0] for item in env.sim.captures] == expected_steps
    assert env.cat_state.bank.online.versions.tolist() == (
        [3, 3] if execution == "async" else [11, 11]
    )
    # Physics/contacts still refresh every control step, even without a scan.
    history = env.cat_state.history
    assert len(history) == 50
    torch.testing.assert_close(history[0][1][:, 0], torch.full((2,), 0.21))
    torch.testing.assert_close(history[-1][1][:, 0], torch.full((2,), 0.70))
    assert not history[0][2].any() and history[-1][2].all()


def test_reset_captures_new_pose_without_resubmitting_other_slots(monkeypatch):
    # Removing reset invalidation or replaying all rays after reset must fail.
    env, clock = _native_loop(monkeypatch, terminate_at=5)
    for step in range(1, 7):
        clock.now = 10.0 + step * 0.02
        ManagerBasedRlEnv.step(env, torch.zeros(2, 12))
    assert [item[0] for item in env.sim.captures] == [5, 5]
    assert env.cat_state.bank.online.versions.tolist() == [1, 2]
    pre, post = [item for item in env.cat_state.history if item[0] == 5]
    torch.testing.assert_close(pre[1][:, 0], torch.tensor([0.25, 0.25]))
    torch.testing.assert_close(post[1][:, 0], torch.tensor([0.20, 0.25]))
    assert pre[2].tolist() == [True, True]
    assert post[2].tolist() == [False, True]
    torch.testing.assert_close(env.sim.captures[0][2], pre[1])
    torch.testing.assert_close(env.sim.captures[1][2], post[1])


def test_goal_change_gets_immediate_scan_without_advancing_other_slots(monkeypatch):
    env, clock = _native_loop(monkeypatch)
    for step in range(1, 8):
        clock.now = 10.0 + step * 0.02
        if step == 2:
            env.cat_state.goals[0, 1] += 0.04
        ManagerBasedRlEnv.step(env, torch.zeros(2, 12))
    assert [item[0] for item in env.sim.captures] == [2, 5, 7]
    assert env.cat_state.bank.online.versions.tolist() == [3, 2]


def test_busy_submission_does_not_raise_sensor_rate_to_control_rate(monkeypatch):
    env, clock = _native_loop(monkeypatch)
    # Model rejected mailbox admission at the real submission boundary: no
    # attempt/measurement timestamp advances. Acquisition must still be paced.
    monkeypatch.setattr(env.cat_state.bank.online, "update", lambda *a, **k: False)
    for step in range(1, 11):
        clock.now = 10.0 + step * 0.02
        ManagerBasedRlEnv.step(env, torch.zeros(2, 12))
    assert [item[0] for item in env.sim.captures] == [5, 10]
    assert env.cat_state.bank.online.versions.tolist() == [1, 1]


def test_busy_scan_then_partial_reset_does_not_drop_other_slots_next_scan(
    monkeypatch,
):
    env, clock = _native_loop(monkeypatch, terminate_at=6)
    update = env.cat_state.bank.online.update

    def busy_once(*args, **kwargs):
        if env.common_step_counter == 5:
            return False
        return update(*args, **kwargs)

    monkeypatch.setattr(env.cat_state.bank.online, "update", busy_once)
    for step in range(1, 11):
        clock.now = 10.0 + step * 0.02
        ManagerBasedRlEnv.step(env, torch.zeros(2, 12))
    assert [item[0] for item in env.sim.captures] == [5, 6, 10]
    assert env.cat_state.bank.online.versions.tolist() == [1, 2]
    torch.testing.assert_close(
        env.cat_state.bank.online._stamp,
        torch.tensor([0.12, 0.20], dtype=torch.float64),
    )


def test_async_measurement_keeps_acquisition_time_and_pose(monkeypatch):
    env, clock = _native_loop(monkeypatch, execution="async", sensor_seconds=0.03)
    clock.now = 10.1
    ManagerBasedRlEnv.step(env, torch.zeros(2, 12))
    assert len(env.sim.captures) == 1
    _, stamp, measured_pose = env.sim.captures[0]
    assert stamp == pytest.approx(10.1)
    torch.testing.assert_close(measured_pose[:, 0], torch.tensor([0.21, 0.21]))
    assert env.cat_state.bank.timestamp == pytest.approx(10.13)
    torch.testing.assert_close(
        env.cat_state.bank.online._stamp,
        torch.full((2,), 10.1, dtype=torch.float64),
    )


def test_spawned_worker_accepts_preselected_scans_without_producer_rethrottling():
    import time

    from embodied_learning.perception.async_field import AsyncFieldMap
    from embodied_learning.perception.occupancy import OccupancyMapper

    pipeline = AsyncFieldMap(
        OccupancyMapper(2, (20, 20, 25), voxel_size=0.04),
        max_rays=1,
        update_period=0.1,
        max_age=0.5,
        preprocess=False,
    )
    target = torch.tensor([[0.413, 0.427, 0.613]]).repeat(2, 1)
    origins = torch.tensor([[[0.213, 0.217, 0.413]]]).repeat(2, 1, 1)
    endpoints = origins.clone()
    endpoints[..., 0] = 0.69
    hits = torch.ones(2, 1, dtype=torch.bool)

    def wait_result():
        deadline = time.monotonic() + 5.0
        while not pipeline.poll():
            assert time.monotonic() < deadline, pipeline.diagnostics()
            time.sleep(0.005)

    with pipeline:
        stamp = time.monotonic() - 1.0
        # The slot-0-only publication at .12 must not suppress the explicitly
        # scheduled slot-1 scan at .20, despite the producer's shared admission.
        for offset, selected in [
            (0.0, [True, True]),
            (0.12, [True, False]),
            (0.20, [False, True]),
        ]:
            rays = RayMeasurements(
                origins,
                endpoints,
                torch.tensor(selected)[:, None],
                hits,
            )
            assert pipeline.update(
                rays,
                target,
                timestamp=stamp + offset,
                apply_cadence=False,
            )
            wait_result()
        assert pipeline.versions.tolist() == [2, 2]
        torch.testing.assert_close(
            pipeline._stamp,
            torch.tensor([stamp + 0.12, stamp + 0.20], dtype=torch.float64),
        )
        with pytest.raises(ValueError, match="future"):
            pipeline.update(
                rays,
                target,
                timestamp=time.monotonic() + 1.0,
                apply_cadence=False,
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires native CUDA sensors"
)
@pytest.mark.parametrize("sensor", ["depth", "lidar"])
def test_native_camera_and_lidar_acquisition_is_throttled(monkeypatch, sensor):
    from pathlib import Path

    from mjlab.sim import Simulation

    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_online import make_cat_online_env
    from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig

    sense = Simulation.sense
    captures = []

    def counted(sim):
        captures.append(sim)
        return sense(sim)

    monkeypatch.setattr(Simulation, "sense", counted)
    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    env = make_cat_online_env(
        manifest,
        scene_ids=("wall_detour",),
        num_envs=1,
        device="cuda:0",
        seed=42,
        options=dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False),
        sensor=OnlineSensorConfig(source=sensor),
    )
    try:
        captures.clear()
        state = env.cat_state
        before = state.bank.online.versions.clone()
        generation = state.generation
        expected_collision = torch.zeros(1, dtype=torch.bool, device="cuda:0")
        for _ in range(10):
            env.step(torch.zeros(1, 12, device="cuda:0"))
            assert state.generation == generation, (
                "Test unexpectedly reset its robot"
            )
            fresh = cat_traversal.read_cat_reward_inputs(env.unwrapped, state)
            torch.testing.assert_close(state.last_positions, fresh["body_positions"])
            expected_collision |= fresh["physical_collision"]
            torch.testing.assert_close(
                state.saved_outcomes["physical_collision"],
                expected_collision,
            )
        assert len(captures) == 2
        torch.testing.assert_close(state.bank.online.versions, before + 2)
        assert state.bank.ray_counts.min() > 0
        assert state.bank.query_available.all()
    finally:
        env.close()
