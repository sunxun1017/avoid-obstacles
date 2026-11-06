"""Actual sensor -> map -> source field -> CAT observation integration."""

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)


@pytest.mark.parametrize("sensor", ["lidar", "depth"])
def test_online_native_without_offline_cache(sensor, monkeypatch):
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_online import make_cat_online_env
    from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig
    from embodied_learning.tasks import field_assets

    def forbidden(*args, **kwargs):
        raise AssertionError("Online deployment must not load/voxelize offline maps")

    monkeypatch.setattr(field_assets, "load_field_cache", forbidden)
    monkeypatch.setattr(field_assets, "prepare_fields", forbidden)
    monkeypatch.setattr(
        "embodied_learning.data.field_scenes.voxelize_scene", forbidden
    )
    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    env = make_cat_online_env(
        manifest,
        scene_ids=("wall_detour",),
        num_envs=2,
        device="cuda:0",
        seed=42,
        options=dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False),
        sensor=OnlineSensorConfig(source=sensor),
    )
    try:
        obs = env.get_observations()
        provider = env.cat_state.bank
        assert obs["actor"].shape == (2, 162)
        assert obs["critic"].shape == (2, 250)
        assert torch.isfinite(obs["actor"]).all()
        assert provider.online.versions.min() >= 1, provider.online.errors
        snapshot = provider.online.mapper.snapshot()
        assert snapshot.observed.flatten(1).any(1).all()
        assert snapshot.occupied.flatten(1).any(1).all()
        assert provider.last_query.sample.guidance_valid.all()
        assert env.schema["field_cache_identity"] is None
        before = provider.online.versions.clone()
        for _ in range(6):
            obs, _, _, _ = env.step(torch.zeros(2, 12, device="cuda:0"))
            assert torch.isfinite(obs["actor"]).all()
        assert (provider.online.versions > before).all()
        slot1_map = provider.online.mapper.snapshot().log_odds[1].clone()
        slot1_version = provider.online.versions[1].clone()
        env.reset_slots(torch.tensor([0], device="cuda:0"))
        assert provider.online.versions[1] == slot1_version
        assert torch.equal(provider.online.mapper.snapshot().log_odds[1], slot1_map)
        assert provider.online.versions[0] == 1
    finally:
        env.close()


@pytest.mark.parametrize("sensor", ["lidar", "depth"])
def test_online_loads_checkpoint_from_actual_sensorless_model(tmp_path, sensor):
    from types import SimpleNamespace

    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.perception.online_field import default_field_sample
    from embodied_learning.policies.cat_online import load_online_policy
    from embodied_learning.policies.field_policy import FieldStudent
    from embodied_learning.protocols.imitation import PolicyContract
    from embodied_learning.simulators.mjlab.cat_online import make_cat_online_env
    from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env

    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    kwargs = dict(
        scene_ids=("wall_detour",),
        num_envs=1,
        device="cuda:0",
        seed=42,
        options=dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False),
    )
    # Numerical fields are irrelevant to compiled model/observation compatibility.
    bank = SimpleNamespace(
        cache_identity="test-only",
        source_profile={"version": "cat_source_v1"},
        sample_source=lambda p, *args: default_field_sample(p),
    )
    training = make_cat_env(manifest, bank, **kwargs)
    try:
        contract = PolicyContract.from_env(training, (16,))
        student = FieldStudent(contract, "cpu")
        path = tmp_path / "untrained-test-only.pt"
        torch.save(
            dict(field_contract=contract.to_dict(), student=student.state_dict()),
            path,
        )
    finally:
        training.close()
    online = make_cat_online_env(
        manifest, sensor=OnlineSensorConfig(source=sensor), **kwargs
    )
    try:
        expected = PolicyContract.from_env(online, (16,))
        policy = load_online_policy(path, online_contract=expected, device="cuda:0")
        assert torch.isfinite(policy.predict(online.get_observations())).all()
        assert (
            online.schema["online_perception"]["sensorless_model_identity"]
            == contract.schema["robot_model_identity"]
        )
    finally:
        online.close()


@pytest.mark.parametrize("sensor", ["lidar", "depth"])
def test_async_native_consumes_measured_gpu_fields_and_closes_worker(sensor):
    import time

    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_online import make_cat_online_env
    from embodied_learning.simulators.mjlab.cat_sensors import OnlineSensorConfig

    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    env = make_cat_online_env(
        manifest,
        scene_ids=("wall_detour",),
        num_envs=1,
        device="cuda:0",
        seed=42,
        options=dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False),
        sensor=OnlineSensorConfig(source=sensor),
        execution="async",
        field_backend="cuda",
        mapping_backend="warp",
    )
    bank = env.cat_state.bank
    try:
        # Simulate a paused consumer after startup. The first publication is
        # deliberately stale, so finite default vectors cannot satisfy this test.
        until = time.monotonic() + 30.0
        channel = bank.online._transport
        while channel.arrays["out_sequence"][0] < channel.arrays["in_sequence"][0]:
            assert time.monotonic() < until, bank.diagnostics()
            time.sleep(0.005)
        time.sleep(bank.online.max_age + 0.02)
        until = time.monotonic() + 30.0
        saw_stale = False
        while time.monotonic() < until:
            # Recompute the native actor/critic observations without moving the
            # robot. Both current and delayed field queries must recover.
            env.unwrapped.common_step_counter += 1
            env.unwrapped.episode_length_buf += 1
            obs = env.unwrapped.observation_manager.compute(update_history=True)
            if bank.query_available.all():
                break
            saw_stale = True
            time.sleep(0.02)
        assert saw_stale
        assert bank.query_available.all(), bank.diagnostics()
        assert torch.isfinite(obs["actor"]).all()
        assert obs["actor"][0, 77:79].norm() > 0.1
        assert bank.online.versions[0] > 0, bank.diagnostics()
        assert bank.diagnostics()["measured_voxels"][0] > 0
        assert bank.diagnostics()["occupied_voxels"][0] > 0
        assert env.schema["online_perception"]["execution"] == "async"
        assert env.schema["online_perception"]["field_backend"] == "cuda"
        sample = bank.sample_source(
            env.cat_state.last_positions,
            env.cat_state.scene_ids,
            env.cat_state.goal_ids,
        )
        assert torch.isfinite(sample.guidance).all()
        assert sample.guidance_valid.all()
        assert (bank.last_query.age <= bank.online.max_age).all()
        assert bank.online.worker_alive
    finally:
        env.close()
    assert not bank.online.worker_alive
