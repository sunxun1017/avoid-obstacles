"""Research-only native envelope payloads and unchanged source physics."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from embodied_learning.data.field_scenes import (
    FieldScene,
    SceneBox,
    SceneManifest,
    load_scene_manifest,
)
from embodied_learning.simulators.mjlab.cat_geometry import CollisionEnvelope
from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env


def test_native_envelope_uses_full_geom_arrays_and_typed_scene_origins():
    from embodied_learning.simulators.mjlab.cat_traversal import _read_body_envelope

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="terrain" type="plane" size="1 1 .1"/>'
        '<body pos="11.5 20 1"><geom name="robot/head_collision" '
        'type="sphere" size=".1"/></body></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    envelope = CollisionEnvelope.from_model(model, "cpu")
    native = SimpleNamespace(
        cat_collision_envelope=envelope,
        sim=SimpleNamespace(
            data=SimpleNamespace(
                geom_xpos=torch.from_numpy(np.array(data.geom_xpos)).repeat(2, 1, 1),
                geom_xmat=torch.from_numpy(np.array(data.geom_xmat))
                .reshape(1, model.ngeom, 3, 3)
                .repeat(2, 1, 1, 1),
            )
        ),
        scene=SimpleNamespace(env_origins=torch.tensor([[10, 20, 0]] * 2)),
    )
    state = SimpleNamespace(
        scene_ids=torch.tensor([0, 1]),
        manifest=SceneManifest(
            tuple(
                FieldScene(str(i), (-3, -3, -1), (80, 80, 80), 0.1, boxes, (), ())
                for i, boxes in enumerate(
                    ((SceneBox((1, 0, 1), (0.2, 0.2, 0.2)),), ())
                )
            )
        ),
    )
    sampled = _read_body_envelope(native, state)
    assert sampled.clearance.dtype == torch.float64
    assert sampled.clearance[0, 0].item() == pytest.approx(0.2)
    torch.testing.assert_close(
        sampled.gradient[0, 0], torch.tensor([1.0, 0, 0]).double()
    )
    assert torch.isposinf(sampled.clearance[1]).all()
    assert sampled.valid.tolist() == [True, True]


def test_recovery_rejects_online_sensor_before_native_initialization():
    with pytest.raises(ValueError, match="[Oo]nline"):
        make_cat_env(
            None,
            None,
            scene_ids=("wall_detour",),
            num_envs=1,
            device="cpu",
            seed=42,
            options={},
            online_sensor=object(),
            task_profile="cat_recovery_v1",
            recovery_options={},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native G1 requires CUDA")
def test_research_native_payload_schema_and_source_physics_are_consistent():
    from embodied_learning.perception.online_field import default_field_sample
    from embodied_learning.simulators.mjlab.cat_traversal import (
        read_cat_reward_inputs,
    )

    manifest = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    bank = SimpleNamespace(
        cache_identity="test-only",
        source_profile={"version": "cat_source_v1"},
        sample_source=lambda positions, *args: default_field_sample(positions),
    )
    common = dict(
        scene_ids=("wall_detour",),
        num_envs=2,
        device="cuda:0",
        seed=42,
        options=dict(noise_level=0.0, random_pd=False, random_rfi=False, push=False),
    )
    source = make_cat_env(manifest, bank, **common)
    try:
        source_schema = source.schema
        assert "recovery" not in source_schema
        assert "body_envelope" not in read_cat_reward_inputs(
            source.unwrapped, source.cat_state
        )
    finally:
        source.close()
    research = make_cat_env(
        manifest,
        bank,
        **common,
        task_profile="cat_recovery_v1",
        recovery_options={"max_speed": 0.4},
    )
    try:
        obs = research.get_observations()
        assert obs["actor"].shape == (2, 171)
        assert obs["critic"].shape == (2, 256)
        assert torch.isfinite(obs["actor"]).all()
        metadata = research.schema["recovery"]
        assert metadata["config"]["max_speed"] == 0.4
        assert (
            metadata["envelope"]["provenance"]
            == "known_scene_boxes_not_online_perception"
        )
        for key in source_schema.keys() - {"task_profile", "observation"}:
            assert research.schema[key] == source_schema[key]
        payload = read_cat_reward_inputs(research.unwrapped, research.cat_state)
        assert payload["body_envelope"].clearance.shape == (2, 11)
        assert payload["body_envelope"].valid.all()
        for _ in range(3):
            obs, reward, _, extras = research.step(
                torch.zeros(2, 12, device="cuda:0")
            )
            assert torch.isfinite(reward).all()
            assert "cat_metrics" in extras
    finally:
        research.close()
