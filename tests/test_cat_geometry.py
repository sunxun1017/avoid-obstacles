"""Full native collision-volume coverage, independent of source site queries."""

import importlib
import importlib.util
import json
import math
from pathlib import Path

import mujoco
import pytest
import torch

from embodied_learning.data.field_scenes import FieldScene, SceneBox, SceneManifest
from embodied_learning.robots.g1 import CAT_DEFAULT_JOINTS, CAT_JOINT_NAMES


@pytest.fixture(scope="module")
def geometry():
    name = "embodied_learning.simulators.mjlab.cat_geometry"
    assert importlib.util.find_spec(name) is not None, "Collision envelope is absent"
    return importlib.import_module(name)


def manifest(*box_sets):
    return SceneManifest(
        tuple(
            FieldScene(str(i), (-3, -3, -1), (80, 80, 80), 0.1, tuple(boxes), (), ())
            for i, boxes in enumerate(box_sets)
        )
    )


def test_capsule_cover_includes_between_centers_and_end_caps(geometry):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom name="left_foot4_collision" '
        'type="capsule" size=".1 .2"/></body></worldbody></mujoco>'
    )
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu", max_segment=0.2)
    pos = torch.zeros(1, model.ngeom, 3)
    rot = torch.eye(3).repeat(1, model.ngeom, 1, 1)
    centers = envelope.world_centers(pos, rot, torch.zeros(1, 3))[0]
    # Includes cylinder-side midpoint and both axial end-cap tips.
    surface = torch.tensor([[0.1, 0, -0.1], [0, 0, 0.3], [0, 0, -0.3]])
    covered = torch.cdist(surface, centers) <= envelope.radii[None] + 1e-7
    assert covered.any(1).all(), "A center-only sparse sampler misses the volume"
    assert envelope.groups.unique().tolist() == [3]


def test_world_transform_uses_compiled_geom_rotation_and_env_origin(geometry):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom name="head_collision" '
        'type="sphere" size=".06"/><geom name="left_shin_collision" '
        'type="capsule" size=".045 .1"/></body></worldbody></mujoco>'
    )
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu", max_segment=0.1)
    pos = torch.tensor([[[10, 20, 1], [10, 20, 1]]], dtype=torch.float32)
    # Local capsule z points along world x.
    rot = torch.tensor([[[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]] * 2]).float()
    points = envelope.world_centers(pos, rot, torch.tensor([[10.0, 20, 0]]))
    head = points[0, envelope.groups == 0]
    shin = points[0, envelope.groups == 7]
    torch.testing.assert_close(head, torch.tensor([[0.0, 0, 1]]))
    torch.testing.assert_close(shin[:, 0].min(), torch.tensor(-0.1))
    torch.testing.assert_close(shin[:, 0].max(), torch.tensor(0.1))
    torch.testing.assert_close(shin[:, 2], torch.ones(len(shin)))
    assert envelope.radii[envelope.groups == 0].item() == pytest.approx(0.06)


def test_analytic_aabb_distance_normal_and_radius_are_scene_routed(geometry):
    scenes = manifest((SceneBox((0, 0, 0), (1, 1, 1)),), ())
    points = torch.tensor([[[2.0, 0, 0], [0.5, 0, 0]]] * 2)
    result = geometry.sample_box_envelope(
        points,
        torch.tensor([0.2, 0.1]),
        torch.tensor([3, 7]),
        torch.tensor([0, 1]),
        scenes,
    )
    torch.testing.assert_close(
        result.clearance[0, [3, 7]], torch.tensor([0.8, -0.6])
    )
    torch.testing.assert_close(
        result.gradient[0, [3, 7]], torch.tensor([[1.0, 0, 0]] * 2)
    )
    assert result.minimum[0].item() == pytest.approx(-0.6)
    assert torch.isposinf(result.clearance[1]).all()
    assert torch.isposinf(result.minimum[1])
    assert (result.gradient[1] == 0).all()
    assert result.valid.tolist() == [True, True]


def test_group_minimum_keeps_normal_of_the_limiting_geometry(geometry):
    scenes = manifest((SceneBox((0, 0, 0), (1, 1, 1)),))
    result = geometry.sample_box_envelope(
        torch.tensor([[[2.0, 0, 0], [0, -1.1, 0]]]),
        torch.tensor([0.1, 0.2]),
        torch.tensor([3, 3]),
        torch.tensor([0]),
        scenes,
    )
    assert result.clearance[0, 3].item() == pytest.approx(-0.1)
    torch.testing.assert_close(result.gradient[0, 3], torch.tensor([0.0, -1, 0]))


def test_nonfinite_environment_is_invalid_even_without_obstacles(geometry):
    result = geometry.sample_box_envelope(
        torch.tensor([[[float("nan"), 0, 0]], [[1.0, 0, 0]]]),
        torch.tensor([0.1]),
        torch.tensor([0]),
        torch.tensor([0, 0]),
        manifest(()),
    )
    assert result.valid.tolist() == [False, True]
    assert torch.isneginf(result.clearance[0]).all()
    assert torch.isneginf(result.minimum[0])
    assert (result.gradient[0] == 0).all()


@pytest.mark.parametrize("bad", ["radius", "group", "scene", "shape"])
def test_malformed_query_contract_is_rejected(geometry, bad):
    points = torch.zeros(1, 1, 3)
    radii, groups, ids = torch.tensor([0.1]), torch.tensor([0]), torch.tensor([0])
    if bad == "radius":
        radii[0] = -0.1
    elif bad == "group":
        groups[0] = 11
    elif bad == "scene":
        ids[0] = 1
    else:
        points = points[:, :, :2]
    with pytest.raises(ValueError):
        geometry.sample_box_envelope(points, radii, groups, ids, manifest(()))


def test_unsupported_collision_box_is_explicitly_rejected(geometry):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom name="head_collision" '
        'type="box" size=".1 .1 .1"/></body></worldbody></mujoco>'
    )
    with pytest.raises(ValueError, match="Unsupported"):
        geometry.CollisionEnvelope.from_model(model, "cpu")


@pytest.fixture(scope="module")
def native_geometry():
    from mjlab.entity import EntityCfg
    from mjlab.scene import Scene

    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.simulators.mjlab.cat_traversal import build_cat_cfg
    from embodied_learning.simulators.mjlab.field_traversal import _box_spec

    scenes = load_scene_manifest(Path("configs/experiments/g1_cat_source.yaml"))
    cfg = build_cat_cfg(num_envs=1, seed=42)
    cfg.scene.sensors = ()
    for i, box in enumerate(scenes.scenes[2].boxes):
        cfg.scene.entities[f"obstacle_{i}"] = EntityCfg(
            spec_fn=lambda half=box.half_size: _box_spec(half),
            init_state=EntityCfg.InitialStateCfg(pos=box.center),
        )
    model = Scene(cfg.scene, device="cpu").compile()
    return model, scenes


def test_all_native_collision_geoms_are_covered_and_schema_serializes(
    geometry, native_geometry
):
    model, _ = native_geometry
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu")
    enabled = {
        i
        for i in range(model.ngeom)
        if model.geom(i).name.startswith("robot/")
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    }
    assert len(enabled) == 33
    assert set(envelope.geom_ids.tolist()) == enabled
    assert set(envelope.groups.tolist()) == set(range(11))
    schema = json.loads(json.dumps(envelope.schema(), allow_nan=False))
    assert schema["version"] == "offline_box_envelope_v1"
    assert schema == geometry.CollisionEnvelope.from_model(model, "cpu").schema()
    assert (
        schema
        != geometry.CollisionEnvelope.from_model(
            model, "cpu", max_segment=0.02
        ).schema()
    )


@pytest.mark.parametrize("root_x,group", [(-0.60, 2), (0.30, 3), (0.35, 7)])
def test_native_torso_toe_and_shin_contact_is_not_reported_as_clear(
    geometry, native_geometry, root_x, group
):
    model, scenes = native_geometry
    data = mujoco.MjData(model)
    data.qpos[:3] = (root_x, 0, 0.8)
    data.qpos[3:7] = (1, 0, 0, 0)
    for name, angle in zip(CAT_JOINT_NAMES, CAT_DEFAULT_JOINTS, strict=True):
        data.qpos[model.jnt_qposadr[model.joint(f"robot/{name}").id]] = angle
    # Native fixed bodies are carried by wrapper mocap roots.
    for i, box in enumerate(scenes.scenes[2].boxes):
        ids = [
            int(model.body_mocapid[j])
            for j in range(model.nbody)
            if model.body(j).name.startswith(f"obstacle_{i}/")
            and model.body_mocapid[j] >= 0
        ]
        data.mocap_pos[ids[0]] = box.center
    mujoco.mj_forward(model, data)
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu")
    points = envelope.world_centers(
        torch.tensor(data.geom_xpos[None], dtype=torch.float32),
        torch.tensor(data.geom_xmat.reshape(1, -1, 3, 3), dtype=torch.float32),
        torch.zeros(1, 3),
    )
    result = geometry.sample_box_envelope(
        points, envelope.radii, envelope.groups, torch.tensor([2]), scenes
    )
    cached = geometry.BoxEnvelopeQuery(scenes, envelope).sample(
        points, torch.tensor([2])
    )
    for name in ("clearance", "gradient", "minimum", "valid"):
        torch.testing.assert_close(getattr(cached, name), getattr(result, name))
    assert result.valid.item()
    assert result.clearance[0, group] < 0
    names = {
        "robot/torso_collision",
        "robot/left_foot4_collision",
        "robot/left_shin_collision",
    }
    assert any(
        contact.dist < 0
        and model.geom(contact.geom1).name in names
        and model.geom(contact.geom2).name.startswith("obstacle_")
        for contact in data.contact
    )
    # Compare the selected native geom with CPU exact narrow phase.
    native_name = {2: "torso", 3: "left_foot4", 7: "left_shin"}[group]
    geom = model.geom(f"robot/{native_name}_collision").id
    obstacle = model.geom(f"obstacle_{0 if group == 2 else 1}/collision").id
    exact = mujoco.mj_geomDistance(model, data, geom, obstacle, math.inf, None)
    assert exact < 0
    assert result.clearance[0, group].item() <= exact + 1e-5


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_query_matches_strict_query_for_rotated_mixed_scenes(geometry, dtype):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom name="head_collision" '
        'type="sphere" size=".06"/><geom name="left_shin_collision" '
        'type="capsule" size=".045 .1"/></body></worldbody></mujoco>'
    )
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu")
    scenes = manifest(
        (SceneBox((0, 0, 1), (0.2, 0.3, 0.4)), SceneBox((1, 0, 1), (0.1, 0.2, 0.3))),
        (SceneBox((0, -1, 1), (0.3, 0.3, 0.3)),),
        (),
    )
    query = geometry.BoxEnvelopeQuery(scenes, envelope, dtype=dtype)
    positions = torch.tensor([[[10.3, 20, 1], [10.3, 20, 1]]] * 5, dtype=dtype)
    rotation = torch.tensor(
        [[[[0, 0, 1], [0, 1, 0], [-1, 0, 0]]] * 2] * 5, dtype=dtype
    )
    origins = torch.tensor([[10, 20, 0]] * 5, dtype=dtype)
    centers = envelope.world_centers(positions, rotation, origins)
    centers[3, 0, 0] = float("nan")  # An empty scene cannot hide invalid physics.
    centers[4, 0, 0] = float("inf")
    ids = torch.tensor([0, 1, 2, 2, 0])
    expected = geometry.sample_box_envelope(
        centers, envelope.radii.to(dtype), envelope.groups, ids, scenes
    )
    actual = query.sample(centers, ids)
    for name in ("clearance", "gradient", "minimum", "valid"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


def test_cached_query_has_no_tensor_scalar_or_dynamic_index_host_reads(
    geometry, monkeypatch
):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom name="head_collision" '
        'type="sphere" size=".06"/></body></worldbody></mujoco>'
    )
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu")
    query = geometry.BoxEnvelopeQuery(
        manifest((SceneBox((0, 0, 0), (0.2, 0.3, 0.4)),), ()), envelope
    )
    centers, ids = torch.zeros(2, 1, 3), torch.tensor([0, 1])

    def reject(*args, **kwargs):
        raise AssertionError("Query hot path must not read GPU values on the host")

    with monkeypatch.context() as patch:
        for name in ("__bool__", "item", "tolist", "cpu", "nonzero", "new_tensor"):
            patch.setattr(torch.Tensor, name, reject)
        patch.setattr(torch, "nonzero", reject)
        patch.setattr(torch, "tensor", reject)
        result = query.sample(centers, ids)
    assert result.valid.tolist() == [True, True]
    assert result.minimum[0] < 0
    assert torch.isposinf(result.minimum[1])


def test_cached_query_empty_manifest_and_invalid_scene_ids_fail_closed(geometry):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><geom name="head_collision" '
        'type="sphere" size=".06"/></body></worldbody></mujoco>'
    )
    envelope = geometry.CollisionEnvelope.from_model(model, "cpu")
    with pytest.raises(ValueError, match="scene"):
        geometry.BoxEnvelopeQuery(manifest(), envelope)
    query = geometry.BoxEnvelopeQuery(manifest(()), envelope)
    result = query.sample(torch.zeros(3, 1, 3), torch.tensor([-1, 0, 1]))
    assert result.valid.tolist() == [False, True, False]
    assert torch.isneginf(result.clearance[[0, 2]]).all()
    assert torch.isposinf(result.clearance[1]).all()
    assert not result.gradient.any()
