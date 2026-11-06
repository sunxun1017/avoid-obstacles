"""Known geometry, offline field identity and per-environment goal routing."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.data.field_scenes import load_scene_manifest, voxelize_scene
from embodied_learning.tasks.field_assets import load_field_cache, prepare_fields


def write_manifest(path, *, center=0.0):
    payload = {
        "scenes": [
            {
                "id": name,
                "origin": [-1.5, -1.5, 0.0],
                "shape": [16, 16, 11],
                "voxel_size": 0.2,
                "boxes": [
                    {"center": [center, offset, 0.9], "half_size": [0.2, 0.3, 0.3]}
                ],
                "starts": [[-1.1, 0.7, 0.8]],
                "goals": [[1.1, 0.7, 0.8], [-1.1, -0.7, 0.8]],
            }
            for name, offset in (("a", 0.0), ("b", -0.7))
        ]
    }
    path.write_text(json.dumps(payload))
    return load_scene_manifest(path)


def test_box_voxel_geometry(tmp_path):
    manifest = write_manifest(tmp_path / "scenes.json")
    scene = manifest.scenes[0]
    occupied = voxelize_scene(scene)
    # Grid node [7,7,4] = (-.1,-.1,.8), inside the declared box.
    assert occupied[7, 7, 4]
    assert not occupied[7, 3, 4]
    assert not occupied[7, 7, 0]


def test_no_floor_obstacle(tmp_path):
    manifest = write_manifest(tmp_path / "scenes.json")
    assert not voxelize_scene(manifest.scenes[0])[:, :, 0].any()


def test_cache_roundtrip(tmp_path):
    manifest = write_manifest(tmp_path / "scenes.json")
    cached = prepare_fields(manifest, tmp_path / "fields")
    bank = load_field_cache(cached.path, manifest, device="cpu")
    assert bank.num_scenes == 2
    assert bank.num_goals == (2, 2)
    q = torch.tensor([[[-1.1, 0.7, 0.8]]])
    result = bank.sample(q, torch.tensor([0]), torch.tensor([0]))
    assert result.geodesic_valid.all()
    assert torch.isfinite(result.geodesic).all()
    before = cached.path.read_bytes()
    prepare_fields(manifest, tmp_path / "fields")
    assert cached.path.read_bytes() == before
    with np.load(tmp_path / "fields" / "a-goal-0.npz", allow_pickle=False) as data:
        assert data["occupied"].dtype == np.bool_


def test_scene_goal_routing(tmp_path):
    manifest = write_manifest(tmp_path / "scenes.json")
    cached = prepare_fields(manifest, tmp_path / "fields")
    bank = load_field_cache(cached.path, manifest, device="cpu")
    q = torch.tensor([[[-1.1, 0.7, 0.8]]] * 3)
    scene_ids, goal_ids = torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])
    result = bank.sample(q, scene_ids, goal_ids)
    for i, (scene, goal) in enumerate(zip(scene_ids, goal_ids, strict=True)):
        direct = bank.fields[scene.item()][goal.item()].sample(q[i : i + 1])
        torch.testing.assert_close(result.geodesic[i], direct.geodesic[0])
        torch.testing.assert_close(result.guidance[i], direct.guidance[0])
    assert not torch.allclose(result.geodesic[0], result.geodesic[2])


def test_cache_rejects_geometry_change(tmp_path):
    manifest = write_manifest(tmp_path / "scenes.json")
    cached = prepare_fields(manifest, tmp_path / "fields")
    changed = write_manifest(tmp_path / "changed.json", center=0.4)
    with pytest.raises(ValueError, match="identity"):
        load_field_cache(cached.path, changed, device="cpu")


def test_all_start_goal_pairs_reachable():
    config = Path(__file__).parents[1] / "configs/experiments/g1_field.yaml"
    manifest = load_scene_manifest(config)
    assert len(manifest.scenes) >= 3
    from embodied_learning.perception.voxel_field import build_voxel_field

    for scene in manifest.scenes:
        for goal in scene.goals:
            field = build_voxel_field(
                voxelize_scene(scene),
                goal=goal,
                voxel_size=scene.voxel_size,
                origin=scene.origin,
            )
            positions = torch.tensor([scene.starts])
            assert field.sample(positions).geodesic_valid.all(), scene.id


def test_rejects_bad_goal_and_duplicate_ids(tmp_path):
    path = tmp_path / "scene.json"
    write_manifest(path)
    payload = json.loads(path.read_text())
    payload["scenes"][0]["goals"] = [[100, 0, 0]]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_scene_manifest(path)
    write_manifest(path)
    payload = json.loads(path.read_text())
    payload["scenes"][1]["id"] = "a"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_scene_manifest(path)


def test_cache_rejects_corrupt_arrays(tmp_path):
    manifest = write_manifest(tmp_path / "scenes.json")
    cached = prepare_fields(manifest, tmp_path / "fields")
    np.savez(tmp_path / "fields" / "a-goal-0.npz", sdf=np.zeros((1, 1, 1)))
    with pytest.raises(ValueError):
        load_field_cache(cached.path, manifest, device="cpu")
