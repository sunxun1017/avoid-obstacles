"""Short native integration checks; no learned-policy quality assertions."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodied_learning.data.config import load_experiment_config
from embodied_learning.data.field_scenes import load_scene_manifest
from embodied_learning.perception.online_field import default_field_sample
from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env
from embodied_learning.tasks.cat_curriculum import stage_settings

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native G1 requires CUDA"
)


def test_v2_native_contacts_reset_and_mixed_scene_selection(tmp_path):
    cfg = stage_settings(
        load_experiment_config(Path("configs/experiments/g1_cat_source.yaml")),
        profile="cat_obstacle_expert_v2",
    )[-1]
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(cfg))
    manifest = load_scene_manifest(path)
    bank = SimpleNamespace(
        cache_identity="native-test",
        source_profile={"version": "cat_source_v1"},
        sample_source=lambda positions, *args: default_field_sample(positions),
    )
    options = {
        **cfg["task"],
        "noise_level": 0.0,
        "random_pd": False,
        "random_rfi": False,
        "push": False,
    }
    env = make_cat_env(
        manifest,
        bank,
        scene_ids=["hurdle_crouch"] * 5
        + ["anchor_flat", "anchor_hurdle", "anchor_ceiling"],
        num_envs=32,
        device="cuda:0",
        seed=42,
        options=options,
    )
    try:
        assert set(env.scene_ids.tolist()) == {2, 3, 4, 5}
        obs = env.get_observations()
        assert obs["actor"].shape == (32, 162)
        assert obs["critic"].shape == (32, 250)
        native = env.unwrapped
        env.cat_state.allowed = (2,)
        env.reset_slots(torch.arange(32, device="cuda:0"))
        assert (env.scene_ids == 2).all()
        robot = native.scene["robot"]
        root = robot.data.default_root_state.clone()
        root[:, :3] = native.scene.env_origins + root.new_tensor([0.52, 0.0, 0.8])
        robot.write_root_state_to_sim(root)
        native.sim.forward()
        native.sim.sense()
        from embodied_learning.evaluation.cat_substep_contacts import (
            capture_prohibited_contacts,
        )

        with capture_prohibited_contacts(env):
            obs, reward, done, extras = env.step(
                torch.zeros(32, 12, device="cuda:0")
            )
        assert extras["contact_geom_pairs"]
        assert all(
            any("foot" in pair[0] for pair in pairs)
            for pairs in extras["contact_geom_pairs"].values()
        )
        hit = extras["field_outcomes"]["prohibited_contact"]
        assert hit.any(), (
            "Native foot/bar intersection must be detected on the first step"
        )
        assert done[hit].all()
        assert (reward[hit] == 0).all()
        assert extras["contact_sensor_hits"][hit].any(1).all()
        assert torch.isfinite(obs["actor"]).all()
        _, _, _, extras = env.step(torch.zeros(32, 12, device="cuda:0"))
        assert not extras["field_outcomes"]["prohibited_contact"].any(), (
            "Old collision must not leak after auto-reset"
        )
    finally:
        env.close()
