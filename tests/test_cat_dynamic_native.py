"""Native checks use actual mocap motion and prohibited physical contacts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Native G1 requires CUDA')


def test_joint_native_substep_motion_contact_and_terminal_timeout(tmp_path):
    from embodied_learning.data.config import load_experiment_config
    from embodied_learning.data.field_scenes import load_scene_manifest
    from embodied_learning.perception.online_field import default_field_sample
    from embodied_learning.simulators.mjlab.cat_traversal import make_cat_env
    from embodied_learning.tasks.cat_curriculum import stage_settings
    cfg = stage_settings(load_experiment_config(Path('configs/experiments/g1_cat_source.yaml')),
                         profile='cat_obstacle_expert_v2')[-1]
    path = tmp_path/'scenes.json'
    path.write_text(json.dumps(cfg))
    manifest = load_scene_manifest(path)
    bank = SimpleNamespace(cache_identity='native-joint', source_profile={'version':'cat_source_v1'},
                           sample_source=lambda p,*args:default_field_sample(p))
    config = dict(stage=1, gait_scale=1., scenarios={'anchor_flat':[
        dict(position=[-1.5, 1., .9], velocity=[0., -.3, 0.], radius=.08, trigger=0)]})
    env = make_cat_env(manifest, bank, scene_ids=['anchor_flat'], num_envs=4,
        device='cuda:0', seed=42,
        options={**cfg['task'], 'noise_level':0., 'random_pd':False, 'random_rfi':False, 'push':False},
        joint_config=config)
    try:
        obs = env.get_observations()
        assert obs['actor'].shape == (4,191) and obs['critic'].shape == (4,279)
        runtime = env.unwrapped.cat_dynamic
        before = runtime.position.clone()
        obs, _, _, _ = env.step(torch.zeros(4,12,device='cuda:0'))
        assert runtime.substeps == 10
        assert (runtime.position[:,1] < before[:,1]).all()
        # Place the real active collision sphere in the pelvis in one world.
        robot = env.unwrapped.scene['robot']
        runtime.position[0] = robot.data.root_link_pos_w[0]-env.unwrapped.scene.env_origins[0]
        runtime.velocity[0] = 0
        runtime.target_velocity[0] = 0
        env.unwrapped.episode_length_buf[1] = 999
        obs, reward, done, extras = env.step(torch.zeros(4,12,device='cuda:0'))
        assert done[0] and extras['field_outcomes']['collision'][0]
        assert reward[0] <= -25
        assert done[1] and extras['field_outcomes']['timeout'][1]
        assert not extras['time_outs'][1]  # budget deadline must reach GAE as terminal
        assert reward[1] <= -10
        assert torch.isfinite(obs['actor']).all()
    finally:
        env.close()
