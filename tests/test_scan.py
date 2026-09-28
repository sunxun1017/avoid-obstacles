import numpy as np
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

task_id = "Mjlab-Velocity-Rough-Unitree-G1"
cfg = load_env_cfg(task_id)
cfg.scene.num_envs = 1

native = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")

try:
    native.reset()

    scan = native.scene["terrain_scan"]
    data = scan.data

    valid = data.distances[0] >= 0
    points = data.hit_pos_w[0][valid].detach().cpu().numpy()

    print("点云形状:", points.shape)
    print("前几个点:", points[:5])
    np.save("terrain_scan.npy", points)
finally:
    native.close()