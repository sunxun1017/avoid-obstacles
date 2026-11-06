"""Optional native GPU scan check; collection itself must remain CPU-safe."""

import numpy as np
import pytest
import torch


def scan_points():
    """Read the native rough-terrain scan and close its environment."""
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg("Mjlab-Velocity-Rough-Unitree-G1")
    cfg.scene.num_envs = 1
    native = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        native.reset()
        data = native.scene["terrain_scan"].data
        valid = data.distances[0] >= 0
        return data.hit_pos_w[0][valid].detach().cpu().numpy()
    finally:
        native.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native scan needs CUDA.")
def test_native_scan_points_are_finite_xyz():
    points = scan_points()
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) > 0 and np.isfinite(points).all()


if __name__ == "__main__":
    points = scan_points()
    print("点云形状:", points.shape)
    print("前几个点:", points[:5])
    np.save("terrain_scan.npy", points)
