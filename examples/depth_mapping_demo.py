'''
Author: sunxun sx2728977548@163.com
Date: 2026-10-08 16:44:01
LastEditors: sunxun sx2728977548.com
LastEditTime: 2026-10-08 20:17:21
FilePath: /embodied-learning/examples/depth_mapping_demo.py
Description: A simple demo of depth mapping and body observation sampling.
'''
"""Fuse synthetic depth frames, build a field, and query body observations."""

import argparse

import torch

from embodied_learning.perception.occupancy import OccupancyMapper
from embodied_learning.perception.rays import rays_from_depth
from embodied_learning.perception.voxel_field import build_voxel_field


def main() -> None:
    """Use a planar depth scene without importing a simulator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    mapper = OccupancyMapper(
        1,
        (21, 21, 31),
        voxel_size=0.1,
        origin=(-1.0, -1.0, 0.0),
        device=args.device,
    )
    # 人为构建一个深度图
    depth = torch.full((1, 21, 21), 2.0, device=args.device)
    # 人为构建内参
    intrinsics = depth.new_tensor(          # 这个是为了让新矩阵自动使用与depth相同的设备和数据类型
        [[10.0, 0.0, 10.5], [0.0, 10.0, 10.5], [0.0, 0.0, 1.0]]
    )
    rotation = torch.eye(3, device=args.device)[None]       # 旋转矩阵，这是为了描述相机光学坐标系相对于世界坐标系的朝向
    # Two views of the same plane z=2, with known optical-to-world poses.
    for camera_x in (0.0, 0.1):         # 为了模拟相机在两个位置采集深度图
        position = depth.new_tensor([[camera_x, 0.0, 0.0]])
        mapper.update(      # 用这些射线占据地图
            rays_from_depth(    # 把深度图换成世界坐标系中的射线，只有不同相机采集的点都落在同一个世界坐标系中，才能融合进同一张地图
                depth,          # 两次都用同一张深度图，但是对应的世界坐标系不同，可以理解为墙很大，两次拍到了同一面墙，x方向平移一下，那堵墙还是那个墙
                intrinsics,
                rotation=rotation,
                position=position,
                max_depth=3.0,
            )
        )
    state = mapper.snapshot()
    assert state.occupied.any()
    assert not state.observed[0, :, :, 21:].any(), "Behind the plane is unknown."
    # Field construction is explicit and CPU-based; mapping itself stays on device.
    field = build_voxel_field(
        state.occupied[0].cpu().numpy(),
        known_mask=state.known_mask[0].cpu().numpy(),
        origin=tuple(state.origin[0].cpu().tolist()),
        voxel_size=state.voxel_size,
        goal=(0.0, 0.0, 1.0),
    ).to(args.device)
    body_positions = depth.new_tensor([[[0.0, 0.0, 0.8], [0.1, 0.1, 0.9]]])
    sample = field.sample(body_positions)
    assert sample.guidance_valid.all()
    print("Observed cells:", state.observed.sum().item())
    print("Occupied cells:", state.occupied.sum().item())
    print("Known cells:", state.known_mask.sum().item())
    print("Body guidance:", sample.guidance.cpu().tolist())
    print("Observation shape:", tuple(sample.guidance.flatten(1).shape))


if __name__ == "__main__":
    main()
