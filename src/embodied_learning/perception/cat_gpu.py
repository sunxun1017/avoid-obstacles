"""GPU CAT field backend; source CPU FMM remains the numerical reference."""

import math

import torch

from embodied_learning.perception.eikonal_warp import CudaEikonal
from embodied_learning.perception.voxel_field import VoxelPotentialField


class CudaCatFieldBuilder:
    def __init__(
        self, shape: tuple[int, int, int], device: str | torch.device
    ) -> None:
        self.shape = tuple(shape)
        self.solver = CudaEikonal(shape, device)
        self.coordinates = torch.stack(
            torch.meshgrid(
                *(
                    torch.arange(n, device=device, dtype=torch.float32)
                    for n in shape
                ),
                indexing="ij",
            ),
            -1,
        )
        self.last_iterations = ()

    @torch.no_grad()
    def build(
        self,
        occupied: torch.Tensor,
        *,
        origin: tuple[float, float, float],
        voxel_size: float,
        goal: tuple[float, float, float],
    ) -> VoxelPotentialField:
        if (
            occupied.shape != self.shape
            or occupied.dtype != torch.bool
            or occupied.device != self.coordinates.device
            or len(origin) != 3
            or len(goal) != 3
            or not all(math.isfinite(v) for v in (*origin, *goal, voxel_size))
            or voxel_size <= 0
        ):
            raise ValueError("Invalid CUDA CAT field layout or parameters.")
        index = torch.round(
            (
                occupied.new_tensor(goal, dtype=torch.float32)
                - occupied.new_tensor(origin, dtype=torch.float32)
            )
            / voxel_size
        ).long()
        if (index < 0).any() or (index >= index.new_tensor(self.shape)).any():
            raise ValueError("goal is outside the online grid.")
        if occupied[tuple(index)]:
            raise ValueError("goal is occupied in the online grid.")
        blocked = occupied.clone()
        if not blocked.any():
            blocked[0, 0, -1] = blocked[0, -1, -1] = True
        delta = (
            self.coordinates * voxel_size
            + self.coordinates.new_tensor(origin)
            - self.coordinates.new_tensor(goal)
        )
        seed = delta.norm(dim=-1) <= 0.12
        phi = torch.where(seed, -1.0, 1.0)
        geodesic = self.solver.solve(phi, blocked, spacing=voxel_size)
        iterations1 = self.solver.iterations
        valid = torch.isfinite(geodesic)
        maximum = torch.where(valid, geodesic, 0.0).max()
        t = torch.where(valid, geodesic, maximum)
        gradient = torch.stack(
            torch.gradient(t, spacing=voxel_size, edge_order=2), -1
        )
        sdf = self.solver.solve(
            torch.where(blocked, -1.0, 1.0),
            torch.zeros_like(blocked),
            spacing=voxel_size,
        )
        self.last_iterations = (iterations1, self.solver.iterations)
        bf = torch.stack(torch.gradient(sdf, spacing=voxel_size, edge_order=2), -1)
        norm = bf.norm(dim=-1, keepdim=True)
        bunit = torch.where(norm > 1.0e-9, bf / norm.clamp_min(1.0e-9), 0.0)
        g = -gradient
        perp = g - (g * bunit).sum(-1, keepdim=True) * bunit
        frac = (sdf.clamp_min(0) / (5 * voxel_size + 1.0e-9)).clamp(0, 1)
        weight = (1 - frac * frac * (3 - 2 * frac))[..., None]
        mixed = torch.where(
            blocked[..., None], bunit, (1 - weight) * g + weight * perp
        )
        norm = mixed.norm(dim=-1, keepdim=True)
        unit = torch.where(norm > 1.0e-9, mixed / norm.clamp_min(1.0e-9), 0.0)
        xy = delta[..., :2].norm(dim=-1)
        speed = torch.where(xy > 0.3, 0.6, 0.6 * (xy / 0.3) ** 3)
        return VoxelPotentialField(
            sdf=sdf,
            geodesic=t,
            sdf_gradient=bf,
            geodesic_gradient=gradient,
            guidance=unit * speed[..., None],
            known=torch.ones_like(blocked),
            reachable=valid & ~blocked,
            occupied=blocked,
            origin=tuple(origin),
            voxel_size=voxel_size,
        )
