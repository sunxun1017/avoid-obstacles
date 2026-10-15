"""CUDA ray-parallel DDA with the reference mapper's per-scan semantics."""

import torch
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def flat_cell(cell: wp.vec3i, nx: int, ny: int, nz: int):
    x = wp.clamp(cell[0], 0, nx - 1)
    y = wp.clamp(cell[1], 0, ny - 1)
    z = wp.clamp(cell[2], 0, nz - 1)
    return (x * ny + y) * nz + z


@wp.kernel
def traverse(
    origins: wp.array2d(dtype=wp.vec3),
    ends: wp.array2d(dtype=wp.vec3),
    valid: wp.array2d(dtype=wp.bool),
    hit: wp.array2d(dtype=wp.bool),
    map_origins: wp.array(dtype=wp.vec3),
    spacing: float,
    nx: int,
    ny: int,
    nz: int,
    flags: wp.array2d(dtype=wp.int32),
):
    b, ray = wp.tid()
    if not valid[b, ray]:
        return
    lower = map_origins[b] - wp.vec3(spacing * 0.5)
    o = (origins[b, ray] - lower) / spacing
    e = (ends[b, ray] - lower) / spacing
    d = e - o
    size = wp.vec3(float(nx), float(ny), float(nz))
    for axis in range(3):
        if (
            not wp.isfinite(o[axis])
            or not wp.isfinite(e[axis])
            or not wp.isfinite(d[axis])
        ):
            return
    if d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0:
        return
    entry = 0.0
    end_time = 1.0
    for axis in range(3):
        if d[axis] != 0.0:
            a = -o[axis] / d[axis]
            c = (size[axis] - o[axis]) / d[axis]
            entry = wp.max(entry, wp.min(a, c))
            end_time = wp.min(end_time, wp.max(a, c))
        elif o[axis] < 0.0 or o[axis] >= size[axis]:
            return
    tolerance = 9.5367431640625e-7  # 8 * float32 eps in ray parameter space.
    hit_valid = hit[b, ray]
    hit_cell = wp.vec3i()
    for axis in range(3):
        hit_valid = hit_valid and e[axis] >= 0.0 and e[axis] < size[axis]
        hit_cell[axis] = int(wp.floor(e[axis]))
    if hit_valid:
        wp.atomic_or(flags, b, flat_cell(hit_cell, nx, ny, nz), 2)
    if end_time - entry <= tolerance:
        return
    start = o + entry * d
    cell = wp.vec3i()
    direction = wp.vec3i()
    crossing = wp.vec3(1.0e30)
    for axis in range(3):
        floor = wp.floor(start[axis])
        if d[axis] < 0.0 and start[axis] == floor:
            floor -= 1.0
        cell[axis] = int(wp.clamp(floor, 0.0, size[axis] - 1.0))
        if d[axis] != 0.0:
            direction[axis] = int(wp.sign(d[axis]))
            boundary = float(cell[axis])
            if direction[axis] > 0:
                boundary += 1.0
            crossing[axis] = (boundary - o[axis]) / d[axis]
    current = entry
    for _ in range(nx + ny + nz + 1):
        next_time = wp.min(crossing[0], wp.min(crossing[1], crossing[2]))
        at_hit = (
            hit_valid
            and cell[0] == hit_cell[0]
            and cell[1] == hit_cell[1]
            and cell[2] == hit_cell[2]
        )
        if wp.min(next_time, end_time) - current > tolerance and not at_hit:
            wp.atomic_or(flags, b, flat_cell(cell, nx, ny, nz), 1)
        for axis in range(3):
            if d[axis] != 0.0 and crossing[axis] <= next_time + tolerance:
                cell[axis] += direction[axis]
            if d[axis] != 0.0:
                boundary = float(cell[axis])
                if direction[axis] > 0:
                    boundary += 1.0
                crossing[axis] = (boundary - o[axis]) / d[axis]
        current = wp.max(current, next_time)
        inside = True
        for axis in range(3):
            inside = inside and cell[axis] >= 0 and float(cell[axis]) < size[axis]
        if next_time >= end_time - tolerance or not inside:
            break


def update_warp(mapper, rays):
    wp.init()
    flags = torch.zeros(
        (mapper.num_envs, *mapper.shape), dtype=torch.int32, device=mapper.device
    )
    with wp.ScopedStream(
        wp.stream_from_torch(torch.cuda.current_stream(mapper.device))
    ):
        wp.launch(
            traverse,
            dim=rays.valid_mask.shape,
            inputs=[
                wp.from_torch(rays.origins.contiguous(), dtype=wp.vec3),
                wp.from_torch(rays.endpoints.contiguous(), dtype=wp.vec3),
                wp.from_torch(rays.valid_mask.contiguous()),
                wp.from_torch(rays.hit_mask.contiguous()),
                wp.from_torch(mapper._origin, dtype=wp.vec3),
                mapper.voxel_size,
                *mapper.shape,
                wp.from_torch(flags.flatten(1)),
            ],
        )
    hit = (flags & 2) != 0
    free = ((flags & 1) != 0) & ~hit
    mapper._observed |= hit | free
    mapper._log_odds += hit.to(mapper.dtype) * mapper.hit_log_odds
    mapper._log_odds += free.to(mapper.dtype) * mapper.miss_log_odds
    mapper._log_odds.clamp_(mapper.min_log_odds, mapper.max_log_odds)
    return (flags != 0).flatten(1).any(-1)
