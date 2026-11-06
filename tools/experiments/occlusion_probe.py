"""EXPERIMENTAL/THROWAWAY finite-grid occlusion probe, not a CMPC reproduction.

The world is private to the sensor simulator and evaluator. Mapping/planning
consume Beam evidence only. This is a planar unicycle with a fine 3-D geometric
observation check, not a humanoid controller, learned ADAPT policy, or ADMM MPC.
Four-neighbor reachability is exact only for the declared grid-walking model.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class Grid:
    """Uniform cubic cells; origin is the lower corner, unlike production maps."""

    shape: tuple[int, int, int]
    cell_size: float
    origin: tuple[float, float, float]

    def index(self, point):
        return np.floor((np.asarray(point) - self.origin) / self.cell_size).astype(
            int
        )

    def center(self, index):
        return np.asarray(self.origin) + (np.asarray(index) + 0.5) * self.cell_size


@dataclass(frozen=True)
class Beam:
    origin: tuple[float, float, float]
    endpoint: tuple[float, float, float]
    hit: bool
    valid: bool
    stamp: float


def _walk(grid, origin, endpoint):
    """Exact cell traversal for a zero-width line; no clearing between rays."""
    start, end = np.asarray(origin, float), np.asarray(endpoint, float)
    delta = end - start
    cell = grid.index(start)
    step = np.sign(delta).astype(int)
    enter = 0.0
    t_delta = np.full(3, np.inf)
    t_max = np.full(3, np.inf)
    moving = delta != 0
    t_delta[moving] = grid.cell_size / np.abs(delta[moving])
    boundary = np.asarray(grid.origin) + (cell + (step > 0)) * grid.cell_size
    t_max[moving] = (boundary[moving] - start[moving]) / delta[moving]
    while (cell >= 0).all() and (cell < grid.shape).all() and enter <= 1 + 1e-12:
        yield tuple(cell), enter
        following = float(t_max.min())
        if not math.isfinite(following) or following > 1 + 1e-12:
            break
        axes = np.abs(t_max - following) < 1e-12
        cell[axes] += step[axes]
        t_max[axes] += t_delta[axes]
        enter = following


class VoxelWorld:
    """Ideal noiseless sensor fixture. Never pass this object to a planner."""

    def __init__(self, grid: Grid, occupied: np.ndarray):
        self.grid = grid
        self.occupied = occupied.copy()
        assert self.occupied.shape == grid.shape

    def ray(self, origin, endpoint, stamp):
        start, end = np.asarray(origin, float), np.asarray(endpoint, float)
        for cell, fraction in _walk(self.grid, start, end):
            if self.occupied[cell]:
                # A tiny displacement into the hit cell preserves endpoint class.
                fraction = min(1.0, fraction + 1e-9)
                end = start + fraction * (end - start)
                return Beam(tuple(start), tuple(end), True, True, float(stamp))
        return Beam(tuple(start), tuple(end), False, True, float(stamp))

    def far_scan(self, xy, count, heights, radius, stamp):
        return [
            self.ray(
                (*xy, z),
                (xy[0] + radius * math.cos(a), xy[1] + radius * math.sin(a), z),
                stamp,
            )
            for z in heights
            for a in np.linspace(-math.pi, math.pi, count, endpoint=False)
        ]

    def near_scan(self, xy, radius, stamp):
        """Dense center-targeted virtual depth rays at fixed voxel resolution.

        Targets depend only on pose/grid/radius, not scene occupancy. All targets
        are ray-cast, so occluded cells remain unknown. This idealized scanner
        costs more rays as radius grows; the report includes that cost.
        """
        sensor_z = self.grid.origin[2] + self.grid.cell_size * (
            min(self.grid.shape[2] - 1, 4) + 0.5
        )
        sensor = (*xy, sensor_z)
        rays = []
        for i, j in np.ndindex(self.grid.shape[:2]):
            center = self.grid.center((i, j, 0))
            if np.linalg.norm(center[:2] - xy) > radius:
                continue
            for k in range(self.grid.shape[2]):
                target = self.grid.center((i, j, k))
                rays.append(self.ray(sensor, target, stamp))
        return rays


class Evidence:
    """Latest per-voxel evidence; occupied wins over free within a scan."""

    def __init__(self, grid):
        self.grid = grid
        self.observed = np.zeros(grid.shape, bool)
        self.occupied = np.zeros(grid.shape, bool)
        self.stamps = np.full(grid.shape, -np.inf)

    def update(self, beams):
        free, hits = {}, {}
        for beam in beams:
            if (
                not beam.valid
                or not np.isfinite((*beam.origin, *beam.endpoint, beam.stamp)).all()
            ):
                continue
            traversed = list(_walk(self.grid, beam.origin, beam.endpoint))
            if not traversed:
                continue
            endpoint_cell = tuple(self.grid.index(beam.endpoint))
            for cell, _ in traversed:
                is_hit = beam.hit and cell == endpoint_cell
                target = hits if is_hit else free
                target[cell] = max(target.get(cell, -np.inf), beam.stamp)
        for cells, occupied in ((free, False), (hits, True)):
            for cell, stamp in cells.items():
                if stamp >= self.stamps[cell]:
                    self.occupied[cell] = occupied
                    self.observed[cell] = True
                    self.stamps[cell] = stamp

    def age(self, now):
        return now - self.stamps

    def free(self, now, max_age):
        return self.observed & ~self.occupied & (self.age(now) <= max_age)

    def planar(self, now, max_age=0.6):
        # Body is the whole vertical span except first/last guard layers.
        layers = slice(1, -1) if self.grid.shape[2] > 2 else slice(None)
        walls = self.occupied[:, :, layers].any(-1)
        free = self.free(now, max_age)[:, :, layers].all(-1) & ~walls
        age = self.age(now)[:, :, layers].max(-1)
        return free, walls, age


def near_radius(
    speed, *, latency=0.2, braking=0.8, body_radius=0.2, margin=0.2, minimum=2.0
):
    """Stopping-distance envelope; cell size never changes with this radius."""
    if braking <= 0 or latency < 0:
        raise ValueError("Positive braking and nonnegative latency are required.")
    v = abs(speed)
    return max(minimum, body_radius + margin + v * latency + v * v / (2 * braking))


@dataclass
class FarProjection:
    edges: np.ndarray
    ranges: np.ndarray
    occlusion_start: np.ndarray
    unknown: np.ndarray
    stamps: np.ndarray
    beams: list[Beam]

    def age(self, now):
        return now - self.stamps


def _angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def compress_far(beams, bins, *, adaptive, heading=0.0, speed=0.0, yaw_rate=0.0):
    """Compress a FIXED raw scan, retaining <= one actual beam per height/bin.

    All methods acquire identical raw rays. This tests representation allocation,
    not active sensor acquisition savings. Bin validity never clears its wedge.
    A hit creates an occlusion start only on its retained measured ray.
    """
    heights = sorted({float(b.origin[2]) for b in beams})
    angles = np.array(
        [
            math.atan2(b.endpoint[1] - b.origin[1], b.endpoint[0] - b.origin[0])
            for b in beams
        ]
    )
    lengths = np.array([math.dist(b.origin, b.endpoint) for b in beams])
    valid = np.array([b.valid for b in beams])
    hits = np.array([b.hit and b.valid for b in beams])
    edges = np.linspace(-math.pi, math.pi, bins + 1)
    if adaptive and len(beams):
        dense = np.linspace(-math.pi, math.pi, 4097)
        weights = np.ones_like(dense)
        weights += (
            2.5
            * min(abs(speed), 1.5)
            * np.exp(-((_angle_delta(dense, heading) / 0.55) ** 2))
        )
        turn = heading + yaw_rate * 0.8
        weights += (
            2
            * min(abs(yaw_rate), 1)
            * np.exp(-((_angle_delta(dense, turn) / 0.45) ** 2))
        )
        edge_angles = []
        for height in heights:
            ids = np.array([i for i, b in enumerate(beams) if b.origin[2] == height])
            ids = ids[np.argsort(angles[ids])]
            for left, right in zip(ids, np.roll(ids, -1), strict=True):
                if (
                    valid[left]
                    and valid[right]
                    and (
                        hits[left] != hits[right]
                        or (
                            hits[left] and abs(lengths[left] - lengths[right]) > 0.75
                        )
                    )
                ):
                    edge_angles.append(angles[left])
        for angle in sorted(set(np.round(edge_angles, 5))):
            weights += 2 * np.exp(-((_angle_delta(dense, angle) / 0.12) ** 2))
        cdf = np.concatenate(([0.0], np.cumsum((weights[:-1] + weights[1:]) / 2)))
        edges = np.interp(np.linspace(0, cdf[-1], bins + 1), cdf, dense)
        edges[[0, -1]] = (-math.pi, math.pi)
    ranges = np.full((len(heights), bins), np.nan)
    occlusion = np.full_like(ranges, np.nan)
    stamps = np.full_like(ranges, -np.inf)
    retained = []
    assignments = np.clip(
        np.searchsorted(edges, angles, side="right") - 1, 0, bins - 1
    )
    for layer, height in enumerate(heights):
        height_mask = np.array([b.origin[2] == height for b in beams])
        for column in range(bins):
            ids = np.flatnonzero((assignments == column) & height_mask & valid)
            if not len(ids):
                continue
            surface = ids[hits[ids]]
            if len(surface):
                selected = surface[np.argmin(lengths[surface])]
                occlusion[layer, column] = lengths[selected]
            else:
                center = (edges[column] + edges[column + 1]) / 2
                selected = ids[np.argmin(np.abs(_angle_delta(angles[ids], center)))]
            ranges[layer, column] = lengths[selected]
            stamps[layer, column] = beams[selected].stamp
            retained.append(beams[selected])
    return FarProjection(
        edges, ranges, occlusion, np.isnan(ranges), stamps, retained
    )


def frontier_sources(free, walls, ages):
    """Every unknown/free boundary is eligible, including beam and FoV gaps."""
    sources = np.zeros_like(free)
    source_ages = np.zeros_like(ages)
    for i, j in np.argwhere(free):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = i + di, j + dj
            if (
                0 <= x < free.shape[0]
                and 0 <= y < free.shape[1]
                and not free[x, y]
                and not walls[x, y]
            ):
                sources[x, y] = True
                source_ages[x, y] = max(source_ages[x, y], ages[i, j])
    return sources, source_ages


def reachable_risk(
    walls, sources, *, speed, times, cell_size, age=0.0, inflation=0.0
):
    """Multi-source grid Dijkstra, with age credited before future expansion.

    Only observed walls constrain CENTER propagation. Unknown cells are passable
    for a hypothetical hidden agent. Four-neighbor grid motion is an explicit
    model restriction, not a continuous-space conservative reachability guarantee.
    Optional Euclidean inflation creates a conservative body-overlap/collision
    mask; it can cover the far side of a thin wall. That is extra conservatism,
    not evidence that the hidden center crossed the wall.
    """
    if speed < 0 or cell_size <= 0 or np.any(np.asarray(age) < 0):
        raise ValueError("Speed/age must be nonnegative and cell size positive.")
    ages = np.broadcast_to(age, walls.shape)
    distance = np.full(walls.shape, np.inf)
    queue = []
    for x, y in np.argwhere(sources & ~walls):
        distance[x, y] = -speed * ages[x, y]
        heapq.heappush(queue, (distance[x, y], int(x), int(y)))
    while queue:
        d, x, y = heapq.heappop(queue)
        if d > distance[x, y] + 1e-12:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if (
                0 <= nx < walls.shape[0]
                and 0 <= ny < walls.shape[1]
                and not walls[nx, ny]
            ):
                following = d + cell_size
                if following < distance[nx, ny] - 1e-12:
                    distance[nx, ny] = following
                    heapq.heappush(queue, (following, nx, ny))
    result = []
    for horizon in times:
        mask = (distance <= speed * horizon + 1e-10) & ~walls
        if inflation > 0 and mask.any():
            mask = distance_transform_edt(~mask) * cell_size <= inflation
        result.append(mask)
    return np.asarray(result)


@dataclass(frozen=True)
class PlanConfig:
    dt: float = 0.2
    steps: int = 10
    prefix_steps: int = 2
    max_speed: float = 0.8
    acceleration: float = 0.8
    braking: float = 0.8
    robot_radius: float = 0.18
    hidden_radius: float = 0.15
    max_age: float = 0.6
    latency: float = 0.1
    substeps: int = 4

    @property
    def brake_steps(self):
        return math.ceil(self.max_speed / (self.braking * self.dt))

    @property
    def total_steps(self):
        return self.steps + self.brake_steps


@dataclass
class PlanResult:
    feasible: bool
    action: tuple[float, float] | None
    prefix: np.ndarray
    branches: np.ndarray
    controls: np.ndarray
    reason: str
    candidates_checked: int


def _candidate_library(state, config):
    actions = np.array(
        [
            (v, w)
            for v in (0, config.max_speed / 2, config.max_speed)
            for w in (-0.8, 0, 0.8)
        ]
    )
    ids = np.array(list(np.ndindex(9, 9, 9)))
    controls = np.zeros((len(ids), config.total_steps, 2))
    middle = config.prefix_steps + (config.steps - config.prefix_steps) // 2
    controls[:, : config.prefix_steps] = actions[ids[:, 0], None]
    controls[:, config.prefix_steps : middle] = actions[ids[:, 1], None]
    controls[:, middle : config.steps] = actions[ids[:, 2], None]
    states = np.broadcast_to(np.asarray(state, float), (len(ids), 4)).copy()
    trajectory = [states.copy()]
    h = config.dt / config.substeps
    for k in range(config.total_steps):
        for _ in range(config.substeps):
            requested = controls[:, k]
            old_speed = states[:, 3].copy()
            states[:, 3] += np.clip(
                requested[:, 0] - states[:, 3],
                -config.braking * h,
                config.acceleration * h,
            )
            velocity = (states[:, 3] + old_speed) / 2
            angle = states[:, 2] + requested[:, 1] * h / 2
            states[:, 0] += velocity * np.cos(angle) * h
            states[:, 1] += velocity * np.sin(angle) * h
            states[:, 2] += requested[:, 1] * h
            trajectory.append(states.copy())
    return ids[:, 0], controls, np.stack(trajectory, axis=1)


def plan_candidates(grid, free, risks, state, goal, config=None):
    """Enumerate 9 shared prefixes x 81 suffixes per risk branch.

    Only common-prefix actions may be executed, and only the first is returned.
    All branches include an explicitly checked stopping tail. Failure is a
    diagnostic with action=None: it is not silently relabeled a safe stop.
    """
    config = config or PlanConfig()
    prefix_ids, controls, trajectories = _candidate_library(state, config)
    indices = np.floor(
        (trajectories[:, :, :2] - np.asarray(grid.origin[:2])) / grid.cell_size
    ).astype(int)
    valid = ((indices >= 0) & (indices < np.asarray(free.shape))).all(-1)
    clipped = np.minimum(np.maximum(indices, 0), np.asarray(free.shape) - 1)
    # Every footprint must fit entirely in measured 3-D free cells.
    clearance = distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * grid.cell_size
    clearance -= grid.cell_size * math.sqrt(2) / 2
    centers = np.asarray(grid.origin[:2]) + (clipped + 0.5) * grid.cell_size
    point_offset = np.linalg.norm(trajectories[:, :, :2] - centers, axis=-1)
    # EDT is sampled at cell centers. Account for arbitrary subcell query
    # positions and the maximum travel between adjacent collision samples.
    travel_margin = (
        max(config.max_speed, abs(state[3])) * config.dt / (2 * config.substeps)
    )
    query_clearance = clearance[clipped[:, :, 0], clipped[:, :, 1]] - point_offset
    valid &= query_clearance >= config.robot_radius + travel_margin
    geometry_ok = valid.all(-1) & (np.abs(trajectories[:, -1, 3]) <= 1e-7)
    stages = np.minimum(
        np.ceil(np.arange(trajectories.shape[1]) / config.substeps).astype(int),
        config.total_steps,
    )
    if risks.shape[1:] != (config.total_steps + 1, *free.shape):
        raise ValueError("Risk tensor must cover every planned and braking stage.")
    cost = 5 * np.linalg.norm(
        trajectories[:, config.steps * config.substeps, :2] - goal, axis=-1
    )
    sampled = trajectories[:, :: config.substeps]
    cost += 0.2 * ((sampled[:, : config.steps, 3] - config.max_speed) ** 2).sum(-1)
    cost += 0.05 * (controls[:, :, 1] ** 2).sum(-1)
    cost += (
        0.05
        * (np.diff(sampled[:, :, 3], axis=1) / config.dt) ** 2
        @ np.ones(config.total_steps)
    )
    branch_valid = []
    for risk in risks:
        threatened = risk[stages[None], clipped[:, :, 0], clipped[:, :, 1]].any(-1)
        branch_valid.append(geometry_ok & ~threatened)
    best = None
    for prefix in range(9):
        selected, score = [], 0.0
        for feasible in branch_valid:
            ids = np.flatnonzero(feasible & (prefix_ids == prefix))
            if not len(ids):
                break
            choice = ids[np.argmin(cost[ids])]
            selected.append(choice)
            score += cost[choice]
        if len(selected) == len(risks) and (best is None or score < best[0]):
            best = (score, selected)
    if best is None:
        return PlanResult(
            False,
            None,
            np.empty((0, 4)),
            np.empty((0, 0, 4)),
            np.empty((0, 0, 2)),
            "No candidate has a measured-free, risk-feasible branch and "
            "braking tail for every configuration.",
            len(controls),
        )
    selected = best[1]
    branches = sampled[selected]
    chosen_controls = controls[selected]
    return PlanResult(
        True,
        tuple(chosen_controls[0, 0]),
        branches[0, : config.prefix_steps + 1].copy(),
        branches,
        chosen_controls,
        "",
        len(controls),
    )


def risk_from_evidence(memory, now, config, speeds):
    free, walls, age = memory.planar(now, config.max_age)
    sources, source_age = frontier_sources(free, walls, age)
    times = np.arange(config.total_steps + 1) * config.dt + config.latency
    inflation = (
        config.hidden_radius
        + config.robot_radius
        + memory.grid.cell_size * math.sqrt(2)
    )
    risk = np.array(
        [
            reachable_risk(
                walls,
                sources,
                speed=s,
                times=times,
                cell_size=memory.grid.cell_size,
                age=source_age,
                inflation=inflation,
            )
            for s in speeds
        ]
    )
    return free, risk


def plan_from_evidence(
    memory, state, goal, *, now, speeds=(0.0, 0.25, 0.5), config=None
):
    config = config or PlanConfig()
    free, risks = risk_from_evidence(memory, now, config, speeds)
    return plan_candidates(memory.grid, free, risks, state, goal, config)


def published_risk_disks(
    obstacle_centers,
    obstacle_radii,
    robot_xy,
    robot_speed,
    hidden_speed,
    *,
    spacing=0.5,
    per_side=2,
    sigma=1e-4,
):
    """Equations 7/8 of arXiv:2503.04563v5, for supplied visible disk detections.

    This formula reference is not a reproduction of the paper's optimization,
    visible-object detector, or experiments. In particular no cone rooted at
    the robot is substituted for the paper's tangent-aligned risk disks.
    """
    centers, radii = [], []
    robot = np.asarray(robot_xy)
    for center, radius in zip(obstacle_centers, obstacle_radii, strict=True):
        relative = np.asarray(center) - robot
        distance = np.linalg.norm(relative)
        if distance <= radius:
            raise ValueError("External tangents require robot outside the disk.")
        angle = math.atan2(relative[1], relative[0])
        tangent = math.asin(radius / distance)
        for side in (-1, 1):
            direction = np.array(
                [math.cos(angle + side * tangent), math.sin(angle + side * tangent)]
            )
            for i in range(per_side):
                length = math.sqrt(distance**2 - radius**2) + i * spacing
                centers.append(robot + length * direction)
                radii.append(length / (robot_speed + sigma) * hidden_speed + radius)
    return np.asarray(centers), np.asarray(radii)


def _scene(name):
    grid = Grid((48, 40, 8), 0.25, (-4.0, -5.0, 0.0))
    solid = np.zeros(grid.shape, bool)
    solid[:, :, 0] = True  # Ground, excluded from the navigation body envelope.
    centers = np.stack(
        np.meshgrid(
            *[
                grid.origin[k] + (np.arange(grid.shape[k]) + 0.5) * grid.cell_size
                for k in range(3)
            ],
            indexing="ij",
        ),
        axis=-1,
    )

    def block(low, high):
        solid[np.all((centers >= low) & (centers < high), axis=-1)] = True

    source = None
    if name == "corner":
        block((0.0, -3.0, 0.25), (0.5, 1.25, 2.0))
        block((0.0, 1.0, 0.25), (3.0, 1.5, 2.0))
        source = (1.125, 0.125)
    elif name == "door":
        block((0.25, -4.0, 0.25), (0.75, -0.75, 2.0))
        block((0.25, 0.75, 0.25), (0.75, 4.0, 2.0))
        source = (1.125, 1.625)
    elif name == "rear_poles":
        for y in (-2.375, -0.875, 1.375, 2.875):
            block((-3.5, y, 0.25), (-3.25, y + 0.25, 2.0))
    elif name == "overhang":
        # Deliberately falls between far sampled heights: a 3-D counterexample.
        block((-0.5, -0.5, 1.5), (0.0, 0.75, 1.75))
    else:
        raise ValueError(f"Unknown scene: {name}")
    return VoxelWorld(grid, solid), (-1.875, 0.125), (3.125, 0.125), source


def _representation_metrics(raw, projection):
    heights = sorted({float(b.origin[2]) for b in raw})
    errors, edge_errors = [], []
    for height in heights:
        layer = heights.index(height)
        rays = [b for b in raw if b.origin[2] == height and b.valid]
        angles = np.array(
            [
                math.atan2(b.endpoint[1] - b.origin[1], b.endpoint[0] - b.origin[0])
                for b in rays
            ]
        )
        lengths = np.array([math.dist(b.origin, b.endpoint) for b in rays])
        columns = np.clip(
            np.searchsorted(projection.edges, angles, side="right") - 1,
            0,
            len(projection.edges) - 2,
        )
        estimates = projection.ranges[layer, columns]
        errors.extend(np.abs(estimates - lengths)[np.isfinite(estimates)].tolist())
        order = np.argsort(angles)
        for left, right in zip(order, np.roll(order, -1), strict=True):
            if rays[left].hit != rays[right].hit or (
                rays[left].hit and abs(lengths[left] - lengths[right]) > 0.75
            ):
                edge_errors.append(
                    float(
                        np.min(np.abs(_angle_delta(projection.edges, angles[left])))
                    )
                )
    return {
        "display_range_mae_m": float(np.mean(errors)) if errors else None,
        "occlusion_edge_bin_error_rad": float(np.mean(edge_errors))
        if edge_errors
        else None,
        "unknown_far_bins": int(projection.unknown.sum()),
    }


def _plan_metrics(result, state, goal, elapsed):
    return {
        "feasible": result.feasible,
        "action_target_v_omega": list(result.action) if result.action else None,
        "prefix_goal_progress_m": float(
            np.linalg.norm(np.asarray(state[:2]) - goal)
            - np.linalg.norm(result.prefix[-1, :2] - goal)
        )
        if result.feasible
        else None,
        "branch_count": len(result.branches),
        "prefix_equal": bool(
            np.allclose(result.branches[:, : len(result.prefix)], result.prefix)
        )
        if result.feasible
        else None,
        "terminal_max_speed": float(result.branches[:, -1, 3].max())
        if result.feasible
        else None,
        "reason": result.reason,
        "candidate_sequences": result.candidates_checked,
        "elapsed_ms": elapsed * 1000,
    }


def _designated_actor_coverage(
    world, source_xy, xy, free, predicted, config, age, hidden_speed
):
    """Ground truth is permitted here only AFTER planning, for evaluation."""
    if source_xy is None:
        return {"designated_hidden_actor": False, "risk_recall_on_free_cells": None}
    line = world.ray((*xy, 0.875), (*source_xy, 0.875), 0)
    if not line.hit:
        return {"designated_hidden_actor": False, "risk_recall_on_free_cells": None}
    sources = np.zeros(world.grid.shape[:2], bool)
    cell = world.grid.index((*source_xy, 0.875))[:2]
    sources[tuple(cell)] = True
    truth_walls = world.occupied[:, :, 1:-1].any(-1)
    times = np.arange(config.total_steps + 1) * config.dt + age + config.latency
    inflation = (
        config.robot_radius
        + config.hidden_radius
        + world.grid.cell_size * math.sqrt(2)
    )
    actual = reachable_risk(
        truth_walls,
        sources,
        speed=hidden_speed,
        times=times,
        cell_size=world.grid.cell_size,
        inflation=inflation,
    )
    relevant = actual & free[None]
    denominator = int(relevant.sum())
    return {
        "designated_hidden_actor": True,
        "reachable_free_cell_times": denominator,
        "risk_recall_on_free_cells": float(
            (predicted & relevant).sum() / denominator
        )
        if denominator
        else None,
        "extra_risk_free_cell_times_for_this_actor": int(
            (predicted & ~actual & free[None]).sum()
        ),
    }


def run_suite(
    *,
    seed=17,
    scene_names=("corner", "door", "rear_poles", "overhang"),
    budgets=(12, 24),
    robot_speeds=(0.6, 1.6),
    observation_ages=(0.0, 0.4),
):
    """Fixed-seed snapshot/parameter sweep, deliberately not navigation episodes."""
    rng = np.random.default_rng(seed)
    cases = []
    hidden_speeds = (0.0, 0.15, 0.3)
    for scene in scene_names:
        world, xy, goal, actor = _scene(scene)
        acquisition_start = time.perf_counter()
        acquired = world.far_scan(xy, 180, (0.375, 0.875, 1.375), 6.0, 0)
        far_acquisition_ms = (time.perf_counter() - acquisition_start) * 1000
        # Fixed missing-beam pattern shared by every method and parameter row.
        missing = rng.random(len(acquired)) < 0.03
        raw = [
            Beam(b.origin, b.endpoint, b.hit, not missing[i], b.stamp)
            for i, b in enumerate(acquired)
        ]
        raw_digest = hashlib.sha256(repr(raw).encode()).hexdigest()
        for speed in robot_speeds:
            for age in observation_ages:
                config = PlanConfig(max_speed=speed)
                radius = near_radius(speed, latency=config.latency + age)
                acquisition_start = time.perf_counter()
                near = world.near_scan(xy, radius, 0)
                near_acquisition_ms = (
                    time.perf_counter() - acquisition_start
                ) * 1000
                state = (*xy, 0.0, speed)
                for budget in budgets:
                    methods = {}
                    for adaptive in (False, True):
                        start = time.perf_counter()
                        projection = compress_far(
                            raw,
                            budget,
                            adaptive=adaptive,
                            heading=0,
                            speed=speed,
                            yaw_rate=0.5,
                        )
                        compression_time = time.perf_counter() - start
                        start = time.perf_counter()
                        memory = Evidence(world.grid)
                        memory.update(near + projection.beams)
                        map_time = time.perf_counter() - start
                        start = time.perf_counter()
                        free, risks = risk_from_evidence(
                            memory, age, config, hidden_speeds
                        )
                        risk_time = time.perf_counter() - start
                        planners = {}
                        for planner, scenarios in (
                            ("visible_only", np.zeros_like(risks[:1])),
                            ("single_worst", risks[-1:]),
                            ("shared_prefix", risks),
                        ):
                            start = time.perf_counter()
                            result = plan_candidates(
                                world.grid, free, scenarios, state, goal, config
                            )
                            planners[planner] = _plan_metrics(
                                result, state, goal, time.perf_counter() - start
                            )
                        name = "adaptive" if adaptive else "uniform"
                        methods[name] = {
                            **_representation_metrics(raw, projection),
                            "angular_bins": budget,
                            "height_layers": 3,
                            "raw_far_rays": len(raw),
                            "raw_ray_digest": raw_digest,
                            "far_scan_fixture_ms": far_acquisition_ms,
                            "retained_far_rays": len(projection.beams),
                            "near_rays": len(near),
                            "near_scan_fixture_ms": near_acquisition_ms,
                            "near_radius_m": radius,
                            "near_voxel_size_m": world.grid.cell_size,
                            "observed_voxels": int(memory.observed.sum()),
                            "measured_free_body_columns": int(free.sum()),
                            "false_free_voxels": int(
                                (
                                    memory.free(age, config.max_age) & world.occupied
                                ).sum()
                            ),
                            "observed_overhead_voxels": int(
                                memory.occupied[:, :, 6].sum()
                            ),
                            "risk_cell_counts_final_by_speed": risks[:, -1]
                            .sum(axis=(1, 2))
                            .tolist(),
                            "compression_ms": compression_time * 1000,
                            "mapping_ms": map_time * 1000,
                            "risk_ms": risk_time * 1000,
                            "planners": planners,
                            "evaluation_only": _designated_actor_coverage(
                                world,
                                actor,
                                xy,
                                free,
                                risks[-1],
                                config,
                                age,
                                hidden_speeds[-1],
                            ),
                        }
                    cases.append(
                        {
                            "scene": scene,
                            "robot_speed_mps": speed,
                            "observation_age_s": age,
                            "angular_bins": budget,
                            "methods": methods,
                        }
                    )
    summary = {
        "paired_cases": len(cases),
        "independent_geometry_scenes": len(scene_names),
        "unique_compression_settings": (
            len(scene_names) * len(budgets) * len(robot_speeds)
        ),
        "shared_vs_single_first_action_changes": {"uniform": 0, "adaptive": 0},
        "adaptive_lower_range_mae": 0,
        "adaptive_higher_range_mae": 0,
        "adaptive_lower_edge_error": 0,
        "adaptive_higher_edge_error": 0,
        "false_free_voxels": 0,
        "feasible_counts": {
            name: {p: 0 for p in ("visible_only", "single_worst", "shared_prefix")}
            for name in ("uniform", "adaptive")
        },
    }
    counterexamples = []
    for case in cases:
        u, a = case["methods"]["uniform"], case["methods"]["adaptive"]
        for key, lower, higher in (
            (
                "display_range_mae_m",
                "adaptive_lower_range_mae",
                "adaptive_higher_range_mae",
            ),
            (
                "occlusion_edge_bin_error_rad",
                "adaptive_lower_edge_error",
                "adaptive_higher_edge_error",
            ),
        ):
            if u[key] is not None and a[key] is not None:
                summary[lower] += int(a[key] < u[key] - 1e-10)
                summary[higher] += int(a[key] > u[key] + 1e-10)
        for name, values in case["methods"].items():
            summary["false_free_voxels"] += values["false_free_voxels"]
            for planner, result in values["planners"].items():
                summary["feasible_counts"][name][planner] += int(result["feasible"])
            single = values["planners"]["single_worst"]
            multi = values["planners"]["shared_prefix"]
            if single["feasible"] and multi["feasible"]:
                summary["shared_vs_single_first_action_changes"][name] += int(
                    single["action_target_v_omega"] != multi["action_target_v_omega"]
                )
        if a["display_range_mae_m"] > u["display_range_mae_m"] + 1e-10:
            counterexamples.append(
                {
                    k: case[k]
                    for k in (
                        "scene",
                        "robot_speed_mps",
                        "observation_age_s",
                        "angular_bins",
                    )
                }
                | {
                    "finding": (
                        "Adaptive binning increases full-circle display "
                        "range reconstruction error."
                    ),
                    "uniform_mae_m": u["display_range_mae_m"],
                    "adaptive_mae_m": a["display_range_mae_m"],
                }
            )
    centers, radii = published_risk_disks([(3, 0)], [0.5], (0, 0), 1.0, 0.3)
    _, stationary = published_risk_disks([(3, 0)], [0.5], (0, 0), 0.0, 0.3)
    return {
        "claim_scope": "finite_grid_probe_not_cmpc_or_g1",
        "seed": seed,
        "provenance": {
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "experiment_type": (
            "paired fixed-observation snapshots, not closed-loop episodes"
        ),
        "defaults": asdict(PlanConfig()),
        "hidden_speed_bounds_mps": list(hidden_speeds),
        "fairness": (
            "Identical raw scan and near scan within each pair; identical "
            "angular-bin budget, height layers, candidate library and cost. "
            "Adaptive compression "
            "uses current raw scan edges; it saves representation entries, "
            "not raw acquisition work."
        ),
        "risk_model": (
            "Unknown/free frontiers, per-cell observation age, 4-neighbor Dijkstra "
            "through unknown cells and around observed walls. No ground-truth walls "
            "in planning. No probabilities, ADMM, recursive feasibility, "
            "or continuous-space guarantee. Center reachability respects walls; "
            "Euclidean collision-volume inflation can conservatively mark cells "
            "on the far side of thin walls."
        ),
        "planning_baselines": {
            "visible_only": (
                "Same candidate library and measured-free geometry; "
                "no future hidden-agent risk."
            ),
            "single_worst": (
                "Same library, one maximum-speed reachable-risk configuration."
            ),
            "shared_prefix": (
                "Three speed configurations, equal-weight sum of best suffix costs, "
                "a 0.4 s common control prefix, and checked braking tails. "
                "Only the first target control may execute."
            ),
        },
        "nested_branch_interpretation": (
            "The three speed-bound risk sets are nested. With the same finite "
            "candidate library, every feasible worst-case suffix can serve all "
            "branches, so shared-prefix feasibility must equal single-worst "
            "feasibility. This experiment tests consensus consistency and cost "
            "selection, not a larger safety-feasible set."
        ),
        "sampling_interpretation": (
            f"The {len(cases)} rows are parameter combinations over "
            f"{len(scene_names)} geometric scenes, not independent scenarios "
            "or success-rate trials. Age changes "
            "planning/freshness but repeats the same compression evidence."
        ),
        "published_formula_reference": {
            "source": "https://arxiv.org/html/2503.04563v5",
            "scope": (
                "Equations 7/8 only, canonical visible disk fixture; "
                "not a full CMPC performance baseline."
            ),
            "risk_centers_m": centers.tolist(),
            "risk_radii_m": radii.tolist(),
            "stationary_radii_m": stationary.tolist(),
            "caution": (
                "The published speed denominator becomes very conservative near "
                "zero speed; no silent denominator modification was made."
            ),
        },
        "summary": summary,
        "counterexamples": counterexamples,
        "limitations": [
            "No G1/MuJoCo/CAT policy integration, learned locomotion, "
            "or closed-loop collision/success statistics.",
            "The near grid uses 0.25 m voxels: fine only relative to far angular "
            "compression. It supports no centimeter-level foot placement or "
            "body-clearance claims; a finer grid and full body collision envelope "
            "are required before integration.",
            "Navigation uses an upright column body and excludes the lowest ground "
            "layer. It does not model feet, articulated legs, "
            "stepping or crouching.",
            "Zero-width rays certify uniform finite voxels only, "
            "never angular sectors or unsampled height gaps.",
            "Near geometry is an ideal dense center-targeted 3-D scanner; "
            "its ray count grows with radius and its runtime is included.",
            "Unknown is not safe ego space; conservative beam/height gaps "
            "can make every candidate infeasible.",
            "All hypothetical agents follow the declared 4-neighbor grid. "
            "Continuous diagonal motion can be faster "
            "and is outside this guarantee.",
            "No source beyond the finite map boundary; "
            "world extent is an explicit closed test domain.",
            "All surfaces in the mapping fixture are static. "
            "Stale occupied cells stay blocked; stale free cells become unknown.",
            "Branch score weights are heuristics, not probabilities. "
            "A feasible finite horizon does not establish recursive safety.",
            "Display reconstruction error is a compression diagnostic; "
            "the planner uses retained rays, never fills a bin's whole wedge.",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/occlusion-probe-2026-10-19.json"),
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = run_suite(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "summary": report["summary"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
