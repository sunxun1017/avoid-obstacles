"""Behavioral contracts for the isolated, finite-grid research probe."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).parents[1] / "tools/experiments/occlusion_probe.py"
    assert path.exists(), "The experimental occlusion probe is not implemented."
    spec = importlib.util.spec_from_file_location("occlusion_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_and_no_return_have_distinct_meanings(probe):
    grid = probe.Grid((8, 5, 3), 0.5, (0, 0, 0))
    memory = probe.Evidence(grid)
    invalid = probe.Beam((0.25, 1.25, 0.75), (3.75, 1.25, 0.75), False, False, 0)
    memory.update([invalid])
    assert not memory.observed.any()
    clear = probe.Beam(invalid.origin, invalid.endpoint, False, True, 1)
    memory.update([clear])
    assert memory.observed[:, 2, 1].all()
    assert not memory.occupied.any()
    assert not memory.observed[:, 1, 1].any(), "Beam gaps cannot become free."
    compressed = probe.compress_far([clear], 8, adaptive=False)
    assert np.isnan(compressed.occlusion_start).all()
    assert compressed.unknown.sum() == 7


def test_hit_never_clears_behind_the_first_surface(probe):
    grid = probe.Grid((8, 5, 3), 0.5, (0, 0, 0))
    memory = probe.Evidence(grid)
    memory.update(
        [probe.Beam((0.25, 1.25, 0.75), (2.25, 1.25, 0.75), True, True, 0)]
    )
    assert memory.occupied[4, 2, 1]
    assert not memory.observed[5:, 2, 1].any()
    assert not memory.free(0, 1)[5:, 2, 1].any()


def test_local_new_scan_does_not_refresh_old_cells(probe):
    grid = probe.Grid((8, 5, 3), 0.5, (0, 0, 0))
    memory = probe.Evidence(grid)
    memory.update(
        [probe.Beam((0.25, 0.25, 0.75), (3.75, 0.25, 0.75), False, True, 0)]
    )
    memory.update(
        [probe.Beam((0.25, 2.25, 0.75), (3.75, 2.25, 0.75), False, True, 2)]
    )
    assert memory.age(2)[4, 0, 1] == 2
    assert memory.age(2)[4, 4, 1] == 0
    assert not memory.free(2, 0.5)[4, 0, 1]


def test_stopping_radius_keeps_resolution_and_grows_with_speed_latency(probe):
    base = probe.near_radius(0.3, latency=0.1, minimum=0)
    assert probe.near_radius(1.0, latency=0.1, minimum=0) > base
    assert probe.near_radius(0.3, latency=0.5, minimum=0) > base
    assert probe.near_radius(0.3, latency=0.1, minimum=0, braking=0.4) > base


def test_far_budget_and_full_circle_are_preserved(probe):
    beams = []
    for angle in np.linspace(-np.pi, np.pi, 96, endpoint=False):
        distance = 2 if 0.3 < angle < 0.7 else 5
        beams.append(
            probe.Beam(
                (0, 0, 0.8),
                (distance * np.cos(angle), distance * np.sin(angle), 0.8),
                distance == 2,
                True,
                0,
            )
        )
    uniform = probe.compress_far(beams, 12, adaptive=False)
    adaptive = probe.compress_far(beams, 12, adaptive=True, speed=0.8, yaw_rate=0.5)
    assert uniform.ranges.shape == adaptive.ranges.shape == (1, 12)
    assert len(uniform.beams) <= 12 and len(adaptive.beams) <= 12
    assert adaptive.edges[0] == -np.pi and adaptive.edges[-1] == np.pi
    assert np.all(np.diff(adaptive.edges) > 0)
    assert not np.allclose(np.diff(adaptive.edges), np.diff(uniform.edges))


def test_risk_cannot_cross_a_known_wall_and_unknown_wall_is_not_used(probe):
    walls = np.zeros((9, 9), bool)
    walls[4, :] = True
    sources = np.zeros_like(walls)
    sources[1, 4] = True
    blocked = probe.reachable_risk(walls, sources, speed=1, times=[20], cell_size=1)
    assert not blocked[0, 7, 4]
    unknown_wall = probe.reachable_risk(
        np.zeros_like(walls), sources, speed=1, times=[20], cell_size=1
    )
    assert unknown_wall[0, 7, 4]


def test_risk_takes_a_wall_route_instead_of_euclidean_shortcut(probe):
    walls = np.zeros((9, 9), bool)
    walls[4, :7] = True
    sources = np.zeros_like(walls)
    sources[2, 3] = True
    risk = probe.reachable_risk(walls, sources, speed=1, times=[5, 15], cell_size=1)
    assert not risk[0, 6, 3]
    assert risk[1, 6, 3]


def test_risk_is_monotonic_in_speed_age_and_horizon(probe):
    walls = np.zeros((11, 11), bool)
    sources = np.zeros_like(walls)
    sources[5, 5] = True
    low = probe.reachable_risk(
        walls, sources, speed=0.3, times=[1, 3], cell_size=0.2
    )
    fast = probe.reachable_risk(
        walls, sources, speed=0.6, times=[1, 3], cell_size=0.2
    )
    old = probe.reachable_risk(
        walls, sources, speed=0.3, times=[1, 3], cell_size=0.2, age=2
    )
    assert np.all(~low | fast)
    assert np.all(~low | old)
    assert np.all(~low[0] | low[1])


def test_near_field_retains_overhead_geometry(probe):
    grid = probe.Grid((20, 20, 8), 0.25, (-2.5, -2.5, 0))
    solid = np.zeros(grid.shape, bool)
    solid[13, 10, 6] = True
    world = probe.VoxelWorld(grid, solid)
    memory = probe.Evidence(grid)
    memory.update(world.near_scan((0.125, 0.125), 1.5, 0))
    assert memory.occupied[13, 10, 6]
    assert not memory.occupied[13, 10, 2]


def test_multibranch_controls_have_identical_prefix(probe):
    grid = probe.Grid((40, 40, 3), 0.2, (-4, -4, 0))
    free = np.ones(grid.shape[:2], bool)
    config = probe.PlanConfig()
    risks = np.zeros((3, config.total_steps + 1, *free.shape), bool)
    result = probe.plan_candidates(grid, free, risks, (0, 0, 0, 0.4), (2, 0), config)
    assert result.feasible
    assert len(result.branches) == 3
    assert np.allclose(result.branches[:, : config.prefix_steps + 1], result.prefix)
    assert np.allclose(
        result.controls[:, : config.prefix_steps],
        result.controls[0, : config.prefix_steps],
    )
    assert np.allclose(result.branches[:, -1, 3], 0), (
        "Every branch needs a checked braking tail."
    )


def test_infeasible_means_no_fabricated_safe_action(probe):
    grid = probe.Grid((20, 20, 3), 0.2, (-2, -2, 0))
    config = probe.PlanConfig()
    free = np.zeros(grid.shape[:2], bool)
    risks = np.zeros((2, config.total_steps + 1, *free.shape), bool)
    result = probe.plan_candidates(grid, free, risks, (0, 0, 0, 0.5), (2, 0), config)
    assert not result.feasible
    assert result.action is None
    assert result.reason


def test_body_footprint_checks_subcell_position(probe):
    grid = probe.Grid((12, 12, 3), 0.25, (0, 0, 0))
    free = np.ones(grid.shape[:2], bool)
    free[5:, :] = False
    config = probe.PlanConfig(robot_radius=0.3)
    risks = np.zeros((1, config.total_steps + 1, *free.shape), bool)
    # Obstacle starts at x=1.25: a radius .3 body at .99 already overlaps it.
    result = probe.plan_candidates(
        grid, free, risks, (0.99, 1.5, 0, 0), (2, 1.5), config
    )
    assert not result.feasible


def test_braking_tail_must_actually_stop_an_overspeed_initial_state(probe):
    grid = probe.Grid((200, 200, 3), 1.0, (-100, -100, 0))
    free = np.ones(grid.shape[:2], bool)
    config = probe.PlanConfig()
    risks = np.zeros((1, config.total_steps + 1, *free.shape), bool)
    result = probe.plan_candidates(grid, free, risks, (0, 0, 0, 20), (60, 0), config)
    assert not result.feasible
    assert result.action is None


def test_hidden_world_change_cannot_change_pre_visibility_action(probe):
    grid = probe.Grid((32, 24, 8), 0.25, (-3, -3, 0))
    solid = np.zeros(grid.shape, bool)
    solid[18, :, :] = True
    hidden_variant = solid.copy()
    hidden_variant[23:27, 8:12, 1:7] = True
    worlds = [probe.VoxelWorld(grid, value) for value in (solid, hidden_variant)]
    observations = [
        world.far_scan((0.125, 0.125), 64, (0.375, 0.875, 1.375), 5, 0)
        + world.near_scan((0.125, 0.125), 2, 0)
        for world in worlds
    ]
    assert observations[0] == observations[1]
    outcomes = []
    for rays in observations:
        memory = probe.Evidence(grid)
        memory.update(rays)
        outcomes.append(
            probe.plan_from_evidence(
                memory, (0.125, 0.125, 0, 0.4), (2, 0.125), now=0
            )
        )
    assert outcomes[0].feasible == outcomes[1].feasible
    assert np.array_equal(outcomes[0].branches, outcomes[1].branches)
    assert outcomes[0].action == outcomes[1].action


def test_published_disk_reference_starts_at_tangent_not_sensor_origin(probe):
    centers, radii = probe.published_risk_disks(
        [(3.0, 0.0)], [0.5], (0.0, 0.0), 1.0, 0.0
    )
    assert centers.shape == (4, 2)
    assert np.allclose(np.linalg.norm(centers[[0, 2]], axis=1), np.sqrt(9 - 0.25))
    assert np.all(radii == 0.5)
    assert np.all(np.linalg.norm(centers, axis=1) > radii)


def test_report_keeps_equal_raw_scan_and_bin_budget(probe):
    report = probe.run_suite(
        seed=17,
        scene_names=("corner",),
        budgets=(12,),
        robot_speeds=(0.6,),
        observation_ages=(0.0,),
    )
    assert report["claim_scope"] == "finite_grid_probe_not_cmpc_or_g1"
    pair = report["cases"][0]
    uniform, adaptive = pair["methods"]["uniform"], pair["methods"]["adaptive"]
    assert uniform["raw_ray_digest"] == adaptive["raw_ray_digest"]
    assert uniform["angular_bins"] == adaptive["angular_bins"] == 12
    assert uniform["near_rays"] == adaptive["near_rays"]
    assert uniform["retained_far_rays"] <= 36
    assert adaptive["retained_far_rays"] <= 36
    assert set(uniform["planners"]) == {
        "visible_only",
        "single_worst",
        "shared_prefix",
    }
