"""Observed offline navigation failures must not turn into root commands."""

import pytest
import torch

from embodied_learning.tasks.cat_command import corrected_navigation_command


def command(
    d=0.083, clearance=float("inf"), field=(-0.00002458, -0.00001987, 0.00189)
):
    root = torch.tensor([[1.5 - d, 0.0, 0.75]])
    goal = torch.tensor([[1.5, 0.0, 0.8]])
    return corrected_navigation_command(
        torch.tensor([field]),
        torch.zeros(1, 5, 3),
        torch.zeros(1, 5, 3),
        root,
        goal,
        torch.full((1, 11), clearance),
    )


def test_logged_weak_reverse_field_uses_root_goal_direction_in_clear_space():
    c = command()
    torch.testing.assert_close(c[:, 1:3], torch.tensor([[1.05 * 0.083, 0.0]]))


def test_exact_goal_and_existing_stop_radius_are_finite_and_stopped():
    for d in (0.0, 0.04, 0.079):
        assert command(d).count_nonzero() == 0


def test_near_obstacle_preserves_projected_root_field_without_amplification():
    c = command(clearance=0.3)
    torch.testing.assert_close(
        c[0, 1:3], torch.tensor([-0.00002458, -0.00001987]) * 0.525
    )


def test_distance_and_clearance_blends_are_continuous():
    for d in (0.25, 0.35):
        assert (command(d - 1e-6) - command(d + 1e-6)).abs().max() < 1e-5
    for clearance in (0.3, 0.5):
        assert (
            command(clearance=clearance - 1e-6) - command(clearance=clearance + 1e-6)
        ).abs().max() < 1e-5


def test_far_field_retains_direction_and_magnitude():
    c = command(d=0.8, field=(0.2, 0.1, 0.0))
    torch.testing.assert_close(c[0, 1:3], torch.tensor([0.105, 0.0525]))


def test_navigation_transfer_declares_only_allowed_changes_and_keeps_robot(tmp_path):
    import copy

    from test_cat_curriculum import fixture_contract, fixture_metadata

    from embodied_learning.tasks.cat_curriculum import validate_transfer
    from embodied_learning.tasks.cat_foot_guidance import sha256
    from embodied_learning.tasks.cat_navigation_transfer import (
        adapt_navigation_transfer,
        navigation_changes,
    )

    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")
    import hashlib
    import json

    proof_before = {
        "scenes": [
            {
                "id": "hurdle_crouch",
                "boxes": [{"center": [0, 0, 0]}],
                "starts": [[-1, 0, 0.8]],
                "goals": [[1, 0, 0.8]],
            }
        ]
    }
    proof_after = copy.deepcopy(proof_before)
    proof_after["scenes"][0]["goals"].append([1, 0.3, 0.8])

    def scene_hash(v):
        return hashlib.sha256(
            json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    old = fixture_contract(4)
    old["schema"]["scene_hash"] = scene_hash(proof_before)
    old["schema"]["task"]["arrival_control"] = "root_linear_v1"
    old["schema"]["reward_smoothness_action_weight"] = -0.005
    old["schema"]["reward_smoothness_joint_weight"] = -5e-6
    new = copy.deepcopy(old)
    new["schema"]["scene_hash"] = scene_hash(proof_after)
    new["schema"]["task"]["arrival_control"] = "root_planar_blend_v2"
    new["schema"]["reward_smoothness_action_weight"] = -0.001
    new["schema"]["reward_smoothness_joint_weight"] = -1e-6
    src = fixture_metadata(4)
    dst = {**src, "experiment": "new", "protocol_sha256": "new"}
    declaration = dict(
        version="cat_navigation_actor_transfer_v1",
        parent_sha256=sha256(parent),
        parent_experiment=src["experiment"],
        parent_protocol_sha256=src["protocol_sha256"],
        changes=navigation_changes(old, new),
        scene_pool_proof=dict(before=proof_before, after=proof_after),
    )
    dst["navigation_refinement"] = declaration
    adapted, meta, changes = adapt_navigation_transfer(parent, old, new, src, dst)
    validate_transfer(adapted, new, meta, dst, actor_warmstart=True)
    assert old["schema"]["task"]["arrival_control"] == "root_linear_v1"
    assert changes["task.arrival_control"]["after"] == "root_planar_blend_v2"
    for key in ("parent_sha256", "parent_protocol_sha256"):
        bad = copy.deepcopy(dst)
        bad["navigation_refinement"][key] = "wrong"
        with pytest.raises(ValueError):
            adapt_navigation_transfer(parent, old, new, src, bad)
    altered = copy.deepcopy(new)
    altered_proof = copy.deepcopy(proof_after)
    altered_proof["scenes"][0]["boxes"][0]["center"] = [0, 0, 99]
    altered["schema"]["scene_hash"] = scene_hash(altered_proof)
    malicious = copy.deepcopy(dst)
    malicious["navigation_refinement"]["changes"] = navigation_changes(old, altered)
    malicious["navigation_refinement"]["scene_pool_proof"]["after"] = altered_proof
    with pytest.raises(ValueError):
        adapt_navigation_transfer(parent, old, altered, src, malicious)
    bad = copy.deepcopy(dst)
    bad["robot_fingerprint"] = "different"
    with pytest.raises(ValueError):
        adapt_navigation_transfer(parent, old, new, src, bad)
    bad = copy.deepcopy(new)
    bad["schema"]["task"]["noise_level"] = 0
    adapted, meta, _ = adapt_navigation_transfer(parent, old, bad, src, dst)
    with pytest.raises(ValueError):
        validate_transfer(adapted, bad, meta, dst, actor_warmstart=True)


def test_native_refresh_queries_root_and_keeps_legacy_body_query():
    from types import SimpleNamespace

    from test_cat_review_regressions import task_case

    class Base:
        def __init__(self):
            self.queries = []

        def sample_source(self, p, scene, goal):
            assert p.shape[1] == 1
            self.queries.append(p.clone())
            gf = torch.zeros_like(p)
            gf[..., 0] = 0.6
            return SimpleNamespace(guidance=gf)

    class FootBank:
        def __init__(self):
            self.base = Base()

        def sample_source(self, p, scene, goal):
            assert p.shape[1] == 11
            gf = torch.zeros_like(p)
            gf[..., 0] = -0.6
            return SimpleNamespace(
                guidance=gf,
                sdf_gradient=torch.zeros_like(p),
                sdf=torch.full(p.shape[:2], float("inf")),
            )

    for mode, sign in [("root_planar_blend_v2", 1), ("root_linear_v1", -1)]:
        st, c, p, _ = task_case()
        st.bank = FootBank()
        st.arrival_control = mode
        st.command_deadband = 0.01
        st.goals[:] = torch.tensor([0.083, 0.0, 0.8])
        p["body_positions"][:, 1, 0] = 0.15
        for step in (0, 1):
            st.refresh(p, c, torch.full((16,), step, dtype=torch.long), step)
            assert (st.gait.last_command[:, 1] * sign > 0).all()
        if mode == "root_planar_blend_v2":
            assert len(st.bank.base.queries) == 2
            torch.testing.assert_close(
                st.bank.base.queries[-1][:, 0], p["root_position"]
            )
        else:
            assert not st.bank.base.queries
