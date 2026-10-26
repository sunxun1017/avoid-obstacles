"""Explicit reward weighting must reach the task without changing its inputs."""

import math
from types import SimpleNamespace

import pytest
import torch
from test_cat_review_regressions import task_case

from embodied_learning.tasks.cat_reward import SOURCE_SCALES
from embodied_learning.tasks.cat_traversal import CatTaskState


def test_explicit_tracking_weight_changes_only_the_executed_tracking_reward():
    fixture, controller, payload, _ = task_case()
    options = {**fixture.options, "ground": 0.0, "lateral": 0.0, "overhead": 0.0}
    source = CatTaskState(
        fixture.manifest, fixture.bank, fixture.allowed, 16, "cpu", options
    )
    weighted = CatTaskState(
        source.manifest,
        source.bank,
        source.allowed,
        16,
        "cpu",
        {**options, "root_tracking_weight": 4.0},
    )
    source.reset(torch.arange(16))
    weighted.reset(torch.arange(16))
    x = payload["reward_inputs"]
    x.pelvis_rpy.zero_()
    x.torso_rpy.zero_()
    x.torso_ang_vel.zero_()
    x.joint_vel.zero_()
    x.torques.zero_()
    x.joint_pos.fill_(0.2)
    x.pelvis_subtree_com.zero_()
    x.foot_pos[:] = torch.tensor([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    # Use identical phases/RNG and a real source reward input fixture.
    weighted.gait.phase.copy_(source.gait.phase)
    weighted.gait.phase_dt.copy_(source.gait.phase_dt)
    for state in (source, weighted):
        torch.manual_seed(123)
        state.refresh(payload, controller, torch.ones(16, dtype=torch.long), 1)
    for key in source.terms:
        torch.testing.assert_close(weighted.terms[key], source.terms[key])
    for key in source.scales:
        if key != "tracking_root_field":
            assert weighted.scales[key] == source.scales[key]
    # Choose slots without clipping: the extra reward is 3 * tracking * dt.
    active = source.reward() > 0
    assert active.any()
    torch.testing.assert_close(
        (weighted.reward() - source.reward())[active],
        (0.06 * source.terms["tracking_root_field"])[active],
    )
    for key in ("actor", "critic"):
        torch.testing.assert_close(
            weighted.observations()[key], source.observations()[key]
        )
    for key in source.outcomes():
        assert torch.equal(weighted.outcomes()[key], source.outcomes()[key])
    assert SOURCE_SCALES["tracking_root_field"] == 1.0


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, True])
def test_tracking_weight_rejects_nonpositive_or_nonfinite_values(value):
    with pytest.raises(ValueError, match="tracking weight"):
        CatTaskState(
            SimpleNamespace(),
            SimpleNamespace(source_profile={"version": "cat_source_v1"}),
            (0,),
            1,
            "cpu",
            {"root_tracking_weight": value},
        )


def test_default_task_does_not_implicitly_change_its_reward_contract():
    state, _, _, _ = task_case()
    assert "root_tracking_weight" not in state.options
    assert state.scales["tracking_root_field"] == 1.0
