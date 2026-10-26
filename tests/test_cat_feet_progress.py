"""The experimental feet reward must require net body progress."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodied_learning.tasks.cat_reward import (
    CatRewardInputs,
    CatRewardParameters,
    cat_reward_terms,
)
from embodied_learning.tasks.cat_traversal import CatTaskState

PROFILE = "root_displacement_0p5s_v1"


def reward_case():
    f = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    x = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(f[k]).clone()
            for k in f.files
            if k.startswith("input_")
        }
    )
    x.move_flag.fill_(1)
    x.cmd_vel[:] = torch.tensor([0.3, 0.0, 0.0])
    x.gf[:, 3:5] = torch.tensor([0.4, 0.0, 0.9])
    x.sdf[:, 3:5] = 0.15
    x.gait_mask.fill_(-1)
    x.body_velocities[:, 3:5] = torch.tensor([0.0, 0.0, 0.1])
    p = CatRewardParameters(
        torch.from_numpy(f["soft_lower"]), torch.from_numpy(f["soft_upper"])
    )
    return x, p


def test_vertical_foot_motion_without_body_progress_gets_no_positive_feetgf():
    x, p = reward_case()
    old = cat_reward_terms(x, parameters=p)
    assert (old["feetgf"] > 3).all()
    new = cat_reward_terms(x, parameters=p, feet_progress_gate=torch.zeros(16))
    torch.testing.assert_close(new["feetgf"], torch.zeros(16))
    for key in old.keys() - {"feetgf"}:
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)


def test_full_progress_preserves_original_reward_and_negative_values_survive():
    x, p = reward_case()
    old = cat_reward_terms(x, parameters=p)
    full = cat_reward_terms(x, parameters=p, feet_progress_gate=torch.ones(16))
    for key in old:
        torch.testing.assert_close(full[key], old[key], rtol=0, atol=0)
    x.body_velocities[:, 3:5, 2] = -0.1
    old = cat_reward_terms(x, parameters=p)
    assert (old["feetgf"] < 0).all()
    new = cat_reward_terms(x, parameters=p, feet_progress_gate=torch.zeros(16))
    torch.testing.assert_close(new["feetgf"], old["feetgf"])


def test_stop_reward_is_preserved_even_with_zero_progress():
    x, p = reward_case()
    x.cmd_vel.zero_()
    x.move_flag.zero_()
    old = cat_reward_terms(x, parameters=p)
    new = cat_reward_terms(x, parameters=p, feet_progress_gate=torch.zeros(16))
    torch.testing.assert_close(new["feetgf"], old["feetgf"])


@pytest.mark.parametrize(
    "gate",
    [torch.zeros(16, 1), torch.full((16,), -1.0), torch.full((16,), float("nan"))],
)
def test_invalid_progress_gate_is_rejected(gate):
    x, p = reward_case()
    with pytest.raises(ValueError, match="progress gate"):
        cat_reward_terms(x, parameters=p, feet_progress_gate=gate)


def test_net_progress_rejects_lateral_backward_and_full_cycle_wobble():
    from embodied_learning.tasks.cat_foot_progress import RootProgressWindow

    history = RootProgressWindow(4, "cpu")
    command = torch.tensor([[0.3, 0.0]]).repeat(4, 1)
    root = torch.zeros(4, 2)
    history.update(root, command, step=0, initial=torch.ones(4, dtype=torch.bool))
    for step in range(1, 26):
        root[0, 0] = step * 0.02 * 0.3
        root[1, 1] = step * 0.02 * 0.3
        root[2, 0] = -step * 0.02 * 0.3
        root[3, 0] = 0.02 * torch.sin(torch.tensor(step * 2 * torch.pi / 25))
        gate = history.update(
            root, command, step=step, initial=torch.zeros(4, dtype=torch.bool)
        )
    torch.testing.assert_close(
        gate, torch.tensor([1.0, 0.0, 0.0, 0.0]), rtol=1e-5, atol=1e-6
    )


def test_window_uses_current_command_direction_and_does_not_credit_teleport_reset():
    from embodied_learning.tasks.cat_foot_progress import RootProgressWindow

    history = RootProgressWindow(2, "cpu")
    cmd = torch.tensor([[0.0, 0.3], [0.3, 0.0]])
    pos = torch.zeros(2, 2)
    initial = torch.ones(2, dtype=torch.bool)
    history.update(pos, cmd, step=0, initial=initial)
    initial.zero_()
    for step in range(1, 31):
        pos += cmd * 0.02
        gate = history.update(pos, cmd, step=step, initial=initial)
    torch.testing.assert_close(gate, torch.ones(2), rtol=1e-5, atol=1e-6)
    # A duplicate read cannot advance the ring or credit new displacement.
    torch.testing.assert_close(
        history.update(pos + 10, cmd, step=30, initial=initial), gate
    )
    pos[0] = 100
    initial[0] = True
    reset = history.update(pos, cmd, step=30, initial=initial)
    assert reset[0] == 0 and reset[1] == gate[1]
    initial.zero_()
    next_gate = history.update(pos, cmd, step=31, initial=initial)
    assert next_gate[0] == 0


def test_task_opt_in_changes_only_feetgf_and_does_not_change_observations():
    from test_cat_review_regressions import task_case

    source, controller, payload, _ = task_case()
    options = {**source.options, "feetgf_progress_gate": PROFILE}
    adjusted = CatTaskState(
        source.manifest, source.bank, source.allowed, 16, "cpu", options
    )
    adjusted.reset(torch.arange(16))
    adjusted.gait.phase.copy_(source.gait.phase)
    adjusted.gait.phase_dt.copy_(source.gait.phase_dt)
    for step in range(3):
        for state in (source, adjusted):
            torch.manual_seed(123)
            state.refresh(payload, controller, torch.full((16,), step), step)
    positive = source.terms["feetgf"] > 0
    moving = source.gait.last_command[:, 1:].norm(dim=-1) > 1e-6
    assert (positive & moving).any()
    assert (adjusted.terms["feetgf"][positive & moving] == 0).all()
    for key in source.terms.keys() - {"feetgf"}:
        torch.testing.assert_close(source.terms[key], adjusted.terms[key])
    for group in ("actor", "critic"):
        torch.testing.assert_close(
            source.observations()[group], adjusted.observations()[group]
        )
    assert source.scales == adjusted.scales


def test_unknown_progress_profile_is_rejected():
    with pytest.raises(ValueError, match="progress"):
        CatTaskState(
            SimpleNamespace(),
            SimpleNamespace(source_profile={"version": "cat_source_v1"}),
            (0,),
            1,
            "cpu",
            {"feetgf_progress_gate": "typo"},
        )
