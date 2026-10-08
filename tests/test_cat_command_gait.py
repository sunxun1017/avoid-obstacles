from pathlib import Path

import numpy as np
import torch

from embodied_learning.tasks.cat_command import compute_cat_command
from embodied_learning.tasks.cat_gait import CatGaitState


def test_cat_command_matches_author():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    actual = compute_cat_command(
        torch.from_numpy(x["input_gf"][:, 1]),
        torch.from_numpy(x["input_gf"][:, [0, 3, 4, 5, 6]]),
        torch.from_numpy(x["input_bf"][:, [0, 3, 4, 5, 6]]),
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(x["command"]), rtol=1e-5, atol=1e-6
    )


def test_move_flag_is_source_075():
    g = torch.tensor([[1.0, 0, 0]])
    cmd = compute_cat_command(g, torch.zeros(1, 5, 3), torch.zeros(1, 5, 3))
    torch.testing.assert_close(cmd, torch.tensor([[0.75, 0.525, 0, 0]]))


def test_phase_stop_transition_matches_author_sequence():
    x = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    state = CatGaitState(1, "cpu")
    state.reset(torch.tensor([0]), torch.tensor([1.4]), torch.tensor([[0.0, np.pi]]))
    for i, command in enumerate(x["gait_commands"]):
        result = state.advance(torch.from_numpy(command)[None])
        torch.testing.assert_close(
            result[0], torch.from_numpy(x["gait_result"][i]), atol=1e-6, rtol=1e-5
        )
        torch.testing.assert_close(
            state.phase[0],
            torch.from_numpy(x["gait_phase"][i]),
            atol=3e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            state.gait_mask[0], torch.from_numpy(x["gait_mask"][i])
        )


def test_slot_reset_preserves_other_slots():
    state = CatGaitState(2, "cpu")
    state.reset(
        torch.tensor([0, 1]),
        torch.tensor([1.3, 1.5]),
        torch.tensor([[0.0, np.pi], [np.pi, 0.0]]),
    )
    state.advance(torch.tensor([[0.75, 0.4, 0, 0], [0.75, 0.4, 0, 0]]))
    before = state.phase[1].clone()
    state.reset(torch.tensor([0]), torch.tensor([1.4]), torch.zeros(1, 2))
    torch.testing.assert_close(state.phase[1], before)
