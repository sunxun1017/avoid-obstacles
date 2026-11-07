"""Verify the Unitree Isaac Lab deployment history and motor mapping."""
import torch

from embodied_learning.policies.cat_unitree_isaac import IsaacLabVelocityHistory, native_joint_values


def test_history_is_term_major_and_resets_each_slot_independently():
    history = IsaacLabVelocityHistory(2, "cpu")
    reset = torch.tensor([True, True])
    first = torch.arange(96, dtype=torch.float32)[None].expand(2, -1)
    widths = (3, 3, 3, 29, 29, 29)

    def expected(frames):
        return torch.cat([part.flatten(1) for part in frames.split(widths, dim=-1)], dim=-1)

    torch.testing.assert_close(history.push(first, reset), expected(first[:, None].expand(-1, 5, -1)))
    second = first + 100
    actual = history.push(second, torch.tensor([False, True]))
    chronological = torch.stack((first, first, first, first, second), dim=1)
    chronological[1] = second[1]
    torch.testing.assert_close(actual, expected(chronological))


def test_policy_to_native_motor_mapping_round_trips_last_applied_action():
    indexes = torch.arange(28, -1, -1)
    values = torch.arange(58, dtype=torch.float32).reshape(2, 29)
    native = native_joint_values(values, indexes)
    torch.testing.assert_close(native[:, indexes], values)
