"""Reward regressions against independently executed author functions."""

from pathlib import Path

import numpy as np
import pytest
import torch

from embodied_learning.tasks.cat_reward import (
    CatRewardInputs,
    CatRewardParameters,
    aggregate_cat_reward,
    cat_reward_terms,
)


def test_cat_reward_matches_author_fixture():
    fixture = np.load(Path(__file__).parent / "fixtures/cat_source_reference.npz")
    inputs = CatRewardInputs(
        **{
            k[6:]: torch.from_numpy(fixture[k])
            for k in fixture.files
            if k.startswith("input_")
        }
    )
    params = CatRewardParameters(
        soft_lower=torch.from_numpy(fixture["soft_lower"]),
        soft_upper=torch.from_numpy(fixture["soft_upper"]),
    )
    actual = cat_reward_terms(inputs, parameters=params)
    for name, value in actual.items():
        torch.testing.assert_close(
            value, torch.from_numpy(fixture["reward_" + name]), rtol=1e-5, atol=1e-6
        )


def test_cat_reward_source_weights_and_single_dt():
    terms = {"a": torch.tensor([2.0, -2.0, 1e9])}
    torch.testing.assert_close(
        aggregate_cat_reward(terms, {"a": 1.0}, dt=0.02),
        torch.tensor([0.04, 0.0, 10000.0]),
    )
    with pytest.raises(ValueError):
        aggregate_cat_reward(terms, {"wrong": 1}, dt=0.02)


def test_cat_field_masks_and_nan_handling():
    terms = {"a": torch.tensor([float("nan"), 3.0])}
    torch.testing.assert_close(
        aggregate_cat_reward(terms, {"a": 1}, dt=0.02), torch.tensor([0.0, 0.06])
    )
