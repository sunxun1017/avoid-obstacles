"""Isolate interpolation-weight correction from clamping and cached geometry."""

from dataclasses import replace

import pytest
import torch

from embodied_learning.perception.voxel_field import VoxelPotentialField
from embodied_learning.tasks.cat_foot_guidance import FootGuidanceBank
from embodied_learning.tasks.field_assets import FieldBank


def linear_bank():
    xyz = torch.stack(
        torch.meshgrid(*[torch.arange(5) * 0.04] * 3, indexing="ij"), -1
    )
    scalar = 2 * xyz[..., 0] + 3 * xyz[..., 1] + 5 * xyz[..., 2]
    valid = torch.ones_like(scalar, dtype=torch.bool)
    field = VoxelPotentialField(
        scalar, scalar, xyz, xyz, xyz, valid, valid, ~valid, (0.0, 0.0, 0.0), 0.04
    )
    return FieldBank(
        ((field,),), "scene", ("scene",), {"version": "cat_source_v1"}, "cache"
    )


def test_corrected_bank_reproduces_linear_fields_without_changing_grid():
    bank = linear_bank()
    bank = replace(bank, source_interpolation="xyz_weights_v1")
    original = bank.fields[0][0].guidance.clone()
    points = torch.tensor([[[0.008, 0.012, 0.028], [0.091, 0.063, 0.074]]])
    ids = torch.zeros(1, dtype=torch.long)
    result = bank.sample_source(points, ids, ids)
    torch.testing.assert_close(result.guidance, points)
    torch.testing.assert_close(
        result.sdf, (points * torch.tensor([2.0, 3.0, 5.0])).sum(-1)
    )
    assert torch.equal(bank.fields[0][0].guidance, original)


def test_default_retains_legacy_and_correction_retains_source_boundary_clipping():
    bank = linear_bank()
    ids = torch.zeros(1, dtype=torch.long)
    point = torch.tensor([[[0.008, 0.012, 0.028]]])
    torch.testing.assert_close(
        bank.sample_source(point, ids, ids).guidance,
        torch.tensor([[[0.028, 0.012, 0.008]]]),
    )
    fixed = replace(bank, source_interpolation="xyz_weights_v1")
    outside = torch.tensor([[[-1.0, 0.2, 0.4]]])
    torch.testing.assert_close(
        fixed.sample_source(outside, ids, ids).guidance,
        torch.tensor([[[0.0, 0.12, 0.12]]]),
    )
    with pytest.raises(ValueError, match="interpolation"):
        replace(bank, source_interpolation="typo").sample_source(point, ids, ids)


def test_foot_override_preserves_corrected_sampling_for_all_eleven_sites():
    base = replace(linear_bank(), source_interpolation="xyz_weights_v1")
    field = base.fields[0][0]
    bank = FootGuidanceBank(
        base, ((field.guidance * 2,),), radius=0.3, manifest_identity="foot"
    )
    points = torch.tensor([[[0.008, 0.012, 0.028]] * 11])
    ids = torch.zeros(1, dtype=torch.long)
    result = bank.sample_source(points, ids, ids)
    expected = points.clone()
    expected[:, 3:5] *= 2
    torch.testing.assert_close(result.guidance, expected)
    torch.testing.assert_close(result.sdf, base.sample_source(points, ids, ids).sdf)
